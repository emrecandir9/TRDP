"""Train several networks at once (docs/model-method.md §6.13).

A spiking network is simulated one millisecond after another, and each millisecond is a few very
small operations, so one network leaves the GPU waiting on instruction launches. A stack holds K
networks of identical shape with their own weights and trains them in the same operations (batched
matrix products over a leading network dimension). Nothing is shared between the networks: each has
its own initial weights (built by `network.Network` with its own seed), its own trial draws and noise
stream (the same generators, seeded the same way, as `train.train`), its own readout and its own
loss; the summed loss gives each network exactly its own gradient, and Adam acts element by element.
The networks a stack produces are therefore those `train.train` produces, up to floating-point
rounding of the batched products (tested in tests/test_network.py).
"""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

import network as nw
import train as tr
import trials


class Stack(nn.Module):
    """K pooled networks of identical shape, parameters stacked along a leading network dimension."""

    def __init__(self, nets):
        super().__init__()
        first = nets[0]
        trials.require(all(n.config == first.config and n.spiking == first.spiking for n in nets),
                       'Stacked networks must share their configuration')
        trials.require(first.pooled, 'Stacks hold pooled networks (docs/model-method.md §6.10)')
        self.config, self.spiking, self.k = dict(first.config), first.spiking, len(nets)
        stack = lambda get: nn.Parameter(torch.stack([get(n).detach().clone() for n in nets]))
        self.w_in = stack(lambda n: n.w_in.weight.T)          # [K, D, n_v]
        self.b_in = stack(lambda n: n.w_in.bias)              # [K, n_v]
        self.w_vm = stack(lambda n: n.w_vm.weight.T)          # [K, n_v, 2]
        self.b_m = stack(lambda n: n.b_m)                     # [K, 2]
        self.alpha, self.beta, self.scale = first.alpha, first.beta, first.scale
        self.visual, self.motor, self._rate = first.visual, first.motor, first._rate
        self.n_motor = first.n_motor

    def drive(self, x):
        """x [K, B, D] -> visual input current [K, B, n_v]."""
        return torch.baddbmm(self.b_in[:, None], x, self.w_in)

    def step(self, drive, gate, state, noise_v, noise_m):
        """One millisecond for every network: drive, noise [K, B, n]; gate [K, B]. Same arithmetic as
        network.Network.step."""
        syn_v, mem_v, syn_m, mem_m = state
        if self.spiking:
            out_v, syn_v, mem_v = self.visual(self.scale * (drive + noise_v), syn_v, mem_v)
        else:
            out_v, syn_v, mem_v = self._rate(self.scale * (drive + noise_v), syn_v, mem_v)
        visuomotor = torch.bmm(out_v, self.w_vm).repeat_interleave(self.n_motor, -1)
        bias = self.b_m.repeat_interleave(self.n_motor, -1)[:, None]
        current = self.scale * (gate[..., None] * visuomotor + bias + noise_m)
        if self.spiking:
            out_m, syn_m, mem_m = self.motor(current, syn_m, mem_m)
        else:
            out_m, syn_m, mem_m = self._rate(current, syn_m, mem_m)
        return out_v, out_m, [syn_v, mem_v, syn_m, mem_m]

    def unstack(self, dim):
        """The K trained networks as network.Network objects (CPU)."""
        out = []
        for i in range(self.k):
            net = nw.Network(dim, self.config, spiking=self.spiking, seed=0)
            with torch.no_grad():
                net.w_in.weight.copy_(self.w_in[i].T)
                net.w_in.bias.copy_(self.b_in[i])
                net.w_vm.weight.copy_(self.w_vm[i].T)
                net.b_m.copy_(self.b_m[i])
            out.append(net)
        return out


def train_stack(nets, data, seeds, cfg=tr.TRAIN, device='cpu', updates=None, resume=None, every=250, log=None, key=None):
    """Train the networks `nets` together. data[i] = (static, movement, gain) of network i, as for
    train.train; seeds[i] its seed. Returns the trained networks (CPU), their readouts and histories,
    each as train.train would return them for that network alone. `key` identifies the stack's runs
    in its checkpoint, so a checkpoint is only ever resumed by the same stack."""
    k, batch, rest = len(nets), cfg['batch'], cfg['rest_trials']
    stack = Stack(nets).to(device)
    kappa = nn.Parameter(torch.full((k,), 2., device=device))
    n0 = nn.Parameter(torch.full((k,), 1., device=device))
    opt = torch.optim.Adam(list(stack.parameters()) + [kappa, n0], lr=cfg['lr'])
    rngs = [np.random.default_rng(np.random.SeedSequence([trials.SEED, s, 7])) for s in seeds]
    generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]
    tensor = lambda a: None if a is None else torch.as_tensor(a, dtype=torch.float32, device=device)
    data = [(tensor(s), tensor(m), tensor(g)) for s, m, g in data]
    histories, first = [[] for _ in range(k)], 0
    if resume is not None and Path(resume).is_file():
        state = torch.load(resume, weights_only=False, map_location=device)
        trials.require(state['seeds'] == list(seeds) and state.get('key') == key, 'Checkpoint belongs to another stack')
        stack.load_state_dict(state['stack'])
        kappa.data.copy_(state['kappa'])
        n0.data.copy_(state['n0'])
        opt.load_state_dict(state['optimiser'])
        for rng, s in zip(rngs, state['rngs']):
            rng.bit_generator.state = s
        for g, s in zip(generators, state['generators']):
            g.set_state(s)
        histories, first = state['histories'], state['update']
    steps = cfg['settle_ms'] + cfg['second_ms']
    proxy = type('Net', (), dict(n_motor=stack.n_motor))()        # train.objective needs n_motor only
    total = updates or cfg['updates']
    for update in range(first, total):
        statics, movements, labels, gains = [], [], [], []
        for (static, movement, gain), rng in zip(data, rngs):          # same draws, same order, as train.train
            pick = torch.as_tensor(rng.integers(len(static), size=batch), device=device)
            label = torch.as_tensor(rng.permutation(np.arange(batch) % 2), dtype=torch.float32, device=device)
            statics.append(static[pick])
            movements.append(movement[pick])
            labels.append(label)
            gains.append(tr.batch_gates(label, None if gain is None else gain[pick], cfg['settle_ms'], cfg['second_ms']))
        labels = torch.stack(labels)                                    # [K, B]
        first_drive = stack.drive(torch.stack(statics))                 # [K, B, n_v]
        second_drive = torch.where(labels[..., None].bool(), stack.drive(torch.stack(movements)), first_drive)
        zeros = torch.zeros(k, rest, first_drive.shape[-1], device=device)
        first_drive, second_drive = torch.cat([first_drive, zeros], 1), torch.cat([second_drive, zeros], 1)
        gates = torch.cat([torch.stack(gains), torch.ones(k, rest, steps, device=device)], 1)   # [K, B+R, T]
        noise = torch.stack([stack_noise(batch + rest, steps, g, stack.config, device) for g in generators])
        n_v = stack.config['n_visual']
        state = [torch.zeros(k, batch + rest, n, device=device) for n in (n_v, n_v, 2 * stack.n_motor, 2 * stack.n_motor)]
        out_v, out_m = [], []
        for t in range(steps):
            drive = first_drive if t < cfg['settle_ms'] else second_drive
            v, m, state = stack.step(drive, gates[:, :, t], state, noise[:, :, t, :n_v], noise[:, :, t, n_v:])
            out_v.append(v)
            out_m.append(m)
        out_v, out_m = torch.stack(out_v, 2), torch.stack(out_m, 2)     # [K, B+R, T, n]
        losses, infos = [], []
        for i in range(k):
            loss, info = tr.objective(proxy, out_v[i, :batch], out_m[i, :batch], labels[i], dict(kappa=kappa[i], n0=n0[i]),
                                      cfg, rest_m=out_m[i, batch:])
            losses.append(loss)
            infos.append(info)
        opt.zero_grad()
        torch.stack(losses).sum().backward()
        opt.step()
        if update % 50 == 0 or update == total - 1:
            for i in range(k):
                histories[i].append(dict(update=update, loss=losses[i].item(), **infos[i]))
            if log:
                log(dict(update=update, loss=float(np.mean([h[-1]['loss'] for h in histories]))))
        if resume is not None and (update + 1) % every == 0 and update + 1 < total:
            state = dict(seeds=list(seeds), key=key, update=update + 1, stack=stack.state_dict(), kappa=kappa.detach(), n0=n0.detach(),
                         optimiser=opt.state_dict(), rngs=[r.bit_generator.state for r in rngs],
                         generators=[g.get_state() for g in generators], histories=histories)
            tr.atomic(resume, lambda path: torch.save(state, path))
    dim = stack.w_in.shape[1]
    readouts = [dict(kappa=kappa[i].detach().cpu(), n0=n0[i].detach().cpu()) for i in range(k)]
    return stack.unstack(dim), readouts, histories


def stack_noise(rows, steps, generator, config, device):
    """The noise network.Network.noise draws for one network: same shape, generator and scale."""
    alpha, beta = nw.decay(config['tau_syn'], config['dt']), nw.decay(config['tau_mem'], config['dt'])
    sigma = nw.noise_sd(alpha, beta, config['membrane_sd'])
    return sigma * torch.randn(rows, steps, config['n_visual'] + 2 * config['n_motor'], generator=generator, device=device)
