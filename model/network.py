"""Spiking visuomotor network and its non-spiking twin (docs/model-method.md §6).

A visual layer of current-based leaky integrate-and-fire neurons receives the standardised
encoder vector of the displayed image; a motor layer holds an FDI- and an ADM-associated
population, driven by the visual spikes through a gate g (one per hypothesis) and probed by a
brief current, the model's TMS analogue. The probe response is read against a matched no-probe
twin that branches from the identical state with the identical noise, so the difference is
caused by the probe alone.
"""
import math

import torch
from torch import nn

# motor_pool: each motor population is a homogeneous pool, its neurons sharing one visuomotor weight
# vector and one bias and differing only by their noise, so the population's firing and its
# excitability move together (docs/model-method.md §6.10). Checkpoints saved before 24 Oct 2026
# have per-neuron weights (motor_pool=False).
CONFIG = dict(n_visual=256, n_motor=64, tau_syn=5., tau_mem=20., dt=1., membrane_sd=.1, threshold=1., slope=25.,
              rate_slope=4., visual_bias=.8, motor_bias=.6, input_gain=.5, motor_pool=True, rate_transfer='lif')
# rate_transfer: output function of the non-spiking twin. 'lif' (25 Oct 2026): the mean-field rate
# function of the network's own noisy spiking neuron (lif_transfer); 'softplus':
# softplus(k(U - threshold))/k, used before, which gave large output below threshold and did not train
# (docs/model-method.md §6.14).
# ms after the probe. The human MEP window (5-60 ms) follows a corticospinal conduction delay of
# about 20 ms; the model reads its motor population directly, which answers within 1-5 ms, so the
# window keeps the human length (55 ms) but starts at the probe (spec amendment, 24 Oct 2026).
READOUT = (0, 55)
FDI, ADM = 0, 1
# Spinal stage of the second MEP analogue (docs/model-method.md §6.15), used only by health checks and
# the evaluation, never in training; no parameter is fitted. Each muscle's cortical pool drives a pool
# of n_motor motoneurons (same neuron model, own noise), every motoneuron receiving every spike of its
# muscle's pool with one weight. bias 0: at rest the motoneurons are silent, as the paper required no
# background EMG. weight: the largest raw current per cortical spike for which a cortical pool firing
# independently at at_hz (train.TRAIN motor_max_hz, the movement limit) leaves the motoneurons
# essentially silent (at most silent_hz), found by bisection on that synthetic input once
# (tests/test_network.py re-derives it): 0.0193, a mean motoneuron membrane of 0.42 threshold. So
# observation alone does not recruit motoneurons and only the synchronous probe volley can. The
# readout counts recruited motoneurons (each at most once), as an MEP's size reflects how many motor
# units discharge together. NOT ADOPTED (25 Oct 2026, §6.15): recruitment is nearly all-or-none, so the
# pool saturates on task trials; a valid spinal readout needs further physiological structure.
SPINAL = dict(bias=0., weight=.0193, at_hz=3., silent_hz=.1)


def decay(tau, dt):
    return math.exp(-dt / tau)


_TRANSFER = {}


def lif_transfer(alpha, beta, membrane_sd, threshold, low=-1., high=3., step=.01, neurons=400, ms=3000, burn=500, seed=0):
    """Mean-field rate function of the network's noisy spiking neuron, for the non-spiking twin: the
    steady firing rate (spikes per time step) of the spiking neuron, with its noise and reset, under a
    constant input that settles its membrane at x (the network's normalised units), for x on a grid
    from `low` to `high`. The twin's unit computes the same filtered input and emits this rate.
    Returns (grid start, step, table); the table is made monotone to remove sampling noise."""
    key = (alpha, beta, membrane_sd, threshold, low, high, step)
    if key not in _TRANSFER:
        generator = torch.Generator().manual_seed(seed)
        scale = (1 - alpha) * (1 - beta)
        sigma = noise_sd(alpha, beta, membrane_sd)
        bins = int(round((high - low) / step)) + 1
        x = torch.linspace(low, high, bins)[:, None].expand(-1, neurons)
        syn, mem, spikes = torch.zeros_like(x), torch.zeros_like(x), torch.zeros(bins)
        for t in range(ms):
            syn = alpha * syn + scale * (x + sigma * torch.randn(x.shape, generator=generator))
            mem = beta * mem + syn
            spike = (mem >= threshold).float()
            mem = mem - spike * threshold
            if t >= burn:
                spikes += spike.sum(1)
        table = torch.cummax(spikes / (neurons * (ms - burn)), 0).values
        _TRANSFER[key] = (low, step, table)
    return _TRANSFER[key]


def noise_sd(alpha, beta, membrane_sd):
    """Per-step input noise SD that gives the declared membrane SD at rest. Inputs are scaled by
    (1 - alpha)(1 - beta), so a constant input x settles at membrane potential x; white noise
    filtered by the synaptic and membrane stages then has membrane variance
    s^2 sigma^2 (1 + alpha beta) / ((1 - alpha^2)(1 - beta^2)(1 - alpha beta))."""
    s = (1 - alpha) * (1 - beta)
    factor = s ** 2 * (1 + alpha * beta) / ((1 - alpha ** 2) * (1 - beta ** 2) * (1 - alpha * beta))
    return membrane_sd / math.sqrt(factor)


class Network(nn.Module):
    """Feedforward visual -> motor network. spiking=False gives the rate twin of §6.1."""

    def __init__(self, dim, config=CONFIG, spiking=True, seed=0):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate
        self.config, self.spiking = dict(config), spiking
        c = self.config
        torch.manual_seed(seed)
        self.w_in = nn.Linear(dim, c['n_visual'])
        units = 2 if c.get('motor_pool') else 2 * c['n_motor']      # one per population, or per neuron
        self.w_vm = nn.Linear(c['n_visual'], units, bias=False)
        self.b_m = nn.Parameter(torch.full((units,), c['motor_bias']))
        nn.init.normal_(self.w_in.weight, 0, c['input_gain'] / math.sqrt(dim))
        nn.init.constant_(self.w_in.bias, c['visual_bias'])
        nn.init.normal_(self.w_vm.weight, 0, 1 / math.sqrt(c['n_visual']))
        self.alpha, self.beta = decay(c['tau_syn'], c['dt']), decay(c['tau_mem'], c['dt'])
        self.scale = (1 - self.alpha) * (1 - self.beta)     # constant input x -> membrane x at rest
        self.sigma = noise_sd(self.alpha, self.beta, c['membrane_sd'])
        grad = surrogate.fast_sigmoid(slope=c['slope'])
        self.visual = snn.Synaptic(alpha=self.alpha, beta=self.beta, threshold=c['threshold'], spike_grad=grad,
                                   reset_mechanism='subtract')
        self.motor = snn.Synaptic(alpha=self.alpha, beta=self.beta, threshold=c['threshold'], spike_grad=grad,
                                  reset_mechanism='subtract')
        # Not registered as a submodule: the spinal stage has nothing to learn or save (checkpoints stay
        # as they were); it is moved to the input's device when used.
        self.__dict__['spinal_neurons'] = snn.Synaptic(alpha=self.alpha, beta=self.beta, threshold=c['threshold'],
                                                       spike_grad=grad, reset_mechanism='subtract')
        # 'cortical': the motor output is the cortical pools (training, primary readout); 'spinal': the
        # motor output is the motoneuron pools they drive (second readout, §6.15).
        self.readout_stage = 'cortical'
        self.spinal_weight = SPINAL['weight']

    @property
    def n_motor(self):
        return self.config['n_motor']

    @property
    def pooled(self):
        return bool(self.config.get('motor_pool'))

    def per_neuron(self, x):
        """Expand per-population values [..., 2] to the motor neurons [..., 2 n_motor] (pooled network)."""
        return x.repeat_interleave(self.n_motor, dim=-1) if self.pooled else x

    def motor_bias(self):
        return self.per_neuron(self.b_m)

    def fdi_bias_index(self):
        """Entries of b_m that set the FDI population's bias."""
        return self.bias_index(FDI)

    def bias_index(self, population):
        """Entries of b_m that set one population's bias (FDI or ADM)."""
        return slice(population, population + 1) if self.pooled else slice(population * self.n_motor, (population + 1) * self.n_motor)

    def drive(self, x):
        """Visual input current of standardised encoder vectors (constant within a frame)."""
        return self.w_in(x)

    def initial(self, batch, device=None):
        z = lambda n: torch.zeros(batch, n, device=device)
        c = self.config
        state = [z(c['n_visual']), z(c['n_visual']), z(2 * c['n_motor']), z(2 * c['n_motor'])]
        return state + [z(2 * c['n_motor']), z(2 * c['n_motor'])] if self.readout_stage == 'spinal' else state

    def _rate(self, current, syn, mem):
        syn = self.alpha * syn + current
        mem = self.beta * mem + syn
        c = self.config
        if c.get('rate_transfer', 'softplus') == 'softplus':
            k = c['rate_slope']
            return nn.functional.softplus(k * (mem - c['threshold'])) / k, syn, mem
        low, step, table = lif_transfer(self.alpha, self.beta, c['membrane_sd'], c['threshold'])
        table = table.to(mem.device)
        position = ((mem - low) / step).clamp(0, len(table) - 1.001)     # linear interpolation, differentiable in mem
        index = position.floor().long()
        frac = position - index
        return table[index] * (1 - frac) + table[index + 1] * frac, syn, mem

    def step(self, drive, gate, probe, state, noise_v, noise_m, noise_s=None):
        """One time step. drive [B, n_visual]; gate, probe [B]; noise [B, n]. Returns visual and
        motor output (spikes, or rates for the twin) and the new state. With readout_stage 'spinal' the
        motor output is the motoneuron pools' (noise_s their noise)."""
        syn_v, mem_v, syn_m, mem_m = state[:4]
        layer = self.visual if self.spiking else None
        if self.spiking:
            out_v, syn_v, mem_v = layer(self.scale * (drive + noise_v), syn_v, mem_v)
        else:
            out_v, syn_v, mem_v = self._rate(self.scale * (drive + noise_v), syn_v, mem_v)
        # The probe is a raw 1-ms current (not scaled), the model's brief TMS pulse.
        current = self.scale * (gate[:, None] * self.per_neuron(self.w_vm(out_v)) + self.motor_bias() + noise_m) + probe[:, None]
        if self.spiking:
            out_m, syn_m, mem_m = self.motor(current, syn_m, mem_m)
        else:
            out_m, syn_m, mem_m = self._rate(current, syn_m, mem_m)
        if self.readout_stage != 'spinal':
            return out_v, out_m, [syn_v, mem_v, syn_m, mem_m]
        syn_s, mem_s = state[4:]
        n = self.n_motor
        pooled = torch.stack([out_m[:, :n].sum(1), out_m[:, n:].sum(1)], -1).repeat_interleave(n, -1)
        spinal = self.spinal_weight * pooled + self.scale * (SPINAL['bias'] + noise_s)
        if self.spiking:
            neurons = self.spinal_neurons
            if neurons.threshold.device != spinal.device:
                neurons.to(spinal.device)
            out_s, syn_s, mem_s = neurons(spinal, syn_s, mem_s)
        else:
            out_s, syn_s, mem_s = self._rate(spinal, syn_s, mem_s)
        return out_v, out_s, [syn_v, mem_v, syn_m, mem_m, syn_s, mem_s]

    def run(self, drives, gates, probes, state, noise):
        """Simulate T steps. drives [B, T, n_visual]; gates, probes [B, T]; noise [B, T, n_v + 2 n_m].
        Returns visual and motor outputs [B, T, n] and the final state."""
        n_v, n_m = self.config['n_visual'], 2 * self.config['n_motor']
        out_v, out_m = [], []
        for t in range(drives.shape[1]):
            spinal = noise[:, t, n_v + n_m:] if self.readout_stage == 'spinal' else None
            v, m, state = self.step(drives[:, t], gates[:, t], probes[:, t], state, noise[:, t, :n_v],
                                    noise[:, t, n_v:n_v + n_m], spinal)
            out_v.append(v)
            out_m.append(m)
        return torch.stack(out_v, 1), torch.stack(out_m, 1), state

    def noise(self, batch, steps, generator, device=None):
        c = self.config
        extra = 2 * c['n_motor'] if self.readout_stage == 'spinal' else 0
        return self.sigma * torch.randn(batch, steps, c['n_visual'] + 2 * c['n_motor'] + extra, generator=generator,
                                        device=device)


def population_counts(motor, n_motor):
    """Mean output per neuron of the FDI and ADM populations, summed over time: [B, 2]."""
    return torch.stack([motor[..., :n_motor].sum(1).mean(-1), motor[..., n_motor:].sum(1).mean(-1)], -1)


@torch.no_grad()
def probe_response(net, state, drive, gate, amplitude, generator):
    """Probe-evoked FDI and ADM response: population output 0-55 ms after a 1-ms probe of the
    given amplitude, minus the matched no-probe twin (same state, same noise). Returns [B, 2]
    as summed outputs of each population (spikes for the spiking network)."""
    batch, steps = drive.shape[0], READOUT[1]
    noise = net.noise(batch, steps, generator, drive.device)
    drives = drive[:, None].expand(batch, steps, -1)
    gates = gate[:, None].expand(batch, steps)
    probes = torch.zeros(batch, steps, device=drive.device)
    probes[:, 0] = amplitude
    copy = lambda s: [x.clone() for x in s]
    _, with_probe, _ = net.run(drives, gates, probes, copy(state), noise)
    _, without, _ = net.run(drives, gates, torch.zeros_like(probes), copy(state), noise)
    window = slice(READOUT[0], READOUT[1])
    n = net.n_motor
    if net.readout_stage == 'spinal':                     # recruited motoneurons: each counts once
        diff = (with_probe[:, window].sum(1) > 0).float() - (without[:, window].sum(1) > 0).float()
        return torch.stack([diff[..., :n].sum(1), diff[..., n:].sum(1)], -1)
    diff = with_probe[:, window] - without[:, window]
    return torch.stack([diff[..., :n].sum((1, 2)), diff[..., n:].sum((1, 2))], -1)


@torch.no_grad()
def settle(net, drive, gate, steps, generator):
    """State after `steps` ms of a constant display, from rest."""
    batch = drive.shape[0]
    state = net.initial(batch, drive.device)
    noise = net.noise(batch, steps, generator, drive.device)
    _, _, state = net.run(drive[:, None].expand(batch, steps, -1), gate[:, None].expand(batch, steps),
                          torch.zeros(batch, steps, device=drive.device), state, noise)
    return state


def rest_drive(net, batch, device=None):
    """Visual input of the fixation display: no visual stimulus, so no stimulus current (the
    layer receives only its noise). A frozen encoder's description of a black frame lies far
    outside the training images and drove the motor layer to 45 Hz (docs/model-method.md §6.8)."""
    return torch.zeros(batch, net.config['n_visual'], device=device)


@torch.no_grad()
def threshold(net, a_max=20., trials=40, criterion=6, halvings=14, seed=0, settle_ms=300, gate=1., device=None):
    """Resting threshold (docs/model-method.md §6.3): smallest amplitude whose FDI response
    reaches `criterion` spikes on at least half of `trials` fixation trials (rest, §6.8), by bisection."""
    device = device or next(net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    drive = rest_drive(net, trials, device)
    gate = torch.full((trials,), float(gate), device=device)
    low, high = 0., a_max
    def passes(amplitude):
        state = settle(net, drive, gate, settle_ms, generator)
        response = probe_response(net, state, drive, gate, amplitude, generator)[:, FDI]
        return (response >= criterion).float().mean().item() >= .5
    if not passes(a_max):
        return float('nan')
    for _ in range(halvings):
        middle = (low + high) / 2
        low, high = (low, middle) if passes(middle) else (middle, high)
    return high
