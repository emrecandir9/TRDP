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

CONFIG = dict(n_visual=256, n_motor=64, tau_syn=5., tau_mem=20., dt=1., membrane_sd=.1, threshold=1., slope=25.,
              rate_slope=4., visual_bias=.8, motor_bias=.6, input_gain=.5)
# ms after the probe. The human MEP window (5-60 ms) follows a corticospinal conduction delay of
# about 20 ms; the model reads its motor population directly, which answers within 1-5 ms, so the
# window keeps the human length (55 ms) but starts at the probe (spec amendment, 24 Oct 2026).
READOUT = (0, 55)
FDI, ADM = 0, 1


def decay(tau, dt):
    return math.exp(-dt / tau)


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
        self.w_vm = nn.Linear(c['n_visual'], 2 * c['n_motor'], bias=False)
        self.b_m = nn.Parameter(torch.full((2 * c['n_motor'],), c['motor_bias']))
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

    @property
    def n_motor(self):
        return self.config['n_motor']

    def drive(self, x):
        """Visual input current of standardised encoder vectors (constant within a frame)."""
        return self.w_in(x)

    def initial(self, batch, device=None):
        z = lambda n: torch.zeros(batch, n, device=device)
        c = self.config
        return [z(c['n_visual']), z(c['n_visual']), z(2 * c['n_motor']), z(2 * c['n_motor'])]

    def _rate(self, current, syn, mem):
        syn = self.alpha * syn + current
        mem = self.beta * mem + syn
        k = self.config['rate_slope']
        return nn.functional.softplus(k * (mem - self.config['threshold'])) / k, syn, mem

    def step(self, drive, gate, probe, state, noise_v, noise_m):
        """One time step. drive [B, n_visual]; gate, probe [B]; noise [B, n]. Returns visual and
        motor output (spikes, or rates for the twin) and the new state."""
        syn_v, mem_v, syn_m, mem_m = state
        layer = self.visual if self.spiking else None
        if self.spiking:
            out_v, syn_v, mem_v = layer(self.scale * (drive + noise_v), syn_v, mem_v)
        else:
            out_v, syn_v, mem_v = self._rate(self.scale * (drive + noise_v), syn_v, mem_v)
        # The probe is a raw 1-ms current (not scaled), the model's brief TMS pulse.
        current = self.scale * (gate[:, None] * self.w_vm(out_v) + self.b_m + noise_m) + probe[:, None]
        if self.spiking:
            out_m, syn_m, mem_m = self.motor(current, syn_m, mem_m)
        else:
            out_m, syn_m, mem_m = self._rate(current, syn_m, mem_m)
        return out_v, out_m, [syn_v, mem_v, syn_m, mem_m]

    def run(self, drives, gates, probes, state, noise):
        """Simulate T steps. drives [B, T, n_visual]; gates, probes [B, T]; noise [B, T, n_v + 2 n_m].
        Returns visual and motor outputs [B, T, n] and the final state."""
        n_v = self.config['n_visual']
        out_v, out_m = [], []
        for t in range(drives.shape[1]):
            v, m, state = self.step(drives[:, t], gates[:, t], probes[:, t], state, noise[:, t, :n_v], noise[:, t, n_v:])
            out_v.append(v)
            out_m.append(m)
        return torch.stack(out_v, 1), torch.stack(out_m, 1), state

    def noise(self, batch, steps, generator, device=None):
        c = self.config
        return self.sigma * torch.randn(batch, steps, c['n_visual'] + 2 * c['n_motor'], generator=generator,
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
