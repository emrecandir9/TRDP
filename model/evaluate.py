"""Evaluation simulator (spec §5.1, §6.4; docs/model-method.md §6.3-6.4, §6.18).

One model instance (a trained network, a hypothesis's input and gate, one initialisation) runs the
paper's session (model/trials.py) on one source's images; every task and fixation trial is probed
at 1.2 x the resting threshold, and each probe is read against its matched no-probe twin. Per
instance the task readouts are trimmed (± 2 SD of their cell mean), z-normalised over both
populations, averaged into display x trial type x population cells and passed to the human
target's contrast code (analysis/human_target.py), so model and humans go through the same
arithmetic. Instances are aggregated by the declared hierarchical bootstrap (source groups, then
initialisations within each group).

Two simulation modes give the same probe state for the feedforward network (§6.4): 'continuous'
runs the whole session at 1 ms with the state carried and the session continuing from the no-probe
branch; 'trialwise' starts each trial from rest `settle_ms` before its second frame and simulates all
trials as one batch. The residual of earlier frames at the probe is below e^-12 for the feedforward
network, which has no state beyond its 5- and 20-ms filters; the recurrent variant needs 'continuous'.

Development rule (spec §8): no context contrast or MEP-analogue summary on a held-out source before
the evaluation is opened. `simulate_instance` refuses a network's own held-out source unless called
with open_evaluation=True, which only the frozen evaluation may do.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
sys.path.insert(0, str(ROOT / 'analysis'))
import human_target as ht  # noqa: E402
import network as nw  # noqa: E402
import trials  # noqa: E402

PROBE_FACTOR = 1.2        # task probe = 120% of the resting threshold (spec §6.4)
BACKGROUND_MS = 100       # pre-probe window for the background rate (the paper's 100 ms before the pulse)
SETTLE_MS = 300           # trialwise mode: rest -> first frame for this long before the second frame
BOOTSTRAP = dict(samples=10000, seed=20261022)
KINDS = ('static', 'movement')
require = trials.require


@dataclass
class Model:
    """A hypothesis applied to a trained network (docs/model-method.md §6.6, §7.1).

    hypothesis: 'h0' (hand-only input, g = 1), 'h1' (g = 1 + s / mean_salience on movement frames),
    'gated' (g = c + (1 - c) a on every frame of a display, a = attachment(family, display); H2, the
    visible-area model of S3(a) and the learned pathways differ only in `attachment`), 'ungated' (g = 1).
    On the fixation display (no stimulus) g = 1, or c for the gated models (a = 0)."""
    hypothesis: str
    c: float = None
    attachment: object = None
    salience: object = None          # salience(family, display) -> s of the movement frame (H1)
    mean_salience: float = None

    def __post_init__(self):
        require(self.hypothesis in ('h0', 'h1', 'gated', 'ungated'), f'Unknown hypothesis: {self.hypothesis}')
        if self.hypothesis == 'gated':
            require(self.c is not None and 0 <= self.c <= 1 and self.attachment is not None, 'Gated model needs c and attachment')
        if self.hypothesis == 'h1':
            require(self.salience is not None and self.mean_salience, 'H1 needs salience and its training mean')

    def input_key(self, key):
        """Feature key the network sees for a displayed image (H0 sees the hand-only image)."""
        if key == trials.BLACK or self.hypothesis != 'h0':
            return key
        family, _, frame = key
        return family, 'hand_only', frame

    def gate(self, key):
        if key == trials.BLACK:
            return self.c if self.hypothesis == 'gated' else 1.
        family, display, frame = key
        if self.hypothesis == 'gated':
            return self.c + (1 - self.c) * float(self.attachment(family, display))
        if self.hypothesis == 'h1' and frame == 'movement':
            return 1. + float(self.salience(family, display)) / self.mean_salience
        return 1.


def drives(net, model, bank, scaler, keys, device):
    """Visual drive [n, n_visual] per displayed image; the fixation display has none (§6.8)."""
    mean, sd = scaler
    out = torch.zeros(len(keys), net.config['n_visual'], device=device)
    shown = [i for i, k in enumerate(keys) if k != trials.BLACK]
    if shown:
        x = (bank.matrix([model.input_key(keys[i]) for i in shown]) - mean) / sd
        out[shown] = net.drive(torch.as_tensor(x, dtype=torch.float32, device=device))
    return out


def stream(session, salt=0):
    """Seed of a session's noise: shared by every model and gate value run on that session, so
    they are compared with common random numbers."""
    return int(np.random.SeedSequence([trials.SEED, session.source, session.initialisation, salt]).generate_state(1)[0])


def trial_keys(trial):
    if trial['kind'] == 'fixation':
        return trials.BLACK, trials.BLACK
    first = (trial['family'], trial['display'], 'static')
    return first, (trial['family'], trial['display'], 'movement' if trial['kind'] == 'movement' else 'static')


def records(session, responses, background):
    out = pd.DataFrame(session.trials)[['trial', 'block', 'display', 'kind', 'family']]
    out[['fdi', 'adm']] = np.asarray(responses)
    out[['pre_fdi_hz', 'pre_adm_hz']] = np.asarray(background)
    return out


@torch.no_grad()
def simulate_trialwise(net, model, bank, scaler, session, amplitude, device='cpu', settle_ms=SETTLE_MS, salt=0):
    """All trials in one batch, each from rest `settle_ms` before its second frame (§6.4)."""
    keys = [trial_keys(t) for t in session.trials]
    first = drives(net, model, bank, scaler, [k[0] for k in keys], device)
    second = drives(net, model, bank, scaler, [k[1] for k in keys], device)
    gate_first = torch.tensor([model.gate(k[0]) for k in keys], dtype=torch.float32, device=device)
    gate_second = torch.tensor([model.gate(k[1]) for k in keys], dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(stream(session, salt))
    n, before = len(keys), trials.TIMING['probe_after_ms']
    state = nw.settle(net, first, gate_first, settle_ms, generator)
    gates = gate_second[:, None].expand(n, before)
    _, m, state = net.run(second[:, None].expand(n, before, -1), gates, gates * 0, state, net.noise(n, before, generator, device))
    tail = m[:, -BACKGROUND_MS:]
    background = 1000 * torch.stack([tail[..., :net.n_motor].mean((1, 2)), tail[..., net.n_motor:].mean((1, 2))], -1)
    responses = nw.probe_response(net, state, second, gate_second, amplitude, generator)
    return records(session, responses.cpu().numpy(), background.cpu().numpy())


@torch.no_grad()
def simulate_continuous(net, model, bank, scaler, session, amplitude, device='cpu', salt=0):
    """The whole session at 1 ms, never reset; at each probe the probe and no-probe branches share
    state and noise, and the session continues from the no-probe branch (§6.3)."""
    keys = session.keys()
    index = {k: i for i, k in enumerate(keys)}
    drive = drives(net, model, bank, scaler, keys, device)
    gate = torch.tensor([model.gate(k) for k in keys], dtype=torch.float32, device=device)
    step_key = torch.as_tensor(session.steps(), device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(stream(session, salt))
    state, t, motor_tail = net.initial(1, device), 0, None
    responses, background = [], []

    def advance(state, start, end, probe_at=None):
        k = step_key[start:end]
        steps = end - start
        probes = torch.zeros(1, steps, device=device)
        if probe_at is not None:
            probes[0, probe_at] = amplitude
        return net.run(drive[k][None], gate[k][None], probes, state, net.noise(1, steps, generator, device))

    for trial in session.trials:
        _, m, state = advance(state, t, trial['probe_ms'])
        tail = m[0, -BACKGROUND_MS:]
        background.append([1000 * tail[:, :net.n_motor].mean().item(), 1000 * tail[:, net.n_motor:].mean().item()])
        end = trial['probe_ms'] + nw.READOUT[1]
        noise_state = generator.get_state()
        _, with_probe, _ = advance([x.clone() for x in state], trial['probe_ms'], end, probe_at=0)
        generator.set_state(noise_state)                      # the twin sees the identical noise
        _, without, state = advance(state, trial['probe_ms'], end)
        diff = (with_probe - without)[0, nw.READOUT[0]:nw.READOUT[1]]
        responses.append([diff[:, :net.n_motor].sum().item(), diff[:, net.n_motor:].sum().item()])
        t = end
    return records(session, responses, background)       # the session's last 445 ms carry no probe


def instance_cells(table, displays, trim=True):
    """One instance's readouts as the paper treated a participant's MEPs (§6.3): task trials only;
    readouts beyond ± 2 SD of their cell mean removed (cell = display x trial type x population);
    z-normalised over both populations; cell means [len(displays), 2 trial types, 2 populations].
    Also returns the mean pre-probe rate per cell (untrimmed) and the number of readouts removed."""
    task = table[table.kind.isin(KINDS)]
    require(set(task.display) == set(displays), 'Session displays differ from the requested displays')
    long = task.melt(id_vars=['trial', 'display', 'kind'], value_vars=['fdi', 'adm'], var_name='population')
    removed = 0
    if trim:
        group = long.groupby(['display', 'kind', 'population']).value
        keep = (long.value - group.transform('mean')).abs() <= 2 * group.transform(lambda v: v.std(ddof=1))
        removed, long = int((~keep).sum()), long[keep]
    sd = long.value.std(ddof=1)
    require(sd > 0, 'Undefined instance standard deviation')
    long = long.assign(z=(long.value - long.value.mean()) / sd)
    means = long.groupby(['display', 'kind', 'population']).z.mean()
    index = pd.MultiIndex.from_product([displays, KINDS, ['fdi', 'adm']])
    means = means.reindex(index)
    require(not means.isna().any(), 'An instance lacks a cell after trimming')
    rates = task.groupby(['display', 'kind'])[['pre_fdi_hz', 'pre_adm_hz']].mean().reindex(
        pd.MultiIndex.from_product([displays, KINDS]))
    return (means.to_numpy().reshape(len(displays), 2, 2), rates.to_numpy().reshape(len(displays), 2, 2), removed)


def hierarchical_bootstrap(cells, groups, samples=BOOTSTRAP['samples'], seed=BOOTSTRAP['seed']):
    """Mean cell array of each resample [samples, ...]: source groups with replacement, then
    instances (initialisations) with replacement within each drawn group (spec §5.1)."""
    cells, groups = np.asarray(cells, float), np.asarray(groups)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    out = np.empty((samples,) + cells.shape[1:])
    for s in range(samples):
        drawn = [m[rng.integers(0, len(m), len(m))] for m in (members[i] for i in rng.integers(0, len(members), len(members)))]
        out[s] = cells[np.concatenate(drawn)].mean(0)
    return out


def summarise(cells, groups, initialisations, interval=.95, **bootstrap):
    """Human-target contrasts and ratios of the instance mean (cells [instances, 3, 2, 2] in the
    human order: full body, upper limb, detached hand), with hierarchical-bootstrap intervals, and
    P3's per-initialisation C_FDI (averaged over source groups)."""
    cells = np.asarray(cells, float)
    require(cells.shape[1:] == (3, 2, 2), 'Cells must be [instances, 3 bodies, 2 trial types, 2 populations]')
    boot = hierarchical_bootstrap(cells, groups, **bootstrap)
    mean, tail = cells.mean(0), (1 - interval) / 2
    rows = [dict(name=name, kind='contrast', family=family, estimate=float(f(mean)),
                 ci_low=float(np.quantile(f(boot), tail)), ci_high=float(np.quantile(f(boot), 1 - tail)))
            for name, family, f in ht.contrasts()]
    rows += [dict(name=name, kind='ratio', family='ratio', estimate=float(f(mean)),
                  ci_low=float(np.quantile(f(boot), tail)), ci_high=float(np.quantile(f(boot), 1 - tail)))
             for name, f in ht.ratios()]
    name, _, c_fdi = ht.contrasts()[0]
    require(name == 'C_FDI', 'The primary contrast is no longer first in the human target')
    per_initialisation = {int(i): float(c_fdi(cells[np.asarray(initialisations) == i].mean(0)))
                          for i in np.unique(initialisations)}
    return pd.DataFrame(rows), per_initialisation


def families_of(bank, source):
    return sorted(f for f, c, r in bank.rows if c == 'full_body' and r == 'static' and int(f[1:3]) == source)


def simulate_instance(net, model, bank, scaler, report, source, displays=trials.BODIES, kind_code=0, mode='trialwise',
                      device='cpu', open_evaluation=False):
    """One instance's session on `source`: resting threshold with the model's fixation gate, task
    probe at 1.2 x threshold, every trial simulated. Refuses the network's held-out source unless
    the evaluation is opened."""
    require(open_evaluation or source != report['held_out_source_not_used'],
            f'Source {source} is held out for this network; the evaluation is not open')
    amplitude = PROBE_FACTOR * nw.threshold(net, gate=model.gate(trials.BLACK), device=device)
    require(np.isfinite(amplitude), 'No resting threshold')
    session = trials.session(families_of(bank, source), source, source - 1, report['seed'], displays, kind_code)
    simulate = simulate_trialwise if mode == 'trialwise' else simulate_continuous
    table = simulate(net, model, bank, scaler, session, amplitude, device)
    return table.assign(source=source, initialisation=report['seed'], amplitude=amplitude), session


def mode_check(path, repeats=20, device='cpu'):
    """§6.4's equivalence check on a trained network, without any context display: a full-body-only
    session on the network's inner training source. (1) Noise switched off: both modes must give
    the same readout and pre-probe rate on every trial. (2) With noise: one continuous run against
    `repeats` trialwise runs (independent noise); per cell (trial type x measure) a Welch t-test of
    the continuous trials against the pooled trialwise trials, Holm-corrected over the cells."""
    import bank as bk
    import train as tr
    from scipy import stats
    net, _, scaler, _, report = tr.checkpoint(path, device)
    require(report.get('hypothesis', 'h2') == 'h2', 'The check uses H2 networks (ungated on full-body frames)')
    b = bk.Bank(report['representation'])
    source = report['inner_source']
    amplitude = PROBE_FACTOR * nw.threshold(net, device=device)
    s = trials.session(families_of(b, source), source, source - 1, report['seed'], ('full_body',))
    model, measures = Model('ungated'), ['fdi', 'adm', 'pre_fdi_hz', 'pre_adm_hz']
    sigma, net.sigma = net.sigma, 0.
    quiet = [simulate(net, model, b, scaler, s, amplitude, device) for simulate in (simulate_continuous, simulate_trialwise)]
    net.sigma = sigma
    noise_free = dict(trials=len(quiet[0]), identical_trials=int((quiet[0][measures] == quiet[1][measures]).all(axis=1).sum()),
                      max_abs_difference=float((quiet[0][measures] - quiet[1][measures]).abs().max().max()))
    continuous = simulate_continuous(net, model, b, scaler, s, amplitude, device)
    trialwise = pd.concat([simulate_trialwise(net, model, b, scaler, s, amplitude, device, salt=k) for k in range(1, repeats + 1)])
    rows = []
    for kind in ('fixation', 'static', 'movement'):
        for m in measures:
            a, w = continuous.loc[continuous.kind == kind, m], trialwise.loc[trialwise.kind == kind, m]
            test = stats.ttest_ind(a, w, equal_var=False)
            rows.append(dict(kind=kind, measure=m, continuous=a.mean(), trialwise=w.mean(), p=test.pvalue))
    table = pd.DataFrame(rows)
    table['p_holm'] = ht.holm(table.p)
    return noise_free, table


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('step', choices=('mode-check',))
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for p in args.paths:
        noise_free, table = mode_check(p, args.repeats, device)
        print(p, noise_free)
        print(table.round(4).to_string(index=False))
