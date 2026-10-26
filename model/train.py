"""Auxiliary-task training and health check (docs/model-method.md §6.2, §6.5).

The network learns only to detect the index movement from its FDI population, on full-body
trials of training sources. Development diagnostics use training sources only: an inner
held-out training source for task accuracy, never an outer held-out source (§6, development
rule). MEP analogues are never a training target.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import bank as bk  # noqa: E402
import network as nw  # noqa: E402
import trials  # noqa: E402

# Motor populations, trial by trial: held near motor_rest_hz on static second frames and at rest, and
# below motor_max_hz on movement frames. The paper excluded every trial with background EMG before
# the pulse, so its MEPs were taken at rest; and the probe response of the pooled populations grows
# with excitability only up to about 5.5 Hz (docs/model-method.md §6.10, §6.12). The limits apply
# to every trial, not to a batch mean, so that no source or variant is driven past that range.
TRAIN = dict(settle_ms=300, second_ms=350, window=(50, 300), updates=3000, batch=32, lr=1e-3,
             visual_rate_hz=(2., 40.), motor_rest_hz=1., motor_max_hz=3., rest_trials=8)
# Gate 2 health criteria (docs/model-method.md §6.5). The background limit is not a constant: it is
# each network's validated readout range, the pre-probe FDI rate at which its excitability curve peaks
# (the probe response grows with excitability up to there; §6.12).
HEALTH = dict(task_auc=.9, silent_visual=.10, fast_visual=.05, fixation_motor_hz=(.5, 20.), dose_spearman=.9,
              saturated=.9, spinal_background_hz=.5)   # motoneurons before the probe: near silent (no background EMG)
OUTPUT = ROOT / 'output/model'
HYPOTHESES = ('h2', 'h0', 'h1')   # h2: display features, gate 1 on full body (shared by H2, ungated, S3(a))


def atomic(path, write):
    """Write through a temporary file and rename, so an interrupted save never leaves a partial file."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    write(temporary)
    os.replace(temporary, path)


def families_by_source(bank):
    out = {}
    for family, condition, frame in bank.rows:
        if condition == 'full_body' and frame == 'static':
            out.setdefault(int(family[1:3]), []).append(family)
    return {s: sorted(f) for s, f in sorted(out.items())}


def frame_inputs(bank, families, scaler, condition='full_body'):
    """Standardised static and movement vectors [n, D] of the given families."""
    mean, sd = scaler
    get = lambda frame: (bank.matrix([(f, condition, frame) for f in families]) - mean) / sd
    return get('static'), get('movement')


def salience(families, condition='full_body'):
    """H1's low-level change s of each family's movement frame: 64 x 64 luminance distance to the
    static frame (Gate 1 `pixels64`; docs/model-method.md §6.6)."""
    pixels = bk.Bank('pixels64')
    frame = lambda r: pixels.matrix([(f, condition, r) for f in families])
    return np.linalg.norm(frame('movement') - frame('static'), axis=1)


def hypothesis_inputs(bank, hypothesis, train_families, families, scaler=None, mean_salience=None):
    """Inputs of one hypothesis (docs/model-method.md §6.6) for training on full-body trials:
    standardised static and movement vectors and the gate on movement frames (None = 1).
    H0 sees the hand-only image, standardised on the training sources' hand-only images; H1 and
    H2 see the full-body display; H1's gate is 1 + s / mean s over training full-body movement frames."""
    trials.require(hypothesis in HYPOTHESES, f'Unknown hypothesis: {hypothesis}')
    condition = 'hand_only' if hypothesis == 'h0' else 'full_body'
    scaler = scaler if scaler is not None else bank.scaler(train_families, display=condition)
    static, movement = frame_inputs(bank, families, scaler, condition)
    gain = None
    if hypothesis == 'h1':
        mean_salience = mean_salience if mean_salience is not None else float(salience(train_families).mean())
        gain = 1 + salience(families) / mean_salience
    return static, movement, gain, scaler, mean_salience


def batch_gates(labels, gain, steps_first, steps_second):
    """Gate [B, T]: 1 on the first frame and on static second frames, `gain` on movement frames."""
    second = torch.ones_like(labels) if gain is None else torch.where(labels.bool(), gain, torch.ones_like(gain))
    return torch.cat([torch.ones(len(labels), steps_first, device=labels.device),
                      second[:, None].expand(-1, steps_second)], 1)


def batch_drives(net, static, movement, labels, cfg=TRAIN):
    """Visual drive [B, T, n_v]: settle on the static frame, then the second frame."""
    first = net.drive(static)
    second = torch.where(labels[:, None].bool(), net.drive(movement), first)
    return torch.cat([first[:, None].expand(-1, cfg['settle_ms'], -1),
                      second[:, None].expand(-1, cfg['second_ms'], -1)], 1)


def objective(net, out_v, out_m, labels, readout, cfg=TRAIN, rest_m=None):
    """Detection loss plus the declared activity regularisers; returns loss and diagnostics.
    rest_m: motor output of no-stimulus (rest) trials, whose populations are held near the rest rate."""
    n = net.n_motor
    start = cfg['settle_ms']
    window = slice(start + cfg['window'][0], start + cfg['window'][1])
    counts = nw.population_counts(out_m[:, window], n)            # [B, 2] mean count per neuron
    logit = readout['kappa'] * (counts[:, nw.FDI] - readout['n0'])
    task = torch.nn.functional.binary_cross_entropy_with_logits(logit, labels)
    hz = lambda x: 1000 * x.mean((0, 1))                           # per-neuron rate in Hz
    visual = hz(out_v[:, start:])
    low, high = cfg['visual_rate_hz']
    reg_visual = (torch.relu(low - visual) ** 2 + torch.relu(visual - high) ** 2).mean() / 100
    second = out_m[:, start:]
    static = labels == 0
    n_m = net.n_motor
    # Per-trial rate of each population (FDI, ADM) in Hz: [trials, 2].
    per_trial = lambda m: 1000 * torch.stack([m[..., :n_m].mean((1, 2)), m[..., n_m:].mean((1, 2))], -1)
    zero = second.sum() * 0
    # Every static trial: each population near the rest rate.
    pops = per_trial(second[static]) if static.any() else zero + torch.zeros(1, 2, device=second.device) + cfg['motor_rest_hz']
    reg_rest = ((pops - cfg['motor_rest_hz']) ** 2).sum(-1).mean() / 5
    # Observation without execution: every movement trial below motor_max_hz in each population.
    moving = labels == 1
    active = per_trial(second[moving]) if moving.any() else zero + torch.zeros(1, 2, device=second.device)
    reg_cap = (torch.relu(active - cfg['motor_max_hz']) ** 2).sum(-1).mean() / 5
    pops, active = pops.mean(0), active.mean(0)
    rest = pops.mean()
    adm = counts[:, nw.ADM]
    reg_adm = (adm[labels == 1].mean() - adm[static].mean()) ** 2 if static.any() and (~static).any() else adm.sum() * 0
    # At rest (no stimulus, the fixation state where the threshold is measured) each population is
    # also held near the rest rate, so rest is a baseline and not a silent state.
    at_rest = per_trial(rest_m[:, start:]) if rest_m is not None else zero + torch.zeros(1, 2, device=second.device) + cfg['motor_rest_hz']
    reg_fixation = ((at_rest - cfg['motor_rest_hz']) ** 2).sum(-1).mean() / 5
    at_rest = at_rest.mean(0)
    loss = task + reg_visual + reg_rest + reg_cap + reg_adm + reg_fixation
    accuracy = ((logit > 0).float() == labels).float().mean()
    return loss, dict(task=task.item(), visual_hz=visual.mean().item(), silent=(visual == 0).float().mean().item(),
                      fast=(visual > 200).float().mean().item(),
                      rest_hz=rest.item(), fdi_static_hz=pops[0].item(), adm_static_hz=pops[1].item(),
                      fdi_movement_hz=active[0].item(), adm_movement_hz=active[1].item(),
                      fdi_rest_hz=torch.as_tensor(at_rest[0]).item(), adm_rest_hz=torch.as_tensor(at_rest[1]).item(),
                      adm_gap=reg_adm.item(), accuracy=accuracy.item())


def train(net, static, movement, cfg=TRAIN, seed=0, device='cpu', log=None, updates=None, gain=None, resume=None,
          every=250):
    """Adam on random full-body trials (half movement) of the training families; `gain` is the
    gate on each family's movement frame (H1), 1 if None. With `resume` (a path), the whole training
    state (weights, readout, optimiser, both random streams, history) is saved there every `every`
    updates and training continues from it if it exists, so an interrupted run resumes where it
    stopped and draws exactly the trials and noise an uninterrupted run would."""
    net.to(device)
    readout = dict(kappa=torch.nn.Parameter(torch.tensor(2., device=device)),
                   n0=torch.nn.Parameter(torch.tensor(1., device=device)))
    opt = torch.optim.Adam(list(net.parameters()) + list(readout.values()), lr=cfg['lr'])
    rng = np.random.default_rng(np.random.SeedSequence([trials.SEED, seed, 7]))
    generator = torch.Generator(device=device).manual_seed(seed)
    static, movement = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in (static, movement))
    gain = None if gain is None else torch.as_tensor(gain, dtype=torch.float32, device=device)
    history, first = [], 0
    if resume is not None and Path(resume).is_file():
        state = torch.load(resume, weights_only=False, map_location=device)
        net.load_state_dict(state['net'])
        for key, value in state['readout'].items():
            readout[key].data.copy_(value)
        opt.load_state_dict(state['optimiser'])
        rng.bit_generator.state = state['rng']
        generator.set_state(state["generator"].cpu())        # RNG states must be CPU ByteTensors
        history, first = state['history'], state['update']
        if log:
            log(dict(resumed_at=first))
    total = updates or cfg['updates']
    for update in range(first, total):
        pick = torch.as_tensor(rng.integers(len(static), size=cfg['batch']), device=device)
        labels = torch.as_tensor(rng.permutation(np.arange(cfg['batch']) % 2), dtype=torch.float32, device=device)
        drives = batch_drives(net, static[pick], movement[pick], labels, cfg)
        steps, rows = drives.shape[1], cfg['batch'] + cfg['rest_trials']
        # rest trials (no stimulus) run in the same batch: zero visual drive, gate 1
        drives = torch.cat([drives, nw.rest_drive(net, cfg['rest_trials'], device)[:, None].expand(-1, steps, -1)])
        noise = net.noise(rows, steps, generator, device)
        gates = batch_gates(labels, None if gain is None else gain[pick], cfg['settle_ms'], cfg['second_ms'])
        gates = torch.cat([gates, torch.ones(cfg['rest_trials'], steps, device=device)])
        out_v, out_m, _ = net.run(drives, gates, gates * 0, net.initial(rows, device), noise)
        b = cfg['batch']
        loss, info = objective(net, out_v[:b], out_m[:b], labels, readout, cfg, rest_m=out_m[b:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if update % 50 == 0 or update == total - 1:
            history.append(dict(update=update, loss=loss.item(), **info))
            if log:
                log(history[-1])
        if resume is not None and (update + 1) % every == 0 and update + 1 < total:
            state = dict(update=update + 1, net=net.state_dict(), readout={k: v.detach() for k, v in readout.items()},
                         optimiser=opt.state_dict(), rng=rng.bit_generator.state, generator=generator.get_state(),
                         history=history)
            atomic(resume, lambda path: torch.save(state, path))
    return readout, history


@torch.no_grad()
def evaluate_task(net, readout, static, movement, cfg=TRAIN, seed=1, device='cpu', repeats=4, gain=None):
    """Movement detection on every family given (both trial types, `repeats` noise draws): accuracy
    with the trained readout threshold; AUC, the threshold-free probability that a movement trial's
    FDI count exceeds a static trial's (ties count half); mean visual rate; and each visual neuron's
    rate (Hz) over all these trials."""
    static, movement = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in (static, movement))
    gain = None if gain is None else torch.as_tensor(gain, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    correct, total, rates, per_neuron, counts = 0, 0, [], [], {0.: [], 1.: []}
    for _ in range(repeats):
        for label in (0., 1.):
            labels = torch.full((len(static),), label, device=device)
            drives = batch_drives(net, static, movement, labels, cfg)
            steps = drives.shape[1]
            gates = batch_gates(labels, gain, cfg['settle_ms'], cfg['second_ms'])
            out_v, out_m, _ = net.run(drives, gates, gates * 0, net.initial(len(static), device),
                                      net.noise(len(static), steps, generator, device))
            _, info = objective(net, out_v, out_m, labels, readout, cfg)
            correct += info['accuracy'] * len(static)
            total += len(static)
            rates.append(info['visual_hz'])
            per_neuron.append(1000 * out_v[:, cfg['settle_ms']:].mean((0, 1)))
            window = slice(cfg['settle_ms'] + cfg['window'][0], cfg['settle_ms'] + cfg['window'][1])
            counts[label].append(nw.population_counts(out_m[:, window], net.n_motor)[:, nw.FDI].cpu().numpy())
    moving, still = np.concatenate(counts[1.]), np.concatenate(counts[0.])
    auc = float(((moving[:, None] > still[None]).mean() + .5 * (moving[:, None] == still[None]).mean()))
    return dict(accuracy=correct / total, auc=auc, visual_hz=float(np.mean(rates)),
                neuron_hz=torch.stack(per_neuron).mean(0).cpu().numpy())


@torch.no_grad()
def fixation_rates(net, device='cpu', ms=1000, n=16, seed=4):
    """Visual and per-population motor rates (Hz) at rest, the fixation display (§6.8)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    drive = nw.rest_drive(net, n, device)[:, None].expand(n, ms, -1)
    ones = torch.ones(n, ms, device=device)
    v, m, _ = net.run(drive, ones, ones * 0, net.initial(n, device), net.noise(n, ms, generator, device))
    hz = lambda x: 1000 * x[:, 200:].mean().item()
    return dict(visual_hz=hz(v), fdi_hz=hz(m[..., :net.n_motor]), adm_hz=hz(m[..., net.n_motor:]))


@torch.no_grad()
def dose_response(net, amplitude, device='cpu', levels=10, trials_per=40, seed=3):
    """Mean FDI probe response at 0.5-2 x threshold on the fixation display (rest)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    drive = nw.rest_drive(net, trials_per, device)
    gate = torch.ones(trials_per, device=device)
    out = []
    for factor in np.linspace(.5, 2, levels):
        state = nw.settle(net, drive, gate, 300, generator)
        out.append((float(factor), nw.probe_response(net, state, drive, gate, factor * amplitude, generator)[:, nw.FDI]
                    .mean().item()))
    return out


@torch.no_grad()
def task_probe(net, amplitude, static, movement, cfg=TRAIN, device='cpu', repeats=4, seed=5, gain=None):
    """Task-trial probes on full-body frames (docs/model-method.md §6.10): settle on the static
    frame, show the second frame for 250 ms, probe. Returns per-trial FDI/ADM responses [repeats, n, 2]
    for static and movement trials, and each population's rate (Hz) in the 100 ms before the probe."""
    static, movement = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in (static, movement))
    generator = torch.Generator(device=device).manual_seed(seed)
    n, before = len(static), trials.TIMING['probe_after_ms']
    ones = torch.ones(n, device=device)
    out = {}
    for kind, second in (('static', static), ('movement', movement)):
        gate = torch.as_tensor(gain, dtype=torch.float32, device=device) if kind == 'movement' and gain is not None else ones
        responses, background = [], []
        for _ in range(repeats):
            state = nw.settle(net, net.drive(static), ones, cfg['settle_ms'], generator)
            drive = net.drive(second)
            gates = gate[:, None].expand(n, before)
            _, m, state = net.run(drive[:, None].expand(n, before, -1), gates, gates * 0, state,
                                  net.noise(n, before, generator, device))
            background.append([1000 * m[:, -100:, :net.n_motor].mean().item(), 1000 * m[:, -100:, net.n_motor:].mean().item()])
            responses.append(nw.probe_response(net, state, drive, gate, amplitude, generator))
        out[kind] = dict(responses=torch.stack(responses).cpu().numpy(), background_hz=np.mean(background, 0).tolist())
    return out


@torch.no_grad()
def excitability_curve(net, amplitude, device='cpu', levels=25, top=.6, n=200, seed=6):
    """FDI probe response at rest while an extra constant current depolarises the FDI population:
    (pre-probe FDI rate in Hz, mean response) per level. Shows where the readout grows with excitability.
    Steps of 0.025 up to 0.6 (26 Oct 2026; was 0.3, where the non-spiking twin had not yet peaked)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    drive, ones = nw.rest_drive(net, n, device), torch.ones(n, device=device)
    bias = net.b_m.detach().clone()
    out = []
    try:
        for extra in np.linspace(0, top, levels):
            net.b_m.data = bias.clone()
            net.b_m.data[net.fdi_bias_index()] += float(extra)
            state = nw.settle(net, drive, ones, 300, generator)
            ones_t = torch.ones(n, 100, device=device)
            _, m, state = net.run(drive[:, None].expand(n, 100, -1), ones_t, ones_t * 0, state,
                                  net.noise(n, 100, generator, device))
            rate = 1000 * m[..., :net.n_motor].mean().item()
            out.append((rate, nw.probe_response(net, state, drive, ones, amplitude, generator)[:, nw.FDI].mean().item()))
    finally:
        net.b_m.data = bias
    return out


def diagnose(net, readout, inner_static, inner_movement, cfg=TRAIN, device='cpu', gain=None, activity=None):
    """Gate 2 health measures and criteria (docs/model-method.md §6.5) on full-body frames of an
    inner training source and on the fixation display; no other display is used. `activity`
    (static, movement, gain) of training-source frames measures silent and fast visual neurons over
    all training sources, since a neuron may be silent for one source only."""
    from scipy import stats
    task = evaluate_task(net, readout, inner_static, inner_movement, cfg, device=device, gain=gain)
    a_static, a_movement, a_gain = activity if activity is not None else (inner_static, inner_movement, gain)
    neuron_hz = evaluate_task(net, readout, a_static, a_movement, cfg, device=device, gain=a_gain, repeats=1)['neuron_hz']
    amplitude = nw.threshold(net, device=device)
    found = bool(np.isfinite(amplitude))
    curve = dose_response(net, amplitude, device=device) if found else []
    dose = float(stats.spearmanr(*zip(*curve))[0]) if curve else float('nan')
    rest = fixation_rates(net, device)
    report = dict(network_config=dict(net.config), inner_accuracy=task['accuracy'], inner_auc=task['auc'],
                  inner_visual_hz=task['visual_hz'], visual_silent_share=float((neuron_hz == 0).mean()),
                  visual_fast_share=float((neuron_hz > 200).mean()), threshold=amplitude, dose_response=curve,
                  dose_spearman=dose, fixation=rest, readout_ms=list(nw.READOUT), train_config=dict(cfg))
    low, high = HEALTH['fixation_motor_hz']
    criteria = dict(task=task['auc'] >= HEALTH['task_auc'], silent_visual=report['visual_silent_share'] <= HEALTH['silent_visual'],
                    fast_visual=report['visual_fast_share'] <= HEALTH['fast_visual'],
                    threshold=found, dose=bool(dose >= HEALTH['dose_spearman']),
                    fixation_motor=bool(low <= rest['fdi_hz'] <= high and low <= rest['adm_hz'] <= high))
    if found:
        probe = 1.2 * amplitude                                            # the task probe (spec §6.4)
        probes = task_probe(net, probe, inner_static, inner_movement, cfg, device, gain=gain)
        diff = (probes['movement']['responses'] - probes['static']['responses'])[..., nw.FDI].ravel()
        half = stats.t.ppf(.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
        curve2 = excitability_curve(net, probe, device)
        peak = max(range(len(curve2)), key=lambda i: curve2[i][1])
        readout_range = curve2[peak][0]                                     # rate at the curve's peak
        within = [c for c in curve2 if c[0] <= readout_range]               # the validated readout range
        rho = float(stats.spearmanr(*zip(*within))[0]) if len(within) >= 3 else float('nan')
        everything = np.concatenate([probes[k]['responses'] for k in probes], 1)
        background = max(max(probes[k]['background_hz']) for k in probes)
        report.update(task_probe={k: dict(fdi_mean=float(v['responses'][..., nw.FDI].mean()),
                                          adm_mean=float(v['responses'][..., nw.ADM].mean()),
                                          background_hz=v['background_hz']) for k, v in probes.items()},
                      fdi_facilitation=float(diff.mean()), fdi_facilitation_ci=[float(diff.mean() - half), float(diff.mean() + half)],
                      excitability_curve=curve2, excitability_spearman=rho,
                      saturated_share=float((everything >= HEALTH['saturated'] * net.n_motor).mean()))
        report.update(readout_range_hz=readout_range, background_max_hz=background)
        criteria.update(readout_peak_found=peak < len(curve2) - 1,           # else the range is not established
                        facilitation=bool(diff.mean() - half > 0), background=bool(background <= readout_range),
                        excitability=bool(rho >= HEALTH['dose_spearman']), not_saturated=report['saturated_share'] == 0,
                        readout_not_silent=bool(everything[..., nw.FDI].mean() > 0))
    report['criteria'] = criteria
    return report


def activity_sample(bank, hypothesis, sources, held_out, inner, scaler, mean_salience):
    """Two appearance variants of every training source, for the visual-activity criteria."""
    families = [f for s, fam in sources.items() if s not in (held_out, inner) for f in fam[::12]]
    static, movement, gain, _, _ = hypothesis_inputs(bank, hypothesis, None, families, scaler, mean_salience)
    return static, movement, gain


def diagnose_spinal(net, inner_static, inner_movement, cfg=TRAIN, device='cpu', gain=None):
    """Health measures of the second MEP analogue, read from the spinal motoneuron pools (§6.15):
    threshold, dose-response, motoneuron activity at rest and before the probe (the analogue of
    background EMG), readout facilitation on full-body trials of the inner source, saturation."""
    from scipy import stats
    net.readout_stage = 'spinal'
    try:
        amplitude = nw.threshold(net, device=device)
        found = bool(np.isfinite(amplitude))
        curve = dose_response(net, amplitude, device=device) if found else []
        dose = float(stats.spearmanr(*zip(*curve))[0]) if curve else float('nan')
        rest = fixation_rates(net, device)
        report = dict(threshold=amplitude, dose_response=curve, dose_spearman=dose, motoneurons_at_rest_hz=rest)
        criteria = dict(threshold=found, dose=bool(dose >= HEALTH['dose_spearman']),
                        silent_at_rest=bool(max(rest['fdi_hz'], rest['adm_hz']) <= HEALTH['spinal_background_hz']))
        if found:
            probes = task_probe(net, 1.2 * amplitude, inner_static, inner_movement, cfg, device, gain=gain)
            diff = (probes['movement']['responses'] - probes['static']['responses'])[..., nw.FDI].ravel()
            half = stats.t.ppf(.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            everything = np.concatenate([probes[k]['responses'] for k in probes], 1)
            background = max(max(probes[k]['background_hz']) for k in probes)
            report.update(task_probe={k: dict(fdi_mean=float(v['responses'][..., nw.FDI].mean()),
                                              adm_mean=float(v['responses'][..., nw.ADM].mean()),
                                              background_hz=v['background_hz']) for k, v in probes.items()},
                          fdi_facilitation=float(diff.mean()), fdi_facilitation_ci=[float(diff.mean() - half), float(diff.mean() + half)],
                          background_max_hz=background,
                          saturated_share=float((everything >= HEALTH['saturated'] * net.n_motor).mean()))   # recruited share
            criteria.update(facilitation=bool(diff.mean() - half > 0),
                            background=bool(background <= HEALTH['spinal_background_hz']),
                            not_saturated=report['saturated_share'] == 0,
                            readout_not_silent=bool(everything[..., nw.FDI].mean() > 0))
        report['criteria'] = criteria
        return report
    finally:
        net.readout_stage = 'cortical'


def checkpoint(path, device='cpu'):
    ck = torch.load(path, weights_only=False, map_location=device)
    report = json.loads(Path(path).with_suffix('.json').read_text())
    config = ck.get('config', dict(nw.CONFIG, motor_pool=False))   # earlier checkpoints: per-neuron weights
    net = nw.Network(len(ck['scaler'][0]), config, spiking=report['spiking'], seed=report['seed']).to(device)
    net.load_state_dict(ck['state'])
    readout = {k: v.to(device) for k, v in ck['readout'].items()}
    return net, readout, [np.asarray(a) for a in ck['scaler']], ck.get('mean_salience'), report


@torch.no_grad()
def recheck(path, device='cpu'):
    """Recompute every health measure of a saved health-check network without retraining (the
    readout and the health measures do not affect training). Uses the report's training config."""
    net, readout, scaler, mean_salience, report = checkpoint(path, device)
    bank = bk.Bank(report['representation'])
    sources, hypothesis = families_by_source(bank), report.get('hypothesis', 'h2')
    inner_static, inner_movement, gain, _, _ = hypothesis_inputs(bank, hypothesis, None, sources[report['inner_source']],
                                                                 scaler, mean_salience)
    activity = activity_sample(bank, hypothesis, sources, report['held_out_source_not_used'], report['inner_source'],
                               scaler, mean_salience)
    cfg = dict(TRAIN, **report.get('train_config', {}))
    report.update(diagnose(net, readout, inner_static, inner_movement, cfg, device, gain, activity))
    text = json.dumps(report, indent=2) + '\n'
    atomic(Path(path).with_suffix('.json'), lambda target: target.write_text(text))
    return report


def run_name(hypothesis, representation, spiking, held_out, seed):
    tag = '' if hypothesis == 'h2' else f'{hypothesis}_'
    return f"health_{tag}{representation}_{'snn' if spiking else 'rate'}_fold{held_out}_seed{seed}"


def prepare(representation='openclip', held_out=1, inner=2, seed=0, spiking=True, hypothesis='h2', bank=None):
    """Everything one run needs before training (§6.5, §6.6): training and inner-source inputs, with
    preprocessing and H1's mean salience from training sources only, and the initial network."""
    bank = bank or bk.Bank(representation)
    sources = families_by_source(bank)
    trials.require(held_out != inner and held_out in sources and inner in sources, 'Invalid fold')
    train_families = [f for s, fam in sources.items() if s not in (held_out, inner) for f in fam]
    static, movement, gain, scaler, mean_salience = hypothesis_inputs(bank, hypothesis, train_families, train_families)
    inner_static, inner_movement, inner_gain, _, _ = hypothesis_inputs(bank, hypothesis, None, sources[inner],
                                                                       scaler, mean_salience)
    return dict(name=run_name(hypothesis, representation, spiking, held_out, seed),
                meta=dict(representation=representation, hypothesis=hypothesis, spiking=spiking,
                          held_out_source_not_used=held_out, inner_source=inner, seed=seed),
                net=nw.Network(static.shape[1], spiking=spiking, seed=seed), static=static, movement=movement, gain=gain,
                inner=(inner_static, inner_movement, inner_gain), scaler=scaler, mean_salience=mean_salience,
                activity=activity_sample(bank, hypothesis, sources, held_out, inner, scaler, mean_salience))


def finish(run, readout, history, seconds, output=OUTPUT, device='cpu', updates=None, extra=None):
    """Health measures of a trained run, then its network and report (report last, atomically)."""
    net = run['net'].to(device)
    readout = {k: v.to(device) for k, v in readout.items()}
    report = dict(run['meta'], updates=updates or TRAIN['updates'], device=str(device), seconds=seconds,
                  final=history[-1], **(extra or {}))
    inner_static, inner_movement, inner_gain = run['inner']
    report.update(diagnose(net, readout, inner_static, inner_movement, device=device, gain=inner_gain,
                           activity=run['activity']))
    network = dict(state={k: v.detach().cpu() for k, v in net.state_dict().items()},
                   readout={k: v.detach().cpu() for k, v in readout.items()}, config=net.config,
                   scaler=[a.tolist() for a in run['scaler']], mean_salience=run['mean_salience'])
    output.mkdir(parents=True, exist_ok=True)
    atomic(output / f"{run['name']}.pt", lambda path: torch.save(network, path))
    # the report is written last: its presence marks a finished run (model/campaign.py)
    text = json.dumps(dict(report, history=history), indent=2) + '\n'
    atomic(output / f"{run['name']}.json", lambda path: path.write_text(text))
    return report


def finish_trained(path, device=None):
    """Health check and final files of a network trained in a stack (`<run>.trained.pt`)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    saved = torch.load(path, weights_only=False, map_location='cpu')
    run = prepare(**saved['meta_args'])
    run['net'].load_state_dict(saved['state'])
    report = finish(run, saved['readout'], saved['history'], saved['seconds'], Path(path).parent, device,
                    saved['updates'], saved['extra'])
    Path(path).unlink()
    return report


def health(representation='openclip', held_out=1, inner=2, seed=0, device=None, updates=None, spiking=True,
           output=OUTPUT, hypothesis='h2'):
    """Gate 2 style health check on one outer fold, using training sources only (§6.5). Trains the
    network of one hypothesis (§6.6): h2 (shared by H2, the ungated network and the S3(a) model),
    h0 or h1."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    run = prepare(representation, held_out, inner, seed, spiking, hypothesis)
    output.mkdir(parents=True, exist_ok=True)
    partial = output / f"{run['name']}.partial.pt"                 # resumable training state
    started = time.time()
    readout, history = train(run['net'], run['static'], run['movement'], seed=seed, device=device, updates=updates,
                             gain=run['gain'], log=lambda h: print(json.dumps(h), flush=True), resume=partial)
    report = finish(run, readout, history, round(time.time() - started), output, device, updates)
    partial.unlink(missing_ok=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representation', default='openclip')
    parser.add_argument('--held-out', type=int, default=1, help='Outer source never used here')
    parser.add_argument('--inner', type=int, default=2, help='Training source held out for task accuracy')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--updates', type=int)
    parser.add_argument('--rate', action='store_true', help='Non-spiking twin')
    parser.add_argument('--hypothesis', default='h2', choices=HYPOTHESES)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--recheck', type=Path, help='Recompute the health measures of a saved network')
    parser.add_argument('--finish', type=Path, help='Health check of a network trained in a stack (<run>.trained.pt)')
    args = parser.parse_args()
    if args.finish:
        report = finish_trained(args.finish)
    elif args.recheck:
        report = recheck(args.recheck, device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        report = health(args.representation, args.held_out, args.inner, args.seed, updates=args.updates,
                        spiking=not args.rate, output=args.output, hypothesis=args.hypothesis)
    print(json.dumps({k: v for k, v in report.items() if k not in ('dose_response', 'history', 'excitability_curve')},
                     indent=2))
