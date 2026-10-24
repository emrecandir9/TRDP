"""Auxiliary-task training and health check (docs/model-method.md §6.2, §6.5).

The network learns only to detect the index movement from its FDI population, on full-body
trials of training sources. Development diagnostics use training sources only: an inner
held-out training source for task accuracy, never an outer held-out source (§6, development
rule). MEP analogues are never a training target.
"""
import argparse
import json
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

# Motor populations: held near motor_rest_hz on static second frames and below motor_max_hz on
# movement frames. The paper excluded trials with background EMG before the pulse, so MEPs were
# taken at rest; and the probe response of a leaky integrate-and-fire population grows with
# excitability only while the population is near rest (it plateaus at 6-9 Hz and falls below its
# resting value at 30-40 Hz; docs/model-method.md §6.10).
TRAIN = dict(settle_ms=300, second_ms=350, window=(50, 300), updates=3000, batch=32, lr=1e-3,
             visual_rate_hz=(2., 40.), motor_rest_hz=2., motor_max_hz=5., rest_trials=8)
# Gate 2 health criteria that depend on these constants (docs/model-method.md §6.5).
HEALTH = dict(task_auc=.9, silent_visual=.10, fast_visual=.05, fixation_motor_hz=(.5, 20.), background_hz=6., dose_spearman=.9,
              saturated=.9)
OUTPUT = ROOT / 'output/model'
HYPOTHESES = ('h2', 'h0', 'h1')   # h2: display features, gate 1 on full body (shared by H2, ungated, S3(a))


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
    # Each population (FDI, ADM) is held near the rest rate on static frames, not only their mean.
    pops = [1000 * second[static][..., :n_m].mean(), 1000 * second[static][..., n_m:].mean()] if static.any() else [second.sum() * 0] * 2
    rest = (pops[0] + pops[1]) / 2
    reg_rest = sum((r - cfg['motor_rest_hz']) ** 2 for r in pops) / 5
    # Observation without execution: on movement frames each population stays below motor_max_hz.
    moving = labels == 1
    active = [1000 * second[moving][..., :n_m].mean(), 1000 * second[moving][..., n_m:].mean()] if moving.any() else [second.sum() * 0] * 2
    reg_cap = sum(torch.relu(r - cfg['motor_max_hz']) ** 2 for r in active) / 5
    adm = counts[:, nw.ADM]
    reg_adm = (adm[labels == 1].mean() - adm[static].mean()) ** 2 if static.any() and (~static).any() else adm.sum() * 0
    # At rest (no stimulus, the fixation state where the threshold is measured) each population is
    # also held near the rest rate, so rest is a baseline and not a silent state.
    at_rest = [1000 * rest_m[:, start:, :n_m].mean(), 1000 * rest_m[:, start:, n_m:].mean()] if rest_m is not None else [second.sum() * 0 + cfg['motor_rest_hz']] * 2
    reg_fixation = sum((r - cfg['motor_rest_hz']) ** 2 for r in at_rest) / 5
    loss = task + reg_visual + reg_rest + reg_cap + reg_adm + reg_fixation
    accuracy = ((logit > 0).float() == labels).float().mean()
    return loss, dict(task=task.item(), visual_hz=visual.mean().item(), silent=(visual == 0).float().mean().item(),
                      fast=(visual > 200).float().mean().item(),
                      rest_hz=rest.item(), fdi_static_hz=pops[0].item(), adm_static_hz=pops[1].item(),
                      fdi_movement_hz=active[0].item(), adm_movement_hz=active[1].item(),
                      fdi_rest_hz=torch.as_tensor(at_rest[0]).item(), adm_rest_hz=torch.as_tensor(at_rest[1]).item(),
                      adm_gap=reg_adm.item(), accuracy=accuracy.item())


def train(net, static, movement, cfg=TRAIN, seed=0, device='cpu', log=None, updates=None, gain=None):
    """Adam on random full-body trials (half movement) of the training families; `gain` is the
    gate on each family's movement frame (H1), 1 if None."""
    net.to(device)
    readout = dict(kappa=torch.nn.Parameter(torch.tensor(2., device=device)),
                   n0=torch.nn.Parameter(torch.tensor(1., device=device)))
    opt = torch.optim.Adam(list(net.parameters()) + list(readout.values()), lr=cfg['lr'])
    rng = np.random.default_rng(np.random.SeedSequence([trials.SEED, seed, 7]))
    generator = torch.Generator(device=device).manual_seed(seed)
    static, movement = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in (static, movement))
    gain = None if gain is None else torch.as_tensor(gain, dtype=torch.float32, device=device)
    history = []
    for update in range(updates or cfg['updates']):
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
        if update % 50 == 0 or update == (updates or cfg['updates']) - 1:
            history.append(dict(update=update, loss=loss.item(), **info))
            if log:
                log(history[-1])
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
def excitability_curve(net, amplitude, device='cpu', levels=13, top=.3, n=200, seed=6):
    """FDI probe response at rest while an extra constant current depolarises the FDI population:
    (pre-probe FDI rate in Hz, mean response) per level. Shows where the readout grows with excitability."""
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
        within = [c for c in curve2 if c[0] <= cfg['motor_max_hz']]
        rho = float(stats.spearmanr(*zip(*within))[0]) if len(within) >= 3 else float('nan')
        everything = np.concatenate([probes[k]['responses'] for k in probes], 1)
        background = max(max(probes[k]['background_hz']) for k in probes)
        report.update(task_probe={k: dict(fdi_mean=float(v['responses'][..., nw.FDI].mean()),
                                          adm_mean=float(v['responses'][..., nw.ADM].mean()),
                                          background_hz=v['background_hz']) for k, v in probes.items()},
                      fdi_facilitation=float(diff.mean()), fdi_facilitation_ci=[float(diff.mean() - half), float(diff.mean() + half)],
                      excitability_curve=curve2, excitability_spearman=rho,
                      saturated_share=float((everything >= HEALTH['saturated'] * net.n_motor).mean()))
        criteria.update(facilitation=bool(diff.mean() - half > 0), background=bool(background <= HEALTH['background_hz']),
                        excitability=bool(rho >= HEALTH['dose_spearman']), not_saturated=report['saturated_share'] == 0,
                        readout_not_silent=bool(everything[..., nw.FDI].mean() > 0))
    report['criteria'] = criteria
    return report


def activity_sample(bank, hypothesis, sources, held_out, inner, scaler, mean_salience):
    """Two appearance variants of every training source, for the visual-activity criteria."""
    families = [f for s, fam in sources.items() if s not in (held_out, inner) for f in fam[::12]]
    static, movement, gain, _, _ = hypothesis_inputs(bank, hypothesis, None, families, scaler, mean_salience)
    return static, movement, gain


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
    Path(path).with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def health(representation='openclip', held_out=1, inner=2, seed=0, device=None, updates=None, spiking=True,
           output=OUTPUT, hypothesis='h2'):
    """Gate 2 style health check on one outer fold, using training sources only (§6.5). Trains the
    network of one hypothesis (§6.6): h2 (shared by H2, the ungated network and the S3(a) model),
    h0 or h1."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    bank = bk.Bank(representation)
    sources = families_by_source(bank)
    trials.require(held_out != inner and held_out in sources and inner in sources, 'Invalid fold')
    train_families = [f for s, fam in sources.items() if s not in (held_out, inner) for f in fam]
    # preprocessing and H1's mean salience from training sources only
    static, movement, gain, scaler, mean_salience = hypothesis_inputs(bank, hypothesis, train_families, train_families)
    inner_static, inner_movement, inner_gain, _, _ = hypothesis_inputs(bank, hypothesis, None, sources[inner],
                                                                       scaler, mean_salience)
    net = nw.Network(static.shape[1], spiking=spiking, seed=seed)
    started = time.time()
    readout, history = train(net, static, movement, seed=seed, device=device, updates=updates, gain=gain,
                             log=lambda h: print(json.dumps(h), flush=True))
    seconds = round(time.time() - started)
    report = dict(representation=representation, hypothesis=hypothesis, spiking=spiking, held_out_source_not_used=held_out,
                  inner_source=inner, seed=seed, updates=updates or TRAIN['updates'], device=str(device),
                  seconds=seconds, final=history[-1])
    activity = activity_sample(bank, hypothesis, sources, held_out, inner, scaler, mean_salience)
    report.update(diagnose(net, readout, inner_static, inner_movement, device=device, gain=inner_gain, activity=activity))
    output.mkdir(parents=True, exist_ok=True)
    tag = '' if hypothesis == 'h2' else f'{hypothesis}_'
    name = f"health_{tag}{representation}_{'snn' if spiking else 'rate'}_fold{held_out}_seed{seed}"
    (output / f'{name}.json').write_text(json.dumps(dict(report, history=history), indent=2) + '\n')
    torch.save(dict(state={k: v.detach().cpu() for k, v in net.state_dict().items()},
                    readout={k: v.detach().cpu() for k, v in readout.items()}, config=net.config,
                    scaler=[a.tolist() for a in scaler], mean_salience=mean_salience), output / f'{name}.pt')
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
    args = parser.parse_args()
    if args.recheck:
        report = recheck(args.recheck)
    else:
        report = health(args.representation, args.held_out, args.inner, args.seed, updates=args.updates,
                        spiking=not args.rate, output=args.output, hypothesis=args.hypothesis)
    print(json.dumps({k: v for k, v in report.items() if k not in ('dose_response', 'history', 'excitability_curve')},
                     indent=2))
