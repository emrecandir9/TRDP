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

TRAIN = dict(settle_ms=300, second_ms=350, window=(50, 300), updates=3000, batch=32, lr=1e-3,
             visual_rate_hz=(2., 40.), motor_rest_hz=5.)
OUTPUT = ROOT / 'output/model'


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


def batch_drives(net, static, movement, labels, cfg=TRAIN):
    """Visual drive [B, T, n_v]: settle on the static frame, then the second frame."""
    first = net.drive(static)
    second = torch.where(labels[:, None].bool(), net.drive(movement), first)
    return torch.cat([first[:, None].expand(-1, cfg['settle_ms'], -1),
                      second[:, None].expand(-1, cfg['second_ms'], -1)], 1)


def objective(net, out_v, out_m, labels, readout, cfg=TRAIN):
    """Detection loss plus the declared activity regularisers; returns loss and diagnostics."""
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
    rest = 1000 * second[static].mean() if static.any() else second.sum() * 0
    reg_rest = (rest - cfg['motor_rest_hz']) ** 2 / 25
    adm = counts[:, nw.ADM]
    reg_adm = (adm[labels == 1].mean() - adm[static].mean()) ** 2 if static.any() and (~static).any() else adm.sum() * 0
    loss = task + reg_visual + reg_rest + reg_adm
    accuracy = ((logit > 0).float() == labels).float().mean()
    return loss, dict(task=task.item(), visual_hz=visual.mean().item(), silent=(visual == 0).float().mean().item(),
                      rest_hz=rest.item(), adm_gap=reg_adm.item(), accuracy=accuracy.item())


def train(net, static, movement, cfg=TRAIN, seed=0, device='cpu', log=None, updates=None):
    """Adam on random full-body trials (half movement) of the training families."""
    net.to(device)
    readout = dict(kappa=torch.nn.Parameter(torch.tensor(2., device=device)),
                   n0=torch.nn.Parameter(torch.tensor(1., device=device)))
    opt = torch.optim.Adam(list(net.parameters()) + list(readout.values()), lr=cfg['lr'])
    rng = np.random.default_rng(np.random.SeedSequence([trials.SEED, seed, 7]))
    generator = torch.Generator(device=device).manual_seed(seed)
    static, movement = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in (static, movement))
    history = []
    for update in range(updates or cfg['updates']):
        pick = torch.as_tensor(rng.integers(len(static), size=cfg['batch']), device=device)
        labels = torch.as_tensor(rng.permutation(np.arange(cfg['batch']) % 2), dtype=torch.float32, device=device)
        drives = batch_drives(net, static[pick], movement[pick], labels, cfg)
        steps = drives.shape[1]
        noise = net.noise(cfg['batch'], steps, generator, device)
        ones = torch.ones(cfg['batch'], steps, device=device)
        out_v, out_m, _ = net.run(drives, ones, ones * 0, net.initial(cfg['batch'], device), noise)
        loss, info = objective(net, out_v, out_m, labels, readout, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if update % 50 == 0 or update == (updates or cfg['updates']) - 1:
            history.append(dict(update=update, loss=loss.item(), **info))
            if log:
                log(history[-1])
    return readout, history


@torch.no_grad()
def evaluate_task(net, readout, static, movement, cfg=TRAIN, seed=1, device='cpu', repeats=4):
    """Detection accuracy on every family given (both trial types, `repeats` noise draws)."""
    static, movement = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in (static, movement))
    generator = torch.Generator(device=device).manual_seed(seed)
    correct, total, rates = 0, 0, []
    for _ in range(repeats):
        for label in (0., 1.):
            labels = torch.full((len(static),), label, device=device)
            drives = batch_drives(net, static, movement, labels, cfg)
            steps = drives.shape[1]
            ones = torch.ones(len(static), steps, device=device)
            out_v, out_m, _ = net.run(drives, ones, ones * 0, net.initial(len(static), device),
                                      net.noise(len(static), steps, generator, device))
            _, info = objective(net, out_v, out_m, labels, readout, cfg)
            correct += info['accuracy'] * len(static)
            total += len(static)
            rates.append(info['visual_hz'])
    return correct / total, float(np.mean(rates))


@torch.no_grad()
def dose_response(net, black, amplitude, device='cpu', levels=10, trials_per=40, seed=3):
    """Mean FDI probe response at 0.5-2 x threshold on the fixation display."""
    generator = torch.Generator(device=device).manual_seed(seed)
    drive = net.drive(black[None]).expand(trials_per, -1)
    gate = torch.ones(trials_per, device=device)
    out = []
    for factor in np.linspace(.5, 2, levels):
        state = nw.settle(net, drive, gate, 300, generator)
        out.append((float(factor), nw.probe_response(net, state, drive, gate, factor * amplitude, generator)[:, nw.FDI]
                    .mean().item()))
    return out


def health(representation='openclip', held_out=1, inner=2, seed=0, device=None, updates=None, spiking=True,
           output=OUTPUT):
    """Gate 2 style health check on one outer fold, using training sources only (§6.5)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    bank = bk.Bank(representation)
    sources = families_by_source(bank)
    trials.require(held_out != inner and held_out in sources and inner in sources, 'Invalid fold')
    train_families = [f for s, fam in sources.items() if s not in (held_out, inner) for f in fam]
    scaler = bank.scaler(train_families)                            # training sources only
    static, movement = frame_inputs(bank, train_families, scaler)
    inner_static, inner_movement = frame_inputs(bank, sources[inner], scaler)
    net = nw.Network(static.shape[1], spiking=spiking, seed=seed)
    started = time.time()
    readout, history = train(net, static, movement, seed=seed, device=device, updates=updates,
                             log=lambda h: print(json.dumps(h), flush=True))
    accuracy, visual_hz = evaluate_task(net, readout, inner_static, inner_movement, device=device)
    black = torch.as_tensor((bank.vector(trials.BLACK) - scaler[0]) / scaler[1], dtype=torch.float32, device=device)
    amplitude = nw.threshold(net, black)
    curve = dose_response(net, black, amplitude, device=device) if np.isfinite(amplitude) else []
    from scipy import stats
    monotone = float(stats.spearmanr([c[0] for c in curve], [c[1] for c in curve])[0]) if curve else float('nan')
    report = dict(representation=representation, spiking=spiking, held_out_source_not_used=held_out,
                  inner_source=inner, seed=seed, updates=updates or TRAIN['updates'], device=str(device),
                  seconds=round(time.time() - started), final=history[-1], inner_accuracy=accuracy,
                  inner_visual_hz=visual_hz, threshold=amplitude, dose_response=curve, dose_spearman=monotone,
                  criteria=dict(task=accuracy >= .9, threshold=bool(np.isfinite(amplitude)),
                                dose=bool(monotone >= .9) if curve else False,
                                silent_visual=history[-1]['silent'] <= .10))
    output.mkdir(parents=True, exist_ok=True)
    name = f"health_{representation}_{'snn' if spiking else 'rate'}_fold{held_out}_seed{seed}"
    (output / f'{name}.json').write_text(json.dumps(dict(report, history=history), indent=2) + '\n')
    torch.save(dict(state=net.state_dict(), readout={k: v.detach().cpu() for k, v in readout.items()},
                    scaler=[a.tolist() for a in scaler]), output / f'{name}.pt')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representation', default='openclip')
    parser.add_argument('--held-out', type=int, default=1, help='Outer source never used here')
    parser.add_argument('--inner', type=int, default=2, help='Training source held out for task accuracy')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--updates', type=int)
    parser.add_argument('--rate', action='store_true', help='Non-spiking twin')
    args = parser.parse_args()
    report = health(args.representation, args.held_out, args.inner, args.seed, updates=args.updates,
                    spiking=not args.rate)
    print(json.dumps({k: v for k, v in report.items() if k != 'dose_response'}, indent=2))
