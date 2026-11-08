"""How much a trained recurrent network uses its recurrent weights (docs/model-method.md §6.20).

Per network: the Frobenius norm of W_rec (self-connections excluded) beside that of W_in, and, on
full-body trials of the inner training source (both trial types, training windows from rest), the mean
absolute recurrent input of the visual layer relative to the mean absolute feedforward input
|W_in x + b_V|, over the second frame. A network whose training left W_rec near zero has a share near 0
and is effectively feedforward. Training sources only.

    python model/recurrence.py                       # every network of h2_openclip_snn_rec
    python model/recurrence.py --summary
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import bank as bk  # noqa: E402
import train as tr  # noqa: E402

OUTPUT = ROOT / 'output/model/recurrence'


@torch.no_grad()
def recurrent_share(net, static, movement, cfg=tr.TRAIN, seed=8, device='cpu'):
    """Mean |W_rec S_V(t-1)| / mean |W_in x + b_V| over the second frame of full-body trials."""
    static, movement = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in (static, movement))
    generator = torch.Generator(device=device).manual_seed(seed)
    recurrent, feedforward = [], []
    for label in (0., 1.):
        labels = torch.full((len(static),), label, device=device)
        drives = tr.batch_drives(net, static, movement, labels, cfg)
        steps = drives.shape[1]
        gates = tr.batch_gates(labels, None, cfg['settle_ms'], cfg['second_ms'])
        noise = net.noise(len(static), steps, generator, device)
        n_v, n_m = net.config['n_visual'], 2 * net.config['n_motor']
        state = net.initial(len(static), device)
        for t in range(steps):
            if t >= cfg['settle_ms']:
                recurrent.append(net.recurrent_input(state[4]).abs().mean().item())
                feedforward.append(drives[:, t].abs().mean().item())
            _, _, state = net.step(drives[:, t], gates[:, t], gates[:, t] * 0, state, noise[:, t, :n_v], noise[:, t, n_v:n_v + n_m])
    return float(np.mean(recurrent) / np.mean(feedforward))


def measure(path, device='cpu'):
    net, _, scaler, _, report = tr.checkpoint(path, device)
    tr.trials.require(net.recurrent, f'{path} is not a recurrent network')
    bank = bk.Bank(report['representation'])
    families = tr.families_by_source(bank)[report['inner_source']]
    static, movement, _, _, _ = tr.hypothesis_inputs(bank, report.get('hypothesis', 'h2'), None, families, scaler)
    w_rec = (net.w_rec * net.rec_mask).detach()
    return dict(run=Path(path).stem, fold=report['held_out_source_not_used'], initialisation=report['seed'],
                healthy=all(report['criteria'].values()), w_rec_norm=float(w_rec.norm()),
                w_in_norm=float(net.w_in.weight.detach().norm()), w_rec_max=float(w_rec.abs().max()),
                recurrent_share=recurrent_share(net, static, movement, device=device))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--configuration', default='h2_openclip_snn_rec')
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    target = OUTPUT / f'{args.configuration}.json'
    if not args.summary:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        done = {r['run']: r for r in json.loads(target.read_text())} if target.exists() else {}
        for path in sorted(glob.glob(str(ROOT / 'output/model/campaign' / args.configuration / 'health_*.pt'))):
            if Path(path).stem not in done and Path(path).with_suffix('.json').exists():
                done[Path(path).stem] = measure(path, device)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        tr.atomic(target, lambda t: t.write_text(json.dumps(sorted(done.values(), key=lambda r: r['run']), indent=1) + '\n'))
    rows = json.loads(target.read_text())
    share = np.array([r['recurrent_share'] for r in rows])
    print(json.dumps(dict(networks=len(rows), recurrent_share_median=float(np.median(share)),
                          recurrent_share_range=[float(share.min()), float(share.max())],
                          w_rec_norm_median=float(np.median([r['w_rec_norm'] for r in rows])),
                          w_in_norm_median=float(np.median([r['w_in_norm'] for r in rows]))), indent=1))


if __name__ == '__main__':
    main()
