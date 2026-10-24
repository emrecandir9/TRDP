"""Training campaign: one network per outer fold and initialisation (spec §5.1, §7.1).

Every network is trained by `train.health`, the health-checked code path: full-body trials of the
fold's training sources, with one inner training source held out for the run's own health check
(docs/model-method.md §6.5). The outer held-out source is never used. The inner source is the next
source after the held-out one (cyclically), so every source serves once as an inner source.

Interruptions: a finished run (its report is written last, atomically) is skipped; a run stopped
part-way continues from its last checkpoint (every 250 updates, `<run>.partial.pt`) with the same
trials and noise, so rerunning the same command resumes the campaign and loses at most 250 updates
per interrupted run. Several workers share one GPU or the CPUs of a machine.

    python model/campaign.py --hypothesis h2 --representation openclip --seeds 0-9 --workers 3

CREMI lab PCs kill a user's processes when their last SSH session closes, so the campaign runs only
while a session stays open. From the Mac, this reconnects and resumes after any disconnection:

    caffeinate -i sh -c 'until ssh -o BatchMode=yes -o ServerAliveInterval=60 jessica \
        "cd /net/travail/mucandy/TRDP && .venv/bin/python model/campaign.py --seeds 0-9 --workers 3"; \
        do sleep 300; done'
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import train as tr  # noqa: E402

SOURCES = range(1, 21)


def inner_source(held_out):
    return held_out % len(SOURCES) + 1


def runs(hypothesis, representation, spiking, folds, seeds, output):
    tag = '' if hypothesis == 'h2' else f'{hypothesis}_'
    for seed in seeds:
        for fold in folds:
            name = f"health_{tag}{representation}_{'snn' if spiking else 'rate'}_fold{fold}_seed{seed}"
            done = (output / f'{name}.json').is_file() and (output / f'{name}.pt').is_file()
            yield fold, seed, name, done


def command(hypothesis, representation, spiking, fold, seed, output, updates=None):
    return [sys.executable, str(ROOT / 'model/train.py'), '--hypothesis', hypothesis, '--representation', representation,
            '--held-out', str(fold), '--inner', str(inner_source(fold)), '--seed', str(seed), '--output', str(output)] + \
        ([] if spiking else ['--rate']) + ([] if updates is None else ['--updates', str(updates)])


def span(text):
    a, _, b = text.partition('-')
    return range(int(a), int(b or a) + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hypothesis', default='h2', choices=tr.HYPOTHESES)
    parser.add_argument('--representation', default='openclip')
    parser.add_argument('--rate', action='store_true', help='Non-spiking twin')
    parser.add_argument('--folds', type=span, default=SOURCES, help='e.g. 1-20')
    parser.add_argument('--seeds', type=span, default=range(10), help='e.g. 0-9')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--updates', type=int, help='Override the declared number of updates (smoke tests only)')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--logs', type=Path, help='Directory for per-run logs (default: <output>/logs)')
    args = parser.parse_args()
    spiking = not args.rate
    output = args.output or ROOT / 'output/model/campaign' / f"{args.hypothesis}_{args.representation}_{'snn' if spiking else 'rate'}"
    logs = args.logs or output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    todo = [(f, s, n) for f, s, n, done in runs(args.hypothesis, args.representation, spiking, args.folds, args.seeds, output)
            if not done]
    print(json.dumps(dict(output=str(output), runs=len(todo), workers=args.workers)), flush=True)
    active, failed = [], []
    while todo or active:
        while todo and len(active) < args.workers:
            fold, seed, name = todo.pop(0)
            log = open(logs / f'{name}.txt', 'w')
            cmd = command(args.hypothesis, args.representation, spiking, fold, seed, output, args.updates)
            active.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, name, time.time()))
        for item in list(active):
            process, log, name, started = item
            if process.poll() is not None:
                log.close()
                active.remove(item)
                status = 'ok' if process.returncode == 0 else f'failed ({process.returncode})'
                if process.returncode:
                    failed.append(name)
                print(json.dumps(dict(run=name, status=status, minutes=round((time.time() - started) / 60, 1),
                                      remaining=len(todo) + len(active))), flush=True)
        time.sleep(5)
    summary = []
    for fold, seed, name, done in runs(args.hypothesis, args.representation, spiking, args.folds, args.seeds, output):
        if done:
            report = json.loads((output / f'{name}.json').read_text())
            summary.append(dict(run=name, fold=fold, seed=seed, healthy=all(report['criteria'].values()),
                                failed=[k for k, v in report['criteria'].items() if not v]))
    (output / 'campaign.json').write_text(json.dumps(dict(runs=summary, failed_to_run=failed), indent=2) + '\n')
    print(json.dumps(dict(trained=len(summary), healthy=sum(r['healthy'] for r in summary), failed_to_run=failed)))


if __name__ == '__main__':
    main()
