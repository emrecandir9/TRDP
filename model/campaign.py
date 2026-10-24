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
import threading
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


def train_stacks(args, spiking, output, logs):
    """Stack mode (docs/model-method.md §6.13): the pending folds of one seed are trained together
    as one stack on the GPU; each trained network's health check (`train.py --finish`) runs as a
    separate process while the next stack trains."""
    import torch
    import bank as bk
    import stack as st
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bank = bk.Bank(args.representation)
    queue, active, failed, lock, stop = [], [], [], threading.Lock(), threading.Event()

    def pump():
        while True:
            with lock:
                for item in list(active):
                    process, log, name, started = item
                    if process.poll() is not None:
                        log.close()
                        active.remove(item)
                        if process.returncode:
                            failed.append(name)
                        print(json.dumps(dict(run=name, status='ok' if process.returncode == 0 else f'failed ({process.returncode})',
                                              finish_minutes=round((time.time() - started) / 60, 1))), flush=True)
                while queue and len(active) < args.finishers:
                    path = queue.pop(0)
                    name = path.name[:-len('.trained.pt')]
                    log = open(logs / f'{name}.txt', 'a')
                    active.append((subprocess.Popen([sys.executable, str(ROOT / 'model/train.py'), '--finish', str(path)],
                                                    stdout=log, stderr=subprocess.STDOUT), log, name, time.time()))
                idle = not queue and not active
            if stop.is_set() and idle:
                return
            time.sleep(5)

    queue.extend(sorted(output.glob('*.trained.pt')))               # trained before an interruption
    worker = threading.Thread(target=pump, daemon=True)
    worker.start()
    for seed in args.seeds:
        pending = [(fold, name) for fold, s, name, done in runs(args.hypothesis, args.representation, spiking, args.folds, [seed], output)
                   if not done and not (output / f'{name}.trained.pt').is_file()]
        for i in range(0, len(pending), args.stack):
            group = pending[i:i + args.stack]
            names = [name for _, name in group]
            prepared = [tr.prepare(args.representation, fold, inner_source(fold), seed, spiking, args.hypothesis, bank)
                        for fold, _ in group]
            started = time.time()
            print(json.dumps(dict(stack=names, device=device)), flush=True)
            partial = output / f'stack_seed{seed}_{group[0][0]}-{group[-1][0]}.partial.pt'
            nets, readouts, histories = st.train_stack(
                [r['net'] for r in prepared], [(r['static'], r['movement'], r['gain']) for r in prepared],
                [seed] * len(prepared), device=device, updates=args.updates, resume=partial, key=names,
                log=lambda h: print(json.dumps(h), flush=True))
            seconds = round(time.time() - started)
            for run, net, readout, history in zip(prepared, nets, readouts, histories):
                trained = dict(state=net.state_dict(), readout=readout, history=history, seconds=seconds,
                               updates=args.updates,
                               meta_args=dict(representation=args.representation, held_out=run['meta']['held_out_source_not_used'],
                                              inner=run['meta']['inner_source'], seed=seed, spiking=spiking,
                                              hypothesis=args.hypothesis),
                               extra=dict(trainer='stack', stack_size=len(prepared), stack_seconds=seconds))
                path = output / f"{run['name']}.trained.pt"
                tr.atomic(path, lambda target: torch.save(trained, target))
                (output / f"{run['name']}.partial.pt").unlink(missing_ok=True)   # a single-run checkpoint, superseded
                with lock:
                    queue.append(path)
            partial.unlink(missing_ok=True)
            print(json.dumps(dict(stack_done=names, minutes=round(seconds / 60, 1))), flush=True)
    stop.set()
    worker.join()
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hypothesis', default='h2', choices=tr.HYPOTHESES)
    parser.add_argument('--representation', default='openclip')
    parser.add_argument('--rate', action='store_true', help='Non-spiking twin')
    parser.add_argument('--folds', type=span, default=SOURCES, help='e.g. 1-20')
    parser.add_argument('--seeds', type=span, default=range(10), help='e.g. 0-9')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--updates', type=int, help='Override the declared number of updates (smoke tests only)')
    parser.add_argument('--stack', type=int, default=0, help='Train this many folds of a seed together (0: one process per run)')
    parser.add_argument('--finishers', type=int, default=3, help='Parallel health checks in stack mode')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--logs', type=Path, help='Directory for per-run logs (default: <output>/logs)')
    args = parser.parse_args()
    spiking = not args.rate
    output = args.output or ROOT / 'output/model/campaign' / f"{args.hypothesis}_{args.representation}_{'snn' if spiking else 'rate'}"
    logs = args.logs or output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    todo = [(f, s, n) for f, s, n, done in runs(args.hypothesis, args.representation, spiking, args.folds, args.seeds, output)
            if not done]
    print(json.dumps(dict(output=str(output), runs=len(todo), workers=args.workers, stack=args.stack)), flush=True)
    active, failed = [], []
    if args.stack:
        failed, todo = train_stacks(args, spiking, output, logs), []
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
