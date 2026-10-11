"""Log a worker; optionally resume only resource stops after 60 seconds of idle GPU."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from dataset import ROOT, RECIPE, read, write, require, require_canary, verify
from generate import gpu_state, check_checkpoints


def wait_for_idle(output):
    idle_since = None
    previous = None
    while True:
        state = gpu_state()
        now = time.monotonic()
        idle = not state['other_processes'] and state['free_mib'] >= read(RECIPE)['minimum_gpu_vram_mib']
        idle_since = (now if idle_since is None else idle_since) if idle else None
        if state['other_processes'] != previous:
            write(output / 'waiting.json', dict(at_utc=datetime.now(timezone.utc).isoformat(), gpu=state,
                                               worker_exited=True, idle_seconds_required=60))
            print('Waiting for continuous GPU idle:', state, flush=True)
            previous = state['other_processes']
        if idle_since is not None and now - idle_since >= 60:
            return
        time.sleep(2)


def observe_gpu(worker, stop, path):
    previous = None
    with path.open('x') as trace:
        while not stop.is_set():
            try:
                state = gpu_state()
                if state['other_processes'] != previous:
                    import json
                    trace.write(json.dumps(dict(at_utc=datetime.now(timezone.utc).isoformat(),
                                                worker_pid=worker.pid, gpu=state)) + '\n')
                    trace.flush()
                    previous = state['other_processes']
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                trace.write(str(error) + '\n')
                trace.flush()
            stop.wait(2)


def launch(scope, output):
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    record = dict(scope=scope, started_at_utc=name, wrapper_pid=os.getpid())
    with (logs / f'{name}.log').open('x') as log:
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
               'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        worker = subprocess.Popen(['nice', '-n', '19', sys.executable, str(ROOT / 'generate.py'), scope,
                                   '--output', str(output)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        record['worker_pid'] = worker.pid
        write(logs / f'{name}.json', record)
        stop = threading.Event()
        observer = threading.Thread(target=observe_gpu, args=(worker, stop, logs / f'{name}.gpu.jsonl'), daemon=True)
        observer.start()
        try:
            for line in worker.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = worker.wait()
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            stop.set()
            observer.join(timeout=12)
            record.update(exit_code=worker.returncode, finished_at_utc=datetime.now(timezone.utc).isoformat())
            write(logs / f'{name}.json', record)
    if code == 0:
        verify(output)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope', choices=('canary', 'full'))
    parser.add_argument('--cooperative', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not args.cooperative or args.scope == 'full', 'Cooperative resume is for approved full production')
    if args.scope == 'full':
        require_canary(output)
    while True:
        check_checkpoints(output)
        if args.cooperative:
            wait_for_idle(output)
        code = launch(args.scope, output)
        if code != 75 or not args.cooperative:
            sys.exit(code)
