"""Claims whose process died before writing the result. Usage: stale.py [--clean]
Only claims made on this host are judged (their pid is checked here); with --clean they are removed."""
import glob, os, shutil, socket, sys
from pathlib import Path
M = Path('/net/travail/mucandy/TRDP/output/model')
OUT = {'fit_gate': M/'gate_fit', 'gate2_sweep': M/'gate2/sweep', 'gate2_predictions': M/'gate2/predictions',
       'curve_adm': M/'curve_extension_adm'}
me = socket.gethostname()
done = {step: {Path(p).stem for p in glob.glob(str(root/'**/*.json'), recursive=True)} for step, root in OUT.items()}
for step, root in OUT.items():
    stale, live, other = [], 0, 0
    for c in (M/'claims'/step).glob('*'):
        if c.name in done[step]:
            continue
        try:
            host, pid = (c/'host').read_text().split()
        except Exception:
            host, pid = me, '0'
        if host != me:
            other += 1; continue
        try:
            os.kill(int(pid), 0); live += 1
        except (ProcessLookupError, ValueError):
            stale.append(c)
    print(f'{me} {step}: results {len(done[step])}, stale {len(stale)}, running {live}, open on other hosts {other}')
    if '--clean' in sys.argv:
        for c in stale:
            shutil.rmtree(c)
