"""Trial simulator: one model instance's session as a timeline of displayed images and probes.

Follows the paper's procedure (Matamala-Gomez et al., 2025, Methods) and the frozen analysis
spec §2, §5.1 and §6.3–6.4; the rules for details neither specifies are in docs/model-method.md.

A session is: 20 fixation trials (black display), then one block per display (60 s of the
static image, then 30 static and 30 movement trials in random order), then 20 fixation trials.
Every trial shows the static frame for 2,250-3,250 ms (uniform integer ms), then a second
frame for 750 ms (the same static frame, or the movement frame of the same family), with the
probe 250 ms after the second frame starts. Trials follow each other without a gap and the
network state is never reset. All times are integer milliseconds from session start.
"""
import itertools
from dataclasses import dataclass, field

import numpy as np

SEED = 20261022
TIMING = dict(first_min_ms=2250, first_max_ms=3250, second_ms=750, probe_after_ms=250,
              familiarisation_ms=60000, fixation_trials=20, trials_per_type=30)
READOUT_MS = (0, 55)     # readout window after the probe (network.READOUT; spec amendment 24 Oct 2026)
BODIES = ('full_body', 'upper_limb', 'detached_hand_silhouette')            # primary displays, paper order
SENSITIVITY_BODIES = ('full_body', 'upper_limb', 'detached_hand')            # rectangle detached hand
CONTROL_DISPLAYS = ('hand_stub', 'hand_gap_stub', 'window_hand', 'window_arm', 'scrambled', 'occluded_wrist')
BLACK = ('black',)
KINDS = ('fixation', 'static', 'movement')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def williams(n):
    """Balanced Latin square for first-order carry-over (Williams, 1949): n rows if n is even,
    2n rows (the square and its mirror) if n is odd."""
    first = [0] + [((k + 1) // 2 if k % 2 else n - k // 2) for k in range(1, n)]   # 0, 1, n-1, 2, n-2, ...
    rows = [[(x + r) % n for x in first] for r in range(n)]
    if n % 2:
        rows += [row[::-1] for row in rows]
    return rows


def block_orders(n):
    """Six lexicographic permutations for the three body displays (spec §2); a Williams square otherwise."""
    return [list(p) for p in itertools.permutations(range(3))] if n == 3 else williams(n)


def order_index(source_rank, initialisation, n_orders):
    """Cycle orders so every source meets every order across initialisations, and every
    initialisation uses every order across sources."""
    return (source_rank + initialisation) % n_orders


def variant_schedule(families, count, rng):
    """`count` families for one trial type: every family once, then repeats without replacement."""
    out = []
    while len(out) < count:
        out += [families[i] for i in rng.permutation(len(families))]
    return out[:count]


@dataclass
class Session:
    """Timeline of one session. `segments`: (start_ms, end_ms, key), contiguous; `trials`: one
    dict per trial; key = ('black',) or (family_id, condition, frame)."""
    source: int
    initialisation: int
    displays: tuple
    order: list
    segments: list = field(default_factory=list)
    trials: list = field(default_factory=list)
    blocks: list = field(default_factory=list)

    @property
    def duration_ms(self):
        return self.segments[-1][1] if self.segments else 0

    def keys(self):
        """Distinct displayed images in order of first appearance."""
        return list(dict.fromkeys(key for _, _, key in self.segments))

    def steps(self, dt_ms=1):
        """Image index (into keys()) for every time step of length dt_ms."""
        grid = lambda t: int(round(t / dt_ms))
        require(all(abs(t / dt_ms - grid(t)) < 1e-9 for s, e, _ in self.segments for t in (s, e)),
                'Segment boundaries must fall on the time grid')
        index = {k: i for i, k in enumerate(self.keys())}
        out = np.empty(grid(self.duration_ms), np.int32)
        for start, end, key in self.segments:
            out[grid(start):grid(end)] = index[key]
        return out

    def add(self, duration, key):
        start = self.duration_ms
        self.segments.append((start, start + duration, key))
        return start


def _trial(session, rng, kind, block, display, family=None):
    first = int(rng.integers(TIMING['first_min_ms'], TIMING['first_max_ms'] + 1))
    if kind == 'fixation':
        start = session.add(first, BLACK)
        onset = session.add(TIMING['second_ms'], BLACK)
    else:
        start = session.add(first, (family, display, 'static'))
        onset = session.add(TIMING['second_ms'], (family, display, 'static' if kind == 'static' else 'movement'))
    session.trials.append(dict(trial=len(session.trials), block=block, display=display, kind=kind, family=family,
                               start_ms=start, onset2_ms=onset, probe_ms=onset + TIMING['probe_after_ms'],
                               end_ms=onset + TIMING['second_ms'], first_ms=first,
                               noise_seed=int(rng.integers(2 ** 63 - 1))))


def session(families, source, source_rank, initialisation, displays=BODIES, kind_code=0):
    """One model instance's evaluation session on one held-out source.

    families: the source's 24 family ids (appearance variants). kind_code separates the random
    streams of differently composed sessions (0 primary, 1 rectangle sensitivity, 2 controls).
    The rectangle sensitivity session uses kind_code 0 so it is paired with the primary one:
    same jitters, trial orders and variants, only the detached-hand images differ.
    """
    require(len(families) > 0 and len(set(families)) == len(families), 'Families must be distinct')
    orders = block_orders(len(displays))
    order = orders[order_index(source_rank, initialisation, len(orders))]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, source, initialisation, kind_code]))
    s = Session(source=source, initialisation=initialisation, displays=tuple(displays), order=order)
    for _ in range(TIMING['fixation_trials']):
        _trial(s, rng, 'fixation', -1, 'black')
    for block, position in enumerate(order):
        display = displays[position]
        familiar = families[int(rng.integers(len(families)))]
        start = s.add(TIMING['familiarisation_ms'], (familiar, display, 'static'))
        kinds = np.array(['static'] * TIMING['trials_per_type'] + ['movement'] * TIMING['trials_per_type'])
        kinds = kinds[rng.permutation(len(kinds))]
        schedule = {k: iter(variant_schedule(families, TIMING['trials_per_type'], rng)) for k in ('static', 'movement')}
        for kind in kinds:
            _trial(s, rng, kind, block, display, next(schedule[kind]))
        s.blocks.append(dict(block=block, display=display, familiarisation_family=familiar,
                             familiarisation_start_ms=start, movement_count=TIMING['trials_per_type']))
    for _ in range(TIMING['fixation_trials']):
        _trial(s, rng, 'fixation', -1, 'black')
    return s


def training_session(families, n_trials, seed, display='full_body'):
    """Auxiliary-task sequence (spec §6.3): full-body trials from training sources only, half
    with movement, same timing; no probes are used in training."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, seed, 99]))
    s = Session(source=-1, initialisation=seed, displays=(display,), order=[0])
    kinds = np.array(['static'] * (n_trials // 2) + ['movement'] * (n_trials - n_trials // 2))
    for kind in kinds[rng.permutation(n_trials)]:
        _trial(s, rng, kind, 0, display, families[int(rng.integers(len(families)))])
    return s
