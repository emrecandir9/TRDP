"""Frozen-encoder features for every image a session can display (spec §6.3).

The encoders run once per image (analysis/features.py, output/features/); frames are static, so
a network receives a constant vector for the duration of each frame. The black fixation display
is encoded separately (output/features_extra/black.npz). Files are checked against their
recorded hashes before use. Input standardisation is fitted on training sources only, from
full-body images, the only display used in training (spec §6.3), and then applied to every
display, so no statistic of a held-out source or of a test display enters preprocessing.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import trials  # noqa: E402

FEATURES = ROOT / 'output/features'
EXTRA = ROOT / 'output/features_extra'
ENCODERS = ('openclip', 'dinov2_cls', 'dinov2_patchmean')
require = trials.require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Bank:
    """Vectors of one representation, addressed by session keys."""

    def __init__(self, representation, directory=FEATURES, extra=EXTRA, arrays=None, index=None, black=None):
        if arrays is None:
            manifest = json.loads((directory / 'manifest.json').read_text())
            for name in ('features.npz', 'index.csv'):
                require(sha(directory / name) == manifest['files'][name], f'Changed feature file: {directory / name}')
            index = pd.read_csv(directory / 'index.csv')
            with np.load(directory / 'features.npz') as z:
                arrays = {representation: z[representation]}
            path = extra / 'black.npz'
            if path.is_file():
                with np.load(path) as z:
                    black = z[representation] if representation in z.files else None
        require(representation in arrays, f'Unknown representation: {representation}')
        self.representation = representation
        self.x = np.asarray(arrays[representation], np.float64)
        self.rows = {(f, c, r): i for i, (f, c, r) in enumerate(zip(index.family, index.condition, index.frame))}
        require(len(self.rows) == len(self.x), 'Index and features disagree')
        self.black = None if black is None else np.asarray(black, np.float64).ravel()

    @property
    def dim(self):
        return self.x.shape[1]

    def vector(self, key):
        if key == trials.BLACK:
            require(self.black is not None, 'Black-display features missing: output/features_extra/black.npz')
            return self.black
        require(key in self.rows, f'No features for {key}')
        return self.x[self.rows[key]]

    def matrix(self, keys):
        return np.stack([self.vector(k) for k in keys])

    def scaler(self, training_families, display='full_body'):
        """Mean and SD over both frames of the training families' full-body images."""
        x = self.matrix([(f, display, r) for f in training_families for r in ('static', 'movement')])
        sd = x.std(axis=0)
        return x.mean(axis=0), np.where(sd > 0, sd, 1.)

    def inputs(self, session, scaler):
        """Standardised vector per distinct key of a session, in session.keys() order."""
        mean, sd = scaler
        return (self.matrix(session.keys()) - mean) / sd
