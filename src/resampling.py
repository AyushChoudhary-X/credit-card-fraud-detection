"""Class-imbalance handling: random undersampling and a from-scratch SMOTE.

SMOTE is implemented here (rather than imported from imbalanced-learn) so the project has few
dependencies and the algorithm is visible. It follows Chawla et al. (2002): each synthetic fraud
point is a random interpolation between a real fraud point and one of its k nearest fraud neighbours.

IMPORTANT: resample the TRAINING set only. Resampling validation/test data would give fake metrics.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def random_undersample(X, y, ratio: float, seed: int = 42):
    """Keep all fraud rows and randomly keep enough legit rows so that fraud:legit == ratio."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_keep = min(len(neg), int(round(len(pos) / ratio)))
    keep_neg = rng.choice(neg, size=n_keep, replace=False)
    idx = rng.permutation(np.concatenate([pos, keep_neg]))
    return X[idx], y[idx]


def smote(X, y, ratio: float, k: int = 5, seed: int = 42):
    """Oversample the fraud class with synthetic points until fraud:legit == ratio."""
    rng = np.random.default_rng(seed)
    X_pos = X[y == 1]
    n_pos, n_neg = len(X_pos), int((y == 0).sum())
    n_new = int(round(ratio * n_neg)) - n_pos
    if n_new <= 0:
        return X, y
    k = min(k, n_pos - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_pos)
    neigh = nn.kneighbors(X_pos, return_distance=False)[:, 1:]  # drop the point itself

    base = rng.integers(0, n_pos, size=n_new)
    partner = neigh[base, rng.integers(0, k, size=n_new)]
    gap = rng.random((n_new, 1))
    synthetic = X_pos[base] + gap * (X_pos[partner] - X_pos[base])

    X_out = np.vstack([X, synthetic])
    y_out = np.concatenate([y, np.ones(n_new, dtype=y.dtype)])
    return X_out, y_out
