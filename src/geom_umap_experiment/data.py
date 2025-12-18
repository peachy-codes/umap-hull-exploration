from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist_subset(
    n_samples: int | None = 20000,
    seed: int = 0,
    drop_classes: Optional[Iterable[int]] = None,
):
    """
    Load MNIST (784-dim pixels) from OpenML and optionally:
    - drop specific classes
    - subsample without replacement

    Returns
    -------
    X : (N, 784) float array
    y : (N,) int array
    """
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)

    if drop_classes is not None:
        drop_set = set(int(c) for c in drop_classes)
        mask = ~np.isin(y, list(drop_set))
        X = X[mask]
        y = y[mask]

    if n_samples is not None:
        if n_samples > len(X):
            raise ValueError(f"Requested n_samples={n_samples} but only {len(X)} available.")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=int(n_samples), replace=False)
        X = X[idx]
        y = y[idx]

    return X, y
