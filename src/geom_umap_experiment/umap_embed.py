from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import umap


def fit_umap_2d(
    X: np.ndarray,
    seed: int = 0,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "manhattan",
    n_epochs: Optional[int] = None,
    n_jobs: int = 1,
    return_model: bool = False,
):
    """
    Fit a 2D UMAP embedding on raw MNIST pixels (no scaling/PCA).
    """
    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
        n_epochs=n_epochs,
        n_jobs=int(n_jobs) if n_jobs is not None else None,
    )
    emb = reducer.fit_transform(X)
    if return_model:
        return emb, reducer
    return emb
