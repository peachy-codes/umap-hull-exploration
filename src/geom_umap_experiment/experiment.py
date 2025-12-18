from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .umap_embed import fit_umap_2d
from .hull2d import fit_hull_curve_2d, sample_curve_uniform_arclength, curvature_from_spline
from .metrics import normalized_curvature_samples, wasserstein_1d


@dataclass
class StageResult:
    fraction_added: float
    n_stage: int
    n_added: int
    curvature_wasserstein: float


def run_experiment_2d(
    X: np.ndarray,
    y: np.ndarray,
    drop_class: int,
    fractions: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = 0,
    umap_params: Optional[Dict] = None,
    hull_alpha: float = 1.5,
    hull_smooth: float = 0.002,
    n_curve_samples: int = 5000,
    use_abs_curvature: bool = True,
    n_jobs: int = 1,
    return_models: bool = False,
    on_stage: Optional[Callable[[dict], None]] = None,
):
    """
    Runs the 2D version of the experiment and returns:
    - full embedding (N,2)
    - list of StageResult
    - dict of optional artifacts for plotting (embeddings and hull samples)
    """
    if umap_params is None:
        umap_params = dict(n_neighbors=30, min_dist=0.1, metric="manhattan", n_epochs=None)

    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    rng = np.random.default_rng(seed)

    idx_all = np.arange(len(X))
    idx_drop = idx_all[y == int(drop_class)]
    idx_keep = idx_all[y != int(drop_class)]

    idx_drop_shuf = idx_drop.copy()
    rng.shuffle(idx_drop_shuf)

    # Full reference UMAP
    if return_models:
        E_full, umap_full = fit_umap_2d(X, seed=seed, n_jobs=n_jobs, return_model=True, **umap_params)
    else:
        E_full = fit_umap_2d(X, seed=seed, n_jobs=n_jobs, **umap_params)

    # Full hull curvature distribution
    tck_full, _ = fit_hull_curve_2d(E_full, alpha=hull_alpha, smooth=hull_smooth, seed=seed)
    curve_full, u_full = sample_curve_uniform_arclength(tck_full, n=n_curve_samples)
    k_full = curvature_from_spline(tck_full, u_full)
    kn_full = normalized_curvature_samples(k_full, curve_full, use_abs=use_abs_curvature)

    results: List[StageResult] = []
    artifacts = {"E_full": E_full, "curve_full": curve_full}

    if on_stage is not None:
        on_stage({
            "stage": "full",
            "fraction_added": 1.0,
            "idx_stage": np.arange(len(X)),
            "embedding": E_full,
            **({"umap_model": umap_full} if return_models else {}),
            "curve": curve_full,
            "curvature": k_full,
            "curvature_normalized": kn_full,
        })

    for f in fractions:
        f = float(f)
        k = int(round(f * len(idx_drop_shuf)))
        idx_added = idx_drop_shuf[:k]
        idx_stage = np.concatenate([idx_keep, idx_added])

        if return_models:
            E_stage, umap_stage = fit_umap_2d(X[idx_stage], seed=seed, n_jobs=n_jobs, return_model=True, **umap_params)
        else:
            E_stage = fit_umap_2d(X[idx_stage], seed=seed, n_jobs=n_jobs, **umap_params)

        # Hull curvature distribution for stage
        tck_s, _ = fit_hull_curve_2d(E_stage, alpha=hull_alpha, smooth=hull_smooth, seed=seed)
        curve_s, u_s = sample_curve_uniform_arclength(tck_s, n=n_curve_samples)
        k_s = curvature_from_spline(tck_s, u_s)
        kn_s = normalized_curvature_samples(k_s, curve_s, use_abs=use_abs_curvature)

        dist = wasserstein_1d(kn_s, kn_full)

        results.append(StageResult(
            fraction_added=f,
            n_stage=int(len(idx_stage)),
            n_added=int(k),
            curvature_wasserstein=float(dist),
        ))

        # store minimal artifacts for optional plotting
        artifacts[f"curve_stage_{f}"] = curve_s

        if on_stage is not None:
            on_stage({
                "stage": "stage",
                "fraction_added": f,
                "idx_stage": idx_stage,
                "n_stage": int(len(idx_stage)),
                "n_added": int(k),
                "embedding": E_stage,
                **({"umap_model": umap_stage} if return_models else {}),
                "curve": curve_s,
                "curvature": k_s,
                "curvature_normalized": kn_s,
                "curvature_wasserstein": float(dist),
            })

    return E_full, results, artifacts
