from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.stats import wasserstein_distance


def rms_radius(points: np.ndarray) -> float:
    """
    Characteristic length scale for scale-invariance: RMS radius of a point set.
    """
    c = points.mean(axis=0)
    r2 = np.sum((points - c) ** 2, axis=1)
    return float(np.sqrt(np.mean(r2)))


def normalized_curvature_samples(curvature: np.ndarray, curve_points: np.ndarray, use_abs: bool = True) -> np.ndarray:
    """
    Make curvature scale-invariant by multiplying by a characteristic length (RMS radius).
    Optionally take absolute value for sign-robustness.
    """
    k = np.asarray(curvature, dtype=float)
    if use_abs:
        k = np.abs(k)
    L = rms_radius(np.asarray(curve_points, dtype=float))
    return k * L


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    1D Wasserstein distance between two empirical samples.
    """
    return float(wasserstein_distance(a, b))
