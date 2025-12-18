from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import splprep, splev


def alpha_boundary_polyline(points2: np.ndarray, alpha: float, seed: int = 0) -> np.ndarray:
    """
    Compute an ordered boundary polyline using an alpha-shape-like filter on Delaunay triangles.
    Returns a closed polyline (M,2) with last point equal to first.

    alpha higher => tighter/more concave (too high can fragment).
    """
    P = np.asarray(points2, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points2 must have shape (N,2).")

    # Tiny jitter to reduce degeneracy from duplicates
    rng = np.random.default_rng(seed)
    Pj = P + rng.normal(scale=1e-9, size=P.shape)

    tri = Delaunay(Pj)
    simplices = tri.simplices

    A = Pj[simplices[:, 0]]
    B = Pj[simplices[:, 1]]
    C = Pj[simplices[:, 2]]

    a = np.linalg.norm(B - C, axis=1)
    b = np.linalg.norm(C - A, axis=1)
    c = np.linalg.norm(A - B, axis=1)

    s = (a + b + c) / 2.0
    area2 = np.maximum(s * (s - a) * (s - b) * (s - c), 0.0)
    area = np.sqrt(area2)
    R = (a * b * c) / np.maximum(4.0 * area, 1e-12)
    keep = R < (1.0 / float(alpha))

    edges = []
    for t in simplices[keep]:
        edges.append(tuple(sorted((t[0], t[1]))))
        edges.append(tuple(sorted((t[1], t[2]))))
        edges.append(tuple(sorted((t[2], t[0]))))

    edge_counts = Counter(edges)
    boundary_edges = [e for e, ct in edge_counts.items() if ct == 1]
    if not boundary_edges:
        raise ValueError("No boundary edges found. Try decreasing alpha.")

    # adjacency for ordering
    adj = {}
    for i, j in boundary_edges:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    # start node
    start = None
    for k, nbrs in adj.items():
        if len(nbrs) == 1:
            start = k
            break
    if start is None:
        start = boundary_edges[0][0]

    poly_idx = [start]
    prev, cur = None, start
    for _ in range(len(boundary_edges) + 10):
        nbrs = adj[cur]
        nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
        if nxt is None:
            break
        poly_idx.append(nxt)
        if nxt == start:
            break
        prev, cur = cur, nxt

    poly = Pj[poly_idx]
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def fit_hull_curve_2d(
    points2: np.ndarray,
    alpha: float = 1.5,
    smooth: float = 0.002,
    seed: int = 0,
) -> Tuple[tuple, np.ndarray]:
    """
    Fit a smooth periodic spline to the alpha-shape boundary polyline.

    Returns
    -------
    tck : spline representation (for splev)
    boundary_poly : (M,2) closed polyline used for fitting
    """
    boundary_poly = alpha_boundary_polyline(points2, alpha=alpha, seed=seed)
    x = boundary_poly[:, 0]
    y = boundary_poly[:, 1]
    # per=True => periodic/closed; requires first==last (ensured)
    tck, _ = splprep([x, y], s=float(smooth), per=True)
    return tck, boundary_poly


def sample_curve_uniform_arclength(tck: tuple, n: int = 5000, dense: int = 50000) -> np.ndarray:
    """
    Sample a periodic spline approximately uniformly by arc length.

    Approach:
    - evaluate densely in parameter u
    - compute cumulative arc length
    - invert arc length -> u via interpolation
    """
    dense = int(dense)
    n = int(n)
    u_dense = np.linspace(0.0, 1.0, dense, endpoint=False)
    xd, yd = splev(u_dense, tck)
    pts = np.column_stack([xd, yd])

    diffs = np.diff(pts, axis=0, append=pts[:1])
    seg = np.linalg.norm(diffs, axis=1)
    s = np.cumsum(seg)
    total = s[-1]
    s = np.concatenate([[0.0], s[:-1]])  # segment start arclengths

    # target arclengths
    s_t = np.linspace(0.0, total, n, endpoint=False)

    # invert using monotone interpolation on (s, u_dense)
    # ensure s is strictly increasing (handle rare zeros)
    eps = 1e-12
    s_mono = s + eps * np.arange(len(s))
    u_t = np.interp(s_t, s_mono, u_dense)

    x, y = splev(u_t, tck)
    return np.column_stack([x, y]), u_t


def curvature_from_spline(tck: tuple, u: np.ndarray) -> np.ndarray:
    """
    Curvature of a 2D parametric curve r(u) = (x(u), y(u)).
    k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    """
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    dx = np.asarray(dx); dy = np.asarray(dy)
    ddx = np.asarray(ddx); ddy = np.asarray(ddy)

    num = np.abs(dx * ddy - dy * ddx)
    den = (dx*dx + dy*dy) ** 1.5
    k = num / np.maximum(den, 1e-12)
    return k
