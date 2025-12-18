from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

from geom_umap_experiment.data import load_mnist_subset


def _load_stage(run_dir: Path, stage_name: str):
    """
    Returns:
      idx_stage: (k,) indices into the fixed X,y subset (global index space for the run)
      emb: (k,2) embedding for those indices (same row order as idx_stage)
      curve: (m,2) hull curve points
    """
    stage_dir = run_dir / "stages" / stage_name
    if not stage_dir.exists():
        raise FileNotFoundError(f"Missing stage dir: {stage_dir}")

    curve = np.load(stage_dir / "curve.npy")

    if stage_name == "full":
        emb = np.load(stage_dir / "embedding.npy")
        idx_stage = np.arange(emb.shape[0], dtype=int)
    else:
        idx_stage = np.load(stage_dir / "idx_stage.npy")
        emb = np.load(stage_dir / "embedding.npy")

    return idx_stage, emb, curve


def _collect_stage_names(run_dir: Path):
    """
    Returns ordered stage names: ["full", "fraction_0.000", "fraction_0.050", ...]
    If "full" exists, it will be used as the first reference.
    """
    stages_dir = run_dir / "stages"
    if not stages_dir.exists():
        raise FileNotFoundError(f"Missing stages directory: {stages_dir}")

    names = sorted([p.name for p in stages_dir.iterdir() if p.is_dir()])
    # Ensure "full" first if present
    if "full" in names:
        names.remove("full")
        names = ["full"] + names

    # Keep only known patterns
    kept = []
    for n in names:
        if n == "full" or n.startswith("fraction_"):
            kept.append(n)
    if not kept:
        raise RuntimeError(f"No stage directories found under {stages_dir}")
    return kept


def _to_global_arrays(n_total: int, idx_stage: np.ndarray, emb_stage: np.ndarray):
    """
    Return global-structured embedding arrays:
      emb_global: (n_total,2) with NaNs where missing
      present: (n_total,) bool mask
    """
    emb_global = np.full((n_total, 2), np.nan, dtype=float)
    emb_global[idx_stage] = emb_stage
    present = np.zeros(n_total, dtype=bool)
    present[idx_stage] = True
    return emb_global, present


def _interp(a: np.ndarray, b: np.ndarray, t: float):
    return (1.0 - t) * a + t * b


def main():
    ap = argparse.ArgumentParser(description="Animate saved UMAP stage artifacts (points + hull).")
    ap.add_argument("--run-dir", type=str, required=True, help="Path to outputs/runs/<run_id>")
    ap.add_argument("--out", type=str, default="animation.gif", help="Output file (.gif or .mp4)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--frames-per-transition", type=int, default=18, help="Frames between each stage pair")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--point-size", type=float, default=10.0)
    ap.add_argument("--alpha-present", type=float, default=0.70)
    ap.add_argument("--alpha-missing", type=float, default=0.0)
    ap.add_argument("--fade-in", type=float, default=0.85, help="Fraction of transition used to fade new points in")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing run_config.json at: {cfg_path}")

    cfg = json.loads(cfg_path.read_text())
    n_samples = int(cfg["n_samples"])
    seed = int(cfg["seed"])

    # Reconstruct y deterministically (same subsample as the run)
    # Note: we do NOT need X for animation; only y for coloring.
    _, y = load_mnist_subset(n_samples=n_samples, seed=seed, drop_classes=None)
    y = np.asarray(y, dtype=int)
    n_total = len(y)

    stage_names = _collect_stage_names(run_dir)

    # Load all stages and lift to global arrays so points can "move" across stages
    emb_globals = []
    present_masks = []
    curves = []
    stage_titles = []

    for name in stage_names:
        idx_stage, emb, curve = _load_stage(run_dir, name)
        emb_g, present = _to_global_arrays(n_total, idx_stage, emb)
        emb_globals.append(emb_g)
        present_masks.append(present)
        curves.append(curve)

        if name == "full":
            stage_titles.append("full")
        else:
            stage_titles.append(name.replace("fraction_", "f="))

    # Plot styling
    sns.set_theme(style="white", context="talk")
    palette = sns.color_palette("tab10", n_colors=10)
    colors = np.array([palette[int(c) % 10] for c in y], dtype=float)

    # Precompute axis limits across all stages for stable framing
    all_xy = np.vstack([eg[np.isfinite(eg[:, 0])] for eg in emb_globals])
    x_min, y_min = np.nanmin(all_xy, axis=0)
    x_max, y_max = np.nanmax(all_xy, axis=0)
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    sns.despine(ax=ax)

    # One scatter artist for all points (we'll hide missing via alpha=0)
    # Initialize with first stage
    emb0 = emb_globals[0]
    pres0 = present_masks[0]
    xy0 = emb0.copy()
    # For missing points, put them at NaN-safe location and alpha=0
    nan_loc = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=float)
    xy0[~pres0] = nan_loc

    alphas0 = np.full(n_total, args.alpha_missing, dtype=float)
    alphas0[pres0] = args.alpha_present

    rgba0 = np.column_stack([colors, alphas0])

    sc = ax.scatter(
        xy0[:, 0],
        xy0[:, 1],
        s=args.point_size,
        c=rgba0,
        linewidths=0,
        rasterized=True,
    )

    # Hull line
    curve0 = curves[0]
    (ln,) = ax.plot(curve0[:, 0], curve0[:, 1], linewidth=2.6, color="black", alpha=0.9)

    # Legend: digits 0–9
    handles = []
    labels = []
    for d in range(10):
        h = ax.scatter([], [], s=60, color=palette[d], label=str(d))
        handles.append(h)
        labels.append(str(d))
    ax.legend(
        handles=handles,
        labels=labels,
        title="Digit",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        borderpad=0.6,
        handletextpad=0.4,
    )

    title = ax.set_title(f"Stage: {stage_titles[0]}")

    # Animation indexing: each transition is frames-per-transition frames
    n_stages = len(stage_names)
    frames_per = int(args.frames_per_transition)
    total_frames = (n_stages - 1) * frames_per + 1

    def frame_to_stage(frame_idx: int):
        if frame_idx == total_frames - 1:
            return n_stages - 2, 1.0  # last frame = final stage
        k = frame_idx // frames_per
        j = frame_idx % frames_per
        t = j / float(frames_per)
        return k, t

    def update(frame_idx: int):
        k, t = frame_to_stage(frame_idx)

        A = emb_globals[k]
        B = emb_globals[k + 1]
        presA = present_masks[k]
        presB = present_masks[k + 1]
        curveA = curves[k]
        curveB = curves[k + 1]

        # Point positions:
        # - if present in both: interpolate
        # - if only in A: keep at A and fade out (we effectively drop alpha)
        # - if only in B: start at B but fade in
        xy = np.empty((n_total, 2), dtype=float)
        xy[:] = nan_loc

        both = presA & presB
        onlyA = presA & (~presB)
        onlyB = (~presA) & presB

        xy[both] = _interp(A[both], B[both], t)
        xy[onlyA] = A[onlyA]
        xy[onlyB] = B[onlyB]

        # Alpha ramp:
        # present points at full alpha
        # new points fade in over early portion of transition
        # disappearing points fade out over transition
        alphas = np.full(n_total, args.alpha_missing, dtype=float)
        alphas[both] = args.alpha_present

        fade_window = max(1e-6, float(args.fade_in))
        t_fade = min(1.0, t / fade_window)

        alphas[onlyB] = args.alpha_present * t_fade
        alphas[onlyA] = args.alpha_present * (1.0 - t)

        rgba = np.column_stack([colors, alphas])

        sc.set_offsets(xy)
        sc.set_facecolors(rgba)

        # Hull interpolation (assumes same number of samples; if not, we fall back to nearest length)
        m = min(curveA.shape[0], curveB.shape[0])
        curve = _interp(curveA[:m], curveB[:m], t)
        ln.set_data(curve[:, 0], curve[:, 1])

        title.set_text(f"Stage: {stage_titles[k]} → {stage_titles[k+1]}   t={t:.2f}")
        return sc, ln, title

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / args.fps, blit=False)

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".gif":
        anim.save(out_path, writer="pillow", fps=args.fps, dpi=args.dpi)
    elif out_path.suffix.lower() == ".mp4":
        anim.save(out_path, writer="ffmpeg", fps=args.fps, dpi=args.dpi)
    else:
        raise ValueError("Output must end with .gif or .mp4")

    plt.close(fig)
    print(f"Wrote animation to: {out_path.resolve()}")


if __name__ == "__main__":
    main()