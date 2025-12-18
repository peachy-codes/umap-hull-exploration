from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _collect_stage_dirs(run_dir: Path):
    stages_dir = run_dir / "stages"
    if not stages_dir.exists():
        raise FileNotFoundError(f"Missing stages dir: {stages_dir}")

    dirs = [p for p in stages_dir.iterdir() if p.is_dir()]
    names = sorted([d.name for d in dirs])

    # Ensure "full" first if present
    if "full" in names:
        names.remove("full")
        names = ["full"] + names

    stage_dirs = [stages_dir / n for n in names if (n == "full" or n.startswith("fraction_"))]
    if not stage_dirs:
        raise RuntimeError(f"No stage folders found under: {stages_dir}")

    return stage_dirs


def _stage_fraction(stage_dir: Path):
    name = stage_dir.name
    if name == "full":
        return None
    if name.startswith("fraction_"):
        try:
            return float(name.replace("fraction_", ""))
        except ValueError:
            return None
    return None


def _load_curvature(stage_dir: Path, normalized: bool):
    fname = "curvature_normalized.npy" if normalized else "curvature.npy"
    fpath = stage_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Missing {fname} at: {fpath}")
    x = np.load(fpath)
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    return x


def main():
    ap = argparse.ArgumentParser(description="Plot curvature distributions from a saved run.")
    ap.add_argument("--run-dir", type=str, required=True, help="Path to outputs/runs/<run_id>")
    ap.add_argument("--outdir", type=str, default="", help="Defaults to <run-dir>/plots_curvature")
    ap.add_argument("--normalized", action="store_true", help="Use curvature_normalized.npy (recommended)")
    ap.add_argument("--use-abs", action="store_true", help="Plot absolute curvature values")
    ap.add_argument("--max-stages", type=int, default=0, help="0 = all stages; otherwise limit for readability")
    ap.add_argument("--kde", action="store_true", help="Use KDE (can be slow for huge arrays)")
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--sample", type=int, default=0, help="Randomly subsample curvature points per stage for plotting (0 = no subsample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xlim-quantiles", type=str, default="0.001,0.999",
                    help="Quantile range for x-limits (e.g. '0.001,0.999' or '' to disable)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    outdir = Path(args.outdir) if args.outdir else (run_dir / "plots_curvature")
    outdir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    stage_dirs = _collect_stage_dirs(run_dir)
    if args.max_stages and args.max_stages > 0:
        stage_dirs = stage_dirs[: args.max_stages]

    rng = np.random.default_rng(args.seed)

    # Load per-stage curvature arrays
    stage_data = []
    for sd in stage_dirs:
        curv = _load_curvature(sd, normalized=args.normalized)
        if args.use_abs:
            curv = np.abs(curv)

        if args.sample and args.sample > 0 and len(curv) > args.sample:
            idx = rng.choice(len(curv), size=args.sample, replace=False)
            curv = curv[idx]

        stage_data.append({
            "name": sd.name,
            "fraction": _stage_fraction(sd),
            "curv": curv,
        })

    # Choose x-limits consistently across plots (optional robust quantiles)
    xlim = None
    if args.xlim_quantiles.strip():
        qlo, qhi = [float(x.strip()) for x in args.xlim_quantiles.split(",")]
        all_curv = np.concatenate([d["curv"] for d in stage_data if d["curv"].size > 0])
        if all_curv.size:
            lo = np.quantile(all_curv, qlo)
            hi = np.quantile(all_curv, qhi)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                xlim = (lo, hi)

    # Styling
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("viridis", n_colors=len(stage_data))

    # 1) Overlay plot
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, d in enumerate(stage_data):
        label = "full" if d["name"] == "full" else f"f={d['fraction']:.3f}"
        if args.kde:
            sns.kdeplot(d["curv"], ax=ax, linewidth=2.0, label=label, color=palette[i])
        else:
            ax.hist(d["curv"], bins=args.bins, density=True, histtype="step", linewidth=2.0,
                    label=label, color=palette[i])

    title_kind = "normalized" if args.normalized else "raw"
    abs_tag = " | abs" if args.use_abs else ""
    ax.set_title(f"Curvature distributions (overlay) [{title_kind}{abs_tag}]")
    ax.set_xlabel("Curvature value")
    ax.set_ylabel("Density")
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, title="Stage")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "curvature_distributions_overlay.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # 2) Small multiples grid
    n = len(stage_data)
    ncols = 4 if n >= 8 else 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.6 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax, d, c in zip(axes, stage_data, palette):
        label = "full" if d["name"] == "full" else f"f={d['fraction']:.3f}"
        if args.kde:
            sns.kdeplot(d["curv"], ax=ax, linewidth=2.0, color=c)
        else:
            ax.hist(d["curv"], bins=args.bins, density=True, color=c, alpha=0.65, edgecolor=None)
        ax.set_title(label)
        if xlim:
            ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.25)

    # Hide any extra axes
    for ax in axes[len(stage_data):]:
        ax.axis("off")

    fig.suptitle(f"Curvature distributions per stage [{title_kind}{abs_tag}]", y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(outdir / "curvature_distributions_grid.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # 3) Summary statistics vs fraction (excluding full)
    fracs = []
    med = []
    p10 = []
    p90 = []
    for d in stage_data:
        if d["name"] == "full":
            continue
        x = d["curv"]
        if x.size == 0 or d["fraction"] is None:
            continue
        fracs.append(d["fraction"])
        med.append(np.median(x))
        p10.append(np.quantile(x, 0.10))
        p90.append(np.quantile(x, 0.90))

    if fracs:
        order = np.argsort(fracs)
        fracs = np.array(fracs)[order]
        med = np.array(med)[order]
        p10 = np.array(p10)[order]
        p90 = np.array(p90)[order]

        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        ax.plot(fracs, med, marker="o", linewidth=2.5)
        ax.fill_between(fracs, p10, p90, alpha=0.25)
        ax.set_xlabel("Fraction of dropped class re-added")
        ax.set_ylabel("Curvature (median with 10–90% band)")
        ax.set_title(f"Curvature summary vs fraction [{title_kind}{abs_tag}]")
        ax.grid(True, alpha=0.25)
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(outdir / "curvature_summary_vs_fraction.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

    # Write a minimal README with pointers
    readme = outdir / "README.txt"
    lines = [
        "Curvature plots written:",
        "- curvature_distributions_overlay.png",
        "- curvature_distributions_grid.png",
        "- curvature_summary_vs_fraction.png (if stage fractions exist)",
        "",
        f"Run: {run_dir}",
        f"Normalized: {args.normalized}",
        f"Abs: {args.use_abs}",
        f"KDE: {args.kde}",
        f"Sample: {args.sample}",
    ]
    readme.write_text("\n".join(lines))

    print(f"Wrote plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()