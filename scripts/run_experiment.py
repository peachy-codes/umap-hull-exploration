from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import joblib

from geom_umap_experiment.data import load_mnist_subset
from geom_umap_experiment.experiment import run_experiment_2d


def parse_fractions(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description="MNIST UMAP hull curvature deformation experiment (2D).")
    p.add_argument("--n-samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drop-class", type=int, required=True)
    p.add_argument("--fractions", type=str, default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0")
    p.add_argument("--umap-n-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-metric", type=str, default="manhattan")
    p.add_argument("--umap-n-epochs", type=int, default=0, help="0 means default")
    p.add_argument("--hull-alpha", type=float, default=.5)
    p.add_argument("--hull-smooth", type=float, default=0.002)
    p.add_argument("--curve-samples", type=int, default=5000)
    p.add_argument("--save-artifacts", action="store_true", help="Save embeddings, hull samples, curvature samples, and plots per stage")
    p.add_argument("--save-models", action="store_true", help="Also persist fitted UMAP models via joblib (can be large)")
    p.add_argument("--n-jobs", type=int, default=1, help="UMAP parallelism. Use 1 for best determinism.")
    p.add_argument("--outdir", type=str, default="outputs")
    args = p.parse_args()

    fractions = parse_fractions(args.fractions)
    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}_drop{args.drop_class}"
    outdir = base_outdir / "runs" / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "stages").mkdir(parents=True, exist_ok=True)

    # Logging to console + file
    log = logging.getLogger("geom_umap_experiment")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(outdir / "run.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

    # Plot styling
    sns.set_theme(style="whitegrid", context="talk")

    log.info("Loading MNIST via OpenML")
    X, y = load_mnist_subset(n_samples=args.n_samples, seed=args.seed, drop_classes=None)
    log.info("Loaded X=%s y=%s", X.shape, y.shape)

    umap_params = dict(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        n_epochs=None if args.umap_n_epochs == 0 else args.umap_n_epochs,
    )

    # Persist run configuration for reproducibility
    config = {
        "run_id": run_id,
        "timestamp_local": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "umap_params": umap_params,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "drop_class": args.drop_class,
        "fractions": fractions,
        "hull_alpha": args.hull_alpha,
        "hull_smooth": args.hull_smooth,
        "curve_samples": args.curve_samples,
        "n_jobs": args.n_jobs,
        "save_artifacts": bool(args.save_artifacts),
        "save_models": bool(args.save_models),
    }
    (outdir / "run_config.json").write_text(json.dumps(config, indent=2))

    # Stage callback: print progress + optionally persist artifacts
    def _save_stage_plot(stage_dir: Path, emb2: np.ndarray, curve: np.ndarray, y_stage: np.ndarray, title: str):
        # Local imports to keep script startup fast / explicit


        sns.set_theme(style="white", context="talk")

        classes = np.unique(y_stage)
        palette = sns.color_palette("tab10", n_colors=10)
        color_map = {c: palette[int(c) % 10] for c in classes}

        fig, ax = plt.subplots(figsize=(9.0, 6.5))

        # Plot each class separately to get a clean categorical legend
        for c in sorted(classes):
            m = (y_stage == c)
            ax.scatter(
                emb2[m, 0],
                emb2[m, 1],
                s=10,
                alpha=0.55,
                linewidths=0,
                color=color_map[c],
                label=str(c),
                rasterized=True,
            )

        # Hull overlay
        ax.plot(curve[:, 0], curve[:, 1], linewidth=2.5, color="black", alpha=0.9)

        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

        # Legend outside the axes
        leg = ax.legend(
            title="Digit",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            borderpad=0.6,
            handletextpad=0.4,
        )
        # Ensure opaque legend markers
        for h in leg.legend_handles:
            try:
                h.set_alpha(1.0)
            except Exception:
                pass

        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(stage_dir / "embedding_hull.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

    # def _save_stage_plot(stage_dir: Path, emb2: np.ndarray, curve: np.ndarray, title: str):
    #     plt.figure(figsize=(7, 6))
    #     plt.scatter(emb2[:, 0], emb2[:, 1], s=2, alpha=0.5)
    #     plt.plot(curve[:, 0], curve[:, 1], linewidth=2)
    #     plt.title(title)
    #     plt.tight_layout()
    #     plt.savefig(stage_dir / "embedding_hull.png", dpi=170)
    #     plt.close()

    def on_stage(payload: dict):
        stage = payload["stage"]
        f = float(payload["fraction_added"])
        if stage == "full":
            log.info("Fitted FULL UMAP and hull")
            if args.save_artifacts:
                stage_dir = outdir / "stages" / "full"
                stage_dir.mkdir(parents=True, exist_ok=True)
                np.save(stage_dir / "embedding.npy", payload["embedding"])
                np.save(stage_dir / "curve.npy", payload["curve"])
                np.save(stage_dir / "curvature.npy", payload["curvature"])
                np.save(stage_dir / "curvature_normalized.npy", payload["curvature_normalized"])
                if args.save_models and "umap_model" in payload:
                    joblib.dump(payload["umap_model"], stage_dir / "umap_model.joblib")
                _save_stage_plot(stage_dir, payload["embedding"], payload["curve"], y, "Full UMAP + hull")
            return

        idx_stage = payload["idx_stage"]
        n_stage = int(payload.get("n_stage", len(idx_stage)))
        n_added = int(payload.get("n_added", -1))
        dist = float(payload.get("curvature_wasserstein", float("nan")))
        log.info("Stage f=%.3f | n_stage=%d | dist=%.6f", f, n_stage, dist)

        if args.save_artifacts:
            stage_dir = outdir / "stages" / f"fraction_{f:.3f}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            np.save(stage_dir / "idx_stage.npy", idx_stage)
            np.save(stage_dir / "embedding.npy", payload["embedding"])
            np.save(stage_dir / "curve.npy", payload["curve"])
            np.save(stage_dir / "curvature.npy", payload["curvature"])
            np.save(stage_dir / "curvature_normalized.npy", payload["curvature_normalized"])
            if args.save_models and "umap_model" in payload:
                joblib.dump(payload["umap_model"], stage_dir / "umap_model.joblib")
            _save_stage_plot(stage_dir, payload["embedding"], payload["curve"], y[idx_stage], f"Stage f={f:.3f} UMAP + hull")

    log.info("Starting experiment: drop_class=%d", args.drop_class)
    E_full, results, artifacts = run_experiment_2d(
        X=X,
        y=y,
        drop_class=args.drop_class,
        fractions=fractions,
        seed=args.seed,
        umap_params=umap_params,
        hull_alpha=args.hull_alpha,
        hull_smooth=args.hull_smooth,
        n_curve_samples=args.curve_samples,
        use_abs_curvature=True,
        n_jobs=args.n_jobs,
        return_models=bool(args.save_models),
        on_stage=on_stage,
    )

    # Write CSV
    csv_path = outdir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fraction_added", "n_stage", "n_added", "curvature_wasserstein"])
        for r in results:
            w.writerow([r.fraction_added, r.n_stage, r.n_added, r.curvature_wasserstein])

    # Plot deformation curve
    xs = [r.fraction_added for r in results]
    ys = [r.curvature_wasserstein for r in results]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    sns.lineplot(
        x=xs,
        y=ys,
        marker="o",
        markersize=7,
        linewidth=2.7,
        ax=ax,
        errorbar=None,
    )

    ax.set_xlabel("Fraction of dropped class re-added")
    ax.set_ylabel("Wasserstein distance (scale-normalized mean curvature)")
    ax.set_title(f"Deformation vs add-back fraction\n(drop_class={args.drop_class})")

    ax.grid(True, which="major", alpha=0.25)
    sns.despine(ax=ax)

    # Highlight the minimum
    imin = int(np.argmin(ys))
    ax.axhline(ys[imin], linestyle="--", linewidth=1.2, alpha=0.6)
    ax.scatter([xs[imin]], [ys[imin]], s=90, zorder=5)

    fig.tight_layout()
    fig.savefig(outdir / "deformation_curve.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    log.info("Wrote: %s", csv_path)
    log.info("Wrote: %s", outdir / "deformation_curve.png")
    log.info("Wrote: %s", outdir / "run_config.json")
    log.info("Wrote: %s", outdir / "run.log")

    # Optional: persist full embedding + hull even if --save-artifacts is not set
    np.save(outdir / "E_full.npy", E_full)
    np.save(outdir / "curve_full.npy", artifacts["curve_full"])
    log.info("Wrote: %s", outdir / "E_full.npy")
    log.info("Wrote: %s", outdir / "curve_full.npy")


if __name__ == "__main__":
    main()
