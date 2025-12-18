# MNIST UMAP Hull Curvature Experiment (2D)

This project runs the experiment you described in a **2D UMAP embedding**:

1. Fit UMAP on a fixed MNIST subset (the "full" reference).
2. Drop one digit class; then gradually add it back in fractions, refitting UMAP each stage.
3. For each embedding, fit an **enclosing boundary** (concave hull via alpha-shape, then a smooth periodic spline).
4. Sample the boundary approximately uniformly by arc length.
5. Compute **curvature** along the boundary (2D analog of mean curvature).
6. Enforce **scale invariance** by multiplying curvature by a characteristic length scale (RMS radius of the boundary).
7. Compare the **distribution** of normalized curvature to the full reference using Wasserstein distance.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# With a src/ layout, install the package so imports work when running scripts
pip install -e .

python scripts/run_experiment.py --n-samples 20000 --seed 0 --drop-class 7
```

Outputs go under `outputs/runs/<run_id>/`:
- `results.csv` deformation metrics per fraction
- `deformation_curve.png` (seaborn styling)
- `run_config.json` (command + parameters + environment)
- `run.log` (progress logging)
- `E_full.npy`, `curve_full.npy`
- If you pass `--save-artifacts`, per-stage folders under `stages/` with:
  - `embedding.npy`, `curve.npy`, `curvature.npy`, `curvature_normalized.npy`
  - `embedding_hull.png`
  - (optional) `umap_model.joblib` if you also pass `--save-models`

## Notes

- This implementation targets **2D** first. The hull is a closed curve; curvature is computed from spline derivatives.
- Extending to 3D+ will require a different hull representation (e.g., implicit SDF or meshed surface) and curvature estimation on that surface.
- For best determinism, run with `--n-jobs 1` and a fixed `--seed`. UMAP can still exhibit small nondeterminism across platforms/BLAS/numba versions.
