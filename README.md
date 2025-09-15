Neural Style Transfer (CPU) — Prototype

Overview

- CPU-only setup with a single Python entry script (`nst_cpu_main.py`).
- Creates required folders, prints run config, and verifies TensorFlow import with GPUs disabled.

Project Layout

- data/content/
- data/style/
- outputs/runs/
- outputs/plots/
- metrics/
- nst_cpu_main.py

Quickstart (Windows, PowerShell)

1) Create and activate a virtual environment

   - py -3 -m venv .venv
   - .venv\Scripts\Activate.ps1

2) Install CPU-only dependencies (examples)

   - pip install --upgrade pip
   - pip install tensorflow-cpu numpy pillow matplotlib

3) Run (GPU disabled by default in code)

- python nst_cpu_main.py

Interactive Notebook

- Open `nst_interactive.ipynb` to run an interactive NST transfer:
  - Pick a style from `data/style/`
  - Enter a path to your content image (any path on disk)
  - Click “Run NST” to generate and view the result

What the script does

- Forces CPU-only mode by setting `CUDA_VISIBLE_DEVICES=-1` before importing TensorFlow.
- Sets global seeds (Python/NumPy/TensorFlow) for reproducibility.
- Ensures required directories exist and prints the run configuration.
- Verifies TensorFlow import and reports visible CPU/GPU devices.
- Declares an image sizing policy (`square-resize` by default; `center-crop-then-resize` optional).

Configuration

- Edit the small config block inside `nst_cpu_main.py` to change:
  - Paths: content/style/outputs/metrics
  - Hyperparameters: `image_size`, `steps`, `alpha`, `beta`, `gamma`
  - Reproducibility: `seed`
  - Resize policy: `square-resize` (default) or `center-crop-then-resize`

Notes on Reproducibility

- The script sets `PYTHONHASHSEED`, seeds Python/NumPy/TF, and requests CPU-only execution. Some TF ops may still be nondeterministic, but this is a reasonable baseline.

Next Steps

- Add image loading and preprocessing based on `resize_policy`.
- Implement NST optimization loop and save artifacts to `outputs/` and `metrics/`.

Sanity Tests, Batch Runner, Portfolio, Analysis

- Optional toggles in `Config` drive these from `main()`:
  - `do_sanity_tests` (Step 9)
  - `do_batch_grid` (Step 10)
  - `do_portfolio_gallery` (Step 11)
  - `do_analysis_plots` (Step 12)
- Or call functions directly:
  - `run_sanity_tests(cfg)`
  - `run_batch_grid(cfg, max_content=6, max_style=6, betas=[1000,2000,5000], size=384)`
  - `run_portfolio_and_gallery(cfg, preferred_beta=2000.0, grid=6, size=512)`
  - `analysis_plots_from_csv(cfg)`

Portfolio & Analysis

- Step 11 portfolio + gallery: `run_portfolio_and_gallery(cfg, preferred_beta=2000.0, grid=6, run_size=256, render_size=512)`
  - Runs the grid at 256px for speed; renders the gallery at 512px by resizing tiles.
- Step 12 analysis plots from CSV: `analysis_plots_from_csv(cfg)`
  - Produces under `outputs/plots/`:
    - `ssim_vs_beta_by_size.png`
    - `runtime_vs_size.png`, `runtime_vs_beta.png`, `runtime_vs_beta_by_size.png`
    - `runtime_per_step_vs_size.png`, `runtime_per_step_vs_beta.png`
    - `ssim_box_by_beta_{size}px.png` (one per size)
    - `ssim_vs_beta_scatter_by_pair.png`
  - Also writes `metrics/summary_top_by_ssim.csv` (top 20 runs).

Standalone analysis script (no re-run)

- You can generate the same plots directly from CSV without importing the main script:
  - `python analyze_metrics.py --csv metrics/metrics.csv --out outputs/plots --top-k 20`
  - This writes the plots into `outputs/plots/` and a top-SSIM summary to `metrics/summary_top_by_ssim.csv`.

Defaults

- Most runs default to 256px for speed (sanity tests and batch grid).
- Final picks and gallery render at 512px via `run_portfolio_and_gallery`.
