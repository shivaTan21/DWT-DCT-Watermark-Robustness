# Reproducibility Guide

This document maps every table and figure in the paper
(*DWT-DCT Watermark Robustness Under Restoration-Oriented
Super-Resolution*, `paper/main_final.tex` / `main_final.pdf`) to the
experiment, script, and output file that produced it.

There are two ways to check the paper:

1. **Verify without recomputing (minutes).** Every number in the paper is
   read from a CSV that is committed to this repository. The table below
   lists which file backs which table/figure — you can open the CSVs and
   compare against the paper directly.
2. **Recompute from scratch (hours).** Run the commands below. Everything
   is seeded (`RANDOM_SEED = 42` in `src/config.py`) and all model
   weights are committed in `weights/`, so reruns are deterministic; the
   Real-ESRGAN arms dominate the runtime (100 images × 4 embedding
   strengths per arm; the original runs used Apple M2 hardware).

All commands are run from the repository root, inside the environment
from the [Setup](#setup) section.

## Quick reference: paper artifact → source

| Paper artifact | Content | Produced by | Committed output backing the paper |
|---|---|---|---|
| **Table I** | NC/BER/PSNR/SSIM per attack (classical + 5 SR models) | Main pipeline + Experiment I | `results/summary_table.csv` + `experiments/exp_I_sr_model_generalization/results_exp_I.csv`, `results_exp_I_edsr.csv` |
| **Table II** | Real-ESRGAN NC vs. embedding strength α | Main pipeline | `results/metrics.csv` (rows `attack=real_esrgan`, standard pair) |
| **Table III** | Three-pair validation (standard / balanced / HF × 4 attacks) | Experiment F (standard, HF) + Experiment J (balanced) | `experiments/exp_F_realesrgan_pair_verification/exp_F_summary.csv`, `exp_F_tests.csv` + `experiments/exp_J_balanced_like_for_like/summary_balanced_rerun.csv`, `tests_balanced_rerun.csv` |
| **Table IV** | Mechanism quantities per SR model (flip rate, σ, k, r_undo) | Experiment I (4 PSNR models) + Experiment H (Real-ESRGAN row) | `experiments/exp_I_sr_model_generalization/mechanism_summary_exp_I.csv`, `restoration_regression_exp_I.csv` + `experiments/exp_H_margin_analysis/block_margins.csv` |
| **Figure 1** | SSIM vs. NC scatter (quality–watermark asymmetry) | `src/plot_results.py` | `paper/figures/ssim_vs_nc.png` (from `results/metrics.csv`) |
| **Figure 2** | DCT damage heatmaps, Real-ESRGAN vs. JPEG | `src/analyze_dct_stability.py` | `paper/figures/dct_stability_comparison.png` |

In-text results that do not appear in a table are mapped in
[In-text results](#in-text-results) below.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch py-real-esrgan          # Real-ESRGAN backend
pip install opencv-contrib-python        # OpenCV DNN-SR backend (ESPCN/FSRCNN/EDSR/LapSRN)
```

Note: `opencv-contrib-python` conflicts with plain `opencv-python`;
uninstall the latter first (see README §1).

**Model weights** — already committed, no download needed:
`weights/RealESRGAN_x4.pth`, `weights/ESPCN_x4.pb`,
`weights/FSRCNN_x4.pb`, `weights/EDSR_x4.pb`, `weights/LapSRN_x4.pb`.

**Images** — the paper uses 100 images: 24 Kodak + 76 TAMPERE17
(noise-free set), resized to 512×512 by the pipeline. The Kodak suite
can be fetched with `python src/download_images.py --output_dir
data/original_images`; the TAMPERE17 noise-free images must be obtained
from the TAMPERE17 database (see the paper's dataset reference) and
placed in `data/original_images/` alongside them. Filenames matter only
in that Kodak images keep their `kodimNN` names (some plots split
Kodak/TAMPERE by name).

## Table I and Figure 1 — attack survival table and asymmetry scatter

Run the main pipeline (standard pair, all classical attacks +
Real-ESRGAN + ESPCN, α ∈ {0.05, 0.1, 0.2, 0.3}):

```bash
python -m src.run_experiment --real-esrgan-model weights/RealESRGAN_x4.pth \
                             --opencv-sr-model weights/ESPCN_x4.pb
python -m src.plot_results
```

- `results/summary_table.csv` → Table I rows: Watermarked only, Gaussian
  noise, JPEG-Q50, Gaussian blur, Median filter, Crop-Resize, ESPCN,
  Real-ESRGAN.
- `results/plots/ssim_vs_nc.png` → **Figure 1** (the paper copy lives at
  `paper/figures/ssim_vs_nc.png`).

Then run Experiment I for the remaining SR rows (FSRCNN, EDSR, LapSRN):

```bash
python experiments/exp_I_sr_model_generalization/run_exp_I.py --models fsrcnn,edsr,lapsrn
```

- `experiments/exp_I_sr_model_generalization/results_exp_I.csv` (and
  `results_exp_I_edsr.csv`) → Table I rows: FSRCNN, EDSR, LapSRN
  (n = 400 each: 100 images × 4 α).

## Table II — Real-ESRGAN NC by embedding strength

Produced by the same main-pipeline run as Table I. Aggregate
`results/metrics.csv` over rows with `attack = real_esrgan` (standard
pair), grouping by α: mean NC and SSIM per α give the four rows of
Table II.

## Table III — three-pair validation (standard / balanced / HF)

Two experiments, one per pipeline arm:

```bash
# Standard (4,1)/(1,4) and HF (7,5)/(7,7) columns + their statistics
python experiments/exp_F_realesrgan_pair_verification/run_exp_F.py

# Balanced (2,3)/(3,2) column — like-for-like rerun through the
# identical whole-image Real-ESRGAN pipeline (Experiment J)
python experiments/exp_J_balanced_like_for_like/run_exp_J.py
```

- `exp_F_summary.csv` → Table III standard and HF columns (means ± SD at
  α = 0.10, all four attacks, PSNR row).
- `summary_balanced_rerun.csv` → Table III balanced column
  (Real-ESRGAN 0.809 ± 0.112, JPEG 0.998, blur 0.933, PSNR 40.5).
- `exp_F_tests.csv` and `tests_balanced_rerun.csv` → the pairwise
  Wilcoxon/BH-FDR statistics quoted in the caption and §VI-C
  (e.g., balanced−standard ΔNC = +0.025, p_adj < 10⁻⁸; HF−standard
  +0.048, d = 0.89).

Notes for reviewers:

- **Do not use the balanced Real-ESRGAN values inside `exp_F_*` files**
  for Table III — that arm of Exp F used a measurably harsher patch-based
  attack invocation (the pipeline artifact documented in
  `experiments/exp_J_balanced_like_for_like/balanced_vs_standard.md`).
  Experiment J supersedes it; the paper uses Exp J for the balanced
  column and Exp F for the other two.
- `experiments/exp_J_balanced_like_for_like/parity_check.txt` contains a
  bit-identity proof that Exp J's code path reproduces the frozen
  standard/HF images exactly, and `crosscheck_exp_f.csv` shows the
  recomputed standard/HF summaries match `exp_F_summary.csv` with zero
  deviation.

## Table IV — mechanism quantities per SR model

```bash
# Real-ESRGAN row (block-level margin data, standard pair, alpha=0.10)
python experiments/exp_H_margin_analysis/run_exp_H.py
python experiments/exp_H_margin_analysis/analyze_exp_H.py

# ESPCN / FSRCNN / EDSR / LapSRN rows
python experiments/exp_I_sr_model_generalization/run_exp_I.py --models fsrcnn,edsr,lapsrn
python experiments/exp_I_sr_model_generalization/analyze_exp_I.py
```

- `mechanism_summary_exp_I.csv` + `restoration_regression_exp_I.csv` →
  Table IV rows for the four PSNR-oriented models (flip rate,
  σ(ΔC₁−ΔC₂), k, r_undo; 102,400 blocks per model).
- Exp H `block_margins.csv` (via `analyze_restoration.py`) → the
  Real-ESRGAN row (flip 10.9 %, σ = 24.2, k = 32.3 %, r_undo = −0.73).

## Figure 2 — DCT damage heatmaps (Real-ESRGAN vs. JPEG)

```bash
python -m src.analyze_dct_stability
```

Writes `results/plots/dct_stability_comparison.png` (paper copy:
`paper/figures/dct_stability_comparison.png`): mean absolute DCT
perturbation per position across 102,400 blocks, Real-ESRGAN vs.
JPEG-Q50, shared color scale, embedding positions marked.

## In-text results

| Paper passage | Experiment / script | Output file |
|---|---|---|
| §VI-A anti-complementarity across α (ρ = −0.716…−0.706, Jaccard 1.0) and §VI-F stability | Experiment G: `python experiments/exp_G_alpha_damage_ablation/run_exp_G.py` | `alpha_profile_correlations.csv`, `alpha_rank_stability.csv` |
| §VI-B exhaustive 1,953-pair search, 37-pair Pareto frontier, E/J-scores | Experiment D: `python experiments/exp_D_exhaustive_optimization/run.py` | `experiments/exp_D_exhaustive_optimization/outputs/` |
| §VI-B pair-level NC anti-correlation (ρ = −0.488, 43 screened pairs) and §VI-F symmetric-vs-asymmetric pairs (0.821 vs. 0.689) | `python -m src.frequency_sweep` | `results/symmetry_summary.csv` |
| §VI-D decision-margin analysis (307,200 blocks; flip-rate ratios 3.4–9.5×; σ(ΔC₁−ΔC₂) ρ = −0.887; k = 32/30/53 %; R² 0.71→0.83) | Experiment H: `run_exp_H.py` then `analyze_exp_H.py`, `analyze_restoration.py` | `block_margins.csv`, `image_summary.csv`, `pair_mechanism_summary.csv`, `flip_vs_natural_margin.csv`, `restoration_regression.csv` |
| §VI-G DWT-SVD-HH subband comparison (0.18/0.18/0.29 vs. 0.93/0.98/0.74) | `python -m src.run_comparison` | `results/comparison_metrics.csv` |

Full per-experiment write-ups live next to each experiment
(`EXP_F_REPORT.md`, `EXP_G_REPORT.md`, `EXP_H_REPORT.md`,
`EXP_I_REPORT.md`, `balanced_vs_standard.md`), and the repository-wide
number audit is in `AUDIT_REPORT.md`.

## Determinism and provenance notes

- The watermark bit-grid, embedding, and Gaussian noise are seeded
  (seed 42); all SR models run deterministic inference from the
  committed weights, so NC/BER values reproduce exactly. PSNR/SSIM may
  differ in the last displayed digit across platforms due to
  floating-point and codec differences.
- Each table in `paper/main_final.tex` carries a `% SOURCE:` comment
  above the LaTeX `table` environment naming its exact backing file —
  the LaTeX source is itself an index into this repository.
- A sanity gate in the pipeline (embed→extract on a clean image must
  give NC > 0.99) aborts any run with a broken codec, so partial or
  corrupted runs cannot silently produce "robustness" numbers.
