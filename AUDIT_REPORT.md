# Reproducibility & Contamination Audit — DWT-DCT WIFS Project

Date: 2026-07-13 · Read-only audit (no code, CSVs, figures, or experiments modified; Exp I EDSR run untouched and still in progress)
Manuscript audited: `paper/main.tex` (post Exp H/I update of 2026-07-13). `paper/main_revised2.tex` shares the same legacy numbers where noted.

---

## 1. Contamination summary

**Two distinct Real-ESRGAN attack pipelines exist in the repository:**

| | Pipeline P1 ("disk"/baseline) | Pipeline P2 ("live patch") |
|---|---|---|
| Script | `src/attacks_ai.py::real_esrgan_enhance` (via `src/run_experiment.py`) | `experiments/exp_F_realesrgan_pair_verification/patch_balanced_esrgan.py`; `experiments/exp_H_margin_analysis/run_exp_H.py` |
| Model | RealESRGAN x4 (`weights/RealESRGAN_x4.pth`) | same |
| patches_size | 192 (py_real_esrgan `predict` default) | **64** |
| padding / pad_size | 24 / 15 (defaults) | **8 / 8** |
| Downscale 2048→512 | `cv2.resize` **INTER_AREA** | `cv2.resize` **INTER_LANCZOS4** |
| Preprocessing | clip→uint8, RGB (identical) | identical |
| Postprocessing | float64 cast (identical) | identical |
| Output directory | `results/attacked/ai_enhancement/` | `exp_H_margin_analysis/{watermarked_balanced, esrgan_balanced, esrgan_standard_control, esrgan_hf_control}/` (Exp F balanced outputs were not saved) |

**Measured severity difference (Exp H control, n=20):** P2 is harsher by −0.102 NC on the standard pair and −0.138 NC on the HF pair (Wilcoxon p = 1e-4). This is the only pipeline inconsistency found. Every other Real-ESRGAN artifact in the repo (baseline metrics, DCT stability profiles, Exps A–D, G, Exp F standard/HF arms, Exp I reference rows) uses P1 exclusively.

**Contaminated comparison:** Exp F's balanced-vs-{standard, HF} Real-ESRGAN comparison mixed P2 (balanced) with P1 (standard, HF). The frozen −0.1006 balanced deficit is ≈ the pipeline penalty (−0.1023), not a property of the positions. Under like-for-like P2, balanced *beats* standard by +0.0195 (p = 0.038, n = 20). The current `main.tex` already carries the † caveat and control numbers.

**Second, previously unreported problem found by this audit:** several PSNR/SSIM values in manuscript Table I and the entire NC column of the α-table (`tab:alpha_main`) do **not** match any CSV currently in the repository (details in §6). They appear to originate from an older run (several match a Kodak-only, i.e. 24-image, aggregate; others match nothing). NC/BER values throughout the paper are fine.

Other live-ESRGAN callers (`src/adaptive_embedding.py`, `src/majority_vote_embedding.py`) use P1 defaults; their outputs (`results/adaptive_embedding*.csv`, `majority_vote.csv`) are not cited in the manuscript.

---

## 2. Experiment inventory (A–H)

| Exp | Purpose | Input files | Scripts | Output files | ESRGAN pipeline | Status |
|---|---|---|---|---|---|---|
| A | DCT perturbation isotropy (ESRGAN vs JPEG) | `results/dct_stability_{realesrgan,jpeg}.csv`, `frequency_sweep.csv`, `symmetry_summary.csv` | `exp_A_dct_isotropy/run.py` | `outputs/` (A1–A4 PNG/CSV, summary.txt) | P1 (inherited) | **VERIFIED** (isotropy r=0.9929 vs 0.7779 reproduce) |
| B | Pre-embedding NC predictor from image stats | `data/original_images/`, `results/metrics.csv` | `exp_B_nc_predictor/run.py` | `outputs/` (features, correlations) | P1 (inherited) | **VERIFIED** (not cited in manuscript) |
| C | Pair-level design principles; Q-step anti-complementarity | stability CSVs, `frequency_sweep.csv` | `exp_C_design_principles/run.py` | `outputs/` (C1, C3, C_pairs) | P1 (inherited) | **VERIFIED** (ρ=−0.9074 Q-step; ρ=−0.4877 pair trade-off reproduce) |
| D | Exhaustive 1,953-pair analytical optimization | stability CSVs | `exp_D_exhaustive_optimization/run.py` | `outputs/` (D_all_pairs, D_pareto) | P1 (inherited) | **VERIFIED** (E=0.0278/J=0.8773 standard; balanced E=0.8183/J=0.8864; 37-pair Pareto reproduce). Its *predictive model* was falsified by F/H — a scientific result, not a reproducibility defect |
| E | Preliminary pair verification (JPEG/blur only; no ESRGAN installed) | originals; live embedding | `exp_E_pair_verification/run.py` | `outputs/` | none | **VERIFIED**, superseded by F; not cited |
| F | End-to-end 3-pair validation, 100 imgs × 4 α | P1 disk images (standard/HF); live P2 (balanced) | `run_exp_F.py`, `patch_balanced_esrgan.py` | `results_exp_F.csv`, `exp_F_summary.csv`, `exp_F_tests.csv`, PNGs | **P1 + P2 mixed** | **POSSIBLY CONTAMINATED** → balanced-vs-others ESRGAN comparison is a pipeline artifact. Non-ESRGAN arms (JPEG/blur/none) and standard-vs-HF ESRGAN are clean. **REQUIRES RERUN** (balanced arm under P1, or all three under one pipeline) for camera-ready |
| G | α-stability of damage profiles | P1 disk images, all 4 α | `exp_G_alpha_damage_ablation/run_exp_G.py` | `dct_damage_profiles.csv`, `alpha_profile_correlations.csv`, `alpha_rank_stability.csv`, PNGs | P1 | **VERIFIED** (ρ −0.7161…−0.7063; Jaccard 1.0; isotropy 0.9884–0.9888 reproduce) |
| H | Decision-margin mechanism; pipeline-confound control | P1 disk (standard/HF), P2 cached (balanced + controls) | `run_exp_H.py`, `analyze_exp_H.py`, `analyze_restoration.py` | `block_margins.csv`, `image_summary.csv`, 6 analysis CSVs, 6 figures | P1 + P2, **by design, with explicit control** | **VERIFIED** (all mechanism statistics reproduce bit-for-bit from its CSVs; NC validity check matches Exp F exactly) |

Baseline (pre-A) artifacts: `results/metrics.csv`, `summary_table.csv` (`src/run_experiment.py`, P1) — **VERIFIED**; `results/dct_stability_*.csv` (`src/analyze_dct_stability.py`, P1 α=0.1 images) — **VERIFIED**; `results/comparison_metrics.csv` (`src/run_comparison.py`, P1, α=0.1, 100 imgs) — **VERIFIED**.

---

## 3. Numerical audit — every number in the manuscript

Legend: ✅ = reproduced from stated source; ⚠️ = reproducible but source/population differs from what the text implies; ❌ = not reproducible from any current CSV.

### Abstract / Intro / Results §A (Table I)

| Manuscript location | Value | Source CSV | Source script | Repro? | Notes |
|---|---|---|---|---|---|
| Abstract, Tab I: ESRGAN NC | 0.795 | summary_table.csv (0.7948) | run_experiment.py | ✅ | |
| Abstract, Tab I: ESRGAN BER | 0.103 | summary_table.csv (0.1026) | run_experiment.py | ✅ | |
| Abstract, Tab I: ESRGAN SSIM | 0.906 | metrics.csv **Kodak-only** (0.9076); full data = **0.885** | run_experiment.py | ⚠️❌ | Population mismatch: table caption says 100 images; 0.906 is the 24-image Kodak aggregate |
| Tab I: watermarked NC/BER | 0.999/0.001 | summary_table.csv (0.9998/0.0001) | run_experiment.py | ✅ | |
| Tab I: watermarked PSNR/SSIM | 48.0/0.999 | comparison_metrics.csv gives 47.96/**0.997** (α=0.1 only); summary_table gives **40.4/0.981** | run_comparison.py | ⚠️❌ | Value from a different run (α=0.1 comparison subset), not the pooled population in the caption; SSIM 0.999 matches neither |
| Tab I: noise NC/BER | 0.999/0.001 | summary_table (0.9986/0.0007) | | ✅ | |
| Tab I: noise PSNR/SSIM | 28.1/0.809 | summary_table: 27.9/**0.702** | | ❌ | SSIM unmatched anywhere |
| Tab I: JPEG NC/BER | 0.996/0.002 | summary_table (0.9956/0.0022) | | ✅ | |
| Tab I: JPEG PSNR/SSIM | 29.8/0.886 | summary_table: 30.2/0.861 | | ❌ | |
| Tab I: blur NC/BER | 0.922/0.039 | summary_table (0.9215/0.0392) | | ✅ | |
| Tab I: blur PSNR/SSIM | 25.7/0.765 | summary_table: 28.7/0.843 | | ❌ | |
| Tab I: median NC/BER | 0.892/0.054 | summary_table (0.8916/0.0542) | | ✅ | |
| Tab I: median PSNR/SSIM | 26.6/0.785 | summary_table: 29.7/0.849 | | ❌ | |
| Tab I: crop NC/BER | −0.010/0.505 | summary_table (−0.0097/0.5049) | | ✅ | |
| Tab I: crop PSNR/SSIM | 14.4/0.129 | summary_table: 14.1/**0.278** | | ❌ | |
| Tab I: ESPCN NC/BER | 0.999/0.000 | summary_table (0.9993/0.0004) | | ✅ | |
| Tab I: FSRCNN row | 1.000/0.000/38.9/0.977 | exp_I results_exp_I.csv (0.9996/0.0002/38.91/0.9767) | run_exp_I.py | ✅ | |
| Tab I: LapSRN row | 0.999/0.000/39.1/0.978 | exp_I results_exp_I.csv (0.9994/0.0003/39.06/0.9775) | run_exp_I.py | ✅ | |
| Results text: FSRCNN 0.9996 / LapSRN 0.9994 pooled | | results_exp_I.csv | run_exp_I.py | ✅ | |

### α-table (`tab:alpha_main`)

| Manuscript | Value | Current data | Repro? |
|---|---|---|---|
| NC @ α=0.05/0.10/0.20/0.30 | 0.787/0.797/0.815/0.832 | standard, full set: **0.774/0.783/0.802/0.820**; Kodak-only: 0.793/0.802/0.820/0.836 | ❌ matches **nothing** in the repo (closest: Kodak-only, still ~0.005 off — likely an older intermediate dataset) |
| SSIM column | 0.907/0.906/0.906/0.905 | Kodak-only by α: 0.9079/0.9076/0.9072/0.9067; full set ≈0.885 flat | ⚠️ Kodak-only source; population mismatch |
| Qualitative trend (NC rises slowly, SSIM flat) | | holds in full data (0.774→0.820) | ✅ trend survives |

### Mechanism section (damage profiles, Pareto, validation)

| Manuscript | Value | Source | Repro? | Notes |
|---|---|---|---|---|
| ρ(ESRGAN, JPEG measured damage) | −0.716 (p<10⁻¹⁰) | exp_G `alpha_profile_correlations.csv` α=0.05 (−0.7161, p=2.9e-11) | ✅⚠️ | Reproducible from Exp G — but a direct recompute from `results/dct_stability_*.csv` (the CSVs behind Fig. `dct_stability_comparison.png`) gives **−0.630**. The figure and the quoted ρ come from two different profile computations; cite Exp G explicitly or recompute for consistency |
| ρ(ESRGAN damage, JPEG Q-step) | −0.907 (p<10⁻²⁵) | recomputed from `dct_stability_realesrgan.csv` + JPEG table: −0.9074 | ✅ | |
| DC concentration "5–14×" | | dct_stability_realesrgan.csv: DC=54.6; DC/max-other=5.3×, DC/median-other=14.0× | ✅ | Bounds are (max-other, median-other); consider stating the basis |
| 1,953 pairs; 37-pair Pareto | | exp_D `D_pareto.csv`/summary | ✅ | |
| Standard pair E=0.028, J=0.877 | | exp_D summary (0.0278/0.8773) | ✅ | |
| Pair-level trade-off ρ=−0.488 (43 pairs) | | exp_C `C_pairs.csv` (−0.4877, n=43) | ✅ | A naive pivot of frequency_sweep.csv gives −0.472; exp C's aggregation is the source |
| Tab III (pairs @ α=0.10): none 0.9997/0.9997/1.000; ESRGAN 0.783±0.124 / 0.683±0.139 / 0.831±0.109; JPEG 0.995/0.998/0.902; blur 0.910/0.933/0.977; PSNR 40.6/40.9/43.9 | | exp_F `exp_F_summary.csv` (all match to rounding) | ✅ | Balanced ESRGAN 0.683 = P2 pipeline (†, contaminated comparison — correctly flagged in current text) |
| HF vs std ESRGAN +0.048, d=0.89, p<10⁻¹³ | | exp_F `exp_F_tests.csv` (+0.0478, d=0.8914, p_adj≈0) | ✅ | |
| HF vs std JPEG −0.094, p<10⁻¹⁷ | | exp_F tests (−0.0938) | ✅ | |
| P2 pipeline penalty −0.102 (n=20, p=1e-4) | | exp_H `control_pipeline_check.csv` (−0.1023, p=0.0001) | ✅ | |
| Like-for-like balanced +0.020, p=0.038 | | exp_H control (+0.0195, p=0.0382) | ✅ | |
| 307,200 block measurements | 3×100×1024 | exp_H `block_margins.csv` | ✅ | |
| σ(ΔC1−ΔC2) predictor ρ=−0.887 | | exp_H `correlations_nc.csv` (−0.8869) | ✅ | |
| abs-damage predictor ρ=−0.825 | | correlations_nc.csv (−0.8246) | ✅ | |
| common-mode ρ=−0.22 | | correlations_nc.csv (−0.2197) | ✅ | |
| pre-attack margin ρ≈−0.07 n.s. | | correlations_nc.csv (−0.0708, p=0.22) | ✅ | |
| flip≡bit-error agreement <0.1% | | exp_H `success_vs_failure.csv` | ✅ | |
| undo corr −0.73/−0.60/−0.80 | | exp_H `restoration_regression.csv` (−0.7257/−0.6046/−0.8047) | ✅ | |
| k = 32%/30%/53% | | restoration_regression.csv (0.3226/0.3030/0.5320) | ✅ | |
| R² 0.71→0.83 (standard) | | restoration_regression.csv (0.7128→0.8284) | ✅ | |
| risk ratio 3.4–9.5×; χ² p≈0 | | exp_H `flip_vs_natural_margin.csv` (3.36/3.40/9.55) | ✅ | |
| 77–91% of flips in opposing blocks | | derived from flip_vs_natural_margin.csv rates (77.1/77.3/90.5%) | ✅ | |
| \|M_orig\| = 10.1 vs ≈31 | | restoration_regression.csv (10.05 / 30.76 / 30.36) | ✅ | |
| Tab IV mechanism (flip 0.02/0.01/0.02/10.9%; σ 1.9/1.8/3.7/24.2; k 0.5/0.2/4.7/32.3%; r −0.14/−0.05/−0.69/−0.73) | | exp_I `mechanism_summary_exp_I.csv` + `restoration_regression_exp_I.csv` | ✅ | Real-ESRGAN row = frozen Exp H standard rows (P1) |

### Stability / symmetry / subband sections

| Manuscript | Value | Source | Repro? |
|---|---|---|---|
| ρ range −0.716→−0.706 (range 0.010) across α | | exp_G alpha_profile_correlations.csv (−0.7161…−0.7063) | ✅ |
| Jaccard top-10 = 1.0 all α pairs | | exp_G alpha_rank_stability.csv | ✅ |
| isotropy r≈0.989 across α | | exp_G (0.9884–0.9888) | ✅ |
| isotropy 0.993 vs 0.778 (Exp A) | | exp_A summary (0.9929/0.7779) | ✅ |
| Symmetry table (ESRGAN 0.821/0.856/0.689; JPEG 0.866/0.721/0.868; blur 0.959/0.976/0.735) | | results/symmetry_summary.csv | ✅ |
| SVD-HH: 0.18 blur, 0.18 JPEG, 0.29 ESRGAN | | comparison_metrics.csv (0.180/0.181/0.285) | ✅ |
| DWT-DCT-LL survives "0.93, 0.98, 0.75" | | comparison_metrics.csv (0.928/0.983/**0.742**) | ⚠️ 0.75 should be 0.74 |
| PSNR 47.7 vs 48.0 dB | | comparison_metrics.csv (47.71/47.96) | ✅ |

---

## 4. Reproducible findings (safe for submission as-is)

1. Real-ESRGAN NC collapse (0.795) with intact ESPCN/FSRCNN/LapSRN (≥0.999) — baseline + Exp I, single P1 pipeline.
2. ESRGAN/JPEG anti-complementarity: ρ=−0.907 (Q-step), −0.716 (Exp G measured), pair-level −0.488.
3. DC-concentrated ESRGAN damage profile; isotropy 0.993 vs 0.778.
4. 1,953-pair Pareto structure (37 pairs; standard pair off-frontier).
5. HF pair superiority under ESRGAN (+0.048, d=0.89) and its JPEG cost (−0.094) — both arms P1.
6. α-stability of the anti-complementarity (Exp G: ρ range 0.010, Jaccard 1.0).
7. The entire Exp H mechanism suite (endogenous restoration, undo fractions, natural-margin opposition, flip≡BER) — pipeline-internal, verified bit-for-bit.
8. Exp I generalization for ESPCN/FSRCNN/LapSRN (EDSR pending; do not cite EDSR until its run completes).
9. Symmetry table and SVD-HH subband comparison.

## 5. Findings requiring rerun

| Item | Why | Scope of rerun |
|---|---|---|
| Exp F balanced-pair Real-ESRGAN arm | P2 vs P1 pipeline mixing; the −0.10 deficit is an artifact | Re-attack the 400 balanced watermarked images through P1 (≈1–2 h GPU/MPS), or re-run all three pairs through one pipeline; alternatively extend the n=20 like-for-like control to all 100 images |
| Like-for-like control n=20 → n=100 | Reviewer-facing sample size (already flagged in Limitations) | 2 × 100 P2 attacks (standard pair, already cached for 20) |

No other experiment requires rerunning.

## 6. Numbers that must be replaced before submission

All in `paper/main.tex` (and the same values in `main_revised2.tex`):

1. **Table I PSNR/SSIM columns** for: watermarked-only (48.0/0.999 → 40.4/0.981 from `summary_table.csv`, or re-caption the table as α=0.1 Y-channel and regenerate consistently), Gaussian noise (28.1/0.809 → 27.9/0.702), JPEG (29.8/0.886 → 30.2/0.861), blur (25.7/0.765 → 28.7/0.843), median (26.6/0.785 → 29.7/0.849), crop (14.4/0.129 → 14.1/0.278).
2. **Real-ESRGAN SSIM 0.906** (abstract, Table I, asymmetry text, conclusion) → **0.885** for the full 100-image set, or explicitly label as Kodak subset. The asymmetry claim itself survives either way (0.885 is still the highest SSIM among damaging attacks).
3. **α-table NC column** 0.787/0.797/0.815/0.832 → 0.774/0.783/0.802/0.820 (`metrics.csv`, standard variant); SSIM column → full-set values (≈0.885 flat) for population consistency.
4. **"0.75"** for DWT-DCT-LL under ESRGAN in the subband paragraph → **0.74**.
5. Optional consistency: state that −0.716 comes from the Exp G profile computation (the Fig. 2 source CSVs yield −0.630 under a direct recompute).

## 7. Conclusions audit

| Claim | Verdict |
|---|---|
| Quality–watermark asymmetry (high-SSIM, low-NC region unique to Real-ESRGAN) | **SUPPORTED AFTER NUMBER UPDATE** (SSIM 0.906→0.885; ordering unchanged) |
| Damage attributable to learned reconstruction, not resampling (pipeline controls) | **SUPPORTED** (now with 3 controls) |
| Threat is architecture-specific, not "AI enhancement" generally | **SUPPORTED** (Exp I; strengthen to 4 controls when EDSR completes) |
| DC-concentrated vs uniform damage profiles; anti-complementarity ρ≈−0.7…−0.9 | **SUPPORTED** |
| No coefficient pair optimizes both attacks (Pareto) | **SUPPORTED** |
| HF pair best under ESRGAN, worst under JPEG | **SUPPORTED** |
| Balanced pair "lowest Real-ESRGAN NC" (legacy claim) | **NO LONGER SUPPORTED** — pipeline artifact; current main.tex wording (†+control) is correct; frozen Exp F table value must keep the caveat or be rerun |
| Embedding strength does not restore robustness | **SUPPORTED AFTER NUMBER UPDATE** (α-table values stale; trend confirmed: +0.046 NC over 6× α) |
| Endogenous-restoration mechanism (Exp H) | **SUPPORTED** |
| Restoration pressure generalizes in direction, destructive only for GAN restoration (Exp I) | **SUPPORTED** (3 of 4 models measured; EDSR pending) |
| α-stability of anti-complementarity is structural | **SUPPORTED** |
| Symmetry not the cause of vulnerability | **SUPPORTED** |
| Subband escape (SVD-HH) not viable | **SUPPORTED** (fix 0.75→0.74) |

---

## 8. Audit provenance

All verifications recomputed read-only from: `results/{summary_table,metrics,comparison_metrics,symmetry_summary,frequency_sweep,dct_stability_realesrgan,dct_stability_jpeg}.csv`, `experiments/exp_{A,C,D,G}` outputs, `experiments/exp_F_realesrgan_pair_verification/{exp_F_summary,exp_F_tests,results_exp_F}.csv`, `experiments/exp_H_margin_analysis/*.csv`, `experiments/exp_I_sr_model_generalization/{results_exp_I,mechanism_summary_exp_I,restoration_regression_exp_I,flip_vs_natural_margin_exp_I}.csv`. No file was modified. The only file created by this audit is `AUDIT_REPORT.md`.
