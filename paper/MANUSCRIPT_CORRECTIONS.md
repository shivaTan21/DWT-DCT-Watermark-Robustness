# Manuscript Corrections — `main_corrected.tex`

Date: 2026-07-14 · Base: `paper/main.tex` (2026-07-13 revision) · Neither `main.tex` nor `main_revised2.tex` was modified.
Authority: `AUDIT_REPORT.md` (esp. §6) + validated CSVs. Every table in `main_corrected.tex` carries a LaTeX comment naming its source CSV and experiment.

---

## Part 1 — Numerical changes

| # | Location | Original | Corrected | Source CSV | Reason |
|---|---|---|---|---|---|
| 1 | Abstract | SSIM 0.906 | **0.885** | `results/summary_table.csv` (standard, real_esrgan, ssim_after_attack_mean = 0.8850) | 0.906 was a Kodak-only (24-image) aggregate; population must match the stated 100-image set (AUDIT §6.2) |
| 2 | Intro, Observation stage | SSIM 0.906 | 0.885 | same | same |
| 3 | Results §A text | SSIM 0.906 | 0.885 | same | same |
| 4 | Conclusion | "highest perceptual quality of any attack tested" (implicit 0.906) | "highest perceptual quality (SSIM 0.885) of any watermark-degrading attack tested" | same | same + superlative accuracy (see Part 2 #3) |
| 5 | Table I, Watermarked only | 0.999 / 0.001 / 48.0 / 0.999 | **0.9998 / 0.0001 / 40.4 / 0.981** | `summary_table.csv` | 48.0/0.999 came from a different run (α=0.1 comparison subset), inconsistent with caption population (AUDIT §6.1) |
| 6 | Table I, Gaussian noise | 0.999 / 0.001 / 28.1 / 0.809 | 0.9986 / 0.0007 / 27.9 / 0.702 | `summary_table.csv` | PSNR/SSIM unreproducible from any current CSV |
| 7 | Table I, JPEG-Q50 | 0.996 / 0.002 / 29.8 / 0.886 | 0.9956 / 0.0022 / 30.2 / 0.861 | `summary_table.csv` | same |
| 8 | Table I, Gaussian blur | 0.922 / 0.039 / 25.7 / 0.765 | 0.9215 / 0.0392 / 28.7 / 0.843 | `summary_table.csv` | same |
| 9 | Table I, Median filter | 0.892 / 0.054 / 26.6 / 0.785 | 0.8916 / 0.0542 / 29.7 / 0.849 | `summary_table.csv` | same |
| 10 | Table I, Crop-Resize | −0.010 / 0.505 / 14.4 / 0.129 | −0.0097 / 0.5049 / 14.1 / 0.278 | `summary_table.csv` | same |
| 11 | Table I, ESPCN | 0.999 / 0.000 / --- / --- | 0.9993 / 0.0004 / 39.4 / 0.978 | `summary_table.csv` | previously missing values filled from validated data |
| 12 | Table I, FSRCNN | 1.000 / 0.000 / 38.9 / 0.977 | 0.9996 / 0.0002 / 38.9 / 0.977 | `exp_I .../results_exp_I.csv` | 4-dp NC/BER for traceability (1.000 was a rounding of 0.9996) |
| 13 | Table I, LapSRN | 0.999 / 0.000 / 39.1 / 0.978 | 0.9994 / 0.0003 / 39.1 / 0.978 | `results_exp_I.csv` | same |
| 14 | Table I, Real-ESRGAN | 0.795 / 0.103 / --- / 0.906 | 0.7948 / 0.1026 / 29.2 / 0.885 | `summary_table.csv` | SSIM population fix; PSNR filled |
| 15 | α-table NC column | 0.787 / 0.797 / 0.815 / 0.832 | **0.774 / 0.783 / 0.802 / 0.820** | `results/metrics.csv` (standard, real_esrgan, per α) | Original column matches no CSV in the repository (AUDIT §6.3) |
| 16 | α-table SSIM column | 0.907 / 0.906 / 0.906 / 0.905 | 0.886 / 0.885 / 0.885 / 0.884 | `metrics.csv` | Original was Kodak-only; population consistency |
| 17 | α-table text | "improves only marginally as α quadruples from 0.05 to 0.30" | "improves only marginally (+0.046) as α increases sixfold from 0.05 to 0.30" | `metrics.csv` | 0.05→0.30 is 6×, not 4×; delta made explicit (0.8198−0.7738=0.046) |
| 18 | Damage-profile text | "roughly 5 to 14 times greater than any other position" | "5.3× that of the most-damaged non-DC position and 14× the median non-DC position" | `results/dct_stability_realesrgan.csv` (DC=54.6; ratios 5.3/14.0) | basis of the ratio made explicit and traceable |
| 19 | Subband section | "0.93, 0.98, and 0.75" | "0.93, 0.98, and 0.74" | `results/comparison_metrics.csv` (0.928/0.983/0.742) | 0.742 rounds to 0.74; added "α=0.1" qualifier to the 47.7/48.0 dB sentence |
| 20 | Setup §Metrics | "computed on the Y channel" | "PSNR and SSIM are computed on the full RGB images (channel-averaged SSIM)" | `src/metrics.py` | description did not match the implementation that produced every quality number |
| 21 | Anti-complementarity sentence | "between the per-position ESRGAN and JPEG damage profiles is ρ=−0.716" | added "measured on the α-resolved profiles of the embedding-strength stability analysis below" | `exp_G .../alpha_profile_correlations.csv` (−0.7161 at α=0.05) | provenance: the Fig. 2 source CSVs give −0.630 under a direct recompute; −0.716 is Exp G's computation (AUDIT §6.5) |

Values NOT changed (audit-verified): Exp F pairs table (0.9997/0.783±0.124/0.683±0.139†/0.831±0.109; JPEG 0.995/0.998/0.902; blur 0.910/0.933/0.977; PSNR 40.6/40.9/43.9), all test statistics (+0.048, d=0.89, −0.094, −0.102, +0.020, p-values), ρ=−0.907, ρ=−0.488 (43 pairs), E=0.028/J=0.877, 37-pair Pareto, 1,953 pairs, isotropy 0.993/0.778 and 0.989, Jaccard 1.0, ρ range 0.010, all Exp H mechanism numbers (−0.887, −0.825, −0.22, −0.07 n.s., −0.73/−0.60/−0.80, 32/30/53%, R² 0.71→0.83, 3.4–9.5×, 77–91%, |M_orig| 10.1 vs ≈31, 307,200 blocks), the Exp I mechanism table (flip 0.02/0.01/0.02/10.9%; σ 1.9/1.8/3.7/24.2; k 0.5/0.2/4.7/32.3%; r −0.14/−0.05/−0.69/−0.73), symmetry table, SVD-HH 0.18/0.18/0.29, 47.7/48.0 dB, dataset/attack parameters (Q50, σ=10, 3×3, 5×5, 80%, seed 42, 1024-bit, δ=30).

## Part 2 — Sentence/discussion changes (non-numerical)

1. **Abstract, balanced pair**: was "a like-for-like pipeline control shows that an initially observed Real-ESRGAN deficit … was an attack-severity artifact rather than a property of the positions." Now states the balanced arm "was produced by an inconsistent, measurably harsher attack pipeline, so its Real-ESRGAN comparison is reported only with a like-for-like control (n = 20) and **requires a full rerun before it can be treated as conclusive**." *Reason: do not present the contaminated comparison, or even its correction, as final.*
2. **Abstract, mechanism**: "Real-ESRGAN acts as an endogenous restoration attack" → "the perturbation is anti-correlated with the embedded coefficient shift … behavior **consistent with** an endogenous restoration process rather than exogenous noise". *Reason: no overclaiming of internal model behavior.*
3. **Abstract/Intro/Results/Conclusion, superlative**: "highest perceptual quality among all evaluated attacks" → "among all **watermark-degrading** attacks". *Reason: with ESPCN/FSRCNN/LapSRN quality values now tabulated (SSIM ≈ 0.977–0.978 > 0.885), the unqualified superlative is false; the qualified one is verified.*
4. **Intro roadmap + contribution bullet**: "shows that Real-ESRGAN's perturbation is endogenous" → "is anti-correlated with the embedded coefficient shift … consistent with an endogenous restoration process"; "establishing generative restoration as an endogenous attack" → "showing that Real-ESRGAN's perturbation **behaves as** an endogenous, signal-dependent attack".
5. **Three-pair validation, Real-ESRGAN paragraph**: added that the pipeline inconsistency "was identified by the decision-margin control experiment of Section VII" (Exp H), and: "We therefore do not present the balanced-pair Real-ESRGAN comparison as a final result: a like-for-like rerun of the balanced arm on all 100 images is required before camera-ready, and the tabulated 0.683 should be read only as an upper bound on the harsher pipeline's effect."
6. **Key finding paragraph**: "+0.02 like-for-like improvement" now qualified "(n = 20 control; full rerun pending)".
7. **Margin section, endogeneity paragraph**: "Real-ESRGAN does not add fixed position-dependent noise; it pulls embedded statistics back toward natural-image statistics" → "This behavior is inconsistent with fixed, position-dependent noise and is **consistent with learned image restoration reducing the artificial coefficient relationships introduced by comparison-based embedding**… We characterize the model's input–output behavior only; we make no claim about its internal computations." Discussion shifted from absolute coefficient stability to **differential perturbation of the embedded coefficient pair** as the governing quantity (also mirrored in the Conclusion).
8. **HF-pair explanation**: "explains why the analytical E-score mispredicted" → "suggests why…"; "there is almost nothing for restoration to restore" → "consistent with there being almost nothing for a restoration process to restore".
9. **Generalization section**: conclusion sentence recast to current evidence, matching the required wording: "Real-ESRGAN produces substantial degradation while ESPCN, FSRCNN, and LapSRN preserve the watermark under the same evaluation pipeline". No EDSR results are mentioned anywhere (its run has not completed).
10. **Stability section**: "Real-ESRGAN is an endogenous restorer" → "Real-ESRGAN's damage follows the embedded signal itself — consistent with … mechanistically different kinds of noise, which **would explain** why no static position choice escapes the trade-off."
11. **Limitations**: "shows why such proxies mispredict … endogenous, signal-dependent attack" → "suggests why … signal-dependent attack"; "revised selection rule" → "candidate selection rule". (The pre-existing statement that the n=20 control must be extended for camera-ready is retained.)
12. **Conclusion**: mechanism sentence recast around differential perturbation with "consistent with" framing; "the margin analysis explains its advantage" → "suggests an explanation for its advantage"; added "on current evidence" to the architecture-dependence claim.
13. **Table provenance**: LaTeX `% SOURCE:` comments added above Tables I (survival), α-table, pairs table, mechanism table, symmetry table, and the subband paragraph.

## Part 3 — Final validation report (every numerical value in `main_corrected.tex`)

| Value | Status |
|---|---|
| 100 images (24 Kodak + 76 TAMPERE17), 512×512, α ∈ {0.05,0.1,0.2,0.3}, seed 42, 1024-bit (32×32), δ=30, positions (4,1)/(1,4),(2,3)/(3,2),(7,5)/(7,7) | Verified (`src/config.py`, `src/watermark.py`, `data/original_images/`) |
| Attack params: JPEG Q50, noise σ=10, median 3×3, blur 5×5, crop 80% | Verified (`src/attacks_traditional.py`) |
| NC 0.999→0.795; BER 0.103; SSIM 0.885 (abstract, Tab I, text, conclusion) | Verified (`summary_table.csv`: 0.9998→0.7948/0.1026/0.8850) |
| NC ≥ 0.999 for ESPCN/FSRCNN/LapSRN; 0.9996/0.9994 pooled | Verified (`summary_table.csv`, `results_exp_I.csv`) |
| Table I — all 40 cells | Verified (regenerated from `summary_table.csv` + `results_exp_I.csv`; 2026-07-14) |
| α-table — all 8 cells; +0.046 delta | Verified (`metrics.csv`) |
| 102,400 blocks (stability); 307,200 blocks (margins) | Verified (100×1024; 3×100×1024) |
| ρ = −0.716 (p<10⁻¹⁰); range −0.716…−0.706 (0.010); Jaccard 1.0; isotropy ≈0.989 | Verified (`exp_G/alpha_profile_correlations.csv`, `alpha_rank_stability.csv`) |
| ρ = −0.907 (p<10⁻²⁵) | Verified (recompute from `dct_stability_realesrgan.csv` + JPEG table: −0.9074) |
| DC 5.3× / 14× (median) | Verified (`dct_stability_realesrgan.csv`: 54.6; 5.28; 14.0) |
| isotropy 0.993 vs 0.778 | Verified (`exp_A/outputs/summary.txt`: 0.9929/0.7779) |
| 1,953 pairs; 37-pair Pareto; E=0.028/J=0.877 | Verified (`exp_D/outputs/`) |
| ρ = −0.488 (p<0.001), 43 pairs | Verified (`exp_C/outputs/C_pairs.csv`: −0.4877, n=43) |
| Pairs table: 0.9997/0.9997/1.000; 0.783±0.124/0.683±0.139†/0.831±0.109; 0.995/0.998/0.902; 0.910/0.933/0.977; PSNR 40.6/40.9/43.9 | Verified (`exp_F_summary.csv`) — † contaminated-pipeline arm, explicitly flagged non-final |
| +0.048 (p<10⁻¹³, d=0.89); −0.094 (p<10⁻¹⁷); +3.3 dB; +0.30 dB | Verified (`exp_F_tests.csv`, `exp_F_summary.csv`) |
| −0.102 (n=20, p=10⁻⁴); +0.020 (n=20, p=0.038) | Verified (`exp_H/control_pipeline_check.csv`: −0.1023/+0.0195, p=0.0001/0.0382) |
| ρ=−0.887 / −0.825 / −0.22 / −0.07 n.s.; flip≡BER <0.1% | Verified (`exp_H/correlations_nc.csv`, `success_vs_failure.csv`) |
| r = −0.73/−0.60/−0.80; k = 32/30/53%; R² 0.71→0.83; \|M_orig\| 10.1 vs ≈31 | Verified (`exp_H/restoration_regression.csv`) |
| risk ratio 3.4–9.5× (χ² p≈0); 77–91% of flips in opposing blocks; margins clamped ≈30; flip prediction within 1 pp | Verified (`exp_H/flip_vs_natural_margin.csv`, `pair_mechanism_summary.csv`) |
| Mechanism table: 0.02/0.01/0.02/10.9%; 1.9/1.8/3.7/24.2; 0.5/0.2/4.7/32.3%; −0.14/−0.05/−0.69/−0.73 | Verified (`exp_I/mechanism_summary_exp_I.csv`, `restoration_regression_exp_I.csv`) |
| Symmetry table (9 cells) | Verified (`results/symmetry_summary.csv`) |
| SVD-HH: 47.7/48.0 dB; 0.18/0.18/0.29 vs 0.93/0.98/0.74 | Verified (`results/comparison_metrics.csv`) |

**TODO_VERIFY count: 0.** No unsupported numbers remain; no EDSR results appear.
