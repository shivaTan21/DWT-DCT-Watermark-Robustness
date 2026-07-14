# Experiment I: SR Model Generalization of the Restoration-Pressure Mechanism

Date: 2026-07-14 · 100 Kodak/TAMPERE17 images · α ∈ {0.05, 0.10, 0.20, 0.30} · standard pair (4,1)/(1,4), seed 42, 1024-bit watermark
Pipeline: identical ESPCN wrapper (512² → x4 SR → 2048² → Lanczos4 → 512²) via `src/attacks_sr_generalization.py` (new file; frozen attack code untouched)
Data: `results_exp_I.csv` (1,200 rows), `summary_table.csv`, `summary_by_alpha.csv`, `mechanism_summary_exp_I.csv`, `restoration_regression_exp_I.csv`, `flip_vs_natural_margin_exp_I.csv`, `block_margins_exp_I.csv` (409,600 rows), `plots/`.
Frozen inputs reused read-only: `results/watermarked/*.png`, `results/attacked/ai_enhancement/*espcn*.png`, Exp H `block_margins.csv` (Real-ESRGAN reference). Nothing frozen was modified or rerun.

## 1. Robustness summary (pooled over 4 alphas, n=400 per model)

| Model | Mean NC | Mean BER | Mean SSIM | Mean PSNR | Runtime (s/img) |
|---|---|---|---|---|---|
| FSRCNN x4 | 0.9996 | 0.0002 | 0.977 | 38.9 | 0.10 |
| EDSR x4 | 0.9998 | 0.0001 | 0.980 | 40.0 | 43.8 |
| LapSRN x4 | 0.9994 | 0.0003 | 0.978 | 39.1 | 5.8 |
| ESPCN x4 (frozen baseline) | 0.9993 | 0.0004 | 0.978 | 39.4 | — |
| Real-ESRGAN (frozen baseline) | 0.7948 | 0.1026 | 0.885 | 29.2 | — |

Per-alpha: every new model has NC ≥ 0.9992 at every α (full table in `summary_by_alpha.csv`); NC differences across alphas are ≤ 0.0007. Sanity checks passed (512×512 outputs, no NaNs, metrics computable, zero crashes); EDSR exceeded the 4-hour projection (4.4 h) and was run with explicit user approval (290.5 min actual).

## 2. Mechanism (Exp H analysis repeated; standard pair, α=0.10, 102,400 blocks/model)

| Model | Sign-flip rate | σ(ΔC1−ΔC2) | mean common-mode σ | k (embedded differential removed) | corr(pert, embed shift) | flips in nature-opposing blocks |
|---|---|---|---|---|---|---|
| FSRCNN | 0.01% (7 flips) | 1.83 | 0.90 | 0.2% | −0.05 | 100% |
| EDSR | 0.00% (1 flip) | 0.57 | 0.25 | 0.2% | −0.20 | 100% |
| LapSRN | 0.02% (18) | 3.71 | 1.17 | 4.7% | **−0.69** | 100% |
| ESPCN | 0.02% (25) | 1.90 | 0.65 | 0.5% | −0.14 | 100% |
| Real-ESRGAN (Exp H frozen) | **10.9%** (11,135) | **24.2** | 5.64 | **32.3%** | **−0.73** | 90.5% (risk ratio 9.5×) |

Natural-margin relationship: for all models, the (few or many) flips concentrate in blocks whose natural coefficient ordering opposes the embedded bit — 100% of the 1–25 flips for the PSNR-oriented models, 90.5% of 11,135 for Real-ESRGAN, with flip probability rising with opposing |M_orig| (strong-opposing flip rates: Real-ESRGAN 34.5% vs ≤0.13% for all others).

## 3. Answers to the final questions

**1. Which models preserve the watermark?** ESPCN, FSRCNN, EDSR, and LapSRN — all four PSNR-oriented SR models, essentially perfectly (NC ≥ 0.999, BER ≤ 0.0004 at every α). EDSR is the most watermark-transparent model measured (1 flipped block in 102,400).

**2. Which models damage the watermark?** Only Real-ESRGAN (NC 0.795, BER 0.103), the sole GAN-trained blind-restoration model.

**3. Which models exhibit the Exp H restoration-pressure mechanism?** In *direction*, several: LapSRN shows a strong restoration signature (undo correlation −0.69, nearly Real-ESRGAN's −0.73) and EDSR/ESPCN weak ones (−0.20/−0.14); FSRCNN essentially none (−0.05). In *magnitude*, only Real-ESRGAN: its restoration strength (k = 32.3%) and differential noise (σ = 24.2 vs the ~30-unit margin) are 7–160× larger than any PSNR-oriented model, and only it crosses decision boundaries at scale. Even the miniature flip counts of the other models follow the Exp H signature (100% in nature-opposing blocks).

**4. Is the mechanism architecture dependent?** The *pressure* appears widespread but weak across bicubic-trained CNN super-resolvers, with its strength tracking training objective more than topology: the damaging regime is reached only by the GAN-trained, degradation-model-trained restoration network. LapSRN (Laplacian pyramid, Charbonnier loss) is the intermediate case — right direction, ~7× too weak. So: the mechanism is general in kind, but its watermark-destroying magnitude is specific to restoration-oriented (GAN-trained) architectures on current evidence.

**5. Can the paper discuss "AI enhancement" generically?** No. It must attribute damage to *certain learned super-resolution architectures* — specifically GAN-trained restoration models (Real-ESRGAN) — and state that four PSNR-oriented SR models (ESPCN, FSRCNN, EDSR, LapSRN) preserve the watermark under the identical pipeline. Statements like "AI enhancement destroys classical watermarks" are not supported.

**6. Exactly which sentences should change (recommendations only — no manuscript modified by this experiment):**
The 2026-07-13/14 revisions (`main.tex`, `main_corrected.tex`, `main_6page.tex`) already implement the architecture-specific framing with ESPCN/FSRCNN/LapSRN. With EDSR now complete, the following updates are recommended when next editing:
   - Abstract: "ESPCN, FSRCNN, and LapSRN … preserve watermark recovery (NC ≥ 0.999)" → add EDSR: "ESPCN, FSRCNN, EDSR, and LapSRN".
   - Intro (Observation stage), Related Work model list, Setup "AI enhancement attacks", Results §A text, Generalization section, Conclusion: same one-word addition (with `\cite{edsr2017}` — Lim et al., CVPRW 2017, needs a bib entry).
   - Table I (survival): add row "EDSR $4\times$ & 0.9998 & 0.0001 & 40.0 & 0.980".
   - Table IV (mechanism): add row "EDSR $4\times$ & 0.00\% & 0.6 & 0.2\% & $-0.20$".
   - Limitations: "four learned super-resolution architectures" → "five"; the sentence "diffusion-based enhancers and other GAN restorers remain untested" stays.
   - Optionally strengthen the generalization paragraph: EDSR shows that scale alone (43 M-parameter EDSR vs 25 K-parameter FSRCNN) does not create restoration pressure — training objective, not capacity, separates the harmless from the damaging models.

## 4. Provenance & safety

- New scripts only: `src/attacks_sr_generalization.py`, `run_exp_I.py`, `analyze_exp_I.py`, `make_summary.py`. Frozen attack implementations, experiments A–H, and all frozen CSVs/figures/images untouched (see `FINAL_SAFETY_REPORT` section of the project audit).
- Weights added to `weights/`: FSRCNN_x4.pb (41 KB), EDSR_x4.pb (38.5 MB), LapSRN_x4.pb (2.7 MB); all verified by size, load, and inference before use. `opencv-contrib-python` replaced `opencv-python` in the venv (documented requirement in requirements.txt).
- Real-ESRGAN comparison rows come from frozen Exp H CSVs (pipeline P1); ESPCN mechanism computed read-only from frozen disk images. Neither was rerun.
