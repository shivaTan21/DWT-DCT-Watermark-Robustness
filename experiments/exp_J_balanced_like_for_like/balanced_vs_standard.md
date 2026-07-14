# Balanced Pair Like-for-Like Rerun (Experiment J)

Date: 2026-07-14  |  n = 100 images (24 Kodak + 76 TAMPERE17)  |  α ∈ {0.05, 0.1, 0.2, 0.3}

**Purpose.** Eliminate the manuscript's known limitation: in Exp F the balanced pair (2,3)/(3,2) was evaluated with a *different* Real-ESRGAN invocation (patch-based, `patches_size=64`, autocast disabled, Lanczos4 downscale) than the standard (4,1)/(1,4) and HF (7,5)/(7,7) pairs (whole-image canonical wrapper). This experiment reruns ONLY the balanced arm through the identical canonical pipeline used for the other two pairs, on all 100 images and all four embedding strengths.

## 1. Pipeline parity verification

Every element of the balanced arm matches the pipeline that produced the standard/HF results:

| Element | Standard / HF arms | Balanced rerun (this experiment) | Match |
|---|---|---|---|
| Attack implementation | `src.attacks_ai.real_esrgan_enhance` | same function, same module | ✓ |
| ESRGAN weights | `weights/RealESRGAN_x4.pth`, scale 4 | same file | ✓ |
| Invocation | whole-image `model.predict()` defaults (batch_size=4, patches_size=192, padding=24, pad_size=15) | same (no overrides) | ✓ |
| Post-attack resize | `cv2.resize(..., config.IMAGE_SIZE, INTER_AREA)` | same | ✓ |
| Preprocessing | originals resized to 512×512 with INTER_AREA, BGR→RGB float64 | same | ✓ |
| Watermarked quantization | uint8 PNG round trip (run_experiment.py) | uint8 round trip before attack | ✓ |
| Watermark bits | `generate_watermark((32,32), seed=42)` | same | ✓ |
| Embedding | `embed_watermark_rgb`, Y channel, Haar DWT, 8×8 DCT, margin floor 30.0 | same, positions (2,3)/(3,2) | ✓ |
| Random seed | 42 | 42 | ✓ |
| Datasets | 24 Kodak + 76 TAMPERE17, all 100 images | same | ✓ |
| JPEG attack | `jpeg_compression(quality=50)` | same | ✓ |
| Blur attack | `gaussian_blur(ksize=5, sigma=1.0)` | same | ✓ |
| Metrics | `src.metrics` NC/BER/PSNR/SSIM vs original | same | ✓ |
| Statistics | Shapiro-gated paired t / Wilcoxon, BH-FDR (Exp F code) | same | ✓ |

**Bit-identity proof** (`parity_check.txt`): regenerating the standard and HF watermarked and Real-ESRGAN-attacked images live through this rerun's code path reproduces the frozen disk files **bit-for-bit**:

```
Parity check: canonical pipeline vs frozen disk files
date: 2026-07-14 03:13:01

[standard] kodim01 alpha=0.1: watermarked bit-identical=True, real_esrgan bit-identical=True
[hf] kodim01 alpha=0.1: watermarked bit-identical=True, real_esrgan bit-identical=True

RESULT: PASS — the live canonical pipeline is bit-identical to the pipeline that produced the frozen standard/HF files.
```

**Exp F cross-check:** the standard/HF summary values recomputed here match `exp_F_summary.csv` with max |ΔNC| = 0.00e+00 across all 32 (pair, α, attack) cells — exact reproduction.

## 2. Results at α = 0.10 (primary operating point)

| Attack | Metric | Standard (4,1)/(1,4) | Balanced (2,3)/(3,2) | HF (7,5)/(7,7) |
|---|---|---|---|---|
| No attack | NC (±SD) | 0.9997 ± 0.0012 | 0.9998 ± 0.0008 | 1.0000 ± 0.0002 |
|  | BER | 0.0002 | 0.0001 | 0.0000 |
|  | PSNR (dB) | 40.58 | 40.54 | 43.92 |
|  | SSIM | 0.9819 | 0.9825 | 0.9887 |
| Real-ESRGAN | NC (±SD) | 0.7833 ± 0.1241 | 0.8086 ± 0.1115 | 0.8311 ± 0.1085 |
|  | BER | 0.1084 | 0.0957 | 0.0845 |
|  | PSNR (dB) | 29.16 | 29.15 | 29.42 |
|  | SSIM | 0.8853 | 0.8857 | 0.8904 |
| JPEG Q50 | NC (±SD) | 0.9953 ± 0.0193 | 0.9979 ± 0.0042 | 0.9015 ± 0.0899 |
|  | BER | 0.0024 | 0.0011 | 0.0492 |
|  | PSNR (dB) | 30.23 | 30.24 | 30.52 |
|  | SSIM | 0.8612 | 0.8633 | 0.8727 |
| Gaussian Blur | NC (±SD) | 0.9097 ± 0.0629 | 0.9330 ± 0.0553 | 0.9768 ± 0.0250 |
|  | BER | 0.0452 | 0.0335 | 0.0116 |
|  | PSNR (dB) | 28.74 | 28.72 | 28.89 |
|  | SSIM | 0.8436 | 0.8429 | 0.8517 |

### Real-ESRGAN NC across all embedding strengths

| α | Standard | Balanced (rerun) | Balanced (old Exp F, patch pipeline) | HF |
|---|---|---|---|---|
| 0.05 | 0.7738 | 0.7998 | 0.6724 | 0.8293 |
| 0.10 | 0.7833 | 0.8086 | 0.6826 | 0.8311 |
| 0.20 | 0.8023 | 0.8253 | 0.7026 | 0.8342 |
| 0.30 | 0.8198 | 0.8402 | 0.7197 | 0.8377 |

## 3. Statistical tests (paired, BH-FDR corrected — same methodology as the paper)

### Balanced vs Standard

| Attack | α | ΔNC (mean) | 95% CI | p_adj (BH) | sig | effect | test | n |
|---|---|---|---|---|---|---|---|---|
| No attack | 0.05 | -0.0000 | [-0.0002, +0.0002] | 4.11e-01 | n.s. | -0.082 | wilcoxon | 100 |
| Real-ESRGAN | 0.05 | +0.0260 | [+0.0181, +0.0340] | 3.34e-09 | *** | 0.603 | wilcoxon | 100 |
| JPEG Q50 | 0.05 | +0.0028 | [-0.0008, +0.0064] | 1.01e-02 | * | 0.268 | wilcoxon | 100 |
| Gaussian Blur | 0.05 | +0.0239 | [+0.0182, +0.0295] | 1.14e-13 | *** | 0.753 | wilcoxon | 100 |
| No attack | 0.10 | +0.0001 | [-0.0001, +0.0003] | 1.66e-01 | n.s. | 0.143 | wilcoxon | 100 |
| Real-ESRGAN | 0.10 | +0.0254 | [+0.0175, +0.0332] | 8.68e-09 | *** | 0.587 | wilcoxon | 100 |
| JPEG Q50 | 0.10 | +0.0026 | [-0.0009, +0.0061] | 4.48e-02 | * | 0.206 | wilcoxon | 100 |
| Gaussian Blur | 0.10 | +0.0233 | [+0.0180, +0.0286] | 4.68e-14 | *** | 0.766 | wilcoxon | 100 |
| No attack | 0.20 | +0.0000 | [-0.0001, +0.0001] | 2.21e-01 | n.s. | 0.126 | wilcoxon | 100 |
| Real-ESRGAN | 0.20 | +0.0230 | [+0.0153, +0.0306] | 3.11e-08 | *** | 0.564 | wilcoxon | 100 |
| JPEG Q50 | 0.20 | +0.0025 | [-0.0008, +0.0058] | 1.88e-02 | * | 0.244 | wilcoxon | 100 |
| Gaussian Blur | 0.20 | +0.0220 | [+0.0171, +0.0269] | 2.55e-14 | *** | 0.775 | wilcoxon | 100 |
| No attack | 0.30 | +0.0000 | [-0.0001, +0.0001] | 3.60e-01 | n.s. | 0.093 | wilcoxon | 100 |
| Real-ESRGAN | 0.30 | +0.0204 | [+0.0131, +0.0278] | 1.41e-07 | *** | 0.537 | wilcoxon | 100 |
| JPEG Q50 | 0.30 | +0.0023 | [-0.0008, +0.0054] | 5.81e-02 | n.s. | 0.194 | wilcoxon | 100 |
| Gaussian Blur | 0.30 | +0.0191 | [+0.0148, +0.0234] | 1.30e-14 | *** | 0.784 | wilcoxon | 100 |

### HF vs Standard

| Attack | α | ΔNC (mean) | 95% CI | p_adj (BH) | sig | effect | test | n |
|---|---|---|---|---|---|---|---|---|
| No attack | 0.05 | +0.0003 | [+0.0001, +0.0005] | 2.70e-03 | ** | 0.312 | wilcoxon | 100 |
| Real-ESRGAN | 0.05 | +0.0555 | [+0.0445, +0.0665] | 3.08e-16 | *** | 1.001 | t-test | 100 |
| JPEG Q50 | 0.05 | -0.0944 | [-0.1109, -0.0779] | 1.41e-17 | *** | -inf | wilcoxon | 100 |
| Gaussian Blur | 0.05 | +0.0765 | [+0.0661, +0.0869] | 1.99e-17 | *** | inf | wilcoxon | 100 |
| No attack | 0.10 | +0.0003 | [+0.0001, +0.0005] | 4.32e-03 | ** | 0.297 | wilcoxon | 100 |
| Real-ESRGAN | 0.10 | +0.0478 | [+0.0372, +0.0584] | 6.13e-14 | *** | 0.891 | t-test | 100 |
| JPEG Q50 | 0.10 | -0.0938 | [-0.1104, -0.0771] | 1.41e-17 | *** | -inf | wilcoxon | 100 |
| Gaussian Blur | 0.10 | +0.0672 | [+0.0578, +0.0766] | 1.99e-17 | *** | inf | wilcoxon | 100 |
| No attack | 0.20 | +0.0001 | [+0.0000, +0.0002] | 1.31e-02 | * | 0.258 | wilcoxon | 100 |
| Real-ESRGAN | 0.20 | +0.0319 | [+0.0218, +0.0419] | 1.54e-08 | *** | 0.631 | t-test | 100 |
| JPEG Q50 | 0.20 | -0.0915 | [-0.1083, -0.0747] | 1.41e-17 | *** | -inf | wilcoxon | 100 |
| Gaussian Blur | 0.20 | +0.0510 | [+0.0432, +0.0588] | 3.82e-17 | *** | inf | wilcoxon | 100 |
| No attack | 0.30 | +0.0001 | [-0.0000, +0.0002] | 3.53e-02 | * | 0.218 | wilcoxon | 100 |
| Real-ESRGAN | 0.30 | +0.0179 | [+0.0084, +0.0274] | 4.83e-04 | *** | 0.375 | t-test | 100 |
| JPEG Q50 | 0.30 | -0.0891 | [-0.1059, -0.0723] | 1.41e-17 | *** | -inf | wilcoxon | 100 |
| Gaussian Blur | 0.30 | +0.0361 | [+0.0300, +0.0422] | 3.82e-17 | *** | inf | wilcoxon | 100 |

### Balanced vs HF

| Attack | α | ΔNC (mean) | 95% CI | p_adj (BH) | sig | effect | test | n |
|---|---|---|---|---|---|---|---|---|
| No attack | 0.05 | -0.0003 | [-0.0006, -0.0001] | 1.69e-03 | ** | -0.327 | wilcoxon | 100 |
| Real-ESRGAN | 0.05 | -0.0295 | [-0.0410, -0.0179] | 6.84e-07 | *** | -0.507 | wilcoxon | 100 |
| JPEG Q50 | 0.05 | +0.0972 | [+0.0797, +0.1148] | 1.41e-17 | *** | inf | wilcoxon | 100 |
| Gaussian Blur | 0.05 | -0.0527 | [-0.0619, -0.0435] | 3.82e-17 | *** | -inf | wilcoxon | 100 |
| No attack | 0.10 | -0.0002 | [-0.0003, -0.0001] | 5.86e-03 | ** | -0.287 | wilcoxon | 100 |
| Real-ESRGAN | 0.10 | -0.0224 | [-0.0338, -0.0110] | 4.30e-05 | *** | -0.421 | wilcoxon | 100 |
| JPEG Q50 | 0.10 | +0.0963 | [+0.0787, +0.1140] | 1.70e-17 | *** | inf | wilcoxon | 100 |
| Gaussian Blur | 0.10 | -0.0439 | [-0.0519, -0.0359] | 5.25e-17 | *** | -inf | wilcoxon | 100 |
| No attack | 0.20 | -0.0001 | [-0.0002, +0.0000] | 3.53e-02 | * | -0.218 | wilcoxon | 100 |
| Real-ESRGAN | 0.20 | -0.0089 | [-0.0197, +0.0020] | 3.86e-02 | * | -0.213 | wilcoxon | 100 |
| JPEG Q50 | 0.20 | +0.0940 | [+0.0763, +0.1117] | 1.41e-17 | *** | inf | wilcoxon | 100 |
| Gaussian Blur | 0.20 | -0.0290 | [-0.0351, -0.0229] | 1.32e-16 | *** | -inf | wilcoxon | 100 |
| No attack | 0.30 | -0.0001 | [-0.0002, -0.0000] | 2.87e-02 | * | -0.228 | wilcoxon | 100 |
| Real-ESRGAN | 0.30 | +0.0025 | [-0.0081, +0.0132] | 3.60e-01 | n.s. | 0.094 | wilcoxon | 100 |
| JPEG Q50 | 0.30 | +0.0914 | [+0.0737, +0.1091] | 1.41e-17 | *** | inf | wilcoxon | 100 |
| Gaussian Blur | 0.30 | -0.0170 | [-0.0215, -0.0125] | 8.46e-13 | *** | -0.726 | wilcoxon | 100 |

## 4. Does the manuscript conclusion change?

Under the identical canonical pipeline at α=0.10, the balanced pair's Real-ESRGAN NC is **0.8086** vs standard **0.7833** (ΔNC = +0.0254, ***, p_adj = 8.68e-09) and HF **0.8311**.

**Verdict:** The balanced pair does **not** underperform the standard pair — it is significantly (modestly) better. The old tabulated 0.683 was a pipeline artifact, exactly as the manuscript's n=20 control predicted.

HF remains the best pair under Real-ESRGAN (balanced − HF: -0.0224, ***).

### What the rerun establishes

1. **The old balanced deficit was entirely a pipeline artifact.** The Exp F tabulated
   value (0.683 at α=0.10) is 0.126 NC below the like-for-like value (0.809). The
   manuscript's n=20 pipeline-confound control (which measured the patch pipeline as
   −0.102 NC harsher and predicted a like-for-like balanced advantage of +0.020)
   is confirmed on all 100 images: the true advantage is **+0.025**.
2. **Balanced modestly but significantly outperforms standard under Real-ESRGAN at
   every embedding strength**: ΔNC = +0.0260 / +0.0254 / +0.0230 / +0.0204 at
   α = 0.05/0.10/0.20/0.30, all Wilcoxon p_adj < 10⁻⁶ after BH-FDR, effect size
   r ≈ 0.54–0.60. There is **no JPEG penalty** (ΔNC ≈ +0.003, i.e. balanced is
   equal-or-better under JPEG too) and a significant blur improvement (+0.023).
3. **HF remains the best pair under Real-ESRGAN** at the paper's operating point
   (0.831 vs 0.809 at α=0.10, p_adj = 4.3×10⁻⁵), converging with balanced only at
   α=0.30 (n.s.). The three-way ordering under ESRGAN is now
   HF > Balanced > Standard at α ≤ 0.20.
4. **The paper's central negative result is strengthened, not weakened**: the
   analytically predicted *large* balanced advantage (Exp D E-score 0.818 vs 0.028)
   materializes as only +0.025 NC under a strictly identical pipeline — the
   damage-proxy misprediction and the endogenous-restoration explanation
   (Section sec:margin) stand unchanged.

## 5. Secondary finding: the imperceptibility comparison was also inconsistent

Exp F embedded the balanced pair as a float image without the uint8 PNG round trip
that the standard/HF disk arms went through. This inflated the balanced no-attack
PSNR to 40.88 dB (the manuscript's "+0.30 dB" claim). Under identical uint8
treatment (this rerun):

| Metric (no attack, α=0.10) | Standard | Balanced | Paired test (bal − std) |
|---|---|---|---|
| PSNR (dB) | 40.58 | 40.54 | −0.04 dB, Wilcoxon p = 0.70 (n.s.) |
| SSIM | 0.9819 | 0.9825 | +0.0006, Wilcoxon p = 6.1×10⁻⁵ |

The balanced pair's imperceptibility is **statistically indistinguishable from
standard in PSNR** (and marginally better in SSIM), not +0.30 dB better. The HF
pair's +3.3 dB advantage is unaffected (43.92 dB, identical files).

## 6. Does the original manuscript conclusion change?

**No conclusion reverses; two quantitative claims become final and one small claim
must be corrected.**

- **Unchanged:** no pair dominates across attacks; ordering is strongly
  attack-specific; HF is best under Real-ESRGAN and blur and worst under JPEG;
  the analytical Pareto ranking mispredicts magnitude for the generative attack;
  the endogenous-restoration mechanism analysis is untouched.
- **Now final (previously provisional):** the balanced pair does not underperform
  the standard pair under Real-ESRGAN — it is modestly better (+0.025, n=100,
  p_adj < 10⁻⁸), exactly as the n=20 control predicted. The dagger caveat, the
  "non-final" language, and the "full rerun pending" qualifiers can all be removed.
- **Correction required:** the balanced pair's "+0.30 dB PSNR" imperceptibility
  advantage was an artifact of the same inconsistency; the like-for-like PSNR
  difference is −0.04 dB (n.s.).

## 7. Exact manuscript sentences to update (paper/main_6page.tex)

*(No manuscript file has been modified; the edits below are recommendations only.
Line numbers refer to the current main_6page.tex. The same edits apply to
main_corrected.tex / main.tex wherever the corresponding passages survive in the
submission lineage.)*

**(1) Abstract, lines 43–45.** Current:

> "while the analytically balanced pair $(2,3)/(3,2)$ achieves the highest JPEG NC (0.998); its Real-ESRGAN comparison is non-final due to an identified and controlled attack-pipeline inconsistency, pending a full rerun."

Replace with:

> "while the analytically balanced pair $(2,3)/(3,2)$ achieves the highest JPEG NC (0.998) and a modest but significant Real-ESRGAN improvement over the standard pair ($0.809$ vs.\ $0.783$, $n=100$, $p_{\mathrm{adj}}<10^{-8}$) --- far short of its analytically predicted advantage."

**(2) LaTeX comment block, lines 485–488** (the `% NOTE (AUDIT_REPORT.md Sec. 1/5)` block): replace with a pointer to this experiment, e.g. `% SOURCE: experiments/exp_J_balanced_like_for_like/{summary,tests}_balanced_rerun.csv — all three arms now use the identical canonical Real-ESRGAN pipeline (bit-identity proof in parity_check.txt).`

**(3) Table tab:pairs caption, lines 490–498.** Delete the dagger sentence:

> "$^\dagger$The balanced pair's Real-ESRGAN images were produced by a patch-based invocation of the attack that is measurably harsher than the whole-image invocation used for the other two pairs; see the like-for-like control in the text before comparing this value across columns."

and adjust the significance sentence to:

> "All pairwise NC differences are statistically significant after BH--FDR correction except the balanced--standard comparison under no attack (n.s.); the balanced--standard JPEG-Q50 difference is significant at $p_{\mathrm{adj}}=0.045$."

**(4) Table tab:pairs body, line 507.** Current:

> `Real-ESRGAN    & $0.783 \pm 0.124$ & $0.683 \pm 0.139^\dagger$   & $0.831 \pm 0.109$ \\`

Replace with:

> `Real-ESRGAN    & $0.783 \pm 0.124$ & $0.809 \pm 0.112$   & $0.831 \pm 0.109$ \\`

**(5) Table tab:pairs body, line 511 (PSNR row).** Current: `PSNR (dB) & $40.6$ & $40.9$ & $43.9$`. Replace the balanced value: `PSNR (dB) & $40.6$ & $40.5$ & $43.9$`.

**(6) Section sec:pairs, "Real-ESRGAN" paragraph, lines 518–537.** Replace everything from "The balanced pair's tabulated value (0.683) must be read with care" through "the next subsection diagnoses why." with:

> "The balanced pair, rerun through the identical whole-image attack pipeline on all 100 images (the pipeline inconsistency in the original validation run was identified by the decision-margin control experiment of Section~\ref{sec:margin}), reaches NC of 0.809 --- a modest but significant $+0.025$ improvement over standard ($p_{\mathrm{adj}}<10^{-8}$, $r=0.59$), stable across all four embedding strengths ($+0.020$ to $+0.026$), with no JPEG penalty. HF remains significantly better than balanced under Real-ESRGAN at $\alpha \le 0.20$ ($-0.022$ at $\alpha=0.10$, $p_{\mathrm{adj}}=4\times10^{-5}$). The analytically predicted \emph{large} advantage of the balanced pair therefore fails to materialize under a strictly like-for-like pipeline; the next subsection diagnoses why."

**(7) "Imperceptibility" paragraph, lines 547–550.** Current:

> "The balanced pair achieves higher PSNR ($+0.30$\,dB at $\alpha=0.10$), and the HF pair achieves substantially higher PSNR ($+3.3$\,dB), ..."

Replace with:

> "The balanced pair's PSNR is statistically indistinguishable from the standard pair's ($40.5$ vs.\ $40.6$\,dB at $\alpha=0.10$, n.s.), and the HF pair achieves substantially higher PSNR ($+3.3$\,dB), ..."

**(8) "Key finding" paragraph, lines 552–562.** Current:

> "delivers only a $+0.02$ like-for-like improvement over the standard pair under Real-ESRGAN ($n=20$ control; full rerun pending),"

Replace with:

> "delivers only a $+0.025$ improvement over the standard pair under Real-ESRGAN ($n=100$, $p_{\mathrm{adj}}<10^{-8}$),"

**(9) Limitations, lines 708–711.** Delete the clause:

> ", and the balanced-pair like-for-like control (Section~\ref{sec:pairs}) remains $n=20$ pending the camera-ready rerun"

**Not requiring changes:** the JPEG-Q50 and Gaussian-blur paragraphs (all values reproduced exactly), the HF-vs-standard statistics ("0.048 above standard"), the Exp H margin-analysis section, and the Table's standard/HF columns.

---
*Generated by experiments/exp_J_balanced_like_for_like/run_exp_J.py — no existing experiment file was modified.*
