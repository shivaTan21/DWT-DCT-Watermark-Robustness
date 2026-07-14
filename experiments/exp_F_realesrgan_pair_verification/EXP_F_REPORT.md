# Experiment F: Real-ESRGAN Pair Verification Report

Date: 2026-07-13  |  Primary α: 0.1  |  n=100 images (24 Kodak + 76 TAMPERE17)

## 1. Experiment Overview

Three coefficient-pair configurations are compared across four attacks and four embedding
strengths (α ∈ {0.05, 0.10, 0.20, 0.30}):

| Pair key  | Positions        | Origin                          |
|-----------|------------------|---------------------------------|
| standard  | (4,1)/(1,4)      | Current baseline (watermark.py) |
| balanced  | (2,3)/(3,2)      | Exp D global optimum (E=0.818)  |
| hf        | (7,5)/(7,7)      | Empirically best (frequency_sweep) |

**Real-ESRGAN data source:**
- Standard: pre-computed disk files (`results/attacked/ai_enhancement/*__real_esrgan.png`)
- HF (7,5)/(7,7): pre-computed disk files (`*__optimized_positions__real_esrgan.png`)
- Balanced: computed live with py_real_esrgan ✓

## 2. Mean NC Results (α=0.10)

| Attack       | Standard         | Balanced         | HF (7,5)/(7,7)   |
|--------------|------------------|------------------|------------------|
| No attack    | 0.9997±0.0002 | 0.9997±0.0002 | 1.0000±0.0000 |
| Real-ESRGAN  | 0.7833±0.0246 | 0.6826±0.0275 | 0.8311±0.0215 |
| JPEG Q50     | 0.9953±0.0038 | 0.9979±0.0008 | 0.9015±0.0178 |
| Gaussian Blur | 0.9097±0.0125 | 0.9330±0.0110 | 0.9768±0.0050 |

## 3. Statistical Tests — Balanced vs Standard (BH-FDR corrected)

| Attack         | α    | Δ NC   | p_raw  | p_adj  | sig  | Effect | Test     |
|----------------|------|--------|--------|--------|------|--------|----------|
| No attack      | 0.05 | -0.0000 | 3.298e-01 | 3.518e-01 | n.s. | -0.097 | wilcoxon |
| Real-ESRGAN    | 0.05 | -0.1014 | 2.160e-18 | 8.641e-18 | ***  | -inf | wilcoxon |
| JPEG Q50       | 0.05 | +0.0028 | 7.349e-03 | 9.534e-03 | **   | 0.268 | wilcoxon |
| Gaussian Blur  | 0.05 | +0.0239 | 4.990e-14 | 8.260e-14 | ***  | 0.753 | wilcoxon |
| No attack      | 0.10 | +0.0000 | 4.184e-01 | 4.273e-01 | n.s. | 0.081 | wilcoxon |
| Real-ESRGAN    | 0.10 | -0.1006 | 2.624e-18 | 9.095e-18 | ***  | -inf | wilcoxon |
| JPEG Q50       | 0.10 | +0.0026 | 3.918e-02 | 4.373e-02 | *    | 0.206 | wilcoxon |
| Gaussian Blur  | 0.10 | +0.0233 | 1.851e-14 | 3.291e-14 | ***  | 0.766 | wilcoxon |
| No attack      | 0.20 | +0.0000 | 3.527e-01 | 3.681e-01 | n.s. | 0.093 | wilcoxon |
| Real-ESRGAN    | 0.20 | -0.0998 | 1.942e-18 | 8.641e-18 | ***  | -inf | wilcoxon |
| JPEG Q50       | 0.20 | +0.0025 | 1.451e-02 | 1.741e-02 | *    | 0.244 | wilcoxon |
| Gaussian Blur  | 0.20 | +0.0220 | 9.578e-15 | 1.768e-14 | ***  | 0.775 | wilcoxon |
| No attack      | 0.30 | +0.0000 | 5.000e-01 | 5.000e-01 | n.s. | 0.000 | wilcoxon |
| Real-ESRGAN    | 0.30 | -0.1001 | 1.939e-18 | 8.641e-18 | ***  | -inf | wilcoxon |
| JPEG Q50       | 0.30 | +0.0023 | 5.206e-02 | 5.680e-02 | n.s. | 0.194 | wilcoxon |
| Gaussian Blur  | 0.30 | +0.0191 | 4.596e-15 | 8.823e-15 | ***  | 0.784 | wilcoxon |

## 4. Statistical Tests — HF vs Standard (BH-FDR corrected)

| Attack         | α    | Δ NC   | p_raw  | p_adj  | sig  | Effect | Test     |
|----------------|------|--------|--------|--------|------|--------|----------|
| No attack      | 0.05 | +0.0003 | 1.802e-03 | 2.471e-03 | **   | 0.312 | wilcoxon |
| Real-ESRGAN    | 0.05 | +0.0555 | 1.025e-16 | 2.051e-16 | ***  | 1.001 | t-test |
| JPEG Q50       | 0.05 | -0.0944 | 1.937e-18 | 8.641e-18 | ***  | -inf | wilcoxon |
| Gaussian Blur  | 0.05 | +0.0765 | 4.144e-18 | 1.170e-17 | ***  | inf | wilcoxon |
| No attack      | 0.10 | +0.0003 | 2.967e-03 | 3.956e-03 | **   | 0.297 | wilcoxon |
| Real-ESRGAN    | 0.10 | +0.0478 | 2.553e-14 | 4.376e-14 | ***  | 0.891 | t-test |
| JPEG Q50       | 0.10 | -0.0938 | 1.934e-18 | 8.641e-18 | ***  | -inf | wilcoxon |
| Gaussian Blur  | 0.10 | +0.0672 | 4.144e-18 | 1.170e-17 | ***  | inf | wilcoxon |
| No attack      | 0.20 | +0.0001 | 9.815e-03 | 1.240e-02 | *    | 0.258 | wilcoxon |
| Real-ESRGAN    | 0.20 | +0.0319 | 8.034e-09 | 1.244e-08 | ***  | 0.631 | t-test |
| JPEG Q50       | 0.20 | -0.0915 | 1.941e-18 | 8.641e-18 | ***  | -inf | wilcoxon |
| Gaussian Blur  | 0.20 | +0.0510 | 1.021e-17 | 2.365e-17 | ***  | inf | wilcoxon |
| No attack      | 0.30 | +0.0001 | 2.939e-02 | 3.359e-02 | *    | 0.218 | wilcoxon |
| Real-ESRGAN    | 0.30 | +0.0179 | 3.021e-04 | 4.532e-04 | ***  | 0.375 | t-test |
| JPEG Q50       | 0.30 | -0.0891 | 1.938e-18 | 8.641e-18 | ***  | -inf | wilcoxon |
| Gaussian Blur  | 0.30 | +0.0361 | 1.035e-17 | 2.365e-17 | ***  | inf | wilcoxon |

## 5. Primary Question

**Does (2,3)/(3,2) significantly improve Real-ESRGAN NC over (4,1)/(1,4)**
**without a large JPEG penalty?**

**Real-ESRGAN NC**: balanced=0.6826 vs standard=0.7833
  ΔNC = -0.1006  (***, p_adj=9.095e-18)

→ Balanced pair does **not** improve Real-ESRGAN NC over standard.

**JPEG Q50 NC delta**: +0.0026 (*, p_adj=4.373e-02)
→ JPEG robustness does **not** significantly regress under balanced pair.

## 6. Analytical Ranking vs Empirical Performance

Exp D predicted: balanced pair (E-score=0.818) > standard (E-score=0.028) for ESRGAN,
with competitive JPEG NC (J-score=0.886 vs 0.877).

- Balanced ESRGAN NC: 0.6826  (predicted improvement confirmed: NO)
- HF ESRGAN NC:       0.8311  (should exceed balanced for ESRGAN-alone)
- Balanced JPEG NC:   0.9979  Standard JPEG NC: 0.9953

Analytical ranking is **NOT confirmed** empirically for ESRGAN at this alpha.

## 7. Alpha Dependence

| α    | Std ESRGAN NC | Bal ESRGAN NC | HF ESRGAN NC  |
|------|---------------|---------------|---------------|
| 0.05 | 0.7738        | 0.6724        | 0.8293        |
| 0.10 | 0.7833        | 0.6826        | 0.8311        |
| 0.20 | 0.8023        | 0.7026        | 0.8342        |
| 0.30 | 0.8198        | 0.7197        | 0.8377        |

## 8. Imperceptibility

| α    | Std PSNR (dB) | Bal PSNR (dB) | HF PSNR (dB) |
|------|---------------|---------------|--------------|
| 0.05 | 40.72         | 41.03         | 43.99        |
| 0.10 | 40.58         | 40.88         | 43.92        |
| 0.20 | 40.30         | 40.57         | 43.77        |
| 0.30 | 40.03         | 40.27         | 43.63        |

## 9. Conclusions

- Balanced pair ESRGAN: ΔNC=-0.1006 vs standard (significant).
- JPEG Q50 delta for balanced pair: ΔNC=+0.0026 (no significant regression).
- **Balanced pair not empirically validated** as superior to standard under ESRGAN.

---
*Report generated by experiments/exp_F_realesrgan_pair_verification/run_exp_F.py*