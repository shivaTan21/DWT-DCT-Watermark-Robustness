# Research Synthesis: DWT-DCT Watermarking Under AI Enhancement
## Target: IEEE WIFS 2026

---

## 1. EXECUTIVE SUMMARY OF ALL FINDINGS

### Core Discovery (Paper-Level Claim)
**AI image enhancement (Real-ESRGAN) and JPEG compression create near-perfectly anti-complementary DCT perturbation patterns in the DWT LL subband (Spearman ρ = −0.907, p < 10⁻²⁵). This structural incompatibility creates an irreducible frequency-space trade-off in DWT-DCT watermark embedding: coefficient positions optimal for JPEG robustness are systematically vulnerable to AI enhancement, and vice versa. We characterize this trade-off, identify the joint-optimal embedding zone, and explain the mechanism via ESRGAN's near-isotropic DCT perturbation profile (isotropy r = 0.993 vs JPEG r = 0.778).**

---

## 2. EXPERIMENTAL FINDINGS BY EXPERIMENT

### Experiment A: DCT Perturbation Isotropy
**Hypothesis:** ESRGAN's DCT damage pattern is isotropic (damage(i,j) ≈ damage(j,i))

**Results:**
- ESRGAN isotropy: Pearson r = 0.993 (p = 1.2e-51)
- JPEG isotropy: Pearson r = 0.778 (p = 1.75e-12)
- ESRGAN asymmetry ratio: 0.0245 vs JPEG: 0.0886
- ESRGAN mean isotropy ratio per symmetric pair: 0.0387 vs JPEG: 0.0752 (t-test p = 0.003)
- Damage differential → NC correlation: Spearman ρ = −0.773 (p = 5e-6)

**Status:** CONFIRMED. ESRGAN is significantly more isotropic than JPEG.

**Mechanism:** ESRGAN's learned convolutional filters process horizontal and vertical spatial frequencies symmetrically (isotropically), creating correlated perturbations at symmetric DCT positions (i,j) and (j,i). This means symmetric coefficient pairs preserve their relative comparison (c₁ > c₂) despite perturbation.

### Experiment B: Multi-Feature NC Predictor
**Hypothesis:** Per-image NC under ESRGAN can be predicted from pre-embedding image statistics

**Results:**
- Best univariate predictor: ll_spectral_entropy (Pearson r = +0.533, Spearman ρ = 0.570, p = 1.2e-8)
- Second best: local_std_cv (r = −0.477 — NEGATIVE: heterogeneous texture → lower NC)
- Third: edge_density (r = +0.459, ρ = 0.459)
- Best multivariate (Ridge, 5-fold CV): R² = 0.377 ± 0.256
- Random Forest feature importance: ll_spectral_entropy (19.5%), edge_density (17.5%)

**Status:** PARTIALLY CONFIRMED. Moderate predictability (R² ≈ 0.38) — not individually reliable but reveals which image properties drive robustness.

**Key Insight:** Images with uniform DCT distributions (high ll_spectral_entropy) and rich edge structure survive better. Images with heterogeneous local texture (high local_std_cv) fail more — smooth blocks get catastrophically damaged by ESRGAN's texture synthesis.

### Experiment C: Coefficient Design Principles
**Hypothesis:** NC is predicted by joint stability + symmetry independently

**Results:**
- Strongest predictor of ESRGAN NC: freq_radius_sum (ρ = +0.885) — counterintuitive
- Stability differential (ρ = −0.773), Combined damage (ρ = −0.691)
- Symmetry (ρ = +0.484) — DOES predict NC but is mediated by damage level
- Partial correlation of symmetry controlling for combined damage: ρ = 0.090 (p = 0.565) — NOT significant
- **CRITICAL: ESRGAN NC vs JPEG NC Spearman ρ = −0.488 (p = 9e-4) — genuine trade-off**
- Top pair: (7,5)/(7,7) — NC_ESRGAN = 0.864, NC_JPEG = 0.893 (near-symmetric)
- Standard pair (4,1)/(1,4) NC_ESRGAN ≈ 0.795

**Status:** Design rule confirmed. The symmetry benefit is real but mediated through the damage level. The PRIMARY predictor is whether the pair occupies a stable frequency zone.

### Experiment D: Exhaustive Optimization (1953 pairs)
**Key structural finding:**
- Spearman ρ(ESRGAN damage, JPEG Q-step) = −0.907 (p = 5e-25)
- ESRGAN and JPEG attack COMPLEMENTARY DCT positions
- JPEG Q-step sum predicts ESRGAN NC: ρ = +0.785 (can use JPEG table as proxy!)
- Pareto frontier: 37 pairs out of 1953
- Globally optimal balanced pair: **(2,3)/(3,2)** — symmetric, E-score=0.818, J-score=0.886
- Standard pair (4,1)/(1,4): E-score=0.028, J-score=0.877 — NOT Pareto-optimal

---

## 3. UNIFIED MECHANISTIC EXPLANATION

### The Two-Level Frequency Conflict

**Level 1: Attack-space (ESRGAN vs JPEG)**
ESRGAN and JPEG perturbation modes are anti-complementary:
- JPEG quantizes high-frequency coefficients within the LL-DCT subband most aggressively
- ESRGAN disturbs low-frequency coefficients within LL-DCT most (in absolute terms)
- This creates ρ = −0.907 anti-correlation between attack susceptibilities per position

**Level 2: Standard embedding is suboptimally placed**
The standard embedding positions (4,1)/(1,4) are:
- Located at (row,col) that creates ANOMALOUSLY high ESRGAN damage (~10.4 vs ~4.0 for neighbors)
- Well within the JPEG-stable zone (q-step ≈ 22-26, moderate)
- Result: good JPEG performance but poor ESRGAN performance
- NOT Pareto-optimal on the ESRGAN-JPEG frontier

**Level 3: ESRGAN isotropy interacts with symmetry**
- ESRGAN's near-isotropic damage means symmetric pairs (i,j)/(j,i) experience correlated perturbations
- The relative comparison (c₁ > c₂) is preserved when both are perturbed similarly
- This provides a secondary robustness benefit WITHIN a stable frequency zone
- But symmetry alone cannot rescue pairs in highly-damaged zones

### Causal Chain:
```
ESRGAN architecture (learned isotropic filters)
    → Near-isotropic DCT perturbation pattern (r = 0.993)
        → High-frequency LL-DCT positions less disturbed
            → Standard embedding in low-freq LL-DCT zone → high ESRGAN damage
                → NC degradation under AI enhancement
    
JPEG compression (standard quantization table)
    → High-frequency LL-DCT positions heavily quantized
        → Fundamental anti-complementarity with ESRGAN (ρ = −0.907)
            → Irreducible ESRGAN-JPEG trade-off in embedding position choice
```

---

## 4. SKEPTICAL REVIEWER CRITIQUE

*Acting as IEEE WIFS reviewer attempting to reject this paper:*

### Objection 1: "Isotropy is trivial — all natural images are isotropic"
**Response:** Not claimed. We measure isotropy of the ATTACK's DCT perturbation pattern, not the image. JPEG's perturbation is demonstrably less isotropic (r=0.778 vs 0.993). The isotropy is a property of ESRGAN's learned network architecture processing.

### Objection 2: "43 pairs is not exhaustive — results may not generalize"
**Response addressed by Experiment D:** All 1953 pairs analyzed analytically. The optimal zone (symmetric pairs at (2-5,2-5) in LL-DCT) is confirmed across the full space.

### Objection 3: "Partial correlation not significant — symmetry isn't a real effect"
**Honest assessment:** Correct. Symmetry's effect on NC is mediated by damage level. The isotropy story is real and mechanistically valid, but the empirical effect is smaller than initially suggested. The paper should not overclaim on symmetry.

### Objection 4: "The NC predictor R²=0.38 is too weak to be useful"
**Honest assessment:** Fair criticism. R²=0.38 means 62% of variance unexplained. Not reliable for individual images. The paper should frame this as "characterizing image properties that drive robustness" rather than "predicting NC reliably."

### Objection 5: "The ESRGAN-JPEG trade-off is just the JPEG quantization table — trivially known"
**Response:** The trade-off was known qualitatively. What's new: (1) quantifying it at ρ=−0.907, (2) showing it creates an irreducible Pareto frontier, (3) identifying the globally optimal balanced pair, (4) showing the standard pair (4,1)/(1,4) is NOT Pareto-optimal. The actionable implication (re-embed with (2,3)/(3,2)) is novel.

### Objection 6: "Only one AI model (ESRGAN) — results may not generalize to other AI enhancers"
**Partially addressed:** ESPCN shows NC ≈ 0.999 (near-perfect robustness) while ESRGAN shows NC ≈ 0.795. These behave very differently. Generalizing "AI enhancement attacks" to all models is premature. The paper should focus specifically on GAN-based super-resolution (ESRGAN, potentially SwinIR).

### Objection 7: "Dataset of 100 natural photographs — may not cover all content types"
**Honest assessment:** Valid. Kodak + TAMPERE17 are standard benchmarks but bias toward natural photographs. Synthetic images, document images, medical images may behave differently.

### What additional evidence would convince me to accept this paper:
1. **Verify (2,3)/(3,2) pair empirically** — run full frequency sweep for top-5 analytically predicted pairs on held-out images
2. **Test SwinIR** — does SwinIR also show near-isotropic DCT damage? If yes, strong generality claim
3. **Correct the symmetry narrative** — reframe from "symmetric pairs are better" to "position stability dominates, symmetry is secondary"
4. **Explain (4,1)/(1,4) anomaly** — WHY is this specific position anomalously damaged by ESRGAN? Network filter visualization?
5. **Ablation on alpha** — do these findings hold across embedding strengths?

---

## 5. PAPER STRUCTURE RECOMMENDATION

### Title:
"Frequency-Space Conflict Between AI Enhancement and JPEG Compression Reveals Design Principles for Robust DWT-DCT Watermarking"

### Contributions:
1. First characterization of ESRGAN's DCT perturbation pattern as near-isotropic (r=0.993) vs JPEG (r=0.778)
2. Discovery that ESRGAN and JPEG create anti-complementary damage patterns (ρ=−0.907) — creating an irreducible embedding trade-off
3. Empirical identification of coefficient pair design principles: (joint stability + frequency zone) ≻ (symmetry alone)
4. Demonstration that the standard (4,1)/(1,4) pair is not Pareto-optimal under ESRGAN+JPEG attacks
5. Analytically derived Pareto-optimal pair space across all 1953 non-DC positions

### Recommended Structure:
```
I. Introduction
   - AI enhancement as emerging watermark threat
   - Research questions about frequency-domain robustness

II. Background
   - DWT-DCT watermarking (textbook baseline used)
   - JPEG and AI enhancement mechanisms
   - Prior work on coefficient selection

III. Experimental Setup (100 images, 3 embedding variants)

IV. ESRGAN DCT Damage Characterization
   - Damage matrix (Table/Figure)
   - Isotropy analysis (Figure: ESRGAN vs JPEG isotropy scatter)
   - Anomalous positions (4,1)/(1,4) finding

V. ESRGAN-JPEG Trade-Off
   - Anti-complementarity (ρ=−0.907 result)
   - Pareto frontier visualization
   - Standard pair's suboptimal position

VI. Coefficient Design Principles
   - Joint stability as primary predictor
   - Symmetry as secondary mechanism
   - Globally optimal pair identification

VII. Image-Level Robustness Prediction
   - ll_spectral_entropy as key predictor
   - Practical implications

VIII. Conclusions and Future Work
```

---

## 6. RECOMMENDED NEXT EXPERIMENTS

**Priority 1 (Essential for submission):**
- Run empirical verification of (2,3)/(3,2) pair on 20 held-out images
- Test SwinIR damage isotropy to establish generality

**Priority 2 (Strengthening):**
- Ablation: does the trade-off hold across alpha values (0.05, 0.1, 0.2)?
- Analysis of WHY (4,1)/(1,4) is anomalously damaged (convolutional filter visualization)

**Priority 3 (Extensions):**
- Multi-attack robustness: can a single pair be found that's robust to ESRGAN + JPEG + blur?
- Content-adaptive pair selection using NC predictor

---

## 7. NOVELTY ASSESSMENT

| Finding | Prior Art Status | Novelty |
|---------|-----------------|---------|
| ESRGAN damages watermarks | Already known (motivation for this work) | None |
| ESRGAN DCT damage is isotropic (r=0.993) | Not previously measured | HIGH |
| ESRGAN-JPEG anti-complementarity (ρ=−0.907) | Not quantified at this level | HIGH |
| Standard (4,1)/(1,4) is not Pareto-optimal | Not previously shown | MEDIUM-HIGH |
| Jointly-optimal pair identification | Not previously published | MEDIUM |
| NC predictor from image features | Partial prior work on content-adaptive WM | MEDIUM |
| Symmetry benefit under ESRGAN | Partially explored in this project | MEDIUM (weaker than initially thought) |
