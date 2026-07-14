# Experiment H: Decision-Margin Analysis — Why the Analytical Prediction Failed

Date: 2026-07-13 | α = 0.10 | n = 100 images (24 Kodak + 76 TAMPERE17) | 1024 blocks/image
Data: `block_margins.csv` (307,200 block measurements), `image_summary.csv`, controls in
`control_pipeline_check.csv`, `restoration_regression.csv`, `flip_vs_natural_margin.csv`.

---

## 0. Validity check

The measurement pipeline reproduces Exp F's frozen NC values **exactly** (bit-for-bit for the
disk-based pairs, and to the fourth decimal for the mean):

| Pair | Exp F frozen NC | Exp H re-measured |
|---|---|---|
| Standard (4,1)/(1,4) | 0.7833 | 0.7833 |
| Balanced (2,3)/(3,2) | 0.6826 | 0.6826 |
| HF (7,5)/(7,7) | 0.8311 | 0.8311 |

All margins below are measured on the identical images that produced those NC values.

---

## 1. FINDING 0 (unplanned, but load-bearing): Exp F's balanced-vs-standard comparison was confounded

Exp F took standard/HF ESRGAN outputs from disk (default `model.predict`, INTER_AREA
downscale) but computed the balanced pair **live** with a different invocation
(`patches_size=64`, Lanczos4 downscale — the MPS workaround in `patch_balanced_esrgan.py`).
Exp H ran the *standard and HF pairs through that same live pipeline* on a 20-image control
subset:

| Comparison | n | NC (a) | NC (b) | Δ | Wilcoxon p |
|---|---|---|---|---|---|
| standard: live vs disk pipeline | 20 | 0.6947 | 0.7971 | **−0.1023** | 0.0001 |
| hf: live vs disk pipeline | 20 | 0.7266 | 0.8642 | **−0.1376** | 0.0001 |
| **like-for-like (live): balanced vs standard** | 20 | 0.7143 | 0.6947 | **+0.0195** | 0.038 |
| like-for-like (live): hf vs standard | 20 | 0.7266 | 0.6947 | +0.0318 | 0.030 |

The live invocation is a substantially harsher attack (~−0.10 to −0.14 NC on the *same*
watermarked images). Exp F's headline deficit for the balanced pair (ΔNC = −0.1006) is
almost exactly the pipeline penalty measured on the standard pair (−0.1023).

**Under an identical attack pipeline, the balanced pair does not underperform the standard
pair — it is slightly (and significantly) better (+0.02, p = 0.038, n = 20).** The Exp D
analytical ranking (balanced > standard) is directionally correct after all; what was wrong
is its *magnitude* (predicted a large gap; the real like-for-like gap is small) and its
*mechanism* (see §3). HF remains the best pair under both pipelines.

Caveats: n = 20 for the like-for-like tests; the frozen Exp F CSV itself is untouched.
Everything in §2–§4 below is pipeline-internal (each pair's margins measured against its own
attacked images), so the mechanism analysis is unaffected by this confound.

## 2. Which variable predicts NC?

Image-level correlations with NC, pooled across the three pairs (n = 300; full table incl.
per-pair scopes in `correlations_nc.csv`):

| Predictor | Pearson r | Spearman ρ | Mutual info |
|---|---|---|---|
| **Sign-flip rate** | −0.999 | −1.000 | 4.03 |
| **Differential perturbation std, σ(ΔC1−ΔC2)** | −0.843 | −0.887 | 0.92 |
| Absolute damage (|ΔC1|+|ΔC2|)/2 — *Exp D's implicit predictor* | −0.797 | −0.825 | 0.65 |
| Mean \|ΔC1−ΔC2\| | −0.787 | −0.809 | 0.66 |
| Bit-directed margin loss | +0.582 | +0.596 | 0.25 |
| Common-mode perturbation std | −0.243 | −0.220 | 0.01 |
| Mean \|M_before\| | −0.039 | −0.071 (n.s., p = 0.22) | 0.07 |
| Min / median \|M_before\| | ~0 (n.s.) | ~0 (n.s.) | — |

Reading this honestly:

- **Sign-flip rate ↔ NC is essentially an identity**, not a discovery: extraction reads
  `bit = [C1′ > C2′]`, embedding leaves NC ≈ 1 pre-attack, so a flipped margin *is* a bit
  error. Its value is as a confirmed consistency check: bit errors are sign flips, nothing else
  (the two rates agree to <0.1% in `success_vs_failure.csv`).
- **The best genuine predictor is the standard deviation of the differential perturbation**
  σ(ΔC1−ΔC2): ρ = −0.887, beating absolute coefficient damage (ρ = −0.825) and by far the
  common mode (ρ = −0.22).
- **Margin magnitude before attack predicts nothing** (ρ ≈ −0.07, n.s.). This falsifies the
  naive "bigger margins survive" version of the margin hypothesis — because the embedding
  enforces a near-constant margin everywhere (median |M_before| = 30.2–31.8 for all three
  pairs; the ABSOLUTE_MARGIN_FLOOR = 30 dominates α·energy at these positions). With margins
  clamped, all variance in survival comes from the noise side.

### Block level: what makes an individual block flip?

Comparing the 33k flipped blocks against the 274k that held (pooled; per-pair in
`block_flip_logistic.csv`):

| Quantity | Median (flipped) | Median (kept) | Rank-biserial effect |
|---|---|---|---|
| \|ΔC1−ΔC2\| (differential) | 36.6 | 19.0 | **−0.926** |
| \|ΔC1\| | 18.0 | 10.0 | −0.708 |
| \|ΔC2\| | 18.3 | 10.5 | −0.698 |
| \|(ΔC1+ΔC2)/2\| (common mode) | 2.76 | 2.39 | −0.056 |
| \|M_before\| | 31.3 | 31.2 | +0.034 |

A block flips when the **differential-mode perturbation exceeds its margin** — and almost
never otherwise. The fraction of blocks with |ΔC1−ΔC2| > |M_before| reproduces each pair's
flip rate nearly exactly (standard 10.9% vs 10.9%; balanced 16.9% vs 15.9%; HF 8.5% vs 8.5%).
Common-mode noise, though present, is 5–8× smaller than differential noise and irrelevant —
the relative comparator is, by design, blind to it.

## 3. Why the analytical model failed: the attack is *endogenous*

Exp D scored positions by absolute coefficient stability measured as if ESRGAN added fixed,
position-dependent noise. Exp H shows the noise is **signal-dependent — ESRGAN partially
undoes the embedding itself** and regresses each block toward its *natural* statistics:

**(a) The perturbation is anti-correlated with what the embedding inserted.**
corr(ΔC1−ΔC2, embedding-induced margin shift) = −0.73 / −0.60 / −0.80 for
standard / balanced / HF. ESRGAN removes ~32% / 30% / 53% of the embedded differential
(`restoration_regression.csv`). This also explains the anti-correlation between ΔC1 and ΔC2
(−0.64 / −0.45 / −0.80): restoration pushes the two coefficients back *toward each other*,
attacking precisely the differential mode the detector reads.

**(b) The post-attack margin regresses toward the block's natural margin.**
Adding the original (unwatermarked) margin M_orig to the regression M_after ~ M_before raises
R² from 0.71→0.83 (standard), 0.66→0.74 (balanced), 0.48→0.55 (HF), with M_orig carrying a
coefficient of ~0.27–0.29. A model of ESRGAN as a projector toward natural-image statistics,
M_after ≈ (1−k)·M_before + k·M_orig + ε, fits all three pairs.

**(c) Consequently, the watermark dies exactly where it fought nature.**
Blocks whose natural margin sign *opposes* the embedded bit flip at 3.4–9.5× the rate of
agreeing blocks (χ² p ≈ 0), and **77–91% of all flips occur in that ~50% of blocks**
(`flip_vs_natural_margin.csv`). Blocks with strongly opposing natural margins (|M_orig| > 30)
flip at 20–36%.

**(d) This resolves the HF paradox and the balanced non-miracle in one stroke.**
- HF (7,5)/(7,7) suffers the *largest* restoration fraction (53%) yet survives best, because
  its natural margins are tiny (mean |M_orig| = 10.1): there is almost no "nature" for ESRGAN
  to restore against the embedded ordering, so regression toward M_orig rarely crosses zero.
- Standard and balanced positions both have natural margins (|M_orig| ≈ 30–31) comparable to
  the enforced embedding margin (~30), so restoration of an opposing natural margin routinely
  drags the decision statistic across zero. The two pairs share this property — which is why,
  like-for-like, they perform within 0.02 NC of each other, not the large gap Exp D predicted
  from absolute-damage scores.

### Answer to the final question (A/B/C/D)

**B is the proximate mechanism** — sign flips fully account for NC loss (flips ≡ bit errors;
margins do not "gracefully degrade" into low-confidence correct decisions, they cross zero).
But B is caused by a refined **D**: an endogenous, restoration-type mechanism. ESRGAN
regresses coefficient pairs toward their natural relationship; flips concentrate where the
natural ordering opposes the embedded bit and the differential-mode perturbation
(σ = 23–28, of which a large part is the deterministic undo-component) exceeds the ~30-unit
enforced margin. A (margin collapse) is true only in this directed sense — margins collapse
*toward M_orig*, not toward zero generically. C (asymmetric perturbation) is a symptom: the
anti-correlation of ΔC1, ΔC2 *is* the restoring differential mode, not an independent cause.
Additionally, the specific empirical failure that motivated this experiment (balanced ≪
standard) was **substantially an attack-pipeline artifact** (§1); the mechanism above explains
why balanced was never going to deliver a large win, but the sign of Exp D's prediction was
correct.

## 4. Revised conceptual model of watermark robustness under generative enhancement

> A relative-comparison watermark survives a generative restoration attack iff the enforced
> decision margin exceeds the attack's *differential-mode* pull toward the block's natural
> coefficient relationship. Robustness is governed by three quantities at the chosen positions:
> (1) the distribution of **natural margins |M_orig|** (small ⇒ little to restore ⇒ safe);
> (2) the **restoration strength k** and stochastic differential noise σ(ΔC1−ΔC2);
> (3) the **enforced margin** (here clamped ≈ 30 by the floor).
> Flip probability ≈ P( k·(M_orig − M_before) + ε < −M_before ), concentrated in blocks where
> sign(M_orig) opposes the bit.
>
> Absolute coefficient stability is the wrong selection criterion because generative attacks
> are endogenous: they do not add exogenous noise at fixed positions, they *pull embedded
> statistics back toward natural-image statistics*. JPEG, by contrast, is a genuinely
> exogenous quantizer — which is why Exp D's stability model works for JPEG and fails for
> ESRGAN, and it gives the anti-complementarity finding (ρ = −0.907) a cleaner
> interpretation: the two attacks are not just spectrally complementary, they are
> *mechanistically different kinds of noise*.

Practical corollary: good positions for generative-attack robustness are symmetric pairs
whose natural margins are near zero — which is what (7,5)/(7,7) accidentally provides, and
why it keeps winning. Position optimization should rank pairs by the natural-margin
distribution and differential-noise ratio, not by per-position damage tables.

## 5. How the IEEE paper should change

1. **Correct the Exp F comparison before anything else.** The balanced-vs-standard ESRGAN
   numbers mix two attack pipelines of different severity; the −0.10 deficit does not
   survive a like-for-like control. Either re-run all three pairs through one pipeline
   (the honest fix; ~1 hr of compute) or report the like-for-like control (n = 20,
   balanced +0.02 over standard, p = 0.038). Publishing the frozen Exp F table as-is would
   be wrong.
2. **Reframe the contribution.** The paper's mechanism section should center on the
   endogenous-restoration model: generative enhancement attacks the differential mode of
   relative-comparison detectors and regresses embedded statistics toward natural ones
   (corr(perturbation, embedded shift) up to −0.80; 77–91% of failures in nature-opposing
   blocks). This is a general statement about blind QIM/comparison-style watermarks vs
   restoration models, bigger than one coefficient-pair choice.
3. **Keep the anti-complementarity result** (Exp A/D) but re-interpret it: JPEG = exogenous
   quantization vs ESRGAN = endogenous restoration explains *why* their damage profiles
   anti-correlate and why no static position choice fully escapes the trade-off.
4. **Demote the (2,3)/(3,2) optimization narrative** from "analytically optimal pair" to a
   case study in why absolute-stability scores mispredict — the failed prediction plus its
   diagnosis is the story.
5. Report the flip-rate ≡ BER identity and the margin-floor clamp as design notes (margin
   magnitude is not a usable lever at these settings; the floor dominates).

## 6. Is this more interesting than the original hypothesis?

Yes. The original contribution was "we searched 1953 coefficient pairs and found a better
one" — incremental, and (as Exp F/H showed) fragile. The Exp H result is a mechanism:
**generative restoration is a signal-dependent attack that selectively erases whatever the
embedder inserted against natural-image statistics**, with a quantitative model (restoration
fraction k, natural-margin opposition, differential-mode SNR) that predicts per-block failure
(flip rate ≈ P(|ΔC1−ΔC2| > |M|), matched to within 1 pp for all three pairs). It explains the
HF pair's success, the balanced pair's non-miracle, and the JPEG/ESRGAN trade-off in one
framework, and it generalizes beyond this scheme to any blind relative-comparison detector.
Negative-result honesty (margin magnitude predicts nothing; the motivating deficit was partly
an artifact) strengthens rather than weakens the narrative.

## 7. Reviewer assessment (acting as IEEE WIFS reviewer)

*Summary:* The authors diagnose why an analytically selected DCT coefficient pair failed to
deliver predicted robustness gains under Real-ESRGAN. They show extraction failure is
entirely sign-flips of the decision margin, driven by differential-mode perturbation that is
anti-correlated with the embedded shift, and that failures concentrate (risk ratio 3.4–9.5×)
in blocks whose natural coefficient ordering opposes the embedded bit. They further identify
an attack-severity confound in their own earlier comparison and correct it with a
like-for-like control.

*Strengths:* (i) A clear falsification-then-diagnosis arc, rare and valuable; (ii) the
endogenous-restoration model is well supported by three independent statistics (undo
correlation, R² gain from M_orig, opposition risk ratio) on 307k block measurements;
(iii) the self-identified pipeline confound and its control demonstrate unusual rigor;
(iv) block-level flip prediction matched to ~1 pp for all pairs.

*Weaknesses:* (i) the like-for-like pipeline control uses only n = 20 images and should be
extended to all 100 before camera-ready; (ii) single enhancement model (Real-ESRGAN x4) and
single α for the margin data — the k ≈ 0.3–0.5 restoration fractions may be model-specific;
(iii) the flip-rate/NC correlation should be presented as a consistency identity, not a
finding; (iv) the revised position-selection rule (minimize natural |M_orig|) is stated but
not yet validated prospectively on a new pair.

*Verdict:* **This experiment strengthens the paper substantially.** It converts a broken
optimization claim into a mechanistic contribution with explanatory and predictive power, and
it removes a result (Exp F's balanced-pair deficit) that a careful reviewer would eventually
have caught as confounded. Accept-leaning, conditional on: extending the like-for-like
control to the full dataset, and softening any remaining "optimal pair" claims.

---
*Generated by `experiments/exp_H_margin_analysis/run_exp_H.py`, `analyze_exp_H.py`,
`analyze_restoration.py`. All frozen experiments (watermark.py, attacks_ai.py,
run_experiment.py, plot_results.py, Exp A–G outputs) untouched.*
