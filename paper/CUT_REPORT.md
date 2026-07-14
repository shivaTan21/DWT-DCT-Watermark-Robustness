# CUT_REPORT — `main_6page.tex` (WIFS 2026 page-limit reduction)

Base: `paper/main_corrected.tex` (untouched, as are `main.tex`, `main_revised2.tex`, all CSVs/experiments).
Build: pdflatex + bibtex + pdflatex ×2 after each stage; zero errors, zero undefined references/citations at every stage.

## Page counts

| Stage | Physical pages | Last-page fill (text-position measure) | Effective length | Saved |
|---|---|---|---|---|
| Baseline (`main_corrected.tex`) | 8 | overflowing (bibliography fills p.8) | ~8.0 | — |
| After Stage 1 | 8 | ~72% | ~7.7 | ~0.3–0.4 pp |
| After Stage 2 | 8 | ~37% (lowest text at y=473/792) | **~7.4** | ~0.4–0.5 pp |

Last line of text on the final page (both stages): the `ssim2004` bibliography entry — "…Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004."

**Honest status: ~7.4 effective pages, i.e. ~1.8 pages above the 5.6 target.** The cut plan's savings estimates were optimistic for this layout: much of pages 1–7 is fixed-size material (4 tables, 3 column-width figures, 2 equation blocks, ~1 page of bibliography), so prose cuts alone moved less than projected. Stage 3 proposals below.

## Stage 1 — removals (section/paragraph level)

- **Symmetry subsection + Table V (`tab:symmetry`) deleted**; replaced by one sentence at the end of the α-stability subsection reporting symmetric 0.821 vs asymmetric 0.689 ESRGAN NC (43-pair set, α=0.1, 20-image subset) with its `% SOURCE:` comment retained.
- **Related Work: 4 subsections → 2 paragraphs** (~60% shorter). All 17 citation keys retained in the paper; none disappeared (verified: every `\cite` key still occurs ≥1×).
- **Background: 4 subsections → 2** ("Scheme" + "Parameters and Metrics"): overview/embedding/extraction prose merged, the display `cases` equation for extraction replaced by an inline statement, duplicated "fully blind" sentence removed. All parameters kept (positions, δ=30.0, 1024-bit, seed 42, α set, NC formula).
- **Contamination caveats consolidated**: abstract now carries one clause ("its Real-ESRGAN comparison is non-final due to an identified and controlled attack-pipeline inconsistency, pending a full rerun"); Limitations now one clause cross-referencing Section `sec:pairs`. The full disclosure remains stated once, in Three-Pair Full Validation: Exp F pipeline inconsistency, identified by the margin-control experiment, −0.102 penalty (n=20, p=10⁻⁴), † on the 0.683 cell, +0.020 like-for-like control (n=20, p=0.038), "full rerun … required before camera-ready", upper-bound reading.

## Stage 2 — mechanism-section compression

Kept (priority order, all numbers verbatim): σ(ΔC1−ΔC2) predictor (ρ=−0.887 vs −0.825 abs-damage, −0.22 common-mode, −0.07 n.s. margin); undo/restoration regression in one paragraph (r=−0.73/−0.60/−0.80, k=32/30/53%, R² 0.71→0.83) with the "input–output behavior only, no claims about internal computations" sentence; natural-margin opposition (3.4–9.5×, 77–91%, ≈50% of blocks); Table IV unchanged; HF explanation in two sentences (|M_orig| = 10.1 vs ≈31).

Cut: the step-by-step measurement walk-through ("For every embedded block, the extractor reads…", the 3×100×1024 breakdown), the "Three results… First/Second/Third" scaffolding, the explicit M_after ≈ (1−k)M_before + kM_orig + ε model display, "5–8× smaller" common-mode aside, "margins do not degrade into low-confidence decisions" restatement, the redundant E-score-mispredicted recap and the position-selection-rule sentence (both restated in the Conclusion), and the standalone intro paragraph of the Generalization subsection (merged into its results paragraph). Added exactly one sentence: "Full derivations and additional analyses appear in the extended version."

## Constraint compliance

1. **No number altered** — number-integrity diff over all `table` environments: `tab:survival` 45 tokens, `tab:alpha_main` 12, `tab:pairs` 40, `tab:mechanism` 22 — all **byte-identical**; `tab:symmetry` (12 tokens) deleted whole. Zero modifications. Prose numbers were only ever deleted with their sentences, never edited.
2. **Contamination disclosure** — fully stated once (Three-Pair Validation, incl. †, n=20 control, camera-ready rerun); abstract still informs the reader the comparison is non-final. ✓
3. **Hedging retained** — 17 "consistent with"/"suggests" instances survive; every kept hedged sentence kept its hedge. ✓
4. **`% SOURCE:` comments** — all 6 retained (incl. the one for the deleted table, now above its replacement sentence). ✓
5. **Untouched** — title, author block, abstract headline numbers (0.795, 0.885; 0.9996/0.9994 in Results), keywords, acknowledgment, [REPO] placeholder, `\bibliography` line. ✓
6. **No EDSR content** — zero occurrences. ✓

## Stage 3 — PROPOSED cuts (NOT applied; awaiting your approval)

Needed: ~1.8 pages. Prose-only trims cannot deliver that; at least one figure or equivalent fixed-size element must go. Estimated savings:

| # | Proposal | Est. saving |
|---|---|---|
| P1 | Drop Fig. 3 (Pareto scatter, `frequency_sweep_heatmap.png`) — its two headline numbers (ρ=−0.488, 37-pair frontier) are already in the text | ~0.40 pp |
| P2 | Drop Fig. 1 (SSIM-vs-NC scatter) and fold its message into one sentence of Results §B — costlier rhetorically: it is the paper's motivating picture | ~0.40 pp |
| P3 | Shorten the intro's four-stage roadmap paragraph to ~⅓ (keep stage names + one clause each) | ~0.20 pp |
| P4 | Compress the four contribution bullets to two | ~0.15 pp |
| P5 | Reduce α-stability subsection to its two headline statistics (ρ range 0.010; Jaccard 1.0) + the exogenous/endogenous sentence | ~0.15 pp |
| P6 | Trim Exhaustive-Pareto subsection prose (keep E/J-scores, 37/1,953, ρ=−0.488) | ~0.15 pp |
| P7 | Trim figure captions to one line each | ~0.08 pp |
| P8 | Compress Results §A/§B narration (survival table walk-through) | ~0.15 pp |
| P9 | Subband paragraph → two sentences | ~0.05 pp |
| P10 | Compress Three-Pair Validation prose *around* the disclosure (JPEG/blur/imperceptibility paragraphs → one compact paragraph; disclosure untouched) | ~0.15 pp |
| P11 | Layout: `\IEEEtriggeratref` + moving both remaining figures to the same page tops (no content change) | ~0.05–0.10 pp |

Recommended package to hit ≤5.6: **P1 + P3 + P4 + P5 + P6 + P7 + P8 + P10** ≈ 1.4–1.5 pp, then re-measure; add **P2** (second figure) only if still above target (total ≈ 1.8–1.9 pp).

Please approve a subset (or all) of P1–P11 and I will apply them as Stage 3 with the same verification protocol.
