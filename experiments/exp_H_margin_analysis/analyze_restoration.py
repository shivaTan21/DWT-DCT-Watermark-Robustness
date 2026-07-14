"""
Supplementary Exp H analysis: is ESRGAN's differential perturbation ENDOGENOUS?

Hypothesis: ESRGAN acts as a projector toward natural-image statistics, so the
post-attack margin regresses toward the block's NATURAL margin (the C1-C2 the
original, unwatermarked image had), rather than being before-margin plus
exogenous noise. If true, the perturbation is signal-dependent: it actively
removes exactly the asymmetry the watermark inserted, and any position-
selection model that treats per-position damage as fixed exogenous noise
(Exp D's E-score) is structurally wrong.

Test: per pair, regress  m_after ~ a*m_before + b*m_orig  and compare against
      m_after ~ a*m_before  alone. Also correlate the perturbation
      (dc1 - dc2) with the embedding-induced margin shift (m_before - m_orig).

Outputs: restoration_regression.csv + printed summary.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "exph", os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_exp_H.py"))
exph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exph)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCK_CSV = os.path.join(OUT_DIR, "block_margins.csv")

N_BITS = 1024


def main():
    blocks = pd.read_csv(BLOCK_CSV)
    blocks = blocks[blocks["pair"].isin(["standard", "balanced", "hf"])].copy()

    # natural margins from the ORIGINAL (unwatermarked) images
    print("Measuring natural margins on original images...", flush=True)
    nat = {}
    for image_name in sorted(blocks["image"].unique()):
        orig = exph.load_original(image_name)
        for pk, pos in exph.PAIRS.items():
            c1o, c2o = exph.block_coeffs(orig, pos["pos1"], pos["pos2"], N_BITS)
            nat[(image_name, pk)] = c1o - c2o

    blocks["m_orig"] = [
        nat[(im, pk)][bi] for im, pk, bi in zip(blocks["image"], blocks["pair"], blocks["block_idx"])
    ]
    blocks["embed_shift"] = blocks["m_before"] - blocks["m_orig"]
    blocks["diff_pert"] = blocks["dc1"] - blocks["dc2"]

    rows = []
    for pk in ["standard", "balanced", "hf"]:
        b = blocks[blocks["pair"] == pk]
        # (1) does the attack undo the embedding? corr(diff_pert, embed_shift)
        r_undo, p_undo = stats.pearsonr(b["diff_pert"], b["embed_shift"])
        # (2) m_after regression: before-only vs before+orig
        X1 = np.column_stack([b["m_before"], np.ones(len(b))])
        beta1, res1, *_ = np.linalg.lstsq(X1, b["m_after"], rcond=None)
        r2_1 = 1 - res1[0] / ((b["m_after"] - b["m_after"].mean()) ** 2).sum()
        X2 = np.column_stack([b["m_before"], b["m_orig"], np.ones(len(b))])
        beta2, res2, *_ = np.linalg.lstsq(X2, b["m_after"], rcond=None)
        r2_2 = 1 - res2[0] / ((b["m_after"] - b["m_after"].mean()) ** 2).sum()
        # (3) fraction of embedded shift removed by the attack:
        # regress diff_pert on embed_shift -> slope = -k means k of the shift is undone
        slope_k = stats.linregress(b["embed_shift"], b["diff_pert"])
        rows.append({
            "pair": pk,
            "corr_diffpert_embedshift": r_undo,
            "p": p_undo,
            "undo_fraction_k": -slope_k.slope,
            "m_after~m_before_slope": beta1[0],
            "R2_before_only": r2_1,
            "m_after~before_coef": beta2[0],
            "m_after~orig_coef": beta2[1],
            "R2_before_plus_orig": r2_2,
            "mean_abs_embed_shift": b["embed_shift"].abs().mean(),
            "mean_abs_m_orig": b["m_orig"].abs().mean(),
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "restoration_regression.csv"), index=False)
    print("\nEndogeneity / restoration test (per pair):")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
