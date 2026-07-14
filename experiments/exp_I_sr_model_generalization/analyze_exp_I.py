"""
Experiment I mechanism analysis: does the Exp H restoration-pressure mechanism
generalize across SR architectures?

Repeats ONLY the Exp H analysis (its code is REUSED via importlib — nothing in
exp_H_margin_analysis/ is modified or rerun) on the standard pair (4,1)/(1,4)
at alpha = 0.10 for:

  fsrcnn / edsr / lapsrn : NEW attacked images from this experiment's cache
  espcn                  : frozen disk images results/attacked/ai_enhancement/
                           <img>__alpha0.1__espcn_x4.png (read-only)
  real_esrgan            : frozen Exp H block_margins.csv rows (pair="standard")
                           — NOT recomputed, NOT rerun

For every model the Exp H quantities are computed per block:
  margins before/after, sign flips, dC1/dC2, differential perturbation
  (dC1-dC2), common-mode perturbation, natural margin m_orig, embedding shift
and per model:
  flip rate, std(dC1-dC2), undo correlation + fraction k, m_after regression
  R^2 (before vs before+orig), flip-vs-natural-margin opposition table.

Outputs (this folder):
  block_margins_exp_I.csv         per-block rows for the 4 newly measured models
  mechanism_summary_exp_I.csv     one row per model incl. real_esrgan reference
  restoration_regression_exp_I.csv
  flip_vs_natural_margin_exp_I.csv
  plots/exp_I_fig*.png
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config
from src.metrics import normalized_correlation, bit_error_rate

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
ATTACKED_DIR = os.path.join(OUT_DIR, "attacked")
EXP_H_DIR = os.path.join(os.path.dirname(OUT_DIR), "exp_H_margin_analysis")

# Reuse Exp H's code (block_coeffs, load_original, PAIRS) without modifying it.
_spec = importlib.util.spec_from_file_location("exph", os.path.join(EXP_H_DIR, "run_exp_H.py"))
exph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exph)

ALPHA = 0.1
N_BITS = 1024
WM_SHAPE = (32, 32)
POS1, POS2 = exph.PAIRS["standard"]["pos1"], exph.PAIRS["standard"]["pos2"]

MODEL_LABEL = {
    "espcn": "ESPCN x4 (frozen disk)",
    "fsrcnn": "FSRCNN x4",
    "edsr": "EDSR x4",
    "lapsrn": "LapSRN x4",
    "real_esrgan": "Real-ESRGAN x4 (frozen Exp H)",
}
MODEL_COLOR = {
    "espcn": "#95a5a6", "fsrcnn": "#2ecc71", "edsr": "#9b59b6",
    "lapsrn": "#f39c12", "real_esrgan": "#e74c3c",
}


def after_path(model, image_name):
    if model == "espcn":
        return os.path.join(config.ATTACKED_AI_DIR, f"{image_name}__alpha{ALPHA}__espcn_x4.png")
    return os.path.join(ATTACKED_DIR, f"{image_name}__alpha{ALPHA}__{model}_x4.png")


def measure_model(model, images, wm_bits, natural_margins):
    """Exp H block-level measurement (same derived columns as run_exp_H.process)."""
    frames = []
    for image_name in images:
        before = exph.load_bgr_to_rgb(os.path.join(config.WATERMARKED_DIR, f"{image_name}__alpha{ALPHA}.png"))
        after = exph.load_bgr_to_rgb(after_path(model, image_name))
        if before is None or after is None:
            print(f"  [SKIP] {model} {image_name}: missing before/after image", flush=True)
            continue
        c1b, c2b = exph.block_coeffs(before, POS1, POS2, N_BITS)
        c1a, c2a = exph.block_coeffs(after, POS1, POS2, N_BITS)
        m_before, m_after = c1b - c2b, c1a - c2a
        dc1, dc2 = c1a - c1b, c2a - c2b
        frames.append(pd.DataFrame({
            "image": image_name,
            "model": model,
            "block_idx": np.arange(N_BITS),
            "bit": wm_bits,
            "m_before": m_before, "m_after": m_after,
            "abs_m_before": np.abs(m_before),
            "sign_flip": (np.sign(m_before) != np.sign(m_after)).astype(int),
            "dc1": dc1, "dc2": dc2,
            "diff_pert": dc1 - dc2,
            "common_pert": (dc1 + dc2) / 2.0,
            "bit_error": ((m_after > 0).astype(np.uint8) != wm_bits).astype(int),
            "m_orig": natural_margins[image_name],
        }))
    return pd.concat(frames, ignore_index=True)


def load_esrgan_reference(images, natural_margins):
    """Frozen Exp H standard-pair rows, restricted to the same derived columns."""
    b = pd.read_csv(os.path.join(EXP_H_DIR, "block_margins.csv"))
    b = b[b["pair"] == "standard"].copy()
    b["model"] = "real_esrgan"
    b["m_orig"] = [natural_margins[im][bi] for im, bi in zip(b["image"], b["block_idx"])]
    cols = ["image", "model", "block_idx", "bit", "m_before", "m_after", "abs_m_before",
            "sign_flip", "dc1", "dc2", "diff_pert", "common_pert", "bit_error", "m_orig"]
    return b[cols]


def mechanism_rows(blocks, model):
    b = blocks[blocks["model"] == model].copy()
    b["embed_shift"] = b["m_before"] - b["m_orig"]

    # image-level NC (extraction identity: bit = [m_after > 0])
    ncs, bers, flips = [], [], []
    for _, g in b.groupby("image"):
        g = g.sort_values("block_idx")
        wm_grid = g["bit"].to_numpy().reshape(WM_SHAPE)
        ext = (g["m_after"].to_numpy() > 0).astype(np.uint8).reshape(WM_SHAPE)
        ncs.append(normalized_correlation(wm_grid, ext))
        bers.append(bit_error_rate(wm_grid, ext))
        flips.append(g["sign_flip"].mean())

    # restoration regression (same construction as exp_H analyze_restoration.py)
    r_undo, p_undo = stats.pearsonr(b["diff_pert"], b["embed_shift"])
    slope_k = stats.linregress(b["embed_shift"], b["diff_pert"])
    X1 = np.column_stack([b["m_before"], np.ones(len(b))])
    beta1, res1, *_ = np.linalg.lstsq(X1, b["m_after"], rcond=None)
    r2_1 = 1 - res1[0] / ((b["m_after"] - b["m_after"].mean()) ** 2).sum()
    X2 = np.column_stack([b["m_before"], b["m_orig"], np.ones(len(b))])
    beta2, res2, *_ = np.linalg.lstsq(X2, b["m_after"], rcond=None)
    r2_2 = 1 - res2[0] / ((b["m_after"] - b["m_after"].mean()) ** 2).sum()

    # flip vs natural-margin opposition (exp_H flip_vs_natural_margin analysis)
    bit_dir = np.where(b["bit"] == 1, 1.0, -1.0)
    opposing = np.sign(b["m_orig"]) != np.sign(bit_dir)
    flip_opp = b.loc[opposing, "sign_flip"].mean()
    flip_agr = b.loc[~opposing, "sign_flip"].mean()
    n_flips = b["sign_flip"].sum()
    pct_flips_in_opposing = 100.0 * b.loc[opposing, "sign_flip"].sum() / max(n_flips, 1)
    strong_opp = opposing & (b["m_orig"].abs() > 30)
    flip_strong_opp = b.loc[strong_opp, "sign_flip"].mean() if strong_opp.any() else np.nan

    mech = {
        "model": model,
        "n_images": b["image"].nunique(),
        "nc_mean": float(np.mean(ncs)),
        "ber_mean": float(np.mean(bers)),
        "flip_rate": b["sign_flip"].mean(),
        "mean_abs_m_before": b["abs_m_before"].mean(),
        "mean_abs_dc1": b["dc1"].abs().mean(),
        "mean_abs_dc2": b["dc2"].abs().mean(),
        "mean_abs_damage": (b["dc1"].abs() + b["dc2"].abs()).mean() / 2,
        "mean_abs_diff_pert": b["diff_pert"].abs().mean(),
        "std_diff_pert": b["diff_pert"].std(),
        "mean_abs_common_pert": b["common_pert"].abs().mean(),
        "std_common_pert": b["common_pert"].std(),
        "corr_dc1_dc2": stats.pearsonr(b["dc1"], b["dc2"])[0],
        "pct_diffpert_exceeds_margin": 100 * (b["diff_pert"].abs() > b["abs_m_before"]).mean(),
    }
    resto = {
        "model": model,
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
    }
    flipnat = {
        "model": model,
        "flip_rate_opposing": flip_opp,
        "flip_rate_agreeing": flip_agr,
        "risk_ratio": flip_opp / flip_agr if flip_agr > 0 else np.inf,
        "pct_of_flips_in_opposing_blocks": pct_flips_in_opposing,
        "pct_blocks_opposing": 100 * opposing.mean(),
        "flip_rate_strong_opposing(|m_orig|>30)": flip_strong_opp,
        "n_flips_total": int(n_flips),
    }
    return mech, resto, flipnat


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="espcn,fsrcnn,edsr,lapsrn",
                    help="models to measure (only those with COMPLETE attacked caches)")
    args = ap.parse_args()
    requested = [m.strip() for m in args.models.split(",") if m.strip()]

    os.makedirs(PLOTS_DIR, exist_ok=True)
    from src import watermark as wm_module
    wm_grid = wm_module.generate_watermark(shape=WM_SHAPE, seed=42)
    wm_bits = wm_grid.flatten()
    images = exph.list_images()

    print("Measuring natural margins (standard pair) on original images...", flush=True)
    natural_margins = {}
    for image_name in images:
        orig = exph.load_original(image_name)
        c1o, c2o = exph.block_coeffs(orig, POS1, POS2, N_BITS)
        natural_margins[image_name] = c1o - c2o

    new_models = [m for m in requested
                  if all(os.path.exists(after_path(m, im)) for im in images)]
    print(f"Models with attacked images available: {new_models}", flush=True)

    frames = []
    for model in new_models:
        print(f"Measuring blocks: {model}", flush=True)
        frames.append(measure_model(model, images, wm_bits, natural_margins))
    blocks_new = pd.concat(frames, ignore_index=True)
    blocks_new.to_csv(os.path.join(OUT_DIR, "block_margins_exp_I.csv"), index=False)

    print("Loading frozen Exp H Real-ESRGAN reference rows...", flush=True)
    blocks = pd.concat([blocks_new, load_esrgan_reference(images, natural_margins)],
                       ignore_index=True)

    models = new_models + ["real_esrgan"]
    mech_rows_, resto_rows, flipnat_rows = [], [], []
    for model in models:
        m, r, f = mechanism_rows(blocks, model)
        mech_rows_.append(m); resto_rows.append(r); flipnat_rows.append(f)

    mech = pd.DataFrame(mech_rows_)
    resto = pd.DataFrame(resto_rows)
    flipnat = pd.DataFrame(flipnat_rows)
    mech.to_csv(os.path.join(OUT_DIR, "mechanism_summary_exp_I.csv"), index=False)
    resto.to_csv(os.path.join(OUT_DIR, "restoration_regression_exp_I.csv"), index=False)
    flipnat.to_csv(os.path.join(OUT_DIR, "flip_vs_natural_margin_exp_I.csv"), index=False)

    print("\nMechanism summary:");             print(mech.round(4).to_string(index=False))
    print("\nRestoration regression:");        print(resto.round(4).to_string(index=False))
    print("\nFlip vs natural margin:");        print(flipnat.round(4).to_string(index=False))

    # ── Figures ──────────────────────────────────────────────────────────────
    # Fig 1: margin before vs after per model (Exp H fig1 style)
    fig, axes = plt.subplots(1, len(models), figsize=(5.3 * len(models), 5.2))
    for ax, model in zip(np.atleast_1d(axes), models):
        sub = blocks[blocks["model"] == model]
        sub = sub.sample(n=min(20000, len(sub)), random_state=42)
        colors = np.where(sub["sign_flip"] == 1, "#e74c3c", "#7f8c8d")
        ax.scatter(sub["m_before"], sub["m_after"], s=2, c=colors, alpha=0.25, rasterized=True)
        lim = np.percentile(np.abs(sub["m_before"]), 99.5) * 1.4
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.6)
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        flip_pct = 100 * blocks[blocks["model"] == model]["sign_flip"].mean()
        ax.set_title(f"{MODEL_LABEL[model]}\nflips = {flip_pct:.2f}%")
        ax.set_xlabel("M before attack")
    np.atleast_1d(axes)[0].set_ylabel("M after attack")
    fig.suptitle("Decision margin before vs after SR attack — standard pair (4,1)/(1,4), α=0.10", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "exp_I_fig1_margin_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: mechanism bar panel (flip rate, σ(diff), undo k, undo corr)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    x = np.arange(len(models))
    panels = [
        (100 * mech["flip_rate"], "sign-flip rate (%)"),
        (mech["std_diff_pert"], "σ(ΔC1−ΔC2)"),
        (100 * resto["undo_fraction_k"], "embedded differential removed k (%)"),
        (-resto["corr_diffpert_embedshift"], "−corr(ΔC1−ΔC2, embed shift)"),
    ]
    for ax, (vals, title) in zip(axes, panels):
        ax.bar(x, vals, color=[MODEL_COLOR[m] for m in models])
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20)
        ax.set_title(title)
    fig.suptitle("Exp H mechanism quantities across SR models (standard pair, α=0.10)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "exp_I_fig2_mechanism_bars.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: flip rate opposing vs agreeing natural margin
    fig, ax = plt.subplots(figsize=(8.5, 5))
    w = 0.38
    ax.bar(x - w / 2, 100 * flipnat["flip_rate_opposing"], w, label="natural margin OPPOSES bit", color="#e74c3c")
    ax.bar(x + w / 2, 100 * flipnat["flip_rate_agreeing"], w, label="natural margin agrees", color="#2ecc71")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("block flip rate (%)")
    ax.set_title("Restoration pressure: flips vs natural-margin opposition (α=0.10)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "exp_I_fig3_flip_vs_natural_margin.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nCSVs + figures written to", OUT_DIR, "and", PLOTS_DIR, flush=True)


if __name__ == "__main__":
    main()
