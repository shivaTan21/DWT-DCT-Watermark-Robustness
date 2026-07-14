"""
Experiment H analysis: which quantity predicts NC after Real-ESRGAN?

Consumes block_margins.csv + image_summary.csv produced by run_exp_H.py and
produces:
  - correlations_nc.csv          image-level Pearson/Spearman/MI of NC vs candidates
  - block_flip_logistic.csv      block-level: what determines whether a block flips
  - success_vs_failure.csv       image-level comparison of successful vs failed images
  - pair_mechanism_summary.csv   per-pair decomposition of margin vs perturbation
  - control_pipeline_check.csv   standard disk vs standard live-ESRGAN NC comparison
  - exp_H_fig1_margin_scatter.png     M_before vs M_after per pair
  - exp_H_fig2_margin_hist.png        |M_before| distributions per pair
  - exp_H_fig3_flip_hist.png          per-image sign-flip-rate distributions
  - exp_H_fig4_nc_vs_flips.png        NC vs sign-flip rate
  - exp_H_fig5_nc_vs_margin_loss.png  NC vs margin-loss measures
  - exp_H_fig6_flip_heatmap.png       spatial map of where flips occur in the LL grid
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCK_CSV = os.path.join(OUT_DIR, "block_margins.csv")
IMAGE_CSV = os.path.join(OUT_DIR, "image_summary.csv")

PAIR_ORDER = ["standard", "balanced", "hf"]
PAIR_LABEL = {
    "standard": "Standard (4,1)/(1,4)",
    "balanced": "Balanced (2,3)/(3,2)",
    "hf": "HF (7,5)/(7,7)",
    "standard_live": "Standard, live-ESRGAN control",
}
PAIR_COLOR = {"standard": "#95a5a6", "balanced": "#2ecc71", "hf": "#3498db"}

WM_GRID = (32, 32)

try:
    from sklearn.feature_selection import mutual_info_regression
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


def mi(x, y):
    if not HAVE_SKLEARN:
        return np.nan
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    return float(mutual_info_regression(x, np.asarray(y, dtype=float), random_state=42)[0])


def corr_row(name, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {
        "predictor": name, "n": len(x),
        "pearson_r": pr, "pearson_p": pp,
        "spearman_rho": sr, "spearman_p": sp,
        "mutual_info": mi(x, y),
    }


def main():
    blocks = pd.read_csv(BLOCK_CSV)
    imgs = pd.read_csv(IMAGE_CSV)

    main_imgs = imgs[imgs["pair"].isin(PAIR_ORDER)].copy()
    main_blocks = blocks[blocks["pair"].isin(PAIR_ORDER)].copy()

    # ── 0. Pipeline-confound control ─────────────────────────────────────────
    # (a) same pair (standard/hf), disk vs live ESRGAN pipeline;
    # (b) three-way like-for-like comparison under the SAME live pipeline.
    ctrl_rows = []
    for live_key, disk_key in (("standard_live", "standard"), ("hf_live", "hf")):
        live = imgs[imgs["pair"] == live_key][["image", "nc"]]
        if not len(live):
            continue
        disk = imgs[imgs["pair"] == disk_key][["image", "nc"]]
        merged = live.merge(disk, on="image", suffixes=("_live", "_disk"))
        w = stats.wilcoxon(merged["nc_live"], merged["nc_disk"]) if len(merged) >= 6 else None
        ctrl_rows.append({
            "comparison": f"{disk_key}: live vs disk pipeline", "n": len(merged),
            "nc_a": merged["nc_live"].mean(), "nc_b": merged["nc_disk"].mean(),
            "delta": merged["nc_live"].mean() - merged["nc_disk"].mean(),
            "wilcoxon_p": w.pvalue if w else np.nan,
        })
    # like-for-like: all three pairs under the live pipeline (control subset)
    live_std = imgs[imgs["pair"] == "standard_live"][["image", "nc"]]
    if len(live_std):
        ctrl_imgs = set(live_std["image"])
        bal = imgs[(imgs["pair"] == "balanced") & imgs["image"].isin(ctrl_imgs)][["image", "nc"]]
        hfl = imgs[(imgs["pair"] == "hf_live") & imgs["image"].isin(ctrl_imgs)][["image", "nc"]]
        for other_name, other in (("balanced", bal), ("hf", hfl)):
            merged = other.merge(live_std, on="image", suffixes=("_other", "_std"))
            if len(merged) >= 6:
                w = stats.wilcoxon(merged["nc_other"], merged["nc_std"])
                ctrl_rows.append({
                    "comparison": f"LIKE-FOR-LIKE live pipeline: {other_name} vs standard",
                    "n": len(merged),
                    "nc_a": merged["nc_other"].mean(), "nc_b": merged["nc_std"].mean(),
                    "delta": merged["nc_other"].mean() - merged["nc_std"].mean(),
                    "wilcoxon_p": w.pvalue,
                })
    if ctrl_rows:
        ctrl_out = pd.DataFrame(ctrl_rows)
        ctrl_out.to_csv(os.path.join(OUT_DIR, "control_pipeline_check.csv"), index=False)
        print("Pipeline-confound controls:")
        print(ctrl_out.round(4).to_string(index=False))

    # ── 1. Image-level correlations: NC vs candidate predictors ─────────────
    # Pooled across pairs AND within pair (the pooled version is what matters
    # for "which quantity predicts NC"; within-pair checks it isn't just a
    # between-pair artifact).
    candidates = [
        ("mean_abs_damage",        "absolute coefficient damage (|dC1|+|dC2|)/2"),
        ("mean_abs_dc1",           "absolute damage to C1"),
        ("mean_abs_dc2",           "absolute damage to C2"),
        ("mean_margin_loss_signed_to_bit", "bit-directed margin loss"),
        ("flip_rate",              "sign-flip rate"),
        ("mean_abs_diff_pert",     "differential perturbation |dC1-dC2|"),
        ("std_diff_pert",          "differential perturbation std"),
        ("mean_abs_common_pert",   "common-mode perturbation |(dC1+dC2)/2|"),
        ("std_common_pert",        "common-mode perturbation std"),
        ("mean_abs_m_before",      "mean |margin| before"),
        ("min_abs_m_before",       "min |margin| before"),
        ("median_abs_m_before",    "median |margin| before"),
        ("mean_abs_m_after",       "mean |margin| after"),
        ("margin_var_after",       "margin variance after"),
    ]
    rows = []
    for col, desc in candidates:
        r = corr_row(col, main_imgs[col], main_imgs["nc"])
        r["description"] = desc
        r["scope"] = "pooled"
        rows.append(r)
        for pk in PAIR_ORDER:
            sub = main_imgs[main_imgs["pair"] == pk]
            r = corr_row(col, sub[col], sub["nc"])
            r["description"] = desc
            r["scope"] = pk
            rows.append(r)
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(os.path.join(OUT_DIR, "correlations_nc.csv"), index=False)

    pooled = corr_df[corr_df["scope"] == "pooled"].sort_values("spearman_rho", key=np.abs, ascending=False)
    print("\nImage-level predictors of NC (pooled across pairs), by |Spearman|:")
    print(pooled[["predictor", "pearson_r", "spearman_rho", "spearman_p", "mutual_info"]].round(4).to_string(index=False))

    # NC is definitionally tied to flip_rate at the image level (a flipped
    # correct bit IS a bit error). The scientifically meaningful check is
    # which PRE/DURING-attack quantity predicts flips.
    # ── 2. Block-level: what determines whether a block flips? ──────────────
    b = main_blocks
    b["snr_margin"] = b["abs_m_before"] / (np.abs(b["diff_pert"]) + 1e-9)
    flip_rows = []
    for pk in PAIR_ORDER + ["pooled"]:
        sub = b if pk == "pooled" else b[b["pair"] == pk]
        flipped = sub[sub["sign_flip"] == 1]
        kept = sub[sub["sign_flip"] == 0]
        for col in ["abs_m_before", "abs_dc1", "abs_dc2", "diff_pert", "common_pert"]:
            x_f = np.abs(flipped[col]) if col in ("diff_pert", "common_pert") else flipped[col]
            x_k = np.abs(kept[col]) if col in ("diff_pert", "common_pert") else kept[col]
            u = stats.mannwhitneyu(x_f, x_k, alternative="two-sided")
            # rank-biserial effect size
            rb = 1 - 2 * u.statistic / (len(x_f) * len(x_k))
            flip_rows.append({
                "pair": pk, "quantity": col,
                "flipped_median": np.median(x_f), "kept_median": np.median(x_k),
                "mannwhitney_p": u.pvalue, "rank_biserial": rb,
            })
        # point-biserial correlations with flip indicator
        for col in ["abs_m_before", "abs_dc1", "abs_dc2"]:
            pr, pp = stats.pearsonr(sub[col], sub["sign_flip"])
            flip_rows.append({
                "pair": pk, "quantity": f"pointbiserial_{col}",
                "flipped_median": np.nan, "kept_median": np.nan,
                "mannwhitney_p": pp, "rank_biserial": pr,
            })
    flip_df = pd.DataFrame(flip_rows)
    flip_df.to_csv(os.path.join(OUT_DIR, "block_flip_logistic.csv"), index=False)
    print("\nBlock-level flip determinants (median in flipped vs kept blocks):")
    print(flip_df[~flip_df["quantity"].str.startswith("pointbiserial")]
          .round(4).to_string(index=False))

    # ── 3. Success vs failure images ─────────────────────────────────────────
    # Per pair: split at the pair's median NC; also absolute threshold NC<0.75.
    sf_rows = []
    for pk in PAIR_ORDER:
        sub = main_imgs[main_imgs["pair"] == pk]
        med = sub["nc"].median()
        good = sub[sub["nc"] >= med]
        bad = sub[sub["nc"] < med]
        for gname, g in (("success(≥median)", good), ("failure(<median)", bad)):
            sf_rows.append({
                "pair": pk, "group": gname, "n": len(g),
                "nc": g["nc"].mean(),
                "pct_flipped_bits": 100 * g["flip_rate"].mean(),
                "pct_bit_errors": 100 * g["bit_error_rate_blocks"].mean(),
                "avg_margin_before": g["mean_abs_m_before"].mean(),
                "min_margin_before": g["min_abs_m_before"].mean(),
                "median_margin_before": g["median_abs_m_before"].mean(),
                "abs_damage": g["mean_abs_damage"].mean(),
                "diff_pert": g["mean_abs_diff_pert"].mean(),
                "common_pert": g["mean_abs_common_pert"].mean(),
            })
    sf_df = pd.DataFrame(sf_rows)
    sf_df.to_csv(os.path.join(OUT_DIR, "success_vs_failure.csv"), index=False)
    print("\nSuccess vs failure images (per pair, split at median NC):")
    print(sf_df.round(3).to_string(index=False))

    # ── 4. Per-pair mechanism decomposition ──────────────────────────────────
    mech_rows = []
    for pk in PAIR_ORDER:
        sub_b = b[b["pair"] == pk]
        sub_i = main_imgs[main_imgs["pair"] == pk]
        flips = sub_b["sign_flip"].mean()
        mech_rows.append({
            "pair": pk,
            "nc_mean": sub_i["nc"].mean(),
            "flip_rate": flips,
            "mean_abs_m_before": sub_b["abs_m_before"].mean(),
            "median_abs_m_before": sub_b["abs_m_before"].median(),
            "mean_abs_dc1": sub_b["abs_dc1"].mean(),
            "mean_abs_dc2": sub_b["abs_dc2"].mean(),
            "mean_abs_damage": (sub_b["abs_dc1"] + sub_b["abs_dc2"]).mean() / 2,
            "mean_abs_diff_pert": np.abs(sub_b["diff_pert"]).mean(),
            "std_diff_pert": sub_b["diff_pert"].std(),
            "mean_abs_common_pert": np.abs(sub_b["common_pert"]).mean(),
            "std_common_pert": sub_b["common_pert"].std(),
            # THE key ratio: differential noise relative to available margin
            "median_snr": (sub_b["abs_m_before"] / (np.abs(sub_b["diff_pert"]) + 1e-9)).median(),
            "pct_diffpert_exceeds_margin": 100 * (np.abs(sub_b["diff_pert"]) > sub_b["abs_m_before"]).mean(),
            "corr_dc1_dc2": stats.pearsonr(sub_b["dc1"], sub_b["dc2"])[0],
        })
    mech_df = pd.DataFrame(mech_rows)
    mech_df.to_csv(os.path.join(OUT_DIR, "pair_mechanism_summary.csv"), index=False)
    print("\nPer-pair mechanism decomposition:")
    print(mech_df.round(4).to_string(index=False))

    # ── Figures ──────────────────────────────────────────────────────────────
    # Fig 1: margin before vs after scatter
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=False)
    for ax, pk in zip(axes, PAIR_ORDER):
        sub = b[b["pair"] == pk].sample(n=min(20000, (b["pair"] == pk).sum()), random_state=42)
        colors = np.where(sub["sign_flip"] == 1, "#e74c3c", "#7f8c8d")
        ax.scatter(sub["m_before"], sub["m_after"], s=2, c=colors, alpha=0.25, rasterized=True)
        lim = np.percentile(np.abs(sub["m_before"]), 99.5) * 1.4
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        flip_pct = 100 * b[b["pair"] == pk]["sign_flip"].mean()
        ax.set_title(f"{PAIR_LABEL[pk]}\nflips = {flip_pct:.1f}% (red = flipped)")
        ax.set_xlabel("M before ESRGAN")
    axes[0].set_ylabel("M after ESRGAN")
    fig.suptitle("Decision margin before vs after Real-ESRGAN (quadrants II/IV = sign flip)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp_H_fig1_margin_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: histogram of |margins| before and after
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for pk in PAIR_ORDER:
        sub = b[b["pair"] == pk]
        axes[0].hist(sub["abs_m_before"], bins=100, range=(0, 150), density=True,
                     histtype="step", lw=1.6, label=PAIR_LABEL[pk], color=PAIR_COLOR[pk])
        axes[1].hist(sub["abs_m_after"], bins=100, range=(0, 150), density=True,
                     histtype="step", lw=1.6, label=PAIR_LABEL[pk], color=PAIR_COLOR[pk])
    axes[0].set_title("|M| before ESRGAN"); axes[1].set_title("|M| after ESRGAN")
    for ax in axes:
        ax.set_xlabel("|C1 − C2|"); ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Decision-margin magnitude distributions")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp_H_fig2_margin_hist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: histogram of per-image sign-flip rates
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for pk in PAIR_ORDER:
        sub = main_imgs[main_imgs["pair"] == pk]
        ax.hist(100 * sub["flip_rate"], bins=30, range=(0, 40), alpha=0.55,
                label=f"{PAIR_LABEL[pk]} (mean {100*sub['flip_rate'].mean():.1f}%)",
                color=PAIR_COLOR[pk])
    ax.set_xlabel("% of blocks with margin sign flip after ESRGAN")
    ax.set_ylabel("images")
    ax.legend(fontsize=9)
    ax.set_title("Sign-flip rate distribution per pair")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp_H_fig3_flip_hist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 4: NC vs sign-flip rate
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for pk in PAIR_ORDER:
        sub = main_imgs[main_imgs["pair"] == pk]
        ax.scatter(100 * sub["flip_rate"], sub["nc"], s=22, alpha=0.7,
                   label=PAIR_LABEL[pk], color=PAIR_COLOR[pk])
    sr, sp = stats.spearmanr(main_imgs["flip_rate"], main_imgs["nc"])
    ax.set_xlabel("% sign flips"); ax.set_ylabel("NC")
    ax.set_title(f"NC vs sign-flip rate (pooled Spearman ρ = {sr:.3f}, p = {sp:.1e})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp_H_fig4_nc_vs_flips.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 5: NC vs margin loss and vs absolute damage (the falsified predictor)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for pk in PAIR_ORDER:
        sub = main_imgs[main_imgs["pair"] == pk]
        axes[0].scatter(sub["mean_margin_loss_signed_to_bit"], sub["nc"], s=20, alpha=0.7,
                        label=PAIR_LABEL[pk], color=PAIR_COLOR[pk])
        axes[1].scatter(sub["mean_abs_damage"], sub["nc"], s=20, alpha=0.7,
                        label=PAIR_LABEL[pk], color=PAIR_COLOR[pk])
    r0 = stats.spearmanr(main_imgs["mean_margin_loss_signed_to_bit"], main_imgs["nc"])
    r1 = stats.spearmanr(main_imgs["mean_abs_damage"], main_imgs["nc"])
    axes[0].set_xlabel("bit-directed margin loss (mean over blocks)")
    axes[0].set_title(f"NC vs margin loss (ρ = {r0.statistic:.3f})")
    axes[1].set_xlabel("absolute coefficient damage (|ΔC1|+|ΔC2|)/2")
    axes[1].set_title(f"NC vs absolute damage — Exp D's predictor (ρ = {r1.statistic:.3f})")
    for ax in axes:
        ax.set_ylabel("NC"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp_H_fig5_nc_vs_margin_loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 6: spatial heatmap of sign flips in the 32x32 LL block grid
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, pk in zip(axes, PAIR_ORDER):
        sub = b[b["pair"] == pk]
        grid = sub.groupby("block_idx")["sign_flip"].mean().reindex(range(WM_GRID[0]*WM_GRID[1])).values.reshape(WM_GRID)
        im = ax.imshow(100 * grid, cmap="inferno", vmin=0)
        ax.set_title(f"{PAIR_LABEL[pk]}")
        plt.colorbar(im, ax=ax, fraction=0.046, label="% images flipped")
    fig.suptitle("Where sign flips occur in the LL block grid (averaged over images)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp_H_fig6_flip_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nFigures + CSVs written to", OUT_DIR)


if __name__ == "__main__":
    main()
