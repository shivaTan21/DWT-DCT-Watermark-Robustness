"""
Build Exp I summary tables and robustness plots.

Reads (read-only):
  results_exp_I.csv (+ results_exp_I_edsr.csv if present)  — new SR models
  ../../results/metrics.csv                                 — frozen espcn_x4 /
      real_esrgan rows (standard variant) as comparison baselines

Writes (this folder):
  summary_table.csv        Model | Mean NC | Mean BER | Mean SSIM | Mean PSNR | Runtime
  summary_by_alpha.csv     the same metrics per alpha
  plots/exp_I_fig4_nc_vs_alpha.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

MODEL_ORDER = ["fsrcnn_x4", "edsr_x4", "lapsrn_x4", "espcn_x4 (frozen)", "real_esrgan (frozen)"]
COLOR = {"fsrcnn_x4": "#2ecc71", "edsr_x4": "#9b59b6", "lapsrn_x4": "#f39c12",
         "espcn_x4 (frozen)": "#95a5a6", "real_esrgan (frozen)": "#e74c3c"}


def load_new_results():
    frames = []
    for f in ("results_exp_I.csv", "results_exp_I_edsr.csv"):
        p = os.path.join(OUT_DIR, f)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    # results_exp_I_edsr.csv rows were merged into results_exp_I.csv once the
    # EDSR run finished; drop the duplicates so n is correct.
    df = df.drop_duplicates(subset=["image", "alpha", "model"], keep="first")
    df["attack_name"] = df["model"] + "_x4"
    return df


def load_frozen_baselines():
    m = pd.read_csv(config.METRICS_CSV_PATH)
    m = m[(m["embedding_variant"] == "standard")
          & (m["attack_name"].isin(["espcn_x4", "real_esrgan"]))].copy()
    m["attack_name"] = m["attack_name"] + " (frozen)"
    m["runtime_s"] = np.nan  # frozen runs did not record per-image runtime
    return m


def summarize(df, by_alpha):
    keys = ["attack_name"] + (["alpha"] if by_alpha else [])
    g = df.groupby(keys)
    out = pd.DataFrame({
        "n": g.size(),
        "nc_mean": g["nc"].mean(), "nc_std": g["nc"].std(),
        "ber_mean": g["ber"].mean(),
        "ssim_mean": g["ssim_after_attack"].mean(),
        "psnr_mean": g["psnr_after_attack"].mean(),
        "runtime_mean_s": g["runtime_s"].mean(),
    }).reset_index()
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    out["__o"] = out["attack_name"].map(order)
    return out.sort_values(["__o"] + (["alpha"] if by_alpha else [])).drop(columns="__o")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    new = load_new_results()
    frozen = load_frozen_baselines()
    cols = ["attack_name", "alpha", "nc", "ber", "psnr_after_attack", "ssim_after_attack", "runtime_s"]
    allr = pd.concat([new[cols], frozen[cols]], ignore_index=True)

    overall = summarize(allr, by_alpha=False)
    per_alpha = summarize(allr, by_alpha=True)
    overall.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)
    per_alpha.to_csv(os.path.join(OUT_DIR, "summary_by_alpha.csv"), index=False)

    print("Overall summary (pooled over alphas):")
    print(overall.round(4).to_string(index=False))
    print("\nPer-alpha summary:")
    print(per_alpha.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name, g in allr.groupby("attack_name"):
        m = g.groupby("alpha")["nc"].mean()
        s = g.groupby("alpha")["nc"].std()
        ax.errorbar(m.index, m.values, yerr=s.values, marker="o", capsize=3,
                    label=name, color=COLOR.get(name))
    ax.set_xlabel("embedding strength α")
    ax.set_ylabel("mean NC after attack")
    ax.set_title("Watermark robustness vs α — new SR models and frozen baselines\n"
                 "(standard pair (4,1)/(1,4), 100 images)")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "exp_I_fig4_nc_vs_alpha.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nWrote summary_table.csv, summary_by_alpha.csv, plots/exp_I_fig4_nc_vs_alpha.png")


if __name__ == "__main__":
    main()
