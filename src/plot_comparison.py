"""
Generate comparison plots from results/comparison_metrics.csv.

Run after run_comparison.py:
    python -m src.plot_comparison
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

COMPARISON_CSV = os.path.join(config.RESULTS_DIR, "comparison_metrics.csv")

SCHEME_LABELS = {
    "dwt_dct_ll": "DWT-DCT-LL",
    "dwt_svd_hh": "DWT-SVD-HH",
}
SCHEME_COLORS = {
    "dwt_dct_ll": "steelblue",
    "dwt_svd_hh": "tomato",
}
SCHEMES = ["dwt_dct_ll", "dwt_svd_hh"]

ATTACK_ORDER = ["gaussian_blur_5x5", "jpeg_compression_q50", "real_esrgan"]
ATTACK_LABELS = {
    "gaussian_blur_5x5": "Gaussian\nBlur 5×5",
    "jpeg_compression_q50": "JPEG\nQ50",
    "real_esrgan": "Real-\nESRGAN",
}


def plot_nc_by_attack(df):
    """Grouped bar chart: mean NC for each scheme × attack."""
    attack_rows = df[df["attack_name"] != "watermarked_only"]
    present = set(attack_rows["attack_name"].unique())
    attacks = [a for a in ATTACK_ORDER if a in present]

    x = np.arange(len(attacks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, scheme in zip([-width / 2, width / 2], SCHEMES):
        means = [
            attack_rows[
                (attack_rows["scheme"] == scheme) & (attack_rows["attack_name"] == atk)
            ]["nc"].mean()
            for atk in attacks
        ]
        stds = [
            attack_rows[
                (attack_rows["scheme"] == scheme) & (attack_rows["attack_name"] == atk)
            ]["nc"].std()
            for atk in attacks
        ]
        bars = ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=4,
            label=SCHEME_LABELS.get(scheme, scheme),
            color=SCHEME_COLORS.get(scheme, "gray"),
            alpha=0.85,
        )

    ax.set_xlabel("Attack")
    ax.set_ylabel("Normalized Correlation (NC)")
    ax.set_title("Watermark Robustness: DWT-DCT-LL vs DWT-SVD-HH\n(mean ± std over all images, alpha=0.1)")
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABELS.get(a, a.replace("_", "\n")) for a in attacks], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(config.PLOTS_DIR, "comparison_nc_by_attack.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_psnr_imperceptibility(df):
    """Bar chart: mean PSNR for watermarked-only rows (imperceptibility check)."""
    wm_rows = df[df["attack_name"] == "watermarked_only"]
    schemes = [s for s in SCHEMES if s in wm_rows["scheme"].unique()]

    means = [wm_rows[wm_rows["scheme"] == s]["psnr_after_attack"].mean() for s in schemes]
    stds = [wm_rows[wm_rows["scheme"] == s]["psnr_after_attack"].std() for s in schemes]
    labels = [SCHEME_LABELS.get(s, s) for s in schemes]
    colors = [SCHEME_COLORS.get(s, "gray") for s in schemes]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85, width=0.4)
    ax.axhline(40, color="red", linewidth=1, linestyle="--", label="40 dB (imperceptibility threshold)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Watermark Imperceptibility: PSNR vs Original\n(mean ± std, alpha=0.1, no attack)")
    ax.set_ylim(0, max(means) * 1.2 if means else 60)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, (m, s_) in enumerate(zip(means, stds)):
        ax.text(i, m + s_ + 0.3, f"{m:.1f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()

    out = os.path.join(config.PLOTS_DIR, "comparison_psnr_imperceptibility.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    if not os.path.exists(COMPARISON_CSV):
        raise FileNotFoundError(
            f"{COMPARISON_CSV} not found. Run `python -m src.run_comparison` first."
        )
    df = pd.read_csv(COMPARISON_CSV)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plot_nc_by_attack(df)
    plot_psnr_imperceptibility(df)
    print("Done.")


if __name__ == "__main__":
    main()
