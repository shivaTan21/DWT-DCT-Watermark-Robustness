"""
Generate the summary table and comparison plots from results/metrics.csv.

Run after run_experiment.py:
    python -m src.plot_results
"""

import itertools
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

# Scatter-series styling for plot_psnr_vs_nc / plot_ssim_vs_nc. The series
# key is attack_type, except for "ai_enhancement" rows where attack_name is
# used instead so each AI backend gets its own color/marker -- Real-ESRGAN
# is the one attack that reaches a high-SSIM/low-NC region no traditional
# attack does, so it must stay visually distinct (red).
SERIES_STYLES = {
    "none": ("tab:blue", "o"),
    "traditional": ("tab:gray", "s"),
    "ai_fallback_enhancement": ("tab:orange", "^"),
    "real_esrgan": ("red", "*"),
    "opencv_dnn_sr": ("tab:purple", "D"),
}

_fallback_color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
_fallback_styles = {}


def _series_key_column(df):
    """attack_name for ai_enhancement rows, else attack_type."""
    return df["attack_type"].where(df["attack_type"] != "ai_enhancement", df["attack_name"])


def _style_for(key):
    """(color, marker) for a series key, with a cycling fallback for unmapped keys."""
    if key in SERIES_STYLES:
        return SERIES_STYLES[key]
    if key not in _fallback_styles:
        _fallback_styles[key] = (next(_fallback_color_cycle), "o")
    return _fallback_styles[key]


def load_metrics(path=config.METRICS_CSV_PATH):
    if not os.path.exists(path):
        raise RuntimeError(f"{path} not found. Run run_experiment.py first.")
    return pd.read_csv(path)


def _standard_variant_only(df):
    """Filter to embedding_variant == 'standard' so pre-existing plots keep their original meaning
    even though results/metrics.csv may now also contain a 'stable_positions' variant."""
    if "embedding_variant" not in df.columns:
        return df
    return df[df["embedding_variant"] == "standard"]


def build_summary_table(df, output_path=config.SUMMARY_CSV_PATH):
    """Group by (embedding_variant, if present,) attack_type/attack_name and summarize NC, BER, PSNR, SSIM."""
    group_cols = ["attack_type", "attack_name"]
    if "embedding_variant" in df.columns:
        group_cols = ["embedding_variant"] + group_cols

    summary = (
        df.groupby(group_cols)
        .agg(
            n=("nc", "size"),
            nc_mean=("nc", "mean"),
            nc_std=("nc", "std"),
            ber_mean=("ber", "mean"),
            ber_std=("ber", "std"),
            psnr_after_attack_mean=("psnr_after_attack", "mean"),
            ssim_after_attack_mean=("ssim_after_attack", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    summary.to_csv(output_path, index=False)
    print(f"Wrote summary table to {output_path}")
    return summary


def plot_nc_by_attack(df, output_path):
    grouped = df.groupby("attack_name")["nc"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Mean Normalized Correlation (NC)")
    ax.set_title("Watermark NC by Attack")
    ax.axvline(0.99, color="red", linestyle="--", linewidth=1, label="sanity threshold (0.99)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ber_by_attack(df, output_path):
    grouped = df.groupby("attack_name")["ber"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped.plot(kind="barh", ax=ax, color="indianred")
    ax.set_xlabel("Mean Bit Error Rate (BER)")
    ax.set_title("Watermark BER by Attack")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_psnr_vs_nc(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    df = df.assign(_series=_series_key_column(df))
    for key, group in df.groupby("_series"):
        color, marker = _style_for(key)
        ax.scatter(group["psnr_after_attack"], group["nc"], label=key,
                   color=color, marker=marker, alpha=0.7)
    ax.set_xlabel("PSNR after attack (dB)")
    ax.set_ylabel("Normalized Correlation (NC)")
    ax.set_title("PSNR vs NC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ssim_vs_nc(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    df = df.assign(_series=_series_key_column(df))
    for key, group in df.groupby("_series"):
        color, marker = _style_for(key)
        ax.scatter(group["ssim_after_attack"], group["nc"], label=key,
                   color=color, marker=marker, alpha=0.7)
    ax.set_xlabel("SSIM after attack")
    ax.set_ylabel("Normalized Correlation (NC)")
    ax.set_title("SSIM vs NC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_variant_comparison(df, output_path):
    """
    Grouped bar chart comparing mean NC (with std error bars) for the standard vs
    stable_positions embedding variants, one group of bars per attack. This is the
    candidate figure for the paper -- it needs both variants present in `df`.
    """
    variant_labels = {"standard": "Standard DWT-DCT", "stable_positions": "Stable-position DWT-DCT"}
    colors = {"standard": "tab:blue", "stable_positions": "tab:red"}

    stats = df.groupby(["attack_name", "embedding_variant"])["nc"].agg(["mean", "std"]).reset_index()
    attack_names = sorted(stats["attack_name"].unique())
    x = np.arange(len(attack_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, variant in enumerate(variant_labels):
        variant_stats = stats[stats["embedding_variant"] == variant].set_index("attack_name")
        means = [variant_stats["mean"].get(name, np.nan) for name in attack_names]
        stds = [variant_stats["std"].get(name, 0.0) for name in attack_names]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               label=variant_labels[variant], color=colors[variant])

    ax.set_xticks(x)
    ax.set_xticklabels(attack_names, rotation=45, ha="right")
    ax.set_xlabel("Attack")
    ax.set_ylabel("Mean Normalized Correlation (NC)")
    ax.set_title("Watermark Robustness: Standard vs Stable-Position DWT-DCT Embedding")
    ax.legend(title="Embedding variant")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_all(metrics_path=config.METRICS_CSV_PATH):
    config.ensure_directories()
    df = load_metrics(metrics_path)
    build_summary_table(df)

    standard_df = _standard_variant_only(df)
    plot_nc_by_attack(standard_df, os.path.join(config.PLOTS_DIR, "nc_by_attack.png"))
    plot_ber_by_attack(standard_df, os.path.join(config.PLOTS_DIR, "ber_by_attack.png"))
    plot_psnr_vs_nc(standard_df, os.path.join(config.PLOTS_DIR, "psnr_vs_nc.png"))
    plot_ssim_vs_nc(standard_df, os.path.join(config.PLOTS_DIR, "ssim_vs_nc.png"))

    variants_present = set(df.get("embedding_variant", pd.Series(dtype=str)).unique())
    if {"standard", "stable_positions"} <= variants_present:
        plot_variant_comparison(df, os.path.join(config.PLOTS_DIR, "variant_comparison.png"))
    else:
        warnings.warn(
            "Skipping variant_comparison.png: results/metrics.csv does not contain rows for both "
            "'standard' and 'stable_positions' embedding_variant -- run run_experiment.py to regenerate it."
        )

    print(f"Plots written to {config.PLOTS_DIR}")


if __name__ == "__main__":
    generate_all()
