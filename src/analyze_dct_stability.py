"""
Per-DCT-coefficient stability analysis: Real-ESRGAN vs JPEG Q50.

Two parallel analyses, both comparing the *watermarked* image (alpha=0.1)
against an attacked version of it -- this isolates each attack's own effect
on the LL-subband DCT coefficients, without conflating it with the watermark
embedding perturbation itself:

  1. Real-ESRGAN: results/watermarked/<stem>__alpha0.1.png vs
     results/attacked/ai_enhancement/<stem>__alpha0.1__real_esrgan.png
  2. JPEG Q50:    results/watermarked/<stem>__alpha0.1.png vs
     results/attacked/traditional/<stem>__alpha0.1__jpeg_compression_q50.png

The preprocessing, YCbCr conversion, DWT settings, LL-subband selection,
block size, and DCT convention are all reused/mirrored from src/watermark.py
so the analysis matches the watermark embedding pipeline exactly -- this is
what lets the resulting heatmaps be read directly against the embedding
positions (COEFF_POS_1, COEFF_POS_2) used there.

Comparing the two side by side identifies coefficient positions that are
stable under *both* attacks simultaneously -- candidates for relocating the
embedding away from (4,1)/(1,4) if those turn out not to be the most robust
choice against either attack.

Run after run_experiment.py has produced alpha=0.1 watermarked images plus
real_esrgan and jpeg_compression_q50 attacked outputs:
    python -m src.analyze_dct_stability
"""

import glob
import os
import warnings

import cv2
import numpy as np
import pandas as pd
import pywt

from . import config
from .watermark import BLOCK_SIZE, COEFF_POS_1, COEFF_POS_2, DEFAULT_WAVELET, _dct2, rgb_to_ycbcr

ANALYSIS_ALPHA = 0.1
REALESRGAN_ATTACK_NAME = "real_esrgan"
JPEG_ATTACK_NAME = "jpeg_compression_q50"

DCT_STABILITY_REALESRGAN_CSV_PATH = os.path.join(config.RESULTS_DIR, "dct_stability_realesrgan.csv")
DCT_STABILITY_JPEG_CSV_PATH = os.path.join(config.RESULTS_DIR, "dct_stability_jpeg.csv")
DCT_STABILITY_REALESRGAN_HEATMAP_PATH = os.path.join(config.PLOTS_DIR, "dct_stability_heatmap_realesrgan.png")
DCT_STABILITY_JPEG_HEATMAP_PATH = os.path.join(config.PLOTS_DIR, "dct_stability_heatmap_jpeg.png")
DCT_STABILITY_COMPARISON_PATH = os.path.join(config.PLOTS_DIR, "dct_stability_comparison.png")


def _load_rgb_resized(path, size=config.IMAGE_SIZE):
    """Load an image as RGB float64, resizing to `size` (width, height) if needed."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    if (bgr.shape[1], bgr.shape[0]) != size:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)


def _y_ll_subband(rgb_image, wavelet=DEFAULT_WAVELET):
    """Y channel (YCbCr, same conversion as embed_watermark_rgb) -> single-level DWT LL subband."""
    y = rgb_to_ycbcr(rgb_image)[..., 0]
    LL, _ = pywt.dwt2(y, wavelet)
    return LL


def _dct_blocks(ll_subband, block_size=BLOCK_SIZE):
    """Yield the 2D-DCT (same convention as src.watermark._dct2) of every non-overlapping block."""
    rows, cols = ll_subband.shape
    for br in range(0, rows - block_size + 1, block_size):
        for bc in range(0, cols - block_size + 1, block_size):
            block = ll_subband[br:br + block_size, bc:bc + block_size]
            yield _dct2(block)


def find_pairs(attacked_dir, attack_name, alpha=ANALYSIS_ALPHA, watermarked_dir=config.WATERMARKED_DIR):
    """Match results/watermarked/<stem>__alpha<alpha>.png to <stem>__alpha<alpha>__<attack_name>.png."""
    base_suffix = f"__alpha{alpha}.png"
    attack_suffix = f"__alpha{alpha}__{attack_name}.png"

    watermarked_paths = sorted(glob.glob(os.path.join(watermarked_dir, f"*{base_suffix}")))
    if not watermarked_paths:
        raise RuntimeError(
            f"No watermarked files matching '*{base_suffix}' found in {watermarked_dir}. "
            f"Run run_experiment.py with alpha={alpha} first."
        )

    pairs = []
    for watermarked_path in watermarked_paths:
        basename = os.path.basename(watermarked_path)
        stem = basename[: -len(base_suffix)]
        attacked_path = os.path.join(attacked_dir, f"{stem}{attack_suffix}")
        if not os.path.exists(attacked_path):
            warnings.warn(f"Skipping {stem}: no matching '{attack_name}' attacked file at {attacked_path}")
            continue
        pairs.append((stem, watermarked_path, attacked_path))

    if not pairs:
        raise RuntimeError(
            f"Found {len(watermarked_paths)} watermarked file(s) in {watermarked_dir} but none had a "
            f"matching '{attack_name}' attacked file in {attacked_dir}."
        )

    return pairs


def compute_coefficient_changes(pairs):
    """Stack |watermarked - attacked| DCT coefficient matrices over every block/image pair."""
    abs_diffs = []
    for stem, watermarked_path, attacked_path in pairs:
        watermarked_ll = _y_ll_subband(_load_rgb_resized(watermarked_path))
        attacked_ll = _y_ll_subband(_load_rgb_resized(attacked_path))

        if watermarked_ll.shape != attacked_ll.shape:
            warnings.warn(
                f"Skipping {stem}: LL subband shape mismatch ({watermarked_ll.shape} vs {attacked_ll.shape})"
            )
            continue

        for wm_block, atk_block in zip(_dct_blocks(watermarked_ll), _dct_blocks(attacked_ll)):
            abs_diffs.append(np.abs(wm_block - atk_block))

    if not abs_diffs:
        raise RuntimeError("No DCT blocks were analyzed -- check that image pairs share the same dimensions.")

    return np.stack(abs_diffs)


def build_stability_table(abs_diffs):
    """One row per (coeff_row, coeff_col) with mean/std absolute change."""
    mean_abs = abs_diffs.mean(axis=0)
    std_abs = abs_diffs.std(axis=0)

    rows = [
        {
            "coeff_row": r,
            "coeff_col": c,
            "mean_abs_change": mean_abs[r, c],
            "std_abs_change": std_abs[r, c],
        }
        for r in range(BLOCK_SIZE)
        for c in range(BLOCK_SIZE)
    ]
    return pd.DataFrame(rows).sort_values(["coeff_row", "coeff_col"]).reset_index(drop=True)


def run_analysis(name, attacked_dir, attack_name, csv_path, heatmap_path):
    """Find pairs, compute per-coefficient stability, and write the CSV + standalone heatmap for one attack."""
    pairs = find_pairs(attacked_dir=attacked_dir, attack_name=attack_name)
    abs_diffs = compute_coefficient_changes(pairs)
    df = build_stability_table(abs_diffs)

    df.to_csv(csv_path, index=False)
    plot_heatmap(df, heatmap_path, title=f"{name}: mean |Δ| per LL-subband DCT coefficient")

    print(f"[{name}] image pairs processed: {len(pairs)}")
    print(f"[{name}] total DCT blocks analyzed: {abs_diffs.shape[0]}")
    print(f"[{name}] wrote {csv_path}")
    print(f"[{name}] wrote {heatmap_path}")

    return df


def _grid_from_df(df, value_col="mean_abs_change"):
    grid = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    for row in df.itertuples():
        grid[row.coeff_row, row.coeff_col] = getattr(row, value_col)
    return grid


def _style_heatmap_axes(ax, title, mark_positions):
    ax.set_xlabel("DCT coefficient column")
    ax.set_ylabel("DCT coefficient row")
    ax.set_title(title)
    ax.set_xticks(range(BLOCK_SIZE))
    ax.set_yticks(range(BLOCK_SIZE))
    for r, c in mark_positions:
        ax.scatter(c, r, marker="x", s=200, color="red", linewidths=3)


def plot_heatmap(df, output_path, title, mark_positions=(COEFF_POS_1, COEFF_POS_2)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = _grid_from_df(df)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="viridis")  # dark = stable (low change), bright = unstable (high change)
    _style_heatmap_axes(ax, title, mark_positions)
    fig.colorbar(im, ax=ax, label="mean_abs_change")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_comparison(realesrgan_df, jpeg_df, output_path, mark_positions=(COEFF_POS_1, COEFF_POS_2)):
    """Side-by-side heatmaps on a shared color scale -- the figure for the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_re = _grid_from_df(realesrgan_df)
    grid_jpeg = _grid_from_df(jpeg_df)
    vmin = min(grid_re.min(), grid_jpeg.min())
    vmax = max(grid_re.max(), grid_jpeg.max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    im = None
    for ax, grid, title in zip(axes, (grid_re, grid_jpeg), ("Real-ESRGAN", "JPEG Q50")):
        im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax)
        _style_heatmap_axes(ax, f"{title}: mean |Δ| per DCT coefficient", mark_positions)

    fig.suptitle("DCT Coefficient Stability: Real-ESRGAN vs JPEG Q50 (shared color scale)")
    fig.colorbar(im, ax=axes.tolist(), label="mean_abs_change", shrink=0.8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_comparison_summary(realesrgan_df, jpeg_df):
    """
    Merge the two per-coefficient tables and flag "tradeoff" candidates: positions
    whose mean_abs_change is at or below the median for *both* attacks, i.e. more
    stable than average against Real-ESRGAN AND more stable than average against
    JPEG Q50 -- the candidate positions for a relocated watermark embedding.
    """
    merged = realesrgan_df.merge(jpeg_df, on=["coeff_row", "coeff_col"], suffixes=("_realesrgan", "_jpeg"))
    re_median = merged["mean_abs_change_realesrgan"].median()
    jpeg_median = merged["mean_abs_change_jpeg"].median()
    merged["tradeoff_flag"] = (
        (merged["mean_abs_change_realesrgan"] <= re_median)
        & (merged["mean_abs_change_jpeg"] <= jpeg_median)
    )
    merged["combined_score"] = merged["mean_abs_change_realesrgan"] + merged["mean_abs_change_jpeg"]
    return merged.sort_values(["coeff_row", "coeff_col"]).reset_index(drop=True)


def print_summary(merged):
    display_cols = ["coeff_row", "coeff_col", "mean_abs_change_realesrgan", "mean_abs_change_jpeg", "tradeoff_flag"]
    print("\nPer-coefficient stability comparison (tradeoff_flag = stable against both attacks):")
    print(merged[display_cols].round(4).to_string(index=False))

    candidates = merged[merged["tradeoff_flag"]].sort_values("combined_score").head(5)
    print("\nTop 5 candidate positions (low Real-ESRGAN AND low JPEG change):")
    print(
        candidates[["coeff_row", "coeff_col", "mean_abs_change_realesrgan", "mean_abs_change_jpeg", "combined_score"]]
        .round(4)
        .to_string(index=False)
    )

    print("\nCurrent embedding positions:")
    for pos in (COEFF_POS_1, COEFF_POS_2):
        row = merged[(merged.coeff_row == pos[0]) & (merged.coeff_col == pos[1])].iloc[0]
        print(
            f"  {pos}: real_esrgan mean_abs_change={row.mean_abs_change_realesrgan:.4f}, "
            f"jpeg mean_abs_change={row.mean_abs_change_jpeg:.4f}, tradeoff_flag={bool(row.tradeoff_flag)}"
        )


def main():
    np.random.seed(config.RANDOM_SEED)
    config.ensure_directories()

    realesrgan_df = run_analysis(
        name="Real-ESRGAN",
        attacked_dir=config.ATTACKED_AI_DIR,
        attack_name=REALESRGAN_ATTACK_NAME,
        csv_path=DCT_STABILITY_REALESRGAN_CSV_PATH,
        heatmap_path=DCT_STABILITY_REALESRGAN_HEATMAP_PATH,
    )
    jpeg_df = run_analysis(
        name="JPEG Q50",
        attacked_dir=config.ATTACKED_TRADITIONAL_DIR,
        attack_name=JPEG_ATTACK_NAME,
        csv_path=DCT_STABILITY_JPEG_CSV_PATH,
        heatmap_path=DCT_STABILITY_JPEG_HEATMAP_PATH,
    )

    plot_comparison(realesrgan_df, jpeg_df, DCT_STABILITY_COMPARISON_PATH)
    print(f"\nWrote {DCT_STABILITY_COMPARISON_PATH}")

    merged = build_comparison_summary(realesrgan_df, jpeg_df)
    print_summary(merged)


if __name__ == "__main__":
    main()
