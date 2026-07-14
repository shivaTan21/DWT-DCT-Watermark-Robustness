"""
Texture complexity analysis: correlates image texture metrics with Real-ESRGAN
watermark survival (NC at alpha=0.1, standard embedding variant).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pandas as pd
import pywt
from scipy.fftpack import dct as _scipy_dct
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

np.random.seed(42)

BLOCK_SIZE = 8
IMAGE_DIR = Path(__file__).parent.parent / 'data' / 'original_images'
METRICS_CSV = Path(__file__).parent.parent / 'results' / 'metrics.csv'
OUTPUT_CSV = Path(__file__).parent.parent / 'results' / 'texture_complexity.csv'
PLOTS_DIR = Path(__file__).parent.parent / 'results' / 'plots'


def _dct2(block):
    """2D DCT-II with ortho normalisation — same as watermark.py."""
    return _scipy_dct(_scipy_dct(block.T, norm='ortho').T, norm='ortho')


def _rgb_to_y(img_rgb):
    """ITU-R BT.601 luma — same formula as watermark.py's rgb_to_ycbcr."""
    img = img_rgb.astype(np.float64)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def compute_laplacian_var(gray):
    """Laplacian variance — standard sharpness / texture proxy."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_dct_energy(img_rgb):
    """
    Variance of all non-DC DCT coefficients across 8x8 blocks in the DWT LL subband.

    Uses the same DWT (Haar) and DCT as the embedding pipeline so the metric
    directly reflects coefficient-domain texture the watermark competes with.
    """
    y = _rgb_to_y(img_rgb)
    LL, _ = pywt.dwt2(y, 'haar')

    rows, cols = LL.shape
    non_dc = []
    for br in range(0, (rows // BLOCK_SIZE) * BLOCK_SIZE, BLOCK_SIZE):
        for bc in range(0, (cols // BLOCK_SIZE) * BLOCK_SIZE, BLOCK_SIZE):
            block = LL[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE]
            flat = _dct2(block).flatten()
            non_dc.append(flat[1:])   # skip DC (flat index 0 == position [0,0])

    return float(np.var(np.concatenate(non_dc)))


def compute_local_std(gray):
    """Mean of per-block standard deviations across all non-overlapping 8x8 blocks."""
    gray = gray.astype(np.float64)
    rows, cols = gray.shape
    stds = []
    for br in range(0, (rows // BLOCK_SIZE) * BLOCK_SIZE, BLOCK_SIZE):
        for bc in range(0, (cols // BLOCK_SIZE) * BLOCK_SIZE, BLOCK_SIZE):
            stds.append(gray[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE].std())
    return float(np.mean(stds))


def interpret_r(r):
    a = abs(r)
    if a > 0.5:
        return 'strong'
    elif a > 0.3:
        return 'moderate'
    else:
        return 'weak'


def main():
    # ── Step 1: compute texture metrics ───────────────────────────────────────
    image_files = sorted(IMAGE_DIR.glob('*.png'))
    print(f"Processing {len(image_files)} images …")

    records = []
    for img_path in image_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  WARNING: could not read {img_path.name}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        lap_var = compute_laplacian_var(gray)
        dct_energy = compute_dct_energy(img_rgb)
        local_std = compute_local_std(gray)

        records.append({
            'image': img_path.stem,
            'laplacian_var': lap_var,
            'dct_energy': dct_energy,
            'local_std': local_std,
        })
        print(f"  {img_path.stem:12s}  lap_var={lap_var:8.1f}  "
              f"dct_energy={dct_energy:10.1f}  local_std={local_std:.3f}")

    texture_df = pd.DataFrame(records)

    # ── Step 2: load Real-ESRGAN NC (alpha=0.1, standard) ─────────────────────
    metrics_df = pd.read_csv(METRICS_CSV)
    mask = (
        (metrics_df['embedding_variant'] == 'standard') &
        (metrics_df['attack_name'] == 'real_esrgan') &
        (metrics_df['alpha'] == 0.1)
    )
    esrgan_df = (
        metrics_df[mask][['image', 'nc']]
        .rename(columns={'nc': 'realesrgan_nc'})
        .copy()
    )
    # In case multiple rows survive the filter, average per image
    esrgan_df = esrgan_df.groupby('image', as_index=False)['realesrgan_nc'].mean()

    # ── Step 3: merge ─────────────────────────────────────────────────────────
    merged = texture_df.merge(esrgan_df, on='image', how='inner')
    print(f"\nMerged {len(merged)} images with Real-ESRGAN NC data")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

    # ── Step 4: Pearson correlations ───────────────────────────────────────────
    metric_defs = [
        ('laplacian_var', 'Laplacian variance'),
        ('dct_energy',    'DCT LL energy'),
        ('local_std',     'Local std deviation'),
    ]

    print(f"\n{'Texture Metric':<23} | {'Pearson r':>9} | {'p-value':>7} | Interpretation")
    print(f"{'-'*23}-+-{'-'*9}-+-{'-'*7}-+-{'-'*14}")

    correlations = {}
    for col, label in metric_defs:
        r, p = pearsonr(merged[col].values, merged['realesrgan_nc'].values)
        interp = interpret_r(r)
        print(f"{label:<23} | {r:>9.3f} | {p:>7.3f} | {interp}")
        correlations[col] = (r, p, label)

    # ── Step 5: scatter plots ──────────────────────────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    point_colors = merged['image'].apply(
        lambda x: 'steelblue' if x.startswith('kodim') else 'darkorange'
    ).values

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=8, label='Kodak'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange',
               markersize=8, label='TAMPERE17'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Texture Complexity vs Real-ESRGAN Watermark NC (α=0.1, standard)',
                 fontsize=13)

    nc_vals = merged['realesrgan_nc'].values

    for ax, (col, label) in zip(axes, metric_defs):
        r, p, _ = correlations[col]
        x = merged[col].values

        ax.scatter(x, nc_vals, c=point_colors, alpha=0.75,
                   edgecolors='k', linewidths=0.3, s=55, zorder=3)

        # regression line
        m, b = np.polyfit(x, nc_vals, 1)
        x_line = np.linspace(x.min(), x.max(), 300)
        fit_line = Line2D(x_line, m * x_line + b, color='crimson',
                          linestyle='--', linewidth=1.5, label=f'Fit (r={r:.3f})')
        ax.add_line(fit_line)

        # threshold lines when a natural gap is visible
        extra_handles = []
        if (nc_vals > 0.8).any() and (nc_vals < 0.6).any():
            ax.axhline(0.8, color='green', linestyle=':', alpha=0.65, linewidth=1.2)
            ax.axhline(0.6, color='red',   linestyle=':', alpha=0.65, linewidth=1.2)
            extra_handles = [
                Line2D([0], [0], color='green', linestyle=':', label='NC=0.8'),
                Line2D([0], [0], color='red',   linestyle=':', label='NC=0.6'),
            ]

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Real-ESRGAN NC', fontsize=10)
        ax.set_title(f'{label}\nr = {r:.3f},  p = {p:.3f}', fontsize=10)
        ax.legend(handles=legend_handles + [fit_line] + extra_handles, fontsize=8)

    plt.tight_layout()
    plot_path = PLOTS_DIR / 'texture_vs_realesrgan_nc.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {plot_path}")

    # ── Step 6: high / low texture image lists ────────────────────────────────
    lap_median = merged['laplacian_var'].median()
    high_tex = merged[
        (merged['laplacian_var'] > lap_median) & (merged['realesrgan_nc'] > 0.8)
    ]['image'].tolist()
    low_tex = merged[
        (merged['laplacian_var'] < lap_median) & (merged['realesrgan_nc'] < 0.6)
    ]['image'].tolist()

    print(f"\nLaplacian variance median: {lap_median:.1f}")
    print(f"High texture images (lap_var > median AND NC > 0.8): {high_tex}")
    print(f"Low texture images  (lap_var < median AND NC < 0.6): {low_tex}")

    # ── Step 7: recommendation ────────────────────────────────────────────────
    lap_r = correlations['laplacian_var'][0]
    print()
    if abs(lap_r) > 0.5:
        print("STRONG CORRELATION (r > 0.5): texture-aware block selection is motivated")
    elif abs(lap_r) > 0.3:
        print("MODERATE CORRELATION (0.3 < r < 0.5): worth investigating but uncertain")
    else:
        print("WEAK CORRELATION (r < 0.3): texture-aware selection unlikely to help")


if __name__ == '__main__':
    main()
