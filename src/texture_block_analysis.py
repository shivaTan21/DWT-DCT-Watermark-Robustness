"""
Block-level texture vs Real-ESRGAN damage correlation.

For each 8×8 block in the DWT LL subband (1024 per image × 100 images):
  - texture_var: variance of LL pixel values in the block
  - dct_damage:  mean absolute difference of DCT coefficients between the
                 original watermarked-only image and the Real-ESRGAN enhanced image

Uses data/processed_images/ (512×512 resized versions, matching the pipeline)
and results/attacked/ai_enhancement/ at alpha=0.1, standard embedding.
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
PROCESSED_DIR  = Path(__file__).parent.parent / 'data' / 'processed_images'
ATTACKED_DIR   = Path(__file__).parent.parent / 'results' / 'attacked' / 'ai_enhancement'
METRICS_CSV    = Path(__file__).parent.parent / 'results' / 'metrics.csv'
OUTPUT_CSV     = Path(__file__).parent.parent / 'results' / 'block_texture_damage.csv'
PLOTS_DIR      = Path(__file__).parent.parent / 'results' / 'plots'

# Embedding coefficient positions from watermark.py
COEFF_POS_1 = (4, 1)
COEFF_POS_2 = (1, 4)


def _dct2(block):
    """2D DCT-II ortho — identical to watermark.py."""
    return _scipy_dct(_scipy_dct(block.T, norm='ortho').T, norm='ortho')


def _rgb_to_y(img_rgb):
    """BT.601 luma — identical to watermark.py's rgb_to_ycbcr."""
    img = img_rgb.astype(np.float64)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def get_ll(img_rgb):
    """Y-channel Haar DWT, returns LL subband."""
    LL, _ = pywt.dwt2(_rgb_to_y(img_rgb), 'haar')
    return LL


def analyse_blocks(orig_LL, enh_LL):
    """
    Iterate over all non-overlapping 8×8 blocks shared by orig_LL and enh_LL.

    Returns
    -------
    records : list of (block_idx, block_row, block_col, texture_var, dct_damage)
    pos_damage_acc : (8, 8) array of |dct_orig - dct_enh| summed over all blocks
    n_blocks : int
    """
    rows = min(orig_LL.shape[0], enh_LL.shape[0])
    cols = min(orig_LL.shape[1], enh_LL.shape[1])
    orig_LL = orig_LL[:rows, :cols]
    enh_LL  = enh_LL[:rows, :cols]

    blocks_r = rows // BLOCK_SIZE
    blocks_c = cols // BLOCK_SIZE

    records = []
    pos_damage_acc = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)

    for br in range(blocks_r):
        for bc in range(blocks_c):
            r0, c0 = br * BLOCK_SIZE, bc * BLOCK_SIZE
            ob = orig_LL[r0:r0 + BLOCK_SIZE, c0:c0 + BLOCK_SIZE]
            eb = enh_LL [r0:r0 + BLOCK_SIZE, c0:c0 + BLOCK_SIZE]

            texture_var = float(np.var(ob))

            diff = np.abs(_dct2(ob) - _dct2(eb))
            dct_damage = float(diff.mean())

            pos_damage_acc += diff
            records.append((br * blocks_c + bc, br, bc, texture_var, dct_damage))

    return records, pos_damage_acc, len(records)


def interpret(r):
    a = abs(r)
    if a > 0.5:  return 'strong'
    if a > 0.3:  return 'moderate'
    return 'weak'


def main():
    # ── NC lookup for choosing heatmap examples ────────────────────────────
    metrics_df = pd.read_csv(METRICS_CSV)
    nc_df = metrics_df[
        (metrics_df['embedding_variant'] == 'standard') &
        (metrics_df['attack_name']       == 'real_esrgan') &
        (metrics_df['alpha']             == 0.1)
    ][['image', 'nc']].copy()
    nc_map = dict(zip(nc_df['image'], nc_df['nc']))

    # ── Main processing loop ───────────────────────────────────────────────
    image_files = sorted(PROCESSED_DIR.glob('*.png'))
    print(f"Processing {len(image_files)} images …")

    all_records   = []
    global_pos    = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)
    global_n      = 0
    # Per-(block_row, block_col) damage accumulator for "top 5" ranking
    grid_sum      = {}
    grid_cnt      = {}
    per_image_r   = {}
    skipped       = []

    for img_path in image_files:
        name     = img_path.stem
        att_path = ATTACKED_DIR / f'{name}__alpha0.1__real_esrgan.png'

        if not att_path.exists():
            print(f"  WARNING: no attacked file for {name} — skipping")
            skipped.append(name)
            continue

        orig_bgr = cv2.imread(str(img_path))
        att_bgr  = cv2.imread(str(att_path))
        if orig_bgr is None or att_bgr is None:
            print(f"  WARNING: could not read image files for {name} — skipping")
            skipped.append(name)
            continue

        orig_LL = get_ll(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        enh_LL  = get_ll(cv2.cvtColor(att_bgr,  cv2.COLOR_BGR2RGB))

        records, pos_acc, n = analyse_blocks(orig_LL, enh_LL)

        global_pos += pos_acc
        global_n   += n

        tvs, dds = [], []
        for (idx, br, bc, tv, dd) in records:
            all_records.append({
                'image': name, 'block_idx': idx,
                'block_row': br, 'block_col': bc,
                'texture_var': tv, 'dct_damage': dd,
            })
            tvs.append(tv); dds.append(dd)
            grid_sum[(br, bc)] = grid_sum.get((br, bc), 0.0) + dd
            grid_cnt[(br, bc)] = grid_cnt.get((br, bc), 0) + 1

        if n > 2:
            r_val, _ = pearsonr(tvs, dds)
            per_image_r[name] = r_val

        print(f"  {name:12s} {n:5d} blocks | "
              f"mean_damage={np.mean(dds):6.2f} | "
              f"NC={nc_map.get(name, float('nan')):.3f}")

    df = pd.DataFrame(all_records)
    print(f"\nTotal blocks collected : {len(df):,}")
    if skipped:
        print(f"Skipped images ({len(skipped)}): {skipped}")

    # ── Save CSV ───────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}")

    # ── Correlation ────────────────────────────────────────────────────────
    r_all, p_all = pearsonr(df['texture_var'].values, df['dct_damage'].values)
    per_r_vals   = list(per_image_r.values())
    mean_r       = float(np.mean(per_r_vals))
    std_r        = float(np.std(per_r_vals))

    print(f"\nBlock-level texture vs Real-ESRGAN damage correlation:")
    print(f"  Pearson r (all {len(df):,} blocks): {r_all:.3f}")
    print(f"  p-value: {p_all:.3e}")
    print(f"  Interpretation: {interpret(r_all)}")
    print(f"\nPer-image correlation (mean ± std): {mean_r:.3f} ± {std_r:.3f}")

    # ── Top 5 most damaged block positions ────────────────────────────────
    block_mean = {k: grid_sum[k] / grid_cnt[k] for k in grid_sum}
    top5 = sorted(block_mean.items(), key=lambda x: -x[1])[:5]
    print(f"\nTop 5 most damaged block positions (by mean damage across all images):")
    for (br, bc), dmg in top5:
        print(f"  Block position ({br:2d}, {bc:2d}): mean_damage={dmg:.4f}")

    # ── DCT coefficient position damage ranking ────────────────────────────
    if global_n > 0:
        mean_pos_damage = global_pos / global_n   # (8, 8)
        flat            = mean_pos_damage.flatten()
        rank_order      = np.argsort(flat)[::-1]  # index 0 = highest damage

        def flat_idx(pos):
            return pos[0] * BLOCK_SIZE + pos[1]

        rank_p1 = int(np.where(rank_order == flat_idx(COEFF_POS_1))[0][0]) + 1
        rank_p2 = int(np.where(rank_order == flat_idx(COEFF_POS_2))[0][0]) + 1

        print(f"\nEmbedding positions {COEFF_POS_1}/{COEFF_POS_2} damage rank: "
              f"{COEFF_POS_1} → rank {rank_p1} of 64 "
              f"(mean damage={mean_pos_damage[COEFF_POS_1]:.4f}) | "
              f"{COEFF_POS_2} → rank {rank_p2} of 64 "
              f"(mean damage={mean_pos_damage[COEFF_POS_2]:.4f})")

    # ── Plot 1: scatter block texture vs damage ────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    kodak_mask = df['image'].str.startswith('kodim')
    for mask, color, label in [
        (kodak_mask,  'steelblue',   'Kodak'),
        (~kodak_mask, 'darkorange', 'TAMPERE17'),
    ]:
        sub = df[mask]
        ax.scatter(sub['texture_var'], sub['dct_damage'],
                   c=color, alpha=0.07, s=4, linewidths=0, rasterized=True,
                   label=label)

    x, y = df['texture_var'].values, df['dct_damage'].values
    m, b  = np.polyfit(x, y, 1)
    xl    = np.linspace(x.min(), x.max(), 500)
    ax.plot(xl, m * xl + b, 'r-', linewidth=2,
            label=f'Fit (r={r_all:.3f}, p={p_all:.2e})')

    ax.set_xlabel('Block Texture Variance (LL subband pixel values)', fontsize=12)
    ax.set_ylabel('Block DCT Damage  (mean |Δcoeff|, original vs enhanced)', fontsize=12)
    ax.set_title(
        f'Block-level Texture vs Real-ESRGAN DCT Damage\n'
        f'({len(df):,} blocks, 100 images, α=0.1, standard embedding)',
        fontsize=12)
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=8, label='Kodak'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange',
               markersize=8, label='TAMPERE17'),
        Line2D([0], [0], color='r', linewidth=2,
               label=f'Fit (r={r_all:.3f})'),
    ], fontsize=10)
    plt.tight_layout()
    scatter_path = PLOTS_DIR / 'block_texture_vs_damage.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nScatter plot → {scatter_path}")

    # ── Plot 2: damage heatmaps (2 low-NC + 2 high-NC) ────────────────────
    processed_set = set(df['image'].unique())
    nc_avail  = {img: nc_map[img] for img in processed_set if img in nc_map}
    sorted_nc = sorted(nc_avail.items(), key=lambda x: x[1])
    low_examples  = [img for img, _ in sorted_nc[:2]]
    high_examples = [img for img, _ in sorted_nc[-2:]]
    panel_imgs    = low_examples + high_examples

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        'Block DCT Damage Heatmaps — Low-NC vs High-NC Images (α=0.1, standard)',
        fontsize=13)

    for ax, img_name in zip(axes, panel_imgs):
        nc_val = nc_avail.get(img_name, float('nan'))
        survival = 'low survival' if nc_val < 0.7 else 'high survival'

        sub = df[df['image'] == img_name].copy()
        nr  = sub['block_row'].max() + 1
        nc_ = sub['block_col'].max() + 1
        # Fast pivot: sort and reshape
        sub_sorted = sub.sort_values(['block_row', 'block_col'])
        heat = sub_sorted['dct_damage'].values.reshape(nr, nc_)

        im = ax.imshow(heat, cmap='hot', interpolation='nearest', aspect='equal')
        plt.colorbar(im, ax=ax, shrink=0.82, label='DCT damage')
        ax.set_title(f'{img_name}\nNC={nc_val:.3f} ({survival})', fontsize=9)
        ax.set_xlabel('Block column', fontsize=8)
        ax.set_ylabel('Block row',    fontsize=8)

    plt.tight_layout()
    heatmap_path = PLOTS_DIR / 'block_damage_heatmap_examples.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Damage heatmap    → {heatmap_path}")


if __name__ == '__main__':
    main()
