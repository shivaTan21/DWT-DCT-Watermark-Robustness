"""
6×6 Spearman rank correlation matrix of DCT damage profiles across attack types.

For each of six attacks, the damage profile is an 8×8 matrix (flattened to 64
values) of mean absolute DCT-coefficient differences computed over all 102,400
LL-subband blocks (100 images × 1024 blocks each) when comparing the watermarked
image (alpha=0.1) against the attacked version.

The 6×6 Spearman matrix then measures how similarly each pair of attacks perturbs
the DCT coefficient space — high rho means the two attacks damage the same
frequency components; low or negative rho means their damage patterns diverge.

Run:
    python -m src.correlation_matrix
"""

import glob
import os
import warnings

import cv2
import matplotlib
import numpy as np
import pywt
from scipy import stats
from scipy.fft import dctn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

ANALYSIS_ALPHA = 0.1
BLOCK_SIZE = 8

ATTACKS = [
    ("Real-ESRGAN",    config.ATTACKED_AI_DIR,          "real_esrgan"),
    ("ESPCN",          config.ATTACKED_AI_DIR,           "espcn_x4"),
    ("JPEG-Q50",       config.ATTACKED_TRADITIONAL_DIR,  "jpeg_compression_q50"),
    ("Gauss-blur",     config.ATTACKED_TRADITIONAL_DIR,  "gaussian_blur_5x5"),
    ("Median-filt",    config.ATTACKED_TRADITIONAL_DIR,  "median_filter_3x3"),
    ("Gauss-noise",    config.ATTACKED_TRADITIONAL_DIR,  "gaussian_noise_s10"),
]

HEATMAP_PATH = os.path.join(config.PLOTS_DIR, "attack_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Image / signal helpers
# ---------------------------------------------------------------------------

def _load_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read: {path}")
    if (bgr.shape[1], bgr.shape[0]) != config.IMAGE_SIZE:
        bgr = cv2.resize(bgr, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)


def _y_channel(rgb):
    """BT.601 luma: Y = 0.299R + 0.587G + 0.114B."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _ll_subband(y):
    LL, _ = pywt.dwt2(y, "haar")
    return LL


def _damage_profile(pairs):
    """
    Return an (8, 8) array: mean |DCT(wm_block) - DCT(atk_block)| over all blocks.
    Uses scipy.fft.dctn(block, norm='ortho') as the DCT convention.
    """
    accum = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)
    n_blocks = 0

    for _, wm_path, atk_path in pairs:
        wm_ll  = _ll_subband(_y_channel(_load_rgb(wm_path)))
        atk_ll = _ll_subband(_y_channel(_load_rgb(atk_path)))

        if wm_ll.shape != atk_ll.shape:
            warnings.warn(f"Shape mismatch, skipping: {wm_path}")
            continue

        rows, cols = wm_ll.shape
        for br in range(0, rows - BLOCK_SIZE + 1, BLOCK_SIZE):
            for bc in range(0, cols - BLOCK_SIZE + 1, BLOCK_SIZE):
                wm_dct  = dctn(wm_ll[br:br+BLOCK_SIZE,  bc:bc+BLOCK_SIZE],  norm="ortho")
                atk_dct = dctn(atk_ll[br:br+BLOCK_SIZE, bc:bc+BLOCK_SIZE], norm="ortho")
                accum += np.abs(wm_dct - atk_dct)
                n_blocks += 1

    return accum / n_blocks, n_blocks


# ---------------------------------------------------------------------------
# File-pair matching (mirrors analyze_dct_stability.find_pairs)
# ---------------------------------------------------------------------------

def _find_pairs(attacked_dir, attack_name, alpha=ANALYSIS_ALPHA):
    base_suffix   = f"__alpha{alpha}.png"
    attack_suffix = f"__alpha{alpha}__{attack_name}.png"

    wm_paths = sorted(glob.glob(os.path.join(config.WATERMARKED_DIR, f"*{base_suffix}")))
    if not wm_paths:
        raise RuntimeError(f"No watermarked files matching '*{base_suffix}' in {config.WATERMARKED_DIR}")

    pairs = []
    for wm_path in wm_paths:
        basename = os.path.basename(wm_path)
        stem = basename[: -len(base_suffix)]
        atk_path = os.path.join(attacked_dir, f"{stem}{attack_suffix}")
        if not os.path.exists(atk_path):
            warnings.warn(f"Missing attacked file, skipping {stem}: {atk_path}")
            continue
        pairs.append((stem, wm_path, atk_path))

    if not pairs:
        raise RuntimeError(f"No valid pairs found for attack '{attack_name}' in {attacked_dir}")
    return pairs


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def _spearman_matrix(profiles):
    """
    profiles : list of (label, flat_64_vector)
    returns  : (n, n) rho matrix, (n, n) p-value matrix, list of labels
    """
    n = len(profiles)
    labels = [p[0] for p in profiles]
    vecs   = np.stack([p[1] for p in profiles])   # (n, 64)

    rho = np.eye(n)
    pval = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r, p = stats.spearmanr(vecs[i], vecs[j])
            rho[i, j] = rho[j, i] = r
            pval[i, j] = pval[j, i] = p

    return rho, pval, labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_heatmap(rho, pval, labels, output_path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(8, 7))

    vmax = 1.0
    im = ax.imshow(rho, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Spearman rank correlation of DCT damage profiles (alpha=0.1)", fontsize=12, pad=14)

    for i in range(n):
        for j in range(n):
            sig = "*" if (i != j and pval[i, j] < 0.05) else ""
            text = f"{rho[i, j]:.2f}{sig}"
            color = "white" if abs(rho[i, j]) > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config.ensure_directories()

    print("Computing DCT damage profiles …\n")
    profiles = []
    dc_damages = []

    for label, attacked_dir, attack_name in ATTACKS:
        pairs = _find_pairs(attacked_dir, attack_name)
        profile, n_blocks = _damage_profile(pairs)
        dc_damages.append((label, profile[0, 0]))
        profiles.append((label, profile.flatten()))
        print(f"  {label:14s}  pairs={len(pairs):3d}  blocks={n_blocks:,}  DC damage={profile[0,0]:.4f}")

    print()

    # --- DC coefficient damage per attack ---
    print("DC coefficient (0,0) damage:")
    for label, dc in dc_damages:
        print(f"  {label:14s}  {dc:.4f}")
    print()

    # --- Spearman correlation matrix ---
    rho, pval, labels = _spearman_matrix(profiles)

    col_w = 14
    header = " " * col_w + "".join(f"{lb:>{col_w}}" for lb in labels)
    print("6×6 Spearman ρ matrix  (* = p<0.05):")
    print(header)
    for i, row_label in enumerate(labels):
        row_str = f"{row_label:<{col_w}}"
        for j in range(len(labels)):
            sig = "*" if (i != j and pval[i, j] < 0.05) else " "
            row_str += f"  {rho[i, j]:+.3f}{sig}   "
        print(row_str)
    print()

    # --- Interpretation ---
    print("Interpretation summary:")
    significant = [
        (labels[i], labels[j], rho[i, j], pval[i, j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
        if pval[i, j] < 0.05
    ]
    if significant:
        print("  Significantly correlated pairs (p<0.05):")
        for a, b, r, p in sorted(significant, key=lambda x: -abs(x[2])):
            direction = "positively" if r > 0 else "negatively"
            print(f"    {a} vs {b}: ρ={r:+.3f} (p={p:.4f}) — {direction} correlated damage patterns")
    else:
        print("  No pairs reach p<0.05 significance.")

    non_sig = [
        (labels[i], labels[j], rho[i, j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
        if pval[i, j] >= 0.05
    ]
    if non_sig:
        print("  Non-significant pairs (p≥0.05) — independent damage patterns:")
        for a, b, r in sorted(non_sig, key=lambda x: abs(x[2])):
            print(f"    {a} vs {b}: ρ={r:+.3f}")

    # Highest and lowest rho (off-diagonal)
    mask = ~np.eye(len(labels), dtype=bool)
    flat_rho = rho[mask]
    idx_max = np.unravel_index(np.argmax(rho * mask), rho.shape)
    idx_min_val = np.argmin(flat_rho)
    idx_all = [(i, j) for i in range(len(labels)) for j in range(len(labels)) if i != j]
    i_min, j_min = idx_all[idx_min_val]

    print(f"\n  Most similar damage patterns : {labels[idx_max[0]]} vs {labels[idx_max[1]]}  ρ={rho[idx_max]:+.3f}")
    print(f"  Most distinct damage patterns: {labels[i_min]} vs {labels[j_min]}  ρ={rho[i_min,j_min]:+.3f}")

    # --- Heatmap ---
    _plot_heatmap(rho, pval, labels, HEATMAP_PATH)


if __name__ == "__main__":
    main()
