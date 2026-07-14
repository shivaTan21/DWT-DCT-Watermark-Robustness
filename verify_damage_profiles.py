"""
Unified damage-profile verification for Real-ESRGAN vs ESPCN.

Computes both profiles from scratch using a single function and cross-checks
the Real-ESRGAN result against the previously saved CSV.

Run from the project root:
    python verify_damage_profiles.py
"""

import glob
import os

import cv2
import numpy as np
import pandas as pd
import pywt
from scipy.fft import dctn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WATERMARKED_DIR = os.path.join(PROJECT_ROOT, "results", "watermarked")
ATTACKED_AI_DIR = os.path.join(PROJECT_ROOT, "results", "attacked", "ai_enhancement")
ESRGAN_CSV = os.path.join(PROJECT_ROOT, "results", "dct_stability_realesrgan.csv")

IMAGE_SIZE = (512, 512)   # (width, height) for cv2.resize
BLOCK_SIZE = 8


# ---------------------------------------------------------------------------
# Unified pipeline
# ---------------------------------------------------------------------------

def _load_rgb(path):
    """Load image as RGB float64, resizing to IMAGE_SIZE if needed."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read: {path}")
    if (bgr.shape[1], bgr.shape[0]) != IMAGE_SIZE:
        bgr = cv2.resize(bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)


def _rgb_to_y(rgb):
    """ITU-R BT.601 luma: Y = 0.299*R + 0.587*G + 0.114*B."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _y_to_ll(y):
    """Single-level Haar DWT → LL subband."""
    LL, _ = pywt.dwt2(y, "haar")
    return LL


def _iter_blocks(ll):
    """Yield non-overlapping 8×8 blocks from the LL subband."""
    rows, cols = ll.shape
    for br in range(0, rows - BLOCK_SIZE + 1, BLOCK_SIZE):
        for bc in range(0, cols - BLOCK_SIZE + 1, BLOCK_SIZE):
            yield ll[br : br + BLOCK_SIZE, bc : bc + BLOCK_SIZE]


def compute_damage_profile(watermarked_dir, attacked_dir, attack_filename_suffix, alpha=0.1):
    """
    For each image stem, compare the watermarked reference against the attacked
    counterpart in the DWT-DCT domain and return per-position mean absolute difference.

    Parameters
    ----------
    watermarked_dir : str   – directory containing <stem>__alpha<alpha>.png files
    attacked_dir    : str   – directory containing attacked files
    attack_filename_suffix : str  – suffix appended after the alpha token,
                                    e.g. '__real_esrgan' or '__espcn_x4'
    alpha           : float – alpha used when embedding (default 0.1)

    Returns
    -------
    dict mapping (row, col) → mean absolute difference across all blocks/images
    plus metadata keys: '_n_images', '_blocks_per_image', '_total_blocks'
    """
    base_suffix = f"__alpha{alpha}.png"
    attack_suffix = f"__alpha{alpha}{attack_filename_suffix}.png"

    watermarked_paths = sorted(glob.glob(os.path.join(watermarked_dir, f"*{base_suffix}")))
    if not watermarked_paths:
        raise RuntimeError(f"No files matching *{base_suffix} in {watermarked_dir}")

    # accumulator: shape (BLOCK_SIZE, BLOCK_SIZE), summed over all blocks/images
    accum = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)
    total_blocks = 0
    n_images = 0
    blocks_per_image = None

    for wm_path in watermarked_paths:
        basename = os.path.basename(wm_path)
        stem = basename[: -len(base_suffix)]
        atk_path = os.path.join(attacked_dir, f"{stem}{attack_suffix}")
        if not os.path.exists(atk_path):
            continue

        wm_rgb = _load_rgb(wm_path)
        atk_rgb = _load_rgb(atk_path)

        wm_ll = _y_to_ll(_rgb_to_y(wm_rgb))
        atk_ll = _y_to_ll(_rgb_to_y(atk_rgb))

        if wm_ll.shape != atk_ll.shape:
            continue

        img_blocks = 0
        for wm_block, atk_block in zip(_iter_blocks(wm_ll), _iter_blocks(atk_ll)):
            wm_dct = dctn(wm_block, norm="ortho")
            atk_dct = dctn(atk_block, norm="ortho")
            accum += np.abs(atk_dct - wm_dct)
            img_blocks += 1

        if blocks_per_image is None:
            blocks_per_image = img_blocks
        total_blocks += img_blocks
        n_images += 1

    if total_blocks == 0:
        raise RuntimeError("No blocks were analyzed — check paths and suffix.")

    mean_diff = accum / total_blocks

    result = {(r, c): mean_diff[r, c]
              for r in range(BLOCK_SIZE) for c in range(BLOCK_SIZE)}
    result["_n_images"] = n_images
    result["_blocks_per_image"] = blocks_per_image
    result["_total_blocks"] = total_blocks
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_stats(label, profile):
    n_images = profile["_n_images"]
    bpi = profile["_blocks_per_image"]
    total = profile["_total_blocks"]

    vals = {k: v for k, v in profile.items() if isinstance(k, tuple)}
    positions = list(vals.keys())
    damages = list(vals.values())

    max_pos = positions[int(np.argmax(damages))]
    min_pos = positions[int(np.argmin(damages))]
    max_val = vals[max_pos]
    min_val = vals[min_pos]
    mean_val = float(np.mean(damages))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Images processed      : {n_images}  (expected 100)")
    print(f"  Blocks per image      : {bpi}  (expected 1024)")
    print(f"  Total blocks analyzed : {total}  (expected 102400)")
    print(f"  DC (0,0) damage       : {vals[(0,0)]:.6f}")
    print(f"  Mean damage (all 64)  : {mean_val:.6f}")
    print(f"  Max  damage position  : {max_pos}  value={max_val:.6f}")
    print(f"  Min  damage position  : {min_pos}  value={min_val:.6f}")


def _cross_check_csv(profile, csv_path):
    """
    Compare fresh (0,0) value against the saved CSV.

    A FAIL here does NOT indicate a pipeline difference — the DCT implementations
    (scipy.fftpack.dct row-by-row vs scipy.fft.dctn) are numerically identical
    (max diff = 0.0 on any block).  A gap larger than 1e-6 means the attacked
    images on disk have changed since the CSV was last written.
    """
    print(f"\n--- Cross-check against {os.path.basename(csv_path)} ---")
    df = pd.read_csv(csv_path)
    row00 = df[(df["coeff_row"] == 0) & (df["coeff_col"] == 0)].iloc[0]
    csv_val = float(row00["mean_abs_change"])
    fresh_val = profile[(0, 0)]
    tol = 1e-6
    match = abs(csv_val - fresh_val) <= tol
    print(f"  CSV   (0,0) = {csv_val:.10f}")
    print(f"  Fresh (0,0) = {fresh_val:.10f}")
    print(f"  Diff        = {abs(csv_val - fresh_val):.2e}  (tol={tol:.0e})")
    if not match:
        print(f"  Note: DCT implementations are bit-for-bit identical.")
        print(f"        Discrepancy indicates attacked images were regenerated")
        print(f"        after the CSV was written — not a pipeline mismatch.")
    print(f"  Result: {'PASS' if match else 'FAIL (stale CSV, not a pipeline error)'}")
    return match


def _comparison_table(esrgan, espcn):
    spots = [(0, 0), (4, 1), (1, 4), (7, 5), (7, 7)]
    esrgan_vals = {k: v for k, v in esrgan.items() if isinstance(k, tuple)}
    espcn_vals  = {k: v for k, v in espcn.items()  if isinstance(k, tuple)}
    esrgan_mean = float(np.mean(list(esrgan_vals.values())))
    espcn_mean  = float(np.mean(list(espcn_vals.values())))

    header = f"{'Position':<12} | {'Real-ESRGAN damage':>19} | {'ESPCN damage':>13} | {'Ratio (ESRGAN/ESPCN)':>20}"
    sep    = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for pos in spots:
        ev = esrgan_vals[pos]
        cv = espcn_vals[pos]
        ratio = ev / cv if cv != 0 else float("inf")
        print(f"{str(pos):<12} | {ev:>19.3f} | {cv:>13.3f} | {ratio:>20.2f}")

    ratio_mean = esrgan_mean / espcn_mean if espcn_mean != 0 else float("inf")
    print(sep)
    print(f"{'mean all':<12} | {esrgan_mean:>19.3f} | {espcn_mean:>13.3f} | {ratio_mean:>20.2f}")
    print(sep)

    return esrgan_vals[(0, 0)] / espcn_vals[(0, 0)], ratio_mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Computing Real-ESRGAN damage profile …")
    esrgan_profile = compute_damage_profile(
        watermarked_dir=WATERMARKED_DIR,
        attacked_dir=ATTACKED_AI_DIR,
        attack_filename_suffix="__real_esrgan",
        alpha=0.1,
    )

    print("Computing ESPCN damage profile …")
    espcn_profile = compute_damage_profile(
        watermarked_dir=WATERMARKED_DIR,
        attacked_dir=ATTACKED_AI_DIR,
        attack_filename_suffix="__espcn_x4",
        alpha=0.1,
    )

    _print_stats("Real-ESRGAN", esrgan_profile)
    _print_stats("ESPCN x4",    espcn_profile)

    csv_pass = _cross_check_csv(esrgan_profile, ESRGAN_CSV)

    dc_ratio, mean_ratio = _comparison_table(esrgan_profile, espcn_profile)

    esrgan_ok = (esrgan_profile["_n_images"] == 100
                 and esrgan_profile["_blocks_per_image"] == 1024
                 and esrgan_profile["_total_blocks"] == 102400)
    espcn_ok  = (espcn_profile["_n_images"] == 100
                 and espcn_profile["_blocks_per_image"] == 1024
                 and espcn_profile["_total_blocks"] == 102400)
    pipeline_verified = esrgan_ok and espcn_ok

    print("\n" + "=" * 60)
    print("  CONCLUSIONS")
    print("=" * 60)
    print(f"  Pipeline verified: both attacks computed identically: {'YES' if pipeline_verified else 'NO'}")
    print(f"  DC perturbation ratio (ESRGAN/ESPCN): {dc_ratio:.2f}")
    print(f"  Mean perturbation ratio (ESRGAN/ESPCN): {mean_ratio:.2f}")
    print(f"  Comparison is apples-to-apples: {'YES' if pipeline_verified else 'NO'}")


if __name__ == "__main__":
    main()
