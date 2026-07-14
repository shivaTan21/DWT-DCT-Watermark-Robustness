"""
Focused comparison: DWT-DCT-LL vs DWT-SVD-HH watermarking schemes.

Scope: all 100 images in data/original_images/, alpha=0.1 only,
attacks: real_esrgan / jpeg_compression_q50 / gaussian_blur_5x5.

Output: results/comparison_metrics.csv

Run from the project root:
    python -m src.run_comparison
"""

import glob
import os

import cv2
import numpy as np
import pandas as pd

from . import config
from .attacks_ai import get_ai_attacks
from .attacks_traditional import get_traditional_attacks
from .metrics import bit_error_rate, normalized_correlation, psnr, ssim
from .watermark import (
    ABSOLUTE_MARGIN_FLOOR,
    _svd_hh_capacity,
    embed_watermark_rgb,
    embed_watermark_svd_hh_rgb,
    extract_watermark_rgb,
    extract_watermark_svd_hh_rgb,
    generate_watermark,
    sanity_check_svd_hh,
)
import pywt

COMPARISON_ALPHA = 0.1
COMPARISON_ATTACKS = {"real_esrgan", "jpeg_compression_q50", "gaussian_blur_5x5"}
COMPARISON_CSV = os.path.join(config.RESULTS_DIR, "comparison_metrics.csv")
WM_SHAPE = (8, 16)   # 128 bits — fits both SVD-HH and DWT-DCT capacity


def _load_preprocess(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    resized = cv2.resize(img_bgr, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float64)


def _load_images():
    paths = sorted(
        glob.glob(os.path.join(config.ORIGINAL_IMAGES_DIR, "*.png"))
        + glob.glob(os.path.join(config.ORIGINAL_IMAGES_DIR, "*.jpg"))
        + glob.glob(os.path.join(config.ORIGINAL_IMAGES_DIR, "*.jpeg"))
    )
    if not paths:
        raise RuntimeError(f"No images found in {config.ORIGINAL_IMAGES_DIR}")
    return {os.path.splitext(os.path.basename(p))[0]: _load_preprocess(p) for p in paths}


def _run_sanity_checks(images, wm):
    print("\n--- SVD-HH sanity checks ---")
    names = ["kodim01", "kodim05", "t017"]
    all_pass = True
    for name in names:
        if name not in images:
            print(f"  {name}: NOT FOUND in dataset, skipping")
            continue
        nc = sanity_check_svd_hh(images[name], wm, alpha=COMPARISON_ALPHA)
        status = "PASS" if nc > config.SANITY_CHECK_NC_THRESHOLD else "FAIL"
        print(f"  {name}: NC={nc:.4f} -> {status}")
        if nc <= config.SANITY_CHECK_NC_THRESHOLD:
            all_pass = False
    if not all_pass:
        raise RuntimeError(
            "SVD-HH sanity check FAILED on one or more images. "
            "Refusing to run the comparison pipeline."
        )
    print("  All sanity checks PASSED\n")


def _check_capacity(images, wm):
    sample = next(iter(images.values()))
    from .watermark import rgb_to_ycbcr
    y = rgb_to_ycbcr(sample)[..., 0]
    _, (_, _, HH) = pywt.dwt2(y, config.DWT_WAVELET)
    cap = _svd_hh_capacity(HH.shape)
    print(f"SVD-HH capacity: {cap} bits (HH shape {HH.shape}, block size from SVD_HH_BLOCK_SIZE)")
    if cap < 128:
        raise RuntimeError(f"SVD-HH capacity {cap} < 128 bits — image too small.")
    if wm.size > cap:
        raise RuntimeError(f"Watermark ({wm.size} bits) exceeds SVD-HH capacity ({cap} bits).")
    print(f"Watermark size: {wm.size} bits -> OK\n")


def run_comparison():
    np.random.seed(config.RANDOM_SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    print("Loading images...")
    images = _load_images()
    print(f"  {len(images)} images loaded from {config.ORIGINAL_IMAGES_DIR}")

    wm = generate_watermark(shape=WM_SHAPE, seed=config.RANDOM_SEED)

    _check_capacity(images, wm)
    _run_sanity_checks(images, wm)

    # Filter attacks to the three specified ones
    all_traditional = get_traditional_attacks(seed=config.RANDOM_SEED)
    all_ai = get_ai_attacks()
    attacks = {
        k: v
        for k, v in {**all_traditional, **all_ai}.items()
        if k in COMPARISON_ATTACKS
    }
    available = set(attacks)
    missing = COMPARISON_ATTACKS - available
    if missing:
        print(f"  [note] attacks not available and will be skipped: {missing}")
    print(f"  Running attacks: {sorted(available)}\n")

    rows = []

    for scheme_name, embed_fn, extract_fn in [
        (
            "dwt_dct_ll",
            lambda img, w: embed_watermark_rgb(img, w, alpha=COMPARISON_ALPHA),
            lambda img, shape: extract_watermark_rgb(img, shape),
        ),
        (
            "dwt_svd_hh",
            lambda img, w: embed_watermark_svd_hh_rgb(img, w, delta=ABSOLUTE_MARGIN_FLOOR),
            lambda img, shape: extract_watermark_svd_hh_rgb(img, shape),
        ),
    ]:
        print(f"=== Scheme: {scheme_name} ===")
        for image_name, original in images.items():
            watermarked = embed_fn(original, wm)
            watermarked_u8 = np.clip(watermarked, 0, 255).astype(np.uint8)
            watermarked_f = watermarked_u8.astype(np.float64)

            extracted_clean = extract_fn(watermarked_f, WM_SHAPE)
            rows.append({
                "scheme": scheme_name,
                "image": image_name,
                "alpha": COMPARISON_ALPHA,
                "attack_name": "watermarked_only",
                "nc": normalized_correlation(wm, extracted_clean),
                "ber": bit_error_rate(wm, extracted_clean),
                "psnr_after_attack": psnr(original, watermarked_f),
                "ssim_after_attack": ssim(original, watermarked_f),
            })

            for attack_name, attack_fn in attacks.items():
                try:
                    attacked = attack_fn(watermarked_f)
                except Exception as exc:
                    print(f"  [skip] {attack_name} on {image_name}: {exc}")
                    continue
                extracted = extract_fn(attacked, WM_SHAPE)
                rows.append({
                    "scheme": scheme_name,
                    "image": image_name,
                    "alpha": COMPARISON_ALPHA,
                    "attack_name": attack_name,
                    "nc": normalized_correlation(wm, extracted),
                    "ber": bit_error_rate(wm, extracted),
                    "psnr_after_attack": psnr(original, attacked),
                    "ssim_after_attack": ssim(original, attacked),
                })

        print(f"  Done. {len([r for r in rows if r['scheme']==scheme_name])} rows\n")

    df = pd.DataFrame(rows)
    df.to_csv(COMPARISON_CSV, index=False)
    print(f"Wrote {len(df)} rows to {COMPARISON_CSV}")
    return df


if __name__ == "__main__":
    df = run_comparison()
    print("\n--- Mean NC by scheme and attack ---")
    summary = (
        df[df["attack_name"] != "watermarked_only"]
        .groupby(["scheme", "attack_name"])["nc"]
        .mean()
        .round(4)
        .unstack("scheme")
    )
    print(summary.to_string())
