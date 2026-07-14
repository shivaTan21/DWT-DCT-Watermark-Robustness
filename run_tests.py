"""
Comprehensive test suite for the DWT-DCT watermarking pipeline.
Run with: .venv/bin/python run_tests.py
"""

import os
import sys
import subprocess
import traceback

import cv2
import numpy as np
import pandas as pd

# Ensure src is importable from project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.watermark import (
    generate_watermark,
    embed_watermark_rgb,
    extract_watermark_rgb,
    embed_watermark_svd_hh_rgb,
    extract_watermark_svd_hh_rgb,
)
from src.metrics import normalized_correlation, bit_error_rate, psnr, ssim
from src import config
from src.attacks_traditional import get_traditional_attacks
from src.attacks_ai import get_ai_attacks

IMAGES_DIR = config.ORIGINAL_IMAGES_DIR
PLOTS_DIR = config.PLOTS_DIR
METRICS_CSV = config.METRICS_CSV_PATH

# Per-test pass/fail trackers
test_results = {}


def load_image(name):
    """Load and return an image as float64 RGB (512×512)."""
    path = os.path.join(IMAGES_DIR, f"{name}.png")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.resize(img, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    return img


# ---------------------------------------------------------------------------
# TEST 1 — Embed/Extract sanity for all three variants
# ---------------------------------------------------------------------------
def test1_sanity():
    print("\n" + "=" * 68)
    print("TEST 1: Watermark embed/extract sanity for all three variants")
    print("=" * 68)

    images = ["kodim01", "kodim05", "kodim10", "t001", "t017"]
    variants = {
        "standard":            dict(use_stable_positions=False, use_optimized_positions=False),
        "stable_positions":    dict(use_stable_positions=True,  use_optimized_positions=False),
        "optimized_positions": dict(use_stable_positions=False, use_optimized_positions=True),
    }

    wm = generate_watermark(shape=(32, 32), seed=42)
    all_pass = True

    header = f"{'Image':<12} {'Variant':<22} {'NC':>8} {'PSNR':>8} {'Result':>6}"
    print(header)
    print("-" * len(header))

    for variant_name, kwargs in variants.items():
        for img_name in images:
            try:
                img = load_image(img_name)
                watermarked = embed_watermark_rgb(img, wm, alpha=0.1, **kwargs)
                watermarked_u8 = np.clip(watermarked, 0, 255).astype(np.uint8)
                extracted = extract_watermark_rgb(
                    watermarked_u8.astype(np.float64), wm.shape, **kwargs
                )
                nc_val  = normalized_correlation(wm, extracted)
                psnr_val = psnr(img, watermarked_u8.astype(np.float64))

                ok = (nc_val > 0.99) and (psnr_val > 40.0)
                status = "PASS" if ok else "FAIL"
                if not ok:
                    all_pass = False
                    reasons = []
                    if nc_val <= 0.99:
                        reasons.append(f"NC={nc_val:.4f}<=0.99")
                    if psnr_val <= 40.0:
                        reasons.append(f"PSNR={psnr_val:.2f}<=40")
                    status = f"FAIL ({', '.join(reasons)})"

                print(f"{img_name:<12} {variant_name:<22} {nc_val:>8.4f} {psnr_val:>8.2f} {status}")
            except Exception as e:
                all_pass = False
                print(f"{img_name:<12} {variant_name:<22} {'ERR':>8} {'ERR':>8} FAIL ({e})")

    test_results["Test 1"] = "PASS" if all_pass else "FAIL"
    print(f"\n=> Test 1: {'PASS' if all_pass else 'FAIL'}")


# ---------------------------------------------------------------------------
# TEST 2 — All attacks produce valid output
# ---------------------------------------------------------------------------
def test2_attacks():
    print("\n" + "=" * 68)
    print("TEST 2: All attacks produce valid output (kodim01, alpha=0.1, standard)")
    print("=" * 68)

    img = load_image("kodim01")
    wm  = generate_watermark(shape=(32, 32), seed=42)
    watermarked = embed_watermark_rgb(img, wm, alpha=0.1,
                                      use_stable_positions=False,
                                      use_optimized_positions=False)
    watermarked_u8 = np.clip(watermarked, 0, 255).astype(np.uint8).astype(np.float64)

    # Build attack dict via the same helpers run_experiment.py uses
    attacks_combined = {}
    attacks_combined.update(get_traditional_attacks(seed=config.RANDOM_SEED))
    attacks_combined.update(get_ai_attacks())
    attacks_ordered = list(attacks_combined.items())

    print(f"Attacks available via get_traditional_attacks() + get_ai_attacks(): "
          f"{[n for n, _ in attacks_ordered]}")

    col_w = [26, 12, 8, 8, 8, 8, 6]
    header = (f"{'Attack':<26} {'Shape':<12} {'NC':>8} {'BER':>8} "
              f"{'PSNR':>8} {'SSIM':>8} {'Result':>6}")
    print(header)
    print("-" * len(header))

    all_pass = True
    for attack_name, attack_fn in attacks_ordered:
        try:
            attacked = attack_fn(watermarked_u8.copy())
            shape    = attacked.shape

            # Shape check
            shape_ok = (shape == (512, 512, 3))

            # Not-identical check
            not_identical = not np.array_equal(
                attacked.astype(np.uint8),
                watermarked_u8.astype(np.uint8)
            )

            # Metrics
            extracted = extract_watermark_rgb(
                attacked, wm.shape,
                use_stable_positions=False, use_optimized_positions=False
            )
            nc_val  = normalized_correlation(wm, extracted)
            ber_val = bit_error_rate(wm, extracted)

            # Validity ranges
            nc_valid  = -1.0 <= nc_val  <= 1.0
            ber_valid = 0.0  <= ber_val <= 1.0

            # PSNR/SSIM (against watermarked image as reference)
            psnr_val = psnr(watermarked_u8, attacked)
            ssim_val = ssim(watermarked_u8, attacked)
            psnr_ok  = psnr_val > 0.0
            ssim_ok  = ssim_val > 0.0

            ok = shape_ok and not_identical and nc_valid and ber_valid and psnr_ok and ssim_ok
            reasons = []
            if not shape_ok:    reasons.append(f"shape={shape}")
            if not not_identical: reasons.append("output==input")
            if not nc_valid:    reasons.append(f"NC={nc_val:.4f} OOB")
            if not ber_valid:   reasons.append(f"BER={ber_val:.4f} OOB")
            if not psnr_ok:     reasons.append(f"PSNR={psnr_val:.2f}<=0")
            if not ssim_ok:     reasons.append(f"SSIM={ssim_val:.4f}<=0")

            status = "PASS" if ok else f"FAIL ({'; '.join(reasons)})"
            if not ok:
                all_pass = False

            print(f"{attack_name:<26} {str(shape):<12} {nc_val:>8.4f} {ber_val:>8.4f} "
                  f"{psnr_val:>8.2f} {ssim_val:>8.4f} {status}")

        except Exception as e:
            all_pass = False
            print(f"{attack_name:<26} {'ERR':<12} {'ERR':>8} {'ERR':>8} "
                  f"{'ERR':>8} {'ERR':>8} FAIL ({type(e).__name__}: {e})")

    test_results["Test 2"] = "PASS" if all_pass else "FAIL"
    print(f"\n=> Test 2: {'PASS' if all_pass else 'FAIL'}")


# ---------------------------------------------------------------------------
# TEST 3 — CSV integrity
# ---------------------------------------------------------------------------
def test3_csv():
    print("\n" + "=" * 68)
    print("TEST 3: CSV integrity check")
    print("=" * 68)

    df = pd.read_csv(METRICS_CSV)
    total_rows = len(df)
    expected_rows = 10800
    rows_ok = (total_rows == expected_rows)

    expected_variants = {"standard", "stable_positions", "optimized_positions"}
    found_variants = set(df["embedding_variant"].unique())
    variants_ok = expected_variants == found_variants

    nan_nc  = df["nc"].isna().sum()
    nan_ber = df["ber"].isna().sum()
    empty_nc = (df["nc"].astype(str).str.strip() == "").sum() if df["nc"].dtype == object else 0
    empty_ber = (df["ber"].astype(str).str.strip() == "").sum() if df["ber"].dtype == object else 0
    missing_nc  = nan_nc  + empty_nc
    missing_ber = nan_ber + empty_ber
    missing_ok = (missing_nc == 0) and (missing_ber == 0)

    expected_attacks = {
        "watermarked_only", "jpeg_compression_q50", "gaussian_noise_s10",
        "median_filter_3x3", "gaussian_blur_5x5", "crop_resize_80pct",
        "ai_fallback_enhancement", "espcn_x4", "real_esrgan",
    }
    found_attacks = set(df["attack_name"].unique())
    attacks_ok = expected_attacks == found_attacks

    expected_alphas = {0.05, 0.1, 0.2, 0.3}
    found_alphas = set(df["alpha"].unique())
    alphas_ok = expected_alphas == found_alphas

    expected_images = 100
    found_images = df["image"].nunique()
    images_ok = (found_images == expected_images)

    # Per-variant row counts
    variant_counts = df.groupby("embedding_variant").size()

    print(f"\n{'Metric':<40} {'Expected':<12} {'Found':<12} {'OK?':>5}")
    print("-" * 72)
    print(f"{'Total rows':<40} {expected_rows:<12} {total_rows:<12} {'OK' if rows_ok else 'FAIL':>5}")
    print(f"{'Variants present':<40} {'3':<12} {len(found_variants):<12} {'OK' if variants_ok else 'FAIL':>5}")
    if not variants_ok:
        print(f"  Missing: {expected_variants - found_variants}")
        print(f"  Extra:   {found_variants - expected_variants}")
    print(f"{'Missing NC values':<40} {'0':<12} {missing_nc:<12} {'OK' if missing_ok else 'FAIL':>5}")
    print(f"{'Missing BER values':<40} {'0':<12} {missing_ber:<12} {'OK' if missing_ok else 'FAIL':>5}")
    print(f"{'Expected attacks present':<40} {len(expected_attacks):<12} {len(found_attacks):<12} {'OK' if attacks_ok else 'FAIL':>5}")
    if not attacks_ok:
        print(f"  Missing: {expected_attacks - found_attacks}")
        print(f"  Extra:   {found_attacks - expected_attacks}")
    print(f"{'Alpha values':<40} {'4':<12} {len(found_alphas):<12} {'OK' if alphas_ok else 'FAIL':>5}")
    if not alphas_ok:
        print(f"  Missing: {expected_alphas - found_alphas}")
    print(f"{'Unique images per variant':<40} {'100':<12} {found_images:<12} {'OK' if images_ok else 'FAIL':>5}")

    print("\nRow counts per variant:")
    for vname in sorted(expected_variants):
        count = variant_counts.get(vname, 0)
        print(f"  {vname:<28} {count}")

    all_ok = rows_ok and variants_ok and missing_ok and attacks_ok and alphas_ok and images_ok
    test_results["Test 3"] = "PASS" if all_ok else "FAIL"
    print(f"\n=> Test 3: {'PASS' if all_ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# TEST 4 — Optimized positions vs standard for Real-ESRGAN
# ---------------------------------------------------------------------------
def test4_optimized_vs_standard():
    print("\n" + "=" * 68)
    print("TEST 4: Optimized positions vs standard — Real-ESRGAN NC comparison")
    print("=" * 68)

    df = pd.read_csv(METRICS_CSV)
    esrgan = df[df["attack_name"] == "real_esrgan"]

    std_nc  = esrgan[esrgan["embedding_variant"] == "standard"]["nc"].mean()
    opt_nc  = esrgan[esrgan["embedding_variant"] == "optimized_positions"]["nc"].mean()

    expected_std = 0.795
    expected_opt = 0.833

    print(f"\n{'Variant':<28} {'Mean NC (CSV)':>14} {'Expected':>10}")
    print("-" * 55)
    print(f"{'standard':<28} {std_nc:>14.4f} {expected_std:>10.3f}")
    print(f"{'optimized_positions':<28} {opt_nc:>14.4f} {expected_opt:>10.3f}")

    direction_ok = opt_nc > std_nc
    print(f"\nDirection check (optimized_NC > standard_NC): "
          f"{opt_nc:.4f} > {std_nc:.4f} → {'PASS' if direction_ok else 'FAIL'}")

    test_results["Test 4"] = "PASS" if direction_ok else "FAIL"
    print(f"\n=> Test 4: {'PASS' if direction_ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# TEST 5 — DWT-SVD-HH sanity
# ---------------------------------------------------------------------------
def test5_svd_hh():
    print("\n" + "=" * 68)
    print("TEST 5: DWT-SVD-HH embed→extract sanity (kodim01, delta=30.0)")
    print("=" * 68)

    try:
        img = load_image("kodim01")
        wm  = generate_watermark(shape=(8, 16), seed=42)   # 128 bits

        watermarked = embed_watermark_svd_hh_rgb(img, wm, delta=30.0)
        watermarked_u8 = np.clip(watermarked, 0, 255).astype(np.uint8)

        extracted = extract_watermark_svd_hh_rgb(
            watermarked_u8.astype(np.float64), wm.shape
        )

        nc_val   = normalized_correlation(wm, extracted)
        psnr_val = psnr(img, watermarked_u8.astype(np.float64))

        nc_ok   = nc_val > 0.99
        psnr_ok = psnr_val > 40.0

        print(f"\nNC   = {nc_val:.6f}  (threshold > 0.99) → {'OK' if nc_ok else 'FAIL'}")
        print(f"PSNR = {psnr_val:.2f} dB  (threshold > 40 dB) → {'OK' if psnr_ok else 'FAIL'}")

        reasons = []
        if not nc_ok:   reasons.append(f"NC={nc_val:.4f}<=0.99")
        if not psnr_ok: reasons.append(f"PSNR={psnr_val:.2f}<=40")
        ok = nc_ok and psnr_ok

        test_results["Test 5"] = "PASS" if ok else f"FAIL ({', '.join(reasons)})"
        print(f"\n=> Test 5: {'PASS' if ok else f'FAIL ({chr(34).join(reasons)})'}")

    except Exception as e:
        test_results["Test 5"] = "FAIL"
        print(f"ERROR: {traceback.format_exc()}")
        print(f"\n=> Test 5: FAIL ({e})")


# ---------------------------------------------------------------------------
# TEST 6 — Plot generation
# ---------------------------------------------------------------------------
def test6_plots():
    print("\n" + "=" * 68)
    print("TEST 6: Plot generation")
    print("=" * 68)

    # Run plot_results.py
    print("\nRunning: python -m src.plot_results ...")
    r1 = subprocess.run(
        [sys.executable, "-m", "src.plot_results"],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=120
    )
    if r1.returncode != 0:
        print(f"  STDERR: {r1.stderr[-2000:]}")
        print("  plot_results.py FAILED")
    else:
        print("  plot_results.py completed without error")

    # Run plot_comparison.py
    print("\nRunning: python -m src.plot_comparison ...")
    r2 = subprocess.run(
        [sys.executable, "-m", "src.plot_comparison"],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=120
    )
    if r2.returncode != 0:
        print(f"  STDERR: {r2.stderr[-2000:]}")
        print("  plot_comparison.py FAILED")
    else:
        print("  plot_comparison.py completed without error")

    scripts_ok = (r1.returncode == 0) and (r2.returncode == 0)

    expected_files = [
        "nc_by_attack.png",
        "ssim_vs_nc.png",
        "psnr_vs_nc.png",
        "ber_by_attack.png",
        "dct_stability_heatmap_realesrgan.png",
        "dct_stability_heatmap_jpeg.png",
        "dct_stability_comparison.png",
        "frequency_sweep_heatmap.png",
        "frequency_sweep_scatter.png",
        "symmetry_analysis.png",
        "comparison_nc_by_attack.png",
        "comparison_psnr_imperceptibility.png",
    ]

    print(f"\n{'File':<48} {'Size (bytes)':>14} {'Status':>10}")
    print("-" * 75)

    all_files_ok = True
    for fname in expected_files:
        fpath = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            flag = "WARN(small)" if size < 10_000 else "OK"
            if size < 10_000:
                all_files_ok = False
            print(f"{fname:<48} {size:>14,} {flag:>10}")
        else:
            all_files_ok = False
            print(f"{fname:<48} {'MISSING':>14} {'FAIL':>10}")

    all_ok = scripts_ok and all_files_ok
    test_results["Test 6"] = "PASS" if all_ok else "FAIL"
    print(f"\n=> Test 6: {'PASS' if all_ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# TEST 7 — Reproducibility
# ---------------------------------------------------------------------------
def test7_reproducibility():
    print("\n" + "=" * 68)
    print("TEST 7: Reproducibility (same seed → pixel-identical output)")
    print("=" * 68)

    img = load_image("kodim01")
    wm  = generate_watermark(shape=(32, 32), seed=42)

    w1 = embed_watermark_rgb(img, wm, alpha=0.1,
                              use_stable_positions=False,
                              use_optimized_positions=False)
    w2 = embed_watermark_rgb(img, wm, alpha=0.1,
                              use_stable_positions=False,
                              use_optimized_positions=False)

    identical = np.array_equal(w1, w2)
    print(f"\nRun 1 shape: {w1.shape}, dtype: {w1.dtype}")
    print(f"Run 2 shape: {w2.shape}, dtype: {w2.dtype}")
    print(f"Pixel-identical: {identical}")
    if not identical:
        diff = np.abs(w1 - w2)
        print(f"Max pixel diff: {diff.max():.6f}")

    test_results["Test 7"] = "PASS" if identical else "FAIL"
    print(f"\n=> Test 7: {'PASS' if identical else 'FAIL'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("DWT-DCT Watermarking Pipeline — Comprehensive Test Suite")
    print(f"Project root: {PROJECT_ROOT}")

    test1_sanity()
    test2_attacks()
    test3_csv()
    test4_optimized_vs_standard()
    test5_svd_hh()
    test6_plots()
    test7_reproducibility()

    print("\n\n" + "=" * 40)
    print("=== TEST SUMMARY ===")
    print("=" * 40)
    labels = {
        "Test 1": "Test 1 (Sanity checks):    ",
        "Test 2": "Test 2 (Attack outputs):   ",
        "Test 3": "Test 3 (CSV integrity):    ",
        "Test 4": "Test 4 (Optimized vs std): ",
        "Test 5": "Test 5 (SVD-HH sanity):    ",
        "Test 6": "Test 6 (Plot generation):  ",
        "Test 7": "Test 7 (Reproducibility):  ",
    }
    all_pass = True
    for key, label in labels.items():
        result = test_results.get(key, "NOT RUN")
        print(f"{label}{result}")
        if result != "PASS":
            all_pass = False

    print(f"{'Overall:':35}{'PASS' if all_pass else 'FAIL'}")
    print("=" * 40)
