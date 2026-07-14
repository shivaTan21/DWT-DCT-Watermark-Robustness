"""
Main entry point for the DWT-DCT watermark robustness experiment.

Pipeline: load+preprocess images -> sanity-check the embed/extract codec ->
embed watermarks at one or more alpha strengths -> apply traditional and
AI-enhancement attacks -> extract the watermark from every variant ->
write all per-image, per-attack metrics to results/metrics.csv.

The full pipeline is run twice -- once with the standard embedding positions
(COEFF_POS_1/COEFF_POS_2) and once with the empirically stable positions
from src/analyze_dct_stability.py (COEFF_POS_1_STABLE/COEFF_POS_2_STABLE) --
using identical images, watermark bits, alphas, attacks, and random seed.
Both variants are tagged via the `embedding_variant` column and appended to
the same results/metrics.csv so they can be compared directly.

Run from the project root:
    python -m src.run_experiment

Then generate the summary table and plots with:
    python -m src.plot_results
"""

import argparse
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
    DEFAULT_ALPHA,
    embed_watermark_rgb,
    extract_watermark_rgb,
    generate_watermark,
    load_watermark_image,
    COEFF_POS_1_OPTIMIZED,
    COEFF_POS_2_OPTIMIZED,
)


def load_and_preprocess_images(input_dir, output_dir, size=config.IMAGE_SIZE):
    """Resize every PNG/JPG in `input_dir` to RGB `size` and save to `output_dir`."""
    paths = sorted(
        glob.glob(os.path.join(input_dir, "*.png"))
        + glob.glob(os.path.join(input_dir, "*.jpg"))
        + glob.glob(os.path.join(input_dir, "*.jpeg"))
    )
    if not paths:
        raise RuntimeError(
            f"No PNG/JPG images found in {input_dir}. Add at least one image before running the experiment."
        )

    processed = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Could not read image: {path}")
        resized_bgr = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), resized_bgr)
        processed[name] = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    return processed


def get_watermark(seed=config.RANDOM_SEED, shape=config.WATERMARK_SIZE):
    if os.path.exists(config.WATERMARK_IMAGE_PATH):
        return load_watermark_image(config.WATERMARK_IMAGE_PATH, shape=shape)
    return generate_watermark(shape=shape, seed=seed)


def run_sanity_check(sample_image, watermark, alpha=DEFAULT_ALPHA, use_stable_positions=False,
                     use_optimized_positions=False):
    """
    Hard-stop sanity check: embed -> save as 8-bit PNG -> extract on a clean
    (unattacked) image must recover the watermark with NC > threshold. The
    8-bit quantization step is included because that's what every
    downstream attack actually operates on (the saved watermarked PNG), so
    this check needs to mirror the real pipeline, not an idealized
    float64-only round trip.

    If this fails, the embed/extract codec itself is broken and no attack
    results downstream can be trusted -- so we raise instead of merely
    printing a warning.
    """
    watermarked = embed_watermark_rgb(sample_image, watermark, alpha=alpha,
                                      use_stable_positions=use_stable_positions,
                                      use_optimized_positions=use_optimized_positions)
    watermarked_u8 = np.clip(watermarked, 0, 255).astype(np.uint8)
    extracted = extract_watermark_rgb(watermarked_u8.astype(np.float64), watermark.shape,
                                      use_stable_positions=use_stable_positions,
                                      use_optimized_positions=use_optimized_positions)
    nc = normalized_correlation(watermark, extracted)
    if nc <= config.SANITY_CHECK_NC_THRESHOLD:
        raise RuntimeError(
            f"Sanity check FAILED: embed->extract NC={nc:.4f} on an unattacked "
            f"image, below the required threshold of {config.SANITY_CHECK_NC_THRESHOLD}. "
            f"The watermark codec is broken; refusing to run the attack pipeline."
        )
    print(f"[sanity check] (use_stable_positions={use_stable_positions}, "
          f"use_optimized_positions={use_optimized_positions}) embed->extract NC={nc:.4f} "
          f"(passed, threshold={config.SANITY_CHECK_NC_THRESHOLD})")


def run_experiment(alphas=None, real_esrgan_model_path=None, opencv_sr_model_path=None,
                    embedding_variant="standard", use_stable_positions=False,
                    use_optimized_positions=False):
    """Run the full embed/attack/extract pipeline once, for a single embedding variant."""
    alphas = alphas or config.ALPHA_VALUES
    config.ensure_directories()
    np.random.seed(config.RANDOM_SEED)

    processed_images = load_and_preprocess_images(config.ORIGINAL_IMAGES_DIR, config.PROCESSED_IMAGES_DIR)
    watermark = get_watermark()

    sample_name, sample_image = next(iter(processed_images.items()))
    run_sanity_check(sample_image, watermark, alpha=DEFAULT_ALPHA,
                     use_stable_positions=use_stable_positions,
                     use_optimized_positions=use_optimized_positions)

    traditional_attacks = get_traditional_attacks(seed=config.RANDOM_SEED)
    ai_attacks = get_ai_attacks(
        real_esrgan_model_path=real_esrgan_model_path,
        opencv_sr_model_path=opencv_sr_model_path,
    )
    print(f"[{embedding_variant}] Traditional attacks: {list(traditional_attacks)}")
    print(f"[{embedding_variant}] AI attacks: {list(ai_attacks)} "
          f"(install realesrgan/basicsr or opencv-contrib-python + pass model paths for more)")

    # Files for the standard variant keep their original (unsuffixed) names so
    # earlier scripts/analyses that read results/watermarked and
    # results/attacked/* by that naming convention (e.g. analyze_dct_stability.py)
    # keep working unchanged; the stable_positions variant writes alongside
    # under a suffixed name instead of overwriting them.
    variant_suffix = "" if embedding_variant == "standard" else f"__{embedding_variant}"

    rows = []
    for image_name, original in processed_images.items():
        for alpha in alphas:
            watermarked = embed_watermark_rgb(original, watermark, alpha=alpha,
                                               use_stable_positions=use_stable_positions,
                                               use_optimized_positions=use_optimized_positions)
            watermarked_u8 = np.clip(watermarked, 0, 255).astype(np.uint8)
            watermarked_bgr = cv2.cvtColor(watermarked_u8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(config.WATERMARKED_DIR, f"{image_name}__alpha{alpha}{variant_suffix}.png"),
                watermarked_bgr,
            )
            watermarked_for_use = watermarked_u8.astype(np.float64)

            extracted_clean = extract_watermark_rgb(watermarked_for_use, watermark.shape,
                                                     use_stable_positions=use_stable_positions,
                                                     use_optimized_positions=use_optimized_positions)
            rows.append({
                "image": image_name,
                "alpha": alpha,
                "embedding_variant": embedding_variant,
                "attack_type": "none",
                "attack_name": "watermarked_only",
                "psnr_watermark": psnr(original, watermarked_for_use),
                "ssim_watermark": ssim(original, watermarked_for_use),
                "psnr_after_attack": psnr(original, watermarked_for_use),
                "ssim_after_attack": ssim(original, watermarked_for_use),
                "nc": normalized_correlation(watermark, extracted_clean),
                "ber": bit_error_rate(watermark, extracted_clean),
            })

            for attack_category, attacks, out_dir in [
                ("traditional", traditional_attacks, config.ATTACKED_TRADITIONAL_DIR),
                ("ai_enhancement", ai_attacks, config.ATTACKED_AI_DIR),
            ]:
                for attack_name, attack_fn in attacks.items():
                    try:
                        attacked = attack_fn(watermarked_for_use)
                    except Exception as exc:
                        print(f"  [skip] {attack_name} on {image_name} (alpha={alpha}): {exc}")
                        continue

                    out_path = os.path.join(
                        out_dir, f"{image_name}__alpha{alpha}{variant_suffix}__{attack_name}.png"
                    )
                    attacked_u8 = np.clip(attacked, 0, 255).astype(np.uint8)
                    cv2.imwrite(out_path, cv2.cvtColor(attacked_u8, cv2.COLOR_RGB2BGR))

                    extracted = extract_watermark_rgb(attacked, watermark.shape,
                                                     use_stable_positions=use_stable_positions,
                                                     use_optimized_positions=use_optimized_positions)
                    rows.append({
                        "image": image_name,
                        "alpha": alpha,
                        "embedding_variant": embedding_variant,
                        "attack_type": attack_category,
                        "attack_name": attack_name,
                        "psnr_watermark": psnr(original, watermarked_for_use),
                        "ssim_watermark": ssim(original, watermarked_for_use),
                        "psnr_after_attack": psnr(original, attacked),
                        "ssim_after_attack": ssim(original, attacked),
                        "nc": normalized_correlation(watermark, extracted),
                        "ber": bit_error_rate(watermark, extracted),
                    })

    return pd.DataFrame(rows)


def run_all_variants(alphas=None, real_esrgan_model_path=None, opencv_sr_model_path=None):
    """
    Run all embedding variants and write results to results/metrics.csv.

    If metrics.csv already exists, only runs variants not yet present and
    appends rows rather than overwriting, so prior results are preserved.
    All three variants: standard, stable_positions, optimized_positions.
    """
    # (name, use_stable_positions, use_optimized_positions)
    all_variants = [
        ("standard",            False, False),
        ("stable_positions",    True,  False),
        ("optimized_positions", False, True),
    ]

    existing_df = None
    existing_variants = set()
    rows_before = 0

    if os.path.exists(config.METRICS_CSV_PATH):
        existing_df = pd.read_csv(config.METRICS_CSV_PATH)
        rows_before = len(existing_df)
        if rows_before > 0 and "embedding_variant" in existing_df.columns:
            existing_variants = set(existing_df["embedding_variant"].unique())
            print(f"[run_all_variants] metrics.csv exists with {rows_before} rows.")
            print(f"[run_all_variants] Variants already present: {sorted(existing_variants)}")

    variants_to_run = [
        (name, use_stable, use_opt)
        for name, use_stable, use_opt in all_variants
        if name not in existing_variants
    ]

    if not variants_to_run:
        print("[run_all_variants] All variants already present -- nothing to run.")
        return existing_df

    new_dfs = []
    for embedding_variant, use_stable_positions, use_optimized_positions in variants_to_run:
        print(f"\n=== Running variant: {embedding_variant} "
              f"(use_stable_positions={use_stable_positions}, "
              f"use_optimized_positions={use_optimized_positions}) ===")
        new_dfs.append(run_experiment(
            alphas=alphas,
            real_esrgan_model_path=real_esrgan_model_path,
            opencv_sr_model_path=opencv_sr_model_path,
            embedding_variant=embedding_variant,
            use_stable_positions=use_stable_positions,
            use_optimized_positions=use_optimized_positions,
        ))

    parts = ([existing_df] if existing_df is not None and rows_before > 0 else []) + new_dfs
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(config.METRICS_CSV_PATH, index=False)

    rows_added = len(df) - rows_before
    print(f"\nmetrics.csv: {rows_before} rows before + {rows_added} new rows = {len(df)} total")
    print(f"Variants now in CSV: {sorted(df['embedding_variant'].unique())}")
    return df


def summarize_variant_comparison(df):
    """
    Per-attack mean/std NC and BER for both embedding variants, plus the NC
    improvement (stable_positions minus standard, absolute and relative).
    """
    grouped = (
        df.groupby(["attack_name", "embedding_variant"])
        .agg(nc_mean=("nc", "mean"), nc_std=("nc", "std"), ber_mean=("ber", "mean"), ber_std=("ber", "std"))
        .reset_index()
    )
    pivoted = grouped.pivot(index="attack_name", columns="embedding_variant")
    pivoted.columns = [f"{metric}_{variant}" for metric, variant in pivoted.columns]
    pivoted = pivoted.reset_index()

    pivoted["nc_improvement"] = pivoted["nc_mean_stable_positions"] - pivoted["nc_mean_standard"]
    pivoted["nc_improvement_pct"] = 100 * pivoted["nc_improvement"] / pivoted["nc_mean_standard"]
    return pivoted.sort_values("nc_improvement", ascending=False).reset_index(drop=True)


def _print_attack_verdict(summary, attack_name):
    match = summary[summary["attack_name"] == attack_name]
    if not len(match):
        print(f"\n{attack_name}: no rows found for this attack in either variant.")
        return
    row = match.iloc[0]
    if row.nc_improvement > 0:
        verdict = "IMPROVED"
    elif row.nc_improvement < 0:
        verdict = "WORSENED"
    else:
        verdict = "NO CHANGE"
    print(
        f"\n{attack_name}: stable_positions NC {verdict} vs standard "
        f"({row.nc_mean_standard:.4f} -> {row.nc_mean_stable_positions:.4f}, "
        f"Δ={row.nc_improvement:+.4f}, {row.nc_improvement_pct:+.2f}%)"
    )
    return row


def print_research_summary(summary):
    """Print the statistical comparison table (section 3) and the research summary (section 6)."""
    print("\n" + "=" * 100)
    print("VARIANT COMPARISON: standard vs stable_positions embedding (per attack)")
    print("=" * 100)
    table_cols = [
        "attack_name",
        "nc_mean_standard", "nc_std_standard", "nc_mean_stable_positions", "nc_std_stable_positions",
        "nc_improvement", "nc_improvement_pct",
        "ber_mean_standard", "ber_mean_stable_positions",
    ]
    print(summary[table_cols].round(4).to_string(index=False))

    print("\n--- Highlights (do not assume improvement -- this reports what the data shows) ---")
    re_row = _print_attack_verdict(summary, "real_esrgan")
    if re_row is not None and re_row.nc_improvement <= 0:
        print("  -> Hypothesis NOT supported: relocating to empirically stable positions did not "
              "improve Real-ESRGAN robustness.")

    for attack_name in ["jpeg_compression_q50", "gaussian_blur_5x5", "gaussian_noise_s10", "median_filter_3x3"]:
        _print_attack_verdict(summary, attack_name)

    worsened = summary[summary["nc_improvement"] < 0].sort_values("nc_improvement")
    print("\n--- Attacks where stable_positions performs WORSE than standard ---")
    if len(worsened):
        print(worsened[["attack_name", "nc_mean_standard", "nc_mean_stable_positions", "nc_improvement"]]
              .round(4).to_string(index=False))
    else:
        print("None -- no attack showed a regression under stable_positions.")

    print("\n--- Attacks that benefited most (top 3 by absolute NC improvement) ---")
    print(summary.sort_values("nc_improvement", ascending=False).head(3)
          [["attack_name", "nc_improvement", "nc_improvement_pct"]].round(4).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the DWT-DCT watermark robustness experiment.")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                         help=f"Embedding strengths to test (default: {config.ALPHA_VALUES})")
    parser.add_argument("--real-esrgan-model", type=str, default=None,
                         help="Path to a Real-ESRGAN .pth weights file (optional)")
    parser.add_argument("--opencv-sr-model", type=str, default=None,
                         help="Path to an OpenCV DNN super-resolution .pb model (optional)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    combined_df = run_all_variants(
        alphas=args.alphas,
        real_esrgan_model_path=args.real_esrgan_model,
        opencv_sr_model_path=args.opencv_sr_model,
    )
    print_research_summary(summarize_variant_comparison(combined_df))
