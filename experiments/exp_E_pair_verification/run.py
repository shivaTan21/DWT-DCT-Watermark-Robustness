"""
Experiment E: Empirical Verification of Analytically-Predicted Optimal Pairs

HYPOTHESIS:
    The analytically-derived Pareto-optimal pair (2,3)/(3,2) will outperform
    the standard pair (4,1)/(1,4) under ESRGAN while maintaining competitive
    JPEG robustness, on a held-out image set.

PURPOSE:
    Address the key reviewer objection: "Analytical prediction using stability
    data from training images doesn't validate generalization."

    We verify 4 candidate pairs derived from Experiment D:
      1. (2,3)/(3,2) — analytically predicted balanced optimum (symmetric)
      2. (3,3)/(3,4) — high stability, near-symmetric, new candidate
      3. (7,5)/(7,7) — empirically best from frequency_sweep (near-symmetric)
      4. (4,1)/(1,4) — standard baseline for comparison

MODIFICATIONS FROM BASELINE:
    - NEW file; uses watermark.py functions via monkey-patching (same as frequency_sweep.py)
    - Does NOT modify any frozen source file
    - Writes: experiments/exp_E_pair_verification/outputs/

SCIENTIFIC VALUE: HIGH
    - Required for submission: validates analytical predictions
    - Provides the paper's core practical result
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import cv2
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    import torch
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        torch.cuda.amp.autocast = lambda *a, **kw: torch.amp.autocast("cpu", *a, **kw)
except ImportError:
    pass

from src import watermark as wm_module
from src import config
from src.metrics import normalized_correlation, bit_error_rate
from src.attacks_traditional import jpeg_compression, gaussian_blur
from src.attacks_ai import espcn_x4_enhance, OPENCV_SR_AVAILABLE as ESPCN_AVAILABLE

try:
    from src.attacks_ai import real_esrgan_enhance, REALESRGAN_AVAILABLE
except ImportError:
    REALESRGAN_AVAILABLE = False
    def real_esrgan_enhance(*a, **kw): raise RuntimeError("Real-ESRGAN not available")

ALPHA = 0.1
WM_SHAPE = config.WATERMARK_SIZE  # (32, 32)
ESRGAN_MODEL = os.path.join(_ROOT, "weights", "RealESRGAN_x4.pth")
N_IMAGES = 30  # use first 30 images (held-out: not in dct_stability computation)

# Pairs to verify:
# (label, pos1, pos2, description)
CANDIDATE_PAIRS = [
    ("standard (4,1)/(1,4)",   (4, 1), (1, 4), "Current standard — baseline"),
    ("balanced (2,3)/(3,2)",   (2, 3), (3, 2), "Analytically predicted global optimum (D: sym, E=0.818, J=0.886)"),
    ("swept (7,5)/(7,7)",      (7, 5), (7, 7), "Best from empirical sweep (C: near-sym, NC_ESRGAN=0.864)"),
    ("near-sym (1,2)/(2,1)",   (1, 2), (2, 1), "Pareto on sweep: sym, NC_ESRGAN=0.793 NC_JPEG=0.995"),
]

ATTACKS = {
    "real_esrgan":           lambda img: real_esrgan_enhance(img, model_path=ESRGAN_MODEL),
    "jpeg_compression_q50":  lambda img: jpeg_compression(img, quality=50),
    "gaussian_blur_5x5":     lambda img: gaussian_blur(img, ksize=5, sigma=1.0),
}
if ESPCN_AVAILABLE:
    ATTACKS["espcn_x4"] = espcn_x4_enhance

ATK_DISPLAY = {
    "real_esrgan":           "Real-ESRGAN",
    "jpeg_compression_q50":  "JPEG Q50",
    "gaussian_blur_5x5":     "Gaussian Blur 5×5",
    "espcn_x4":              "ESPCN x4",
}


def _embed(image, wm_bits, pos1, pos2):
    orig1, orig2 = wm_module.COEFF_POS_1, wm_module.COEFF_POS_2
    wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = pos1, pos2
    try:
        return wm_module.embed_watermark_rgb(image.astype(np.float64), wm_bits, alpha=ALPHA)
    finally:
        wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = orig1, orig2


def _extract(image, pos1, pos2):
    orig1, orig2 = wm_module.COEFF_POS_1, wm_module.COEFF_POS_2
    wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = pos1, pos2
    try:
        return wm_module.extract_watermark_rgb(image.astype(np.float64), WM_SHAPE)
    finally:
        wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = orig1, orig2


def run_verification():
    np.random.seed(42)
    wm = wm_module.generate_watermark(shape=WM_SHAPE, seed=42)

    image_files = sorted(
        f for f in os.listdir(config.ORIGINAL_IMAGES_DIR)
        if f.lower().endswith(".png")
    )[:N_IMAGES]

    print(f"Verifying {len(CANDIDATE_PAIRS)} pairs on {len(image_files)} images")
    print(f"Attacks: {list(ATTACKS.keys())}\n")

    records = []
    for pair_label, pos1, pos2, description in CANDIDATE_PAIRS:
        print(f"\n--- Pair: {pair_label} ---")
        print(f"    {description}")

        for img_file in image_files:
            img_path = os.path.join(config.ORIGINAL_IMAGES_DIR, img_file)
            raw = cv2.imread(img_path)
            if raw is None:
                print(f"  [WARN] Cannot load {img_file}")
                continue

            img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float64)
            img = cv2.resize(img, config.IMAGE_SIZE).astype(np.float64)

            try:
                wm_img = np.clip(_embed(img, wm, pos1, pos2), 0, 255)
            except Exception as e:
                print(f"  [WARN] Embed failed {img_file}: {e}")
                continue

            # Sanity check
            sanity_nc = normalized_correlation(wm, _extract(wm_img, pos1, pos2))
            if sanity_nc < 0.99:
                print(f"  [WARN] Sanity NC={sanity_nc:.4f} < 0.99 for {img_file}")

            for atk_name, atk_fn in ATTACKS.items():
                try:
                    attacked = np.clip(atk_fn(wm_img), 0, 255)
                    extracted = _extract(attacked, pos1, pos2)
                    nc  = normalized_correlation(wm, extracted)
                    ber = bit_error_rate(wm, extracted)
                except Exception as e:
                    print(f"  [WARN] {atk_name} failed {img_file}: {e}")
                    nc, ber = np.nan, np.nan

                records.append({
                    "pair_label": pair_label,
                    "pos1": str(pos1),
                    "pos2": str(pos2),
                    "image": img_file,
                    "attack": atk_name,
                    "nc": nc,
                    "ber": ber,
                })

        # Print progress
        subset = [r for r in records if r["pair_label"] == pair_label]
        for atk in ATTACKS:
            nc_vals = [r["nc"] for r in subset if r["attack"] == atk and not np.isnan(r["nc"])]
            if nc_vals:
                print(f"  {ATK_DISPLAY.get(atk, atk)}: mean NC = {np.mean(nc_vals):.4f} ± {np.std(nc_vals):.4f}")

    return pd.DataFrame(records)


def analyze_results(df):
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS SUMMARY")
    print("=" * 70)

    pivot = df.groupby(["pair_label", "attack"])["nc"].agg(["mean", "std", "count"]).reset_index()
    print(pivot.round(4).to_string(index=False))

    # Statistical tests: each candidate vs standard
    std_label = "standard (4,1)/(1,4)"
    std_data = df[df["pair_label"] == std_label]

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (each pair vs standard, per attack)")
    print("=" * 70)

    for pair_label, _, _, _ in CANDIDATE_PAIRS:
        if pair_label == std_label:
            continue
        print(f"\n  {pair_label}:")
        pair_data = df[df["pair_label"] == pair_label]

        for atk in ATTACKS:
            std_nc  = std_data[std_data["attack"] == atk]["nc"].dropna()
            pair_nc = pair_data[pair_data["attack"] == atk]["nc"].dropna()

            # Align by image
            std_img  = std_data[std_data["attack"] == atk].set_index("image")["nc"]
            pair_img = pair_data[pair_data["attack"] == atk].set_index("image")["nc"]
            common   = std_img.index.intersection(pair_img.index)
            if len(common) < 5:
                continue
            diffs = pair_img[common].values - std_img[common].values
            mean_diff = diffs.mean()

            try:
                stat, p = wilcoxon(diffs, alternative="greater" if mean_diff > 0 else "less")
                test_name = "Wilcoxon"
            except Exception:
                stat, p = 0, 1.0
                test_name = "N/A"

            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            direction = "+" if mean_diff > 0 else ""
            print(f"    {ATK_DISPLAY.get(atk,atk):<25}: Δ={direction}{mean_diff:+.4f}{sig}  "
                  f"({test_name} p={p:.3e}, n={len(common)})")

    return pivot


def plot_results(df, out_dir):
    attacks_to_plot = [a for a in ATTACKS if a != "espcn_x4"]  # skip ESPCN (near-perfect for all)

    fig, axes = plt.subplots(1, len(attacks_to_plot), figsize=(14, 6))
    if len(attacks_to_plot) == 1:
        axes = [axes]

    colors = ["#95a5a6", "#2ecc71", "#3498db", "#e67e22"]
    pair_labels = [p[0] for p in CANDIDATE_PAIRS]

    for ax, atk in zip(axes, attacks_to_plot):
        means, stds = [], []
        for plabel in pair_labels:
            sub = df[(df["pair_label"] == plabel) & (df["attack"] == atk)]["nc"].dropna()
            means.append(sub.mean() if len(sub) else np.nan)
            stds.append(sub.std() if len(sub) else np.nan)

        x = np.arange(len(pair_labels))
        bars = ax.bar(x, means, yerr=stds, color=colors[:len(pair_labels)],
                      alpha=0.85, capsize=6, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([l.split(" ")[0] for l in pair_labels], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean NC" if ax == axes[0] else "", fontsize=10)
        ax.set_title(ATK_DISPLAY.get(atk, atk), fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="NC=0.8")
        ax.grid(True, axis="y", alpha=0.3)

        # Mark baseline
        std_idx = pair_labels.index("standard (4,1)/(1,4)")
        ax.axhline(means[std_idx], color="red", linestyle=":", linewidth=1.5,
                   label="Standard baseline")
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle(
        "Experiment E: Pair Verification — Standard vs Analytically Predicted Optimal Pairs\n"
        f"(n={N_IMAGES} images, α={ALPHA}, empirical test)",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "E_pair_verification.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out}")


def write_summary(df, pivot, out_dir):
    std_label = "standard (4,1)/(1,4)"
    std_esrgan = df[(df["pair_label"]==std_label) & (df["attack"]=="real_esrgan")]["nc"].mean()

    lines = [
        "=" * 70,
        "EXPERIMENT E: PAIR VERIFICATION SUMMARY",
        "=" * 70,
        "",
        "STANDARD PAIR (4,1)/(1,4):",
        f"  ESRGAN NC = {std_esrgan:.4f}",
        "",
        "CANDIDATE PAIRS vs STANDARD:",
    ]

    for pair_label, _, _, desc in CANDIDATE_PAIRS:
        if pair_label == std_label:
            continue
        sub = df[df["pair_label"]==pair_label]
        nc_e = sub[sub["attack"]=="real_esrgan"]["nc"].mean()
        nc_j = sub[sub["attack"]=="jpeg_compression_q50"]["nc"].mean()
        delta_e = nc_e - std_esrgan
        lines.append(f"\n  {pair_label}:")
        lines.append(f"    Description: {desc}")
        lines.append(f"    ESRGAN NC: {nc_e:.4f} (Δ={delta_e:+.4f})")
        lines.append(f"    JPEG NC:   {nc_j:.4f}")

    lines += [
        "",
        "CONCLUSION:",
        "  Does analytical prediction transfer to held-out images?",
        "  See numerical results above.",
        "",
        "PAPER IMPLICATION:",
        "  If (2,3)/(3,2) matches or exceeds standard pair on ESRGAN while",
        "  maintaining JPEG NC, this confirms the design principles.",
        "=" * 70,
    ]

    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    for line in lines:
        print(line)


def main():
    print("=" * 70)
    print("EXPERIMENT E: ANALYTICAL PAIR PREDICTION VERIFICATION")
    print("=" * 70)

    results_csv = os.path.join(OUT_DIR, "E_results.csv")

    if os.path.exists(results_csv):
        print(f"Loading cached results from {results_csv}")
        df = pd.read_csv(results_csv)
    else:
        df = run_verification()
        df.to_csv(results_csv, index=False)
        print(f"\nSaved → {results_csv}")

    pivot = analyze_results(df)
    pivot.to_csv(os.path.join(OUT_DIR, "E_pivot.csv"), index=False)
    plot_results(df, OUT_DIR)
    write_summary(df, pivot, OUT_DIR)


if __name__ == "__main__":
    main()
