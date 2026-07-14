"""
Experiment F: End-to-End Real-ESRGAN Pair Verification

Goal: Empirically verify whether the analytically selected coefficient pair
(2,3)/(3,2) improves the balance between Real-ESRGAN robustness, JPEG-Q50
robustness, Gaussian-blur robustness, and imperceptibility.

Pairs tested:
  1. standard:  (4,1)/(1,4)  -- current baseline
  2. balanced:  (2,3)/(3,2)  -- Exp D analytically predicted global optimum
  3. hf:        (7,5)/(7,7)  -- empirically best from frequency_sweep

Attacks: none (watermarked only), real_esrgan, jpeg_q50, gaussian_blur
Alphas:  0.05, 0.10, 0.20, 0.30
Images:  all 100 (24 Kodak + 76 TAMPERE17)

Real-ESRGAN strategy:
  Standard (4,1)/(1,4): load from results/attacked/ai_enhancement/<stem>__alpha<a>__real_esrgan.png
  HF (7,5)/(7,7):       load from results/attacked/ai_enhancement/<stem>__alpha<a>__optimized_positions__real_esrgan.png
  Balanced (2,3)/(3,2): run py_real_esrgan live if available; otherwise NaN (library not installed).

Do NOT modify any frozen baseline files.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_rel, wilcoxon, t as t_dist, false_discovery_control

from src import config
from src import watermark as wm_module
from src.metrics import normalized_correlation, bit_error_rate, psnr, ssim
from src.attacks_traditional import jpeg_compression, gaussian_blur

try:
    from src.attacks_ai import real_esrgan_enhance, REALESRGAN_AVAILABLE
except Exception:
    REALESRGAN_AVAILABLE = False
    def real_esrgan_enhance(*a, **kw):
        raise RuntimeError("Real-ESRGAN not installed (pip install py-real-esrgan torch).")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ALPHAS = [0.05, 0.10, 0.20, 0.30]
SEED = 42
WM_SHAPE = config.WATERMARK_SIZE   # (32, 32)
IMAGE_SIZE = config.IMAGE_SIZE      # (512, 512)
ESRGAN_MODEL = os.path.join(_ROOT, "weights", "RealESRGAN_x4.pth")

WATERMARKED_DIR = config.WATERMARKED_DIR
AI_DIR = config.ATTACKED_AI_DIR
TRAD_DIR = config.ATTACKED_TRADITIONAL_DIR
ORIG_DIR = config.ORIGINAL_IMAGES_DIR

# disk_suffix: appended to <stem>__alpha<a> in existing watermarked/attacked filenames.
# esrgan_disk_suffix: appended to <stem>__alpha<a> before __real_esrgan.png.
# None means no pre-computed files exist; fall back to live computation.
PAIRS = {
    "standard": {
        "pos1": (4, 1), "pos2": (1, 4),
        "label": "Standard (4,1)/(1,4)",
        "color": "#95a5a6",
        "disk_suffix": "",
        "esrgan_disk_suffix": "",
    },
    "balanced": {
        "pos1": (2, 3), "pos2": (3, 2),
        "label": "Balanced (2,3)/(3,2)",
        "color": "#2ecc71",
        "disk_suffix": None,   # no pre-computed files; embed fresh
        "esrgan_disk_suffix": None,   # no pre-computed ESRGAN; requires live model
    },
    "hf": {
        "pos1": (7, 5), "pos2": (7, 7),
        "label": "HF (7,5)/(7,7)",
        "color": "#3498db",
        "disk_suffix": "__optimized_positions",
        "esrgan_disk_suffix": "__optimized_positions",
    },
}

ATTACK_DISPLAY = {
    "none": "No attack",
    "real_esrgan": "Real-ESRGAN",
    "jpeg_q50": "JPEG Q50",
    "gaussian_blur": "Gaussian Blur",
}

RESULTS_CSV = os.path.join(OUT_DIR, "results_exp_F.csv")


# ─── Image I/O helpers ───────────────────────────────────────────────────────

def dataset_of(image_name):
    return "kodak" if image_name.lower().startswith("kodim") else "tampere17"


def load_bgr_to_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)


def load_original(image_name):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(ORIG_DIR, image_name + ext)
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    raise FileNotFoundError(f"No original image found for {image_name}")


# ─── Watermark embed/extract with custom positions ───────────────────────────

def _embed_custom(original, wm, pos1, pos2, alpha):
    orig1, orig2 = wm_module.COEFF_POS_1, wm_module.COEFF_POS_2
    wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = pos1, pos2
    try:
        watermarked = wm_module.embed_watermark_rgb(
            original.astype(np.float64), wm, alpha=alpha
        )
        return np.clip(watermarked, 0, 255)
    finally:
        wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = orig1, orig2


def _extract_custom(image, pos1, pos2):
    orig1, orig2 = wm_module.COEFF_POS_1, wm_module.COEFF_POS_2
    wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = pos1, pos2
    try:
        return wm_module.extract_watermark_rgb(image.astype(np.float64), WM_SHAPE)
    finally:
        wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = orig1, orig2


# ─── Per-image attack helpers ─────────────────────────────────────────────────

def _get_watermarked(image_name, original, wm, alpha, pair_key):
    pair = PAIRS[pair_key]
    suffix = pair["disk_suffix"]
    if suffix is not None:
        path = os.path.join(WATERMARKED_DIR, f"{image_name}__alpha{alpha}{suffix}.png")
        img = load_bgr_to_rgb(path)
        if img is not None:
            return img
    return _embed_custom(original, wm, pair["pos1"], pair["pos2"], alpha)


def _get_esrgan(image_name, alpha, pair_key, watermarked_img):
    pair = PAIRS[pair_key]
    esrgan_suffix = pair["esrgan_disk_suffix"]
    if esrgan_suffix is not None:
        path = os.path.join(
            AI_DIR, f"{image_name}__alpha{alpha}{esrgan_suffix}__real_esrgan.png"
        )
        img = load_bgr_to_rgb(path)
        if img is not None:
            return img
    if REALESRGAN_AVAILABLE:
        try:
            out = real_esrgan_enhance(watermarked_img, model_path=ESRGAN_MODEL)
            return np.clip(out, 0, 255)
        except Exception as e:
            print(f"    [WARN] ESRGAN live call failed: {e}")
    return None


# ─── Main data collection ─────────────────────────────────────────────────────

def run_experiment():
    np.random.seed(SEED)
    wm = wm_module.generate_watermark(shape=WM_SHAPE, seed=SEED)

    image_files = sorted(
        f for f in os.listdir(ORIG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    image_names = [os.path.splitext(f)[0] for f in image_files]
    print(f"  {len(image_names)} images, ESRGAN live: {REALESRGAN_AVAILABLE}")

    records = []
    esrgan_nan_count = 0

    for img_idx, image_name in enumerate(image_names):
        if (img_idx + 1) % 20 == 0:
            print(f"  [{img_idx+1}/{len(image_names)}] {image_name}")
        try:
            original = load_original(image_name)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            continue

        dataset = dataset_of(image_name)

        for alpha in ALPHAS:
            for pair_key, pair in PAIRS.items():
                pos1, pos2 = pair["pos1"], pair["pos2"]

                wm_img = _get_watermarked(image_name, original, wm, alpha, pair_key)
                wm_psnr = psnr(original, wm_img)
                wm_ssim = ssim(original, wm_img)

                # No attack
                ext = _extract_custom(wm_img, pos1, pos2)
                records.append({
                    "image": image_name, "dataset": dataset,
                    "coefficient_pair": pair_key, "alpha": alpha,
                    "attack": "none",
                    "nc": normalized_correlation(wm, ext),
                    "ber": bit_error_rate(wm, ext),
                    "psnr": wm_psnr, "ssim": wm_ssim,
                })

                # Real-ESRGAN
                esrgan_img = _get_esrgan(image_name, alpha, pair_key, wm_img)
                if esrgan_img is not None:
                    ext_e = _extract_custom(esrgan_img, pos1, pos2)
                    records.append({
                        "image": image_name, "dataset": dataset,
                        "coefficient_pair": pair_key, "alpha": alpha,
                        "attack": "real_esrgan",
                        "nc": normalized_correlation(wm, ext_e),
                        "ber": bit_error_rate(wm, ext_e),
                        "psnr": psnr(original, esrgan_img),
                        "ssim": ssim(original, esrgan_img),
                    })
                else:
                    esrgan_nan_count += 1
                    records.append({
                        "image": image_name, "dataset": dataset,
                        "coefficient_pair": pair_key, "alpha": alpha,
                        "attack": "real_esrgan",
                        "nc": np.nan, "ber": np.nan,
                        "psnr": np.nan, "ssim": np.nan,
                    })

                # JPEG Q50
                try:
                    jpeg_img = np.clip(jpeg_compression(wm_img, quality=50), 0, 255)
                    ext_j = _extract_custom(jpeg_img, pos1, pos2)
                    records.append({
                        "image": image_name, "dataset": dataset,
                        "coefficient_pair": pair_key, "alpha": alpha,
                        "attack": "jpeg_q50",
                        "nc": normalized_correlation(wm, ext_j),
                        "ber": bit_error_rate(wm, ext_j),
                        "psnr": psnr(original, jpeg_img),
                        "ssim": ssim(original, jpeg_img),
                    })
                except Exception as exc:
                    print(f"  [WARN] JPEG {image_name} α={alpha}: {exc}")
                    records.append({
                        "image": image_name, "dataset": dataset,
                        "coefficient_pair": pair_key, "alpha": alpha,
                        "attack": "jpeg_q50",
                        "nc": np.nan, "ber": np.nan, "psnr": np.nan, "ssim": np.nan,
                    })

                # Gaussian blur (same params as baseline: ksize=5, sigma=1.0)
                try:
                    blur_img = np.clip(gaussian_blur(wm_img, ksize=5, sigma=1.0), 0, 255)
                    ext_b = _extract_custom(blur_img, pos1, pos2)
                    records.append({
                        "image": image_name, "dataset": dataset,
                        "coefficient_pair": pair_key, "alpha": alpha,
                        "attack": "gaussian_blur",
                        "nc": normalized_correlation(wm, ext_b),
                        "ber": bit_error_rate(wm, ext_b),
                        "psnr": psnr(original, blur_img),
                        "ssim": ssim(original, blur_img),
                    })
                except Exception as exc:
                    print(f"  [WARN] Blur {image_name} α={alpha}: {exc}")
                    records.append({
                        "image": image_name, "dataset": dataset,
                        "coefficient_pair": pair_key, "alpha": alpha,
                        "attack": "gaussian_blur",
                        "nc": np.nan, "ber": np.nan, "psnr": np.nan, "ssim": np.nan,
                    })

    if esrgan_nan_count > 0:
        print(
            f"\n[WARNING] {esrgan_nan_count} ESRGAN entries are NaN. "
            f"Balanced pair requires py_real_esrgan + torch (not installed). "
            f"Standard and HF pairs used pre-computed disk files."
        )

    return pd.DataFrame(records)


# ─── Summary statistics ───────────────────────────────────────────────────────

def ci95(vals):
    """95% CI half-width for the mean (t-distribution)."""
    n = len(vals)
    if n < 2:
        return np.nan
    se = np.std(vals, ddof=1) / np.sqrt(n)
    return float(t_dist.ppf(0.975, df=n - 1) * se)


def compute_summary(df):
    """Mean NC/BER/PSNR/SSIM per (coefficient_pair, alpha, attack) with 95% CI on NC."""
    rows = []
    for (pair, alpha, attack), grp in df.groupby(
        ["coefficient_pair", "alpha", "attack"]
    ):
        nc_vals = grp["nc"].dropna().values
        rows.append({
            "coefficient_pair": pair,
            "alpha": alpha,
            "attack": attack,
            "n": len(nc_vals),
            "nc_mean": float(np.mean(nc_vals)) if len(nc_vals) else np.nan,
            "nc_std": float(np.std(nc_vals, ddof=1)) if len(nc_vals) > 1 else np.nan,
            "nc_ci95": ci95(nc_vals),
            "ber_mean": float(grp["ber"].mean()),
            "psnr_mean": float(grp["psnr"].mean()),
            "ssim_mean": float(grp["ssim"].mean()),
        })
    return pd.DataFrame(rows)


# ─── Statistical tests ────────────────────────────────────────────────────────

def _paired_test(diffs):
    """
    Test whether diffs differ from zero.
    Returns (test_name, stat, p_raw, mean_diff, ci_lo, ci_hi, effect_size).
    Uses paired t-test when normality not rejected (Shapiro-Wilk p > 0.05),
    otherwise Wilcoxon signed-rank.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 5:
        return "n/a", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    se_d = std_d / np.sqrt(n)

    # Normality test
    if n >= 3:
        _, p_norm = shapiro(diffs)
    else:
        p_norm = 0.0   # too small — skip t-test

    if p_norm > 0.05:
        # Paired t-test
        t_stat, p_raw = ttest_rel(diffs, np.zeros(n))
        ci_lo = mean_d - t_dist.ppf(0.975, n - 1) * se_d
        ci_hi = mean_d + t_dist.ppf(0.975, n - 1) * se_d
        effect = mean_d / std_d if std_d > 0 else 0.0   # Cohen's d
        return "t-test", float(t_stat), float(p_raw), mean_d, ci_lo, ci_hi, effect
    else:
        # Wilcoxon signed-rank
        try:
            alt = "greater" if mean_d >= 0 else "less"
            stat, p_raw = wilcoxon(diffs, alternative=alt)
            # approximate z for effect size r = Z/sqrt(n)
            from scipy.stats import norm
            z = norm.ppf(1 - p_raw / 2) * np.sign(mean_d)
            effect = float(z / np.sqrt(n))
        except Exception:
            stat, p_raw, effect = np.nan, np.nan, np.nan
        ci_lo = mean_d - 1.96 * se_d
        ci_hi = mean_d + 1.96 * se_d
        return "wilcoxon", float(stat), float(p_raw), mean_d, ci_lo, ci_hi, effect


def run_statistical_tests(df):
    """
    Paired tests comparing (balanced vs standard), (hf vs standard),
    (balanced vs hf) for each (attack, alpha) combination.
    p-values corrected with Benjamini-Hochberg FDR across all tests.
    """
    comparisons = [
        ("balanced", "standard", "balanced_vs_standard"),
        ("hf",       "standard", "hf_vs_standard"),
        ("balanced", "hf",       "balanced_vs_hf"),
    ]

    all_rows = []
    for pair_a, pair_b, comp_label in comparisons:
        sub_a = df[df["coefficient_pair"] == pair_a].set_index(["image", "alpha", "attack"])
        sub_b = df[df["coefficient_pair"] == pair_b].set_index(["image", "alpha", "attack"])
        common = sub_a.index.intersection(sub_b.index)

        for attack in df["attack"].unique():
            for alpha in ALPHAS:
                idx_mask = [(im, al, atk) for (im, al, atk) in common
                            if al == alpha and atk == attack]
                if not idx_mask:
                    continue

                nc_a = sub_a.loc[idx_mask, "nc"].values.astype(float)
                nc_b = sub_b.loc[idx_mask, "nc"].values.astype(float)
                diffs = nc_a - nc_b   # positive = pair_a better

                test, stat, p_raw, mean_d, ci_lo, ci_hi, effect = _paired_test(diffs)
                all_rows.append({
                    "comparison": comp_label,
                    "pair_a": pair_a,
                    "pair_b": pair_b,
                    "attack": attack,
                    "alpha": alpha,
                    "n": len(diffs),
                    "mean_diff": mean_d,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "effect_size": effect,
                    "test": test,
                    "stat": stat,
                    "p_raw": p_raw,
                    "nc_mean_a": float(np.nanmean(nc_a)),
                    "nc_mean_b": float(np.nanmean(nc_b)),
                    "nc_abs_diff": float(np.nanmean(nc_a) - np.nanmean(nc_b)),
                    "nc_rel_diff": float(
                        (np.nanmean(nc_a) - np.nanmean(nc_b)) / max(np.nanmean(nc_b), 1e-9)
                    ),
                })

    tests_df = pd.DataFrame(all_rows)
    if tests_df.empty:
        return tests_df

    # BH-FDR correction across all tests
    valid_mask = tests_df["p_raw"].notna()
    p_vals = tests_df.loc[valid_mask, "p_raw"].values
    if len(p_vals) > 0:
        p_adj = false_discovery_control(p_vals, method="bh")
        tests_df["p_adj_bh"] = np.nan
        tests_df.loc[valid_mask, "p_adj_bh"] = p_adj
    else:
        tests_df["p_adj_bh"] = np.nan

    return tests_df


# ─── Figures ─────────────────────────────────────────────────────────────────

def _pair_labels():
    return {k: v["label"] for k, v in PAIRS.items()}


def _pair_colors():
    return {k: v["color"] for k, v in PAIRS.items()}


def fig_nc_by_pair_attack(df, alpha_display=0.10):
    """Figure 1: grouped mean-NC per attack, one bar series per pair (at alpha_display)."""
    sub = df[df["alpha"] == alpha_display]
    attacks = ["none", "real_esrgan", "jpeg_q50", "gaussian_blur"]
    pair_keys = list(PAIRS.keys())
    labels = _pair_labels()
    colors = _pair_colors()

    x = np.arange(len(attacks))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, pk in enumerate(pair_keys):
        means, cis = [], []
        for atk in attacks:
            grp = sub[(sub["coefficient_pair"] == pk) & (sub["attack"] == atk)]["nc"].dropna()
            means.append(grp.mean() if len(grp) else np.nan)
            cis.append(ci95(grp.values) if len(grp) > 1 else 0)
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=cis, label=labels[pk],
               color=colors[pk], alpha=0.85, capsize=5, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_DISPLAY[a] for a in attacks], fontsize=11)
    ax.set_ylabel("Mean NC", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="NC=0.8 ref")
    ax.legend(fontsize=10)
    ax.set_title(
        f"Experiment F — Mean NC by Attack and Coefficient Pair (α={alpha_display})\n"
        f"Error bars: 95% CI  |  n=100 images",
        fontsize=12,
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "exp_F_nc_by_pair_attack.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def fig_realesrgan_alpha(df):
    """Figure 2: Real-ESRGAN NC vs alpha, one line per pair with CI ribbon."""
    sub = df[df["attack"] == "real_esrgan"]
    labels = _pair_labels()
    colors = _pair_colors()

    fig, ax = plt.subplots(figsize=(9, 6))
    for pk in PAIRS:
        means, lo_bounds, hi_bounds = [], [], []
        for alpha in ALPHAS:
            grp = sub[(sub["coefficient_pair"] == pk) & (sub["alpha"] == alpha)]["nc"].dropna()
            if len(grp) == 0:
                means.append(np.nan)
                lo_bounds.append(np.nan)
                hi_bounds.append(np.nan)
                continue
            m = float(grp.mean())
            hw = ci95(grp.values)
            means.append(m)
            lo_bounds.append(m - hw)
            hi_bounds.append(m + hw)

        alphas_arr = np.array(ALPHAS)
        means_arr = np.array(means, dtype=float)
        lo_arr = np.array(lo_bounds, dtype=float)
        hi_arr = np.array(hi_bounds, dtype=float)

        valid = ~np.isnan(means_arr)
        if valid.any():
            ax.plot(alphas_arr[valid], means_arr[valid], "o-",
                    color=colors[pk], label=labels[pk], linewidth=2, markersize=7)
            ax.fill_between(alphas_arr[valid], lo_arr[valid], hi_arr[valid],
                            color=colors[pk], alpha=0.15)
        else:
            ax.plot([], [], "o--", color=colors[pk],
                    label=f"{labels[pk]} [ESRGAN n/a]")

    ax.set_xlabel("Embedding strength α", fontsize=12)
    ax.set_ylabel("Mean NC under Real-ESRGAN", fontsize=12)
    ax.set_title(
        "Experiment F — Real-ESRGAN NC vs Embedding Strength\n"
        "Shaded bands: 95% CI  |  n=100 images per point",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "exp_F_realesrgan_alpha.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def fig_tradeoff(df, alpha_display=0.10):
    """Figure 3: JPEG-Q50 NC (x) vs ESRGAN NC (y) at alpha_display, one point per pair."""
    labels = _pair_labels()
    colors = _pair_colors()

    fig, ax = plt.subplots(figsize=(8, 7))

    for pk in PAIRS:
        for alpha in ALPHAS:
            sub_e = df[
                (df["coefficient_pair"] == pk) & (df["alpha"] == alpha) &
                (df["attack"] == "real_esrgan")
            ]["nc"]
            sub_j = df[
                (df["coefficient_pair"] == pk) & (df["alpha"] == alpha) &
                (df["attack"] == "jpeg_q50")
            ]["nc"]

            nc_e = float(sub_e.mean()) if len(sub_e.dropna()) else np.nan
            nc_j = float(sub_j.mean()) if len(sub_j.dropna()) else np.nan

            marker_size = 180 if alpha == alpha_display else 60
            alpha_val = 1.0 if alpha == alpha_display else 0.35
            zorder = 5 if alpha == alpha_display else 2
            ax.scatter(nc_j, nc_e, s=marker_size, color=colors[pk],
                       alpha=alpha_val, zorder=zorder, edgecolors="black" if alpha == alpha_display else "none",
                       linewidths=1.2 if alpha == alpha_display else 0)
            if alpha == alpha_display and not np.isnan(nc_j) and not np.isnan(nc_e):
                ax.annotate(
                    labels[pk].split("(")[0].strip(),
                    (nc_j, nc_e), fontsize=9,
                    xytext=(6, 4), textcoords="offset points"
                )

    # Legend proxy artists for pairs
    for pk in PAIRS:
        ax.scatter([], [], color=colors[pk], s=100,
                   label=labels[pk], edgecolors="black", linewidths=1)
    ax.scatter([], [], color="gray", s=60, alpha=0.35, label=f"Other α values")
    ax.scatter([], [], color="gray", s=180, edgecolors="black", linewidths=1.2,
               label=f"α={alpha_display} (primary)")

    ax.set_xlabel("Mean NC under JPEG Q50", fontsize=12)
    ax.set_ylabel("Mean NC under Real-ESRGAN", fontsize=12)
    ax.set_title(
        "Experiment F — ESRGAN vs JPEG Trade-off by Coefficient Pair\n"
        f"Large markers: α={alpha_display}  |  Small: other α  |  n=100 images",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "exp_F_tradeoff.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def fig_imperceptibility(df):
    """Figure 4: PSNR and SSIM for un-attacked watermarked images across pairs and alphas."""
    sub = df[df["attack"] == "none"]
    labels = _pair_labels()
    colors = _pair_colors()
    pair_keys = list(PAIRS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(ALPHAS))
    width = 0.25

    for metric, ax, ylabel in [
        ("psnr", axes[0], "PSNR (dB)"),
        ("ssim", axes[1], "SSIM"),
    ]:
        for i, pk in enumerate(pair_keys):
            means = []
            for alpha in ALPHAS:
                grp = sub[(sub["coefficient_pair"] == pk) & (sub["alpha"] == alpha)][metric].dropna()
                means.append(float(grp.mean()) if len(grp) else np.nan)
            offset = (i - 1) * width
            ax.bar(x + offset, means, width, label=labels[pk],
                   color=colors[pk], alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([f"α={a}" for a in ALPHAS], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} — Watermarked vs Original", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Experiment F — Imperceptibility: Watermarked Image Quality (no attack)\n"
        "Higher is better for both PSNR and SSIM",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "exp_F_imperceptibility.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ─── Report ───────────────────────────────────────────────────────────────────

def _sig_symbol(p):
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def write_report(df, summary_df, tests_df):
    alpha_ref = 0.10
    attacks = ["none", "real_esrgan", "jpeg_q50", "gaussian_blur"]

    esrgan_available_for_balanced = (
        not df[
            (df["coefficient_pair"] == "balanced") & (df["attack"] == "real_esrgan")
        ]["nc"].isna().all()
    )

    def nc_for(pair, attack, alpha=alpha_ref):
        row = summary_df[
            (summary_df["coefficient_pair"] == pair) &
            (summary_df["attack"] == attack) &
            (summary_df["alpha"] == alpha)
        ]
        if row.empty:
            return np.nan, np.nan
        return float(row["nc_mean"].iloc[0]), float(row["nc_ci95"].iloc[0])

    def test_result(comp, attack, alpha=alpha_ref):
        row = tests_df[
            (tests_df["comparison"] == comp) &
            (tests_df["attack"] == attack) &
            (tests_df["alpha"] == alpha)
        ]
        if row.empty:
            return {}
        r = row.iloc[0]
        return {
            "mean_diff": r["mean_diff"],
            "p_raw": r["p_raw"],
            "p_adj": r["p_adj_bh"],
            "effect": r["effect_size"],
            "test": r["test"],
        }

    lines = []
    def emit(*args):
        lines.append(" ".join(str(a) for a in args))

    emit("# Experiment F: Real-ESRGAN Pair Verification Report")
    emit()
    emit(f"Date: 2026-07-13  |  Primary α: {alpha_ref}  |  n=100 images (24 Kodak + 76 TAMPERE17)")
    emit()
    emit("## 1. Experiment Overview")
    emit()
    emit("Three coefficient-pair configurations are compared across four attacks and four embedding")
    emit("strengths (α ∈ {0.05, 0.10, 0.20, 0.30}):")
    emit()
    emit("| Pair key  | Positions        | Origin                          |")
    emit("|-----------|------------------|---------------------------------|")
    emit("| standard  | (4,1)/(1,4)      | Current baseline (watermark.py) |")
    emit("| balanced  | (2,3)/(3,2)      | Exp D global optimum (E=0.818)  |")
    emit("| hf        | (7,5)/(7,7)      | Empirically best (frequency_sweep) |")
    emit()
    emit("**Real-ESRGAN data source:**")
    emit("- Standard: pre-computed disk files (`results/attacked/ai_enhancement/*__real_esrgan.png`)")
    emit("- HF (7,5)/(7,7): pre-computed disk files (`*__optimized_positions__real_esrgan.png`)")
    if esrgan_available_for_balanced:
        emit("- Balanced: computed live with py_real_esrgan ✓")
    else:
        emit("- Balanced: **py_real_esrgan not installed — ESRGAN NC reported as NaN**.")
        emit("  Install with `pip install py-real-esrgan torch` and re-run to complete this column.")
    emit()

    emit("## 2. Mean NC Results (α=0.10)")
    emit()
    emit("| Attack       | Standard         | Balanced         | HF (7,5)/(7,7)   |")
    emit("|--------------|------------------|------------------|------------------|")
    for atk in attacks:
        std_nc, std_ci = nc_for("standard", atk)
        bal_nc, bal_ci = nc_for("balanced", atk)
        hf_nc, hf_ci = nc_for("hf", atk)
        def fmt(nc, ci):
            if np.isnan(nc):
                return "n/a      "
            return f"{nc:.4f}±{ci:.4f}"
        emit(f"| {ATTACK_DISPLAY[atk]:<12} | {fmt(std_nc, std_ci)} | {fmt(bal_nc, bal_ci)} | {fmt(hf_nc, hf_ci)} |")
    emit()

    emit("## 3. Statistical Tests — Balanced vs Standard (BH-FDR corrected)")
    emit()
    emit("| Attack         | α    | Δ NC   | p_raw  | p_adj  | sig  | Effect | Test     |")
    emit("|----------------|------|--------|--------|--------|------|--------|----------|")
    for alpha in ALPHAS:
        for atk in attacks:
            tr = test_result("balanced_vs_standard", atk, alpha)
            if not tr:
                continue
            mean_d = tr.get("mean_diff", np.nan)
            p_r = tr.get("p_raw", np.nan)
            p_a = tr.get("p_adj", np.nan)
            eff = tr.get("effect", np.nan)
            test = tr.get("test", "n/a")
            sig = _sig_symbol(p_a)
            emit(
                f"| {ATTACK_DISPLAY[atk]:<14} | {alpha:.2f} | "
                f"{'n/a' if np.isnan(mean_d) else f'{mean_d:+.4f}'} | "
                f"{'n/a' if np.isnan(p_r) else f'{p_r:.3e}'} | "
                f"{'n/a' if np.isnan(p_a) else f'{p_a:.3e}'} | "
                f"{sig:<4} | "
                f"{'n/a' if np.isnan(eff) else f'{eff:.3f}'} | "
                f"{test} |"
            )
    emit()

    emit("## 4. Statistical Tests — HF vs Standard (BH-FDR corrected)")
    emit()
    emit("| Attack         | α    | Δ NC   | p_raw  | p_adj  | sig  | Effect | Test     |")
    emit("|----------------|------|--------|--------|--------|------|--------|----------|")
    for alpha in ALPHAS:
        for atk in attacks:
            tr = test_result("hf_vs_standard", atk, alpha)
            if not tr:
                continue
            mean_d = tr.get("mean_diff", np.nan)
            p_r = tr.get("p_raw", np.nan)
            p_a = tr.get("p_adj", np.nan)
            eff = tr.get("effect", np.nan)
            test = tr.get("test", "n/a")
            sig = _sig_symbol(p_a)
            emit(
                f"| {ATTACK_DISPLAY[atk]:<14} | {alpha:.2f} | "
                f"{'n/a' if np.isnan(mean_d) else f'{mean_d:+.4f}'} | "
                f"{'n/a' if np.isnan(p_r) else f'{p_r:.3e}'} | "
                f"{'n/a' if np.isnan(p_a) else f'{p_a:.3e}'} | "
                f"{sig:<4} | "
                f"{'n/a' if np.isnan(eff) else f'{eff:.3f}'} | "
                f"{test} |"
            )
    emit()

    emit("## 5. Primary Question")
    emit()
    emit("**Does (2,3)/(3,2) significantly improve Real-ESRGAN NC over (4,1)/(1,4)**")
    emit("**without a large JPEG penalty?**")
    emit()
    if not esrgan_available_for_balanced:
        emit("> **ESRGAN results for the balanced pair are unavailable** (py_real_esrgan not installed).")
        emit("> The primary research question cannot be answered from this run.")
        emit("> Re-run after installing py_real_esrgan and torch.")
        emit()
        emit("Available evidence (JPEG Q50 and Gaussian Blur, α=0.10):")
        tr_j = test_result("balanced_vs_standard", "jpeg_q50")
        if tr_j:
            d = tr_j.get("mean_diff", np.nan)
            p = tr_j.get("p_adj", np.nan)
            emit(f"- JPEG Q50: ΔNC = {d:+.4f} ({_sig_symbol(p)} after FDR, p_adj={p:.3e})")
        tr_b = test_result("balanced_vs_standard", "gaussian_blur")
        if tr_b:
            d = tr_b.get("mean_diff", np.nan)
            p = tr_b.get("p_adj", np.nan)
            emit(f"- Gaussian Blur: ΔNC = {d:+.4f} ({_sig_symbol(p)} after FDR, p_adj={p:.3e})")
    else:
        tr_e = test_result("balanced_vs_standard", "real_esrgan")
        tr_j = test_result("balanced_vs_standard", "jpeg_q50")
        bal_e, _ = nc_for("balanced", "real_esrgan")
        std_e, _ = nc_for("standard", "real_esrgan")
        d_e = tr_e.get("mean_diff", np.nan)
        p_e = tr_e.get("p_adj", np.nan)
        d_j = tr_j.get("mean_diff", np.nan)
        p_j = tr_j.get("p_adj", np.nan)

        esrgan_improved = (not np.isnan(d_e)) and d_e > 0 and p_e < 0.05
        jpeg_degraded = (not np.isnan(d_j)) and d_j < -0.02

        emit(f"**Real-ESRGAN NC**: balanced={bal_e:.4f} vs standard={std_e:.4f}")
        emit(f"  ΔNC = {d_e:+.4f}  ({_sig_symbol(p_e)}, p_adj={p_e:.3e})")
        emit()
        if esrgan_improved:
            emit("→ Balanced pair **significantly improves** Real-ESRGAN NC.")
        elif not np.isnan(d_e) and d_e > 0:
            emit("→ Balanced pair shows higher ESRGAN NC but **not statistically significant**.")
        else:
            emit("→ Balanced pair does **not** improve Real-ESRGAN NC over standard.")
        emit()
        emit(f"**JPEG Q50 NC delta**: {d_j:+.4f} ({_sig_symbol(p_j)}, p_adj={p_j:.3e})")
        if jpeg_degraded:
            emit("→ JPEG robustness **significantly regresses** under balanced pair.")
        else:
            emit("→ JPEG robustness does **not** significantly regress under balanced pair.")

    emit()
    emit("## 6. Analytical Ranking vs Empirical Performance")
    emit()
    emit("Exp D predicted: balanced pair (E-score=0.818) > standard (E-score=0.028) for ESRGAN,")
    emit("with competitive JPEG NC (J-score=0.886 vs 0.877).")
    emit()
    if not esrgan_available_for_balanced:
        emit("Verification status: **INCOMPLETE** — ESRGAN data for balanced pair unavailable.")
        emit("JPEG and blur results available; see sections 3–4 for partial validation.")
    else:
        bal_j, _ = nc_for("balanced", "jpeg_q50")
        std_j, _ = nc_for("standard", "jpeg_q50")
        hf_e, _ = nc_for("hf", "real_esrgan")
        emit(f"- Balanced ESRGAN NC: {bal_e:.4f}  (predicted improvement confirmed: {'YES' if d_e > 0 else 'NO'})")
        emit(f"- HF ESRGAN NC:       {hf_e:.4f}  (should exceed balanced for ESRGAN-alone)")
        emit(f"- Balanced JPEG NC:   {bal_j:.4f}  Standard JPEG NC: {std_j:.4f}")
        emit()
        if d_e > 0:
            emit("Analytical ranking appears **consistent** with empirical results for ESRGAN.")
        else:
            emit("Analytical ranking is **NOT confirmed** empirically for ESRGAN at this alpha.")

    emit()
    emit("## 7. Alpha Dependence")
    emit()
    emit("| α    | Std ESRGAN NC | Bal ESRGAN NC | HF ESRGAN NC  |")
    emit("|------|---------------|---------------|---------------|")
    for alpha in ALPHAS:
        s, _ = nc_for("standard", "real_esrgan", alpha)
        b, _ = nc_for("balanced", "real_esrgan", alpha)
        h, _ = nc_for("hf", "real_esrgan", alpha)
        emit(f"| {alpha:.2f} | {s:.4f}        | {'n/a  ' if np.isnan(b) else f'{b:.4f}'}        | {h:.4f}        |")
    emit()

    emit("## 8. Imperceptibility")
    emit()
    emit("| α    | Std PSNR (dB) | Bal PSNR (dB) | HF PSNR (dB) |")
    emit("|------|---------------|---------------|--------------|")
    for alpha in ALPHAS:
        for pk in ["standard", "balanced", "hf"]:
            row = summary_df[
                (summary_df["coefficient_pair"] == pk) &
                (summary_df["attack"] == "none") &
                (summary_df["alpha"] == alpha)
            ]
        vals = []
        for pk in ["standard", "balanced", "hf"]:
            row = summary_df[
                (summary_df["coefficient_pair"] == pk) &
                (summary_df["attack"] == "none") &
                (summary_df["alpha"] == alpha)
            ]
            vals.append(float(row["psnr_mean"].iloc[0]) if not row.empty else np.nan)
        emit(f"| {alpha:.2f} | {vals[0]:.2f}         | {vals[1]:.2f}         | {vals[2]:.2f}        |")
    emit()

    emit("## 9. Conclusions")
    emit()
    if not esrgan_available_for_balanced:
        emit("- **Primary conclusion deferred**: ESRGAN results for balanced pair require py_real_esrgan.")
        emit("- Standard and HF pair ESRGAN comparison is complete using pre-computed attacked files.")
        emit("- JPEG and blur results for all three pairs are complete and available.")
        tr_j = test_result("balanced_vs_standard", "jpeg_q50", alpha_ref)
        d_j = tr_j.get("mean_diff", np.nan) if tr_j else np.nan
        if not np.isnan(d_j):
            direction = "higher" if d_j > 0 else "lower"
            emit(f"- Balanced pair JPEG NC is {direction} than standard by ΔNC={d_j:+.4f} at α={alpha_ref}.")
        emit("- **Do not claim (2,3)/(3,2) is empirically validated under ESRGAN** until ESRGAN data is collected.")
    else:
        emit(f"- Balanced pair ESRGAN: ΔNC={d_e:+.4f} vs standard ({'significant' if p_e < 0.05 else 'not significant'}).")
        emit(f"- JPEG Q50 delta for balanced pair: ΔNC={d_j:+.4f} ({'significant regression' if jpeg_degraded else 'no significant regression'}).")
        if esrgan_improved and not jpeg_degraded:
            emit("- **Balanced pair is empirically validated**: improves ESRGAN NC without significant JPEG loss.")
        elif esrgan_improved and jpeg_degraded:
            emit("- **Mixed result**: ESRGAN improves but JPEG regresses significantly.")
        else:
            emit("- **Balanced pair not empirically validated** as superior to standard under ESRGAN.")
    emit()
    emit("---")
    emit("*Report generated by experiments/exp_F_realesrgan_pair_verification/run_exp_F.py*")

    report_path = os.path.join(OUT_DIR, "EXP_F_REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report: {report_path}")
    return report_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXPERIMENT F: END-TO-END REAL-ESRGAN PAIR VERIFICATION")
    print("=" * 70)

    if os.path.exists(RESULTS_CSV):
        print(f"[cache] Loading existing results from {RESULTS_CSV}")
        df = pd.read_csv(RESULTS_CSV)
    else:
        print("\n[1/5] Running experiment (embed / attack / extract)...")
        df = run_experiment()
        df.to_csv(RESULTS_CSV, index=False)
        print(f"  Saved {RESULTS_CSV}  ({len(df)} rows)")

    print("\n[2/5] Computing summary statistics...")
    summary_df = compute_summary(df)
    summary_df.to_csv(os.path.join(OUT_DIR, "exp_F_summary.csv"), index=False)

    print("\n[3/5] Running statistical tests...")
    tests_df = run_statistical_tests(df)
    tests_df.to_csv(os.path.join(OUT_DIR, "exp_F_tests.csv"), index=False)

    print("\n[4/5] Generating figures...")
    fig_nc_by_pair_attack(df, alpha_display=0.10)
    fig_realesrgan_alpha(df)
    fig_tradeoff(df, alpha_display=0.10)
    fig_imperceptibility(df)

    print("\n[5/5] Writing report...")
    write_report(df, summary_df, tests_df)

    print("\n" + "=" * 70)
    print("Done. Output in:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
