"""
Content-adaptive DCT coefficient pair selection for DWT-DCT watermarking.

Hypothesis: selecting the coefficient pair per image based on natural DCT
magnitudes in the LL subband reduces embedding perturbation and improves
imperceptibility and/or robustness vs the fixed standard pair (4,1)/(1,4).

Usage:
  python src/adaptive_embedding.py --sanity-only   # sanity check only
  python src/adaptive_embedding.py                 # full experiment
"""

import sys
import os
import re
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pywt
from scipy.fftpack import dct as _scipy_dct

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# ── MPS autocast monkey-patch ─────────────────────────────────────────────────
# Must be applied before py_real_esrgan.model is imported; same fix as in
# attacks_ai._get_real_esrgan_model, applied here at module load time as well.
def _apply_mps_autocast_patch():
    try:
        import torch
        if not torch.cuda.is_available() and torch.backends.mps.is_available():
            torch.cuda.amp.autocast = (
                lambda *a, **kw: torch.amp.autocast("cpu", *a, **kw)
            )
    except ImportError:
        pass

_apply_mps_autocast_patch()

from src import watermark as wm_module
from src import config
from src.metrics import (
    normalized_correlation,
    bit_error_rate,
    psnr as compute_psnr,
    ssim as compute_ssim,
)
from src.attacks_traditional import jpeg_compression, gaussian_blur

try:
    from src.attacks_ai import real_esrgan_enhance, REALESRGAN_AVAILABLE
except ImportError:
    REALESRGAN_AVAILABLE = False

    def real_esrgan_enhance(image, model_path=None, scale=4):
        raise RuntimeError("Real-ESRGAN not available (pip install py-real-esrgan torch).")


# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
ALPHA = 0.1
WM_SHAPE = config.WATERMARK_SIZE  # (32, 32)
ESRGAN_MODEL_PATH = os.path.join(_ROOT, "weights", "RealESRGAN_x4.pth")
BLOCK_SIZE = 8
DWT_WAVELET = config.DWT_WAVELET  # "haar"

ATTACKS = ["real_esrgan", "jpeg_compression_q50", "gaussian_blur_5x5"]
ATTACK_DISPLAY = {
    "real_esrgan":          "Real-ESRGAN",
    "jpeg_compression_q50": "JPEG Q50",
    "gaussian_blur_5x5":    "Gaussian Blur 5×5",
}

# Combined cost weights: 0.7 Real-ESRGAN damage + 0.3 imperceptibility cost.
ESRGAN_DAMAGE_WEIGHT = 0.7
EMBED_COST_WEIGHT = 0.3

# Standard pair: matches watermark.py COEFF_POS_1 / COEFF_POS_2 defaults.
STANDARD_POS1 = (4, 1)
STANDARD_POS2 = (1, 4)
STANDARD_LABEL = "(4,1)/(1,4)"
_STD_KEY = frozenset([STANDARD_POS1, STANDARD_POS2])

# Hardcoded fallback when frequency_sweep.csv is absent.
_FALLBACK_STRINGS = [
    "(7,5)/(7,7)", "(7,6)/(7,7)", "(6,7)/(7,7)", "(7,4)/(7,7)",
    "(6,6)/(6,7)", "(1,7)/(7,1)", "(1,2)/(2,1)", "(1,4)/(4,1)",
    "(7,5)/(5,7)", "(5,7)/(7,5)",
]


# ── Pair helpers ──────────────────────────────────────────────────────────────
def _parse_label(label):
    """'(r1,c1)/(r2,c2)' → two (row, col) tuples."""
    parts = re.findall(r'\((\d+),(\d+)\)', label)
    return (int(parts[0][0]), int(parts[0][1])), (int(parts[1][0]), int(parts[1][1]))


# ── Real-ESRGAN stability loading ─────────────────────────────────────────────
def load_stability():
    """
    Load mean |ΔDCT| under Real-ESRGAN from dct_stability_realesrgan.csv.
    Returns {(row, col): mean_abs_change}.  Falls back to uniform zeros
    (disables the damage term) if the file is not found.
    """
    csv_path = os.path.join(config.RESULTS_DIR, "dct_stability_realesrgan.csv")
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found — ESRGAN damage term disabled (uniform 0).")
        return {(r, c): 0.0 for r in range(8) for c in range(8)}
    df = pd.read_csv(csv_path)
    return {
        (int(row.coeff_row), int(row.coeff_col)): float(row.mean_abs_change)
        for row in df.itertuples(index=False)
    }


# ── Candidate pair loading ────────────────────────────────────────────────────
def _hardcoded_candidates():
    seen = set()
    candidates = []
    for pair_str in _FALLBACK_STRINGS:
        p1, p2 = _parse_label(pair_str)
        key = frozenset([p1, p2])
        if key in seen:
            continue
        seen.add(key)
        if key == _STD_KEY:
            candidates.append((STANDARD_LABEL, STANDARD_POS1, STANDARD_POS2))
        else:
            candidates.append((pair_str, p1, p2))
    if _STD_KEY not in seen:
        candidates.append((STANDARD_LABEL, STANDARD_POS1, STANDARD_POS2))
    return candidates


def load_candidate_pairs():
    """Top 10 unique pairs by Real-ESRGAN NC from frequency_sweep.csv + standard pair."""
    sweep_csv = os.path.join(config.RESULTS_DIR, "frequency_sweep.csv")
    if not os.path.exists(sweep_csv):
        print(f"WARNING: {sweep_csv} not found — using hardcoded fallback pairs.")
        return _hardcoded_candidates()

    df = pd.read_csv(sweep_csv)
    esrgan = df[df["attack_name"] == "real_esrgan"].copy()
    pivot = (
        esrgan.groupby(["coeff_pair", "direction"])["nc"]
        .mean()
        .reset_index()
        .pivot(index="coeff_pair", columns="direction", values="nc")
        .fillna(-np.inf)
    )
    pivot["best_dir"] = pivot[["A", "B"]].idxmax(axis=1)
    pivot["best_nc"] = pivot[["A", "B"]].max(axis=1)
    top10 = pivot.sort_values("best_nc", ascending=False).head(10)

    seen = set()
    candidates = []
    for label, row in top10.iterrows():
        p1, p2 = _parse_label(label)
        key = frozenset([p1, p2])
        if key in seen:
            continue
        seen.add(key)
        if key == _STD_KEY:
            candidates.append((STANDARD_LABEL, STANDARD_POS1, STANDARD_POS2))
        else:
            best_dir = row["best_dir"]
            pos1, pos2 = (p1, p2) if best_dir == "A" else (p2, p1)
            candidates.append((label, pos1, pos2))

    if _STD_KEY not in seen:
        candidates.append((STANDARD_LABEL, STANDARD_POS1, STANDARD_POS2))

    return candidates


# ── DCT utility ───────────────────────────────────────────────────────────────
def _dct2(block):
    return _scipy_dct(_scipy_dct(block.T, norm="ortho").T, norm="ortho")


# ── Embedding cost (imperceptibility term) ────────────────────────────────────
def _embed_cost_raw(image_rgb, pos1, pos2):
    """
    Imperceptibility component: 1 / (mean_abs(pos1) * mean_abs(pos2)) across
    all 8×8 LL blocks of the Y channel.  Lower = larger natural coefficients =
    less perturbation required to enforce the embedding margin.
    """
    ycbcr = wm_module.rgb_to_ycbcr(image_rgb.astype(np.float64))
    y = ycbcr[..., 0]
    LL, _ = pywt.dwt2(y, DWT_WAVELET)
    rows, cols = LL.shape

    mags1, mags2 = [], []
    for br in range(0, rows - BLOCK_SIZE + 1, BLOCK_SIZE):
        for bc in range(0, cols - BLOCK_SIZE + 1, BLOCK_SIZE):
            block = LL[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE]
            dct_block = _dct2(block)
            mags1.append(abs(dct_block[pos1]))
            mags2.append(abs(dct_block[pos2]))

    mean1 = float(np.mean(mags1))
    mean2 = float(np.mean(mags2))
    if mean1 == 0.0 or mean2 == 0.0:
        return np.inf
    return 1.0 / (mean1 * mean2)


def select_pair(image_rgb, candidates, stability):
    """
    Choose the best (label, pos1, pos2) for this image using:

        combined_cost = 0.3 * norm(embed_cost) + 0.7 * norm(realesrgan_damage)

    Both terms are normalised to [0, 1] across the candidate set before
    combining so neither raw scale dominates.  Returns the best label and a
    per-pair score dict with raw + normalised components.
    """
    labels = [lbl for lbl, _, _ in candidates]

    # Raw imperceptibility cost per pair (image-dependent)
    raw_embed = {
        lbl: _embed_cost_raw(image_rgb, p1, p2)
        for lbl, p1, p2 in candidates
    }
    # Raw Real-ESRGAN damage per pair (image-independent, from stability CSV)
    raw_damage = {
        lbl: (stability.get(p1, 0.0) + stability.get(p2, 0.0)) / 2.0
        for lbl, p1, p2 in candidates
    }

    # Normalise each term to [0, 1] across candidates
    ec_vals = np.array([raw_embed[l] for l in labels], dtype=float)
    dm_vals = np.array([raw_damage[l] for l in labels], dtype=float)

    def _norm(arr):
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    norm_ec = _norm(ec_vals)
    norm_dm = _norm(dm_vals)

    scores = {}
    for i, lbl in enumerate(labels):
        combined = EMBED_COST_WEIGHT * norm_ec[i] + ESRGAN_DAMAGE_WEIGHT * norm_dm[i]
        scores[lbl] = {
            "embed_cost_raw":  raw_embed[lbl],
            "damage_raw":      raw_damage[lbl],
            "embed_cost_norm": float(norm_ec[i]),
            "damage_norm":     float(norm_dm[i]),
            "combined":        float(combined),
        }

    best_label = min(labels, key=lambda l: scores[l]["combined"])
    return best_label, scores


# ── Per-image Real-ESRGAN vulnerability ──────────────────────────────────────
def compute_realesrgan_vulnerability(image_rgb):
    """
    Run Real-ESRGAN on the unwatermarked original, then measure how much the
    LL DCT coefficients shift on average:

        1. Convert original and enhanced to Y channel.
        2. Apply DWT (Haar) → LL subband of each.
        3. For every 8×8 block compute DCT2; take mean |Δ| across all 64
           coefficients and all blocks.

    Returns one scalar per image: higher = more vulnerable to ESRGAN damage.
    """
    enhanced = np.clip(
        real_esrgan_enhance(image_rgb, model_path=ESRGAN_MODEL_PATH), 0, 255
    )
    orig_y = wm_module.rgb_to_ycbcr(image_rgb.astype(np.float64))[..., 0]
    enh_y  = wm_module.rgb_to_ycbcr(enhanced.astype(np.float64))[..., 0]

    LL_orig, _ = pywt.dwt2(orig_y, DWT_WAVELET)
    LL_enh,  _ = pywt.dwt2(enh_y,  DWT_WAVELET)

    rows, cols = LL_orig.shape
    block_diffs = []
    for br in range(0, rows - BLOCK_SIZE + 1, BLOCK_SIZE):
        for bc in range(0, cols - BLOCK_SIZE + 1, BLOCK_SIZE):
            dct_orig = _dct2(LL_orig[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE])
            dct_enh  = _dct2(LL_enh [br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE])
            block_diffs.append(float(np.mean(np.abs(dct_orig - dct_enh))))

    return float(np.mean(block_diffs))


def compute_all_vulnerabilities(image_files):
    """
    First pass: run Real-ESRGAN once per original image and cache the
    vulnerability score.  Returns {img_file: score}.
    ESRGAN is only called here — never again for the same image.
    """
    print(f"\nVulnerability pass: running Real-ESRGAN on {len(image_files)} originals …")
    if not REALESRGAN_AVAILABLE:
        print("WARNING: Real-ESRGAN unavailable — all vulnerabilities will be NaN.")
    elif not os.path.exists(ESRGAN_MODEL_PATH):
        print(f"WARNING: weights not found at {ESRGAN_MODEL_PATH}.")

    scores = {}
    t0 = time.time()
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(config.ORIGINAL_IMAGES_DIR, img_file)
        raw = cv2.imread(img_path)
        if raw is None:
            print(f"  [WARN] Cannot load {img_file}")
            scores[img_file] = np.nan
            continue
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float64)
        image = cv2.resize(image, config.IMAGE_SIZE).astype(np.float64)
        try:
            score = compute_realesrgan_vulnerability(image)
        except Exception as exc:
            print(f"  [WARN] {img_file}: {exc}")
            score = np.nan
        scores[img_file] = score
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (len(image_files) - idx - 1) / rate / 60.0
        print(f"  [{idx + 1:3d}/{len(image_files)}] {img_file}"
              f"  vuln={score:.4f}  ETA {eta:.1f} min")
    return scores


def compute_alpha_from_vulnerability(vuln_score, vuln_min, vuln_max):
    """
    alpha_adaptive = 0.1 + 0.2 * normalised_vulnerability

    Least-vulnerable image → alpha = 0.1 (minimum embedding strength).
    Most-vulnerable image  → alpha = 0.3 (maximum embedding strength).
    """
    if np.isnan(vuln_score) or vuln_max == vuln_min:
        return 0.1
    return 0.1 + 0.2 * (vuln_score - vuln_min) / (vuln_max - vuln_min)


# ── Standard-results loader ───────────────────────────────────────────────────
def load_standard_results():
    """
    Load precomputed standard-(4,1)/(1,4) at alpha=0.1 results from
    metrics.csv so we don't re-run those attacks.  Returns a dict keyed by
    (image_stem_without_extension, attack_name).
    """
    csv_path = config.METRICS_CSV_PATH
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found — standard results will be NaN.")
        return {}
    df = pd.read_csv(csv_path)
    std = df[(df["alpha"] == 0.1) & (df["embedding_variant"] == "standard")]
    results = {}
    for row in std.itertuples(index=False):
        key = (str(row.image), str(row.attack_name))
        results[key] = {
            "nc":             float(row.nc),
            "ber":            float(row.ber),
            "psnr":           float(row.psnr_after_attack),
            "ssim":           float(row.ssim_after_attack),
            "psnr_watermark": float(row.psnr_watermark),
        }
    return results


# ── Embed / extract wrappers (monkey-patch pattern from frequency_sweep.py) ───
def _embed(image, wm_bits, pos1, pos2, alpha=ALPHA):
    orig1, orig2 = wm_module.COEFF_POS_1, wm_module.COEFF_POS_2
    wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = pos1, pos2
    try:
        return wm_module.embed_watermark_rgb(image.astype(np.float64), wm_bits, alpha=alpha)
    finally:
        wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = orig1, orig2


def _extract(image, pos1, pos2):
    orig1, orig2 = wm_module.COEFF_POS_1, wm_module.COEFF_POS_2
    wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = pos1, pos2
    try:
        return wm_module.extract_watermark_rgb(image.astype(np.float64), WM_SHAPE)
    finally:
        wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = orig1, orig2


# ── Attack dispatch ───────────────────────────────────────────────────────────
def _apply_attack(attack_name, image):
    if attack_name == "real_esrgan":
        return real_esrgan_enhance(image, model_path=ESRGAN_MODEL_PATH)
    if attack_name == "jpeg_compression_q50":
        return jpeg_compression(image, quality=50)
    if attack_name == "gaussian_blur_5x5":
        return gaussian_blur(image, ksize=5, sigma=1.0)
    raise ValueError(f"Unknown attack: {attack_name}")


# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check_kodim01(candidates, stability):
    """
    Print per-pair cost breakdown for kodim01 and confirm the selected pair
    (minimum combined cost) is actually the minimum.
    """
    img_path = os.path.join(config.ORIGINAL_IMAGES_DIR, "kodim01.png")
    raw = cv2.imread(img_path)
    if raw is None:
        print(f"SANITY CHECK FAILED: cannot load {img_path}")
        return

    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float64)
    image = cv2.resize(image, config.IMAGE_SIZE).astype(np.float64)

    print("\n--- Sanity check: kodim01.png ---")
    hdr = (f"  {'Pair':<25} {'EmbCost(raw)':>13} {'Damage(raw)':>12}"
           f" {'EmbCost(n)':>11} {'Damage(n)':>10} {'Combined':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    best_label, scores = select_pair(image, candidates, stability)

    for label, pos1, pos2 in candidates:
        s = scores[label]
        marker = "  ←" if label == best_label else ""
        print(
            f"  {label:<25} {s['embed_cost_raw']:>13.6f} {s['damage_raw']:>12.4f}"
            f" {s['embed_cost_norm']:>11.4f} {s['damage_norm']:>10.4f}"
            f" {s['combined']:>10.4f}{marker}"
        )

    print()
    print(f"  Selected pair (lowest combined cost): {best_label}"
          f"  combined={scores[best_label]['combined']:.6f}")
    print()

    # Verify the selection logic
    best_combined = scores[best_label]["combined"]
    all_ge = all(s["combined"] >= best_combined - 1e-12 for s in scores.values())
    if all_ge:
        print("  PASS: selected pair has the lowest combined cost among all candidates.")
    else:
        lower = [(k, v["combined"]) for k, v in scores.items()
                 if v["combined"] < best_combined - 1e-12]
        print(f"  FAIL: pairs with lower combined cost than the selected one: {lower}")


def sanity_check_vulnerability():
    """
    Run Real-ESRGAN on kodim01 and kodim05, compute per-image vulnerability
    scores, and show the provisional alpha_adaptive for each.

    Alpha normalisation is relative to just these two images — in the full
    experiment it is normalised across all 100 — but this is sufficient to
    verify that the two images receive meaningfully different alpha values.
    """
    preview = ["kodim01.png", "kodim05.png"]

    print("\n--- Sanity check: per-image vulnerability (kodim01 vs kodim05) ---")

    if not REALESRGAN_AVAILABLE:
        print("  Real-ESRGAN not available — skipping vulnerability sanity check.")
        return
    if not os.path.exists(ESRGAN_MODEL_PATH):
        print(f"  Weights not found at {ESRGAN_MODEL_PATH} — skipping.")
        return

    scores = {}
    for img_file in preview:
        img_path = os.path.join(config.ORIGINAL_IMAGES_DIR, img_file)
        raw = cv2.imread(img_path)
        if raw is None:
            print(f"  [WARN] Cannot load {img_file}")
            scores[img_file] = np.nan
            continue
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float64)
        image = cv2.resize(image, config.IMAGE_SIZE).astype(np.float64)
        try:
            score = compute_realesrgan_vulnerability(image)
        except Exception as exc:
            print(f"  [WARN] {img_file}: {exc}")
            score = np.nan
        scores[img_file] = score
        print(f"  {img_file}: vulnerability = {score:.6f}")

    valid_scores = [v for v in scores.values() if not np.isnan(v)]
    if len(valid_scores) < 2:
        print("  Cannot normalise — fewer than 2 valid scores.")
        return

    vuln_min, vuln_max = min(valid_scores), max(valid_scores)
    diff = vuln_max - vuln_min

    print()
    print(f"  {'Image':<15} {'Vulnerability':>14} {'Norm':>7} {'α_adaptive':>11}")
    print("  " + "-" * 52)
    for img_file in preview:
        v = scores.get(img_file, np.nan)
        if np.isnan(v):
            print(f"  {img_file:<15} {'NaN':>14}")
            continue
        norm  = (v - vuln_min) / (vuln_max - vuln_min)
        alpha = compute_alpha_from_vulnerability(v, vuln_min, vuln_max)
        print(f"  {img_file:<15} {v:>14.6f} {norm:>7.4f} {alpha:>11.4f}")

    print()
    print(f"  Vulnerability difference: {diff:.6f}")
    print(f"  Alpha spread (2-image preview): "
          f"{compute_alpha_from_vulnerability(vuln_min, vuln_min, vuln_max):.4f}"
          f" → {compute_alpha_from_vulnerability(vuln_max, vuln_min, vuln_max):.4f}")
    print(f"  Note: final α_adaptive normalises across all 100 images, so the "
          f"absolute values will shift, but the ordering holds.")


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_nc_comparison(main_df):
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    agg = main_df.groupby("attack_name")[
        ["nc_standard", "nc_adaptive", "nc_adaptive_strength"]
    ].mean()
    attacks_ordered = [a for a in ATTACKS if a in agg.index]
    x = np.arange(len(attacks_ordered))
    width = 0.25

    conditions = [
        ("nc_standard",          f"Standard {STANDARD_LABEL} α=0.1", "#e67e22"),
        ("nc_adaptive",          "Adaptive pos  α=0.1",              "#2980b9"),
        ("nc_adaptive_strength", "Adaptive pos+strength α_adaptive", "#27ae60"),
    ]
    offsets = np.array([-1, 0, 1]) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for (col, label, color), offset in zip(conditions, offsets):
        vals = [agg.loc[a, col] if a in agg.index else np.nan for a in attacks_ordered]
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_DISPLAY.get(a, a) for a in attacks_ordered], fontsize=11)
    ax.set_ylabel("Mean NC", fontsize=11)
    ax.set_title("Adaptive vs Standard Embedding — Mean NC per Attack\n"
                 "(3 conditions: standard, adaptive-position, adaptive-position+strength)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(config.PLOTS_DIR, "adaptive_vs_standard_nc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Main experiment ───────────────────────────────────────────────────────────
def run_experiment(candidates, wm, stability, standard_results,
                   vuln_scores, vuln_min, vuln_max):
    """
    Three conditions per image × attack:
      1. Standard: (4,1)/(1,4) at alpha=0.1 — loaded from metrics.csv
      2. Adaptive position only: selected pair at alpha=0.1
      3. Adaptive position + strength: selected pair at alpha_adaptive
         where alpha_adaptive = 0.1 + 0.2 * normalised_vulnerability
    """
    image_files = sorted(
        f for f in os.listdir(config.ORIGINAL_IMAGES_DIR)
        if f.lower().endswith(".png")
    )
    print(f"\nImages: {len(image_files)}  |  Candidate pairs: {len(candidates)}")
    print(f"Attacks: {ATTACKS}")
    print(f"Standard results preloaded: {len(standard_results)} rows from metrics.csv")

    if not REALESRGAN_AVAILABLE:
        print("WARNING: Real-ESRGAN not installed — real_esrgan results will be NaN.")
    elif not os.path.exists(ESRGAN_MODEL_PATH):
        print(f"WARNING: weights not found at {ESRGAN_MODEL_PATH} — real_esrgan may fail.")

    main_records = []
    selection_records = []
    t0 = time.time()

    for img_idx, img_file in enumerate(image_files):
        img_path = os.path.join(config.ORIGINAL_IMAGES_DIR, img_file)
        raw = cv2.imread(img_path)
        if raw is None:
            print(f"  [WARN] Cannot load {img_file} — skipping")
            continue

        original = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float64)
        original = cv2.resize(original, config.IMAGE_SIZE).astype(np.float64)
        img_stem = os.path.splitext(img_file)[0]  # "kodim01" — matches metrics.csv key

        # ── per-image pair selection + vulnerability-based alpha ──────────────
        best_label, scores = select_pair(original, candidates, stability)
        best_cost = scores[best_label]["combined"]
        best_pos1, best_pos2 = next(
            (p1, p2) for lbl, p1, p2 in candidates if lbl == best_label
        )
        alpha_adaptive = compute_alpha_from_vulnerability(
            vuln_scores.get(img_file, np.nan), vuln_min, vuln_max
        )

        # ── condition 2: adaptive position at alpha=0.1 ───────────────────────
        try:
            wm_adaptive = np.clip(
                _embed(original, wm, best_pos1, best_pos2, alpha=ALPHA), 0, 255
            )
        except Exception as exc:
            print(f"  [WARN] Adaptive embed failed {img_file}: {exc}")
            wm_adaptive = None

        # ── condition 3: adaptive position at alpha_adaptive ──────────────────
        try:
            wm_strength = np.clip(
                _embed(original, wm, best_pos1, best_pos2, alpha=alpha_adaptive), 0, 255
            )
        except Exception as exc:
            print(f"  [WARN] Strength embed failed {img_file}: {exc}")
            wm_strength = None

        # ── watermarked-image imperceptibility (no attack) ────────────────────
        std_wm_row = standard_results.get((img_stem, "watermarked_only"), {})
        standard_wm_psnr = std_wm_row.get("psnr", np.nan)
        adaptive_wm_psnr = (
            compute_psnr(original, wm_adaptive) if wm_adaptive is not None else np.nan
        )
        psnr_improvement = (
            adaptive_wm_psnr - standard_wm_psnr
            if not (np.isnan(adaptive_wm_psnr) or np.isnan(standard_wm_psnr))
            else np.nan
        )
        selection_records.append({
            "image":            img_file,
            "selected_pair":    best_label,
            "embedding_cost":   best_cost,
            "alpha_adaptive":   alpha_adaptive,
            "standard_psnr":    standard_wm_psnr,
            "adaptive_psnr":    adaptive_wm_psnr,
            "psnr_improvement": psnr_improvement,
        })

        # ── attacks ───────────────────────────────────────────────────────────
        for attack_name in ATTACKS:
            # condition 1 — standard: load from metrics.csv
            std_row = standard_results.get((img_stem, attack_name), {})
            nc_s   = std_row.get("nc",   np.nan)
            psnr_s = std_row.get("psnr", np.nan)
            ssim_s = std_row.get("ssim", np.nan)

            # condition 2 — adaptive position at alpha=0.1
            nc_a = psnr_a = ssim_a = np.nan
            if wm_adaptive is not None:
                try:
                    attacked_a = np.clip(_apply_attack(attack_name, wm_adaptive), 0, 255)
                    extracted_a = _extract(attacked_a, best_pos1, best_pos2)
                    nc_a   = normalized_correlation(wm, extracted_a)
                    psnr_a = compute_psnr(original, attacked_a)
                    ssim_a = compute_ssim(original, attacked_a)
                except Exception as exc:
                    print(f"  [WARN] Adaptive {attack_name} failed {img_file}: {exc}")

            # condition 3 — adaptive position + strength
            nc_as = ber_as = psnr_as = ssim_as = np.nan
            if wm_strength is not None:
                try:
                    attacked_as = np.clip(_apply_attack(attack_name, wm_strength), 0, 255)
                    extracted_as = _extract(attacked_as, best_pos1, best_pos2)
                    nc_as   = normalized_correlation(wm, extracted_as)
                    ber_as  = bit_error_rate(wm, extracted_as)
                    psnr_as = compute_psnr(original, attacked_as)
                    ssim_as = compute_ssim(original, attacked_as)
                except Exception as exc:
                    print(f"  [WARN] Strength {attack_name} failed {img_file}: {exc}")

            main_records.append({
                "image":                  img_file,
                "selected_pair":          best_label,
                "embedding_cost":         best_cost,
                "attack_name":            attack_name,
                "nc_adaptive":            nc_a,
                "nc_standard":            nc_s,
                "psnr_adaptive":          psnr_a,
                "psnr_standard":          psnr_s,
                "ssim_adaptive":          ssim_a,
                "ssim_standard":          ssim_s,
                "alpha_adaptive":         alpha_adaptive,
                "nc_adaptive_strength":   nc_as,
                "ber_adaptive_strength":  ber_as,
                "psnr_adaptive_strength": psnr_as,
                "ssim_adaptive_strength": ssim_as,
            })

        elapsed = time.time() - t0
        rate = (img_idx + 1) / elapsed
        eta = (len(image_files) - img_idx - 1) / rate / 60.0
        print(
            f"  [{img_idx + 1:3d}/{len(image_files)}] {img_file}"
            f"  selected={best_label}  α_adapt={alpha_adaptive:.3f}"
            f"  cost={best_cost:.4f}  ETA {eta:.1f} min"
        )

    return pd.DataFrame(main_records), pd.DataFrame(selection_records)


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(main_df, selection_df):
    sep = "=" * 66
    print(f"\n{sep}")
    print("Adaptive Embedding Experiment — Summary")
    print(sep)

    n_images = len(selection_df)
    n_unique = selection_df["selected_pair"].nunique()
    most_common = selection_df["selected_pair"].value_counts().idxmax()
    most_common_count = selection_df["selected_pair"].value_counts().max()
    print(f"Unique pairs selected across {n_images} images: {n_unique}")
    print(f"Most frequently selected pair: {most_common} ({most_common_count}/{n_images} images)")
    print(f"Mean adaptive alpha: {selection_df['alpha_adaptive'].mean():.4f}"
          f"  (range {selection_df['alpha_adaptive'].min():.4f}"
          f"–{selection_df['alpha_adaptive'].max():.4f})")

    # NC improvement under Real-ESRGAN — all three conditions
    esrgan_df = main_df[main_df["attack_name"] == "real_esrgan"]
    nc_s  = esrgan_df["nc_standard"].mean()
    nc_a  = esrgan_df["nc_adaptive"].mean()
    nc_as = esrgan_df["nc_adaptive_strength"].mean()
    print(f"\nMean NC — Real-ESRGAN:")
    print(f"  Standard             : {nc_s:.4f}")
    print(f"  Adaptive pos         : {nc_a:.4f}  (Δ {nc_a - nc_s:+.4f} vs standard)")
    print(f"  Adaptive pos+strength: {nc_as:.4f}  (Δ {nc_as - nc_s:+.4f} vs standard)")

    # PSNR (watermarked image, no attack)
    mean_psnr_a = selection_df["adaptive_psnr"].mean()
    mean_psnr_s = selection_df["standard_psnr"].mean()
    mean_delta  = selection_df["psnr_improvement"].mean()
    print(f"\nMean PSNR (watermarked image vs original, no attack):")
    print(f"  Standard   : {mean_psnr_s:.2f} dB")
    print(f"  Adaptive   : {mean_psnr_a:.2f} dB  (Δ {mean_delta:+.2f} dB)")

    # Fraction beating standard on both NC (avg all attacks) and PSNR (pos-only)
    per_image_nc_a = main_df.groupby("image")["nc_adaptive"].mean()
    per_image_nc_s = main_df.groupby("image")["nc_standard"].mean()
    per_image_nc_as = main_df.groupby("image")["nc_adaptive_strength"].mean()
    sel_idx = selection_df.set_index("image")

    images = sorted(set(per_image_nc_a.index) & set(sel_idx.index))

    beat_both_pos = sum(
        1 for img in images
        if (per_image_nc_a.get(img, np.nan) > per_image_nc_s.get(img, np.nan)
            and sel_idx.loc[img, "psnr_improvement"] > 0)
    )
    beat_both_str = sum(
        1 for img in images
        if (per_image_nc_as.get(img, np.nan) > per_image_nc_s.get(img, np.nan)
            and sel_idx.loc[img, "psnr_improvement"] > 0)
    )
    n = len(images)
    print(f"\nFraction beating standard on BOTH NC and PSNR:")
    print(f"  Adaptive pos         : {beat_both_pos}/{n} = {beat_both_pos/n:.1%}")
    print(f"  Adaptive pos+strength: {beat_both_str}/{n} = {beat_both_str/n:.1%}")
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Adaptive DCT coefficient pair selection experiment"
    )
    parser.add_argument(
        "--sanity-only", action="store_true",
        help="Run sanity check on kodim01 only, then exit without running the full experiment",
    )
    args = parser.parse_args()

    candidates = load_candidate_pairs()
    stability = load_stability()
    standard_results = load_standard_results()

    print(f"\nCandidate pairs ({len(candidates)} total):")
    for i, (label, pos1, pos2) in enumerate(candidates, 1):
        tag = "  ← standard baseline" if label == STANDARD_LABEL else ""
        print(f"  {i:2d}. {label:<25}  pos1={pos1}  pos2={pos2}{tag}")

    print(f"\nCost weights: embed_cost={EMBED_COST_WEIGHT}  esrgan_damage={ESRGAN_DAMAGE_WEIGHT}")

    sanity_check_kodim01(candidates, stability)
    sanity_check_vulnerability()

    print("Confirmation: results/metrics.csv will NOT be modified.")
    print("Output files (written only during the full experiment):")
    print(f"  {os.path.join(config.RESULTS_DIR, 'adaptive_embedding.csv')}")
    print(f"  {os.path.join(config.RESULTS_DIR, 'adaptive_embedding_pair_selection.csv')}")
    print(f"  {os.path.join(config.PLOTS_DIR, 'adaptive_vs_standard_nc.png')}")

    if args.sanity_only:
        print("\n--sanity-only flag set; exiting before full experiment.")
        return

    # ── Vulnerability first pass (ESRGAN runs once per image here) ────────────
    image_files = sorted(
        f for f in os.listdir(config.ORIGINAL_IMAGES_DIR)
        if f.lower().endswith(".png")
    )
    vuln_scores = compute_all_vulnerabilities(image_files)
    valid_vulns = [v for v in vuln_scores.values() if not np.isnan(v)]
    if valid_vulns:
        vuln_min, vuln_max = min(valid_vulns), max(valid_vulns)
        print(f"\nVulnerability range: {vuln_min:.4f} – {vuln_max:.4f}"
              f"  (spread {vuln_max - vuln_min:.4f})")
        print(f"Resulting alpha range: "
              f"{compute_alpha_from_vulnerability(vuln_min, vuln_min, vuln_max):.4f}"
              f" – {compute_alpha_from_vulnerability(vuln_max, vuln_min, vuln_max):.4f}")
    else:
        vuln_min, vuln_max = 0.0, 0.0
        print("WARNING: no valid vulnerability scores — alpha_adaptive will be 0.1 for all.")

    wm = wm_module.generate_watermark(shape=WM_SHAPE, seed=RANDOM_SEED)
    main_df, selection_df = run_experiment(
        candidates, wm, stability, standard_results,
        vuln_scores, vuln_min, vuln_max,
    )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    main_out = os.path.join(config.RESULTS_DIR, "adaptive_embedding.csv")
    sel_out = os.path.join(config.RESULTS_DIR, "adaptive_embedding_pair_selection.csv")
    main_df.to_csv(main_out, index=False)
    print(f"\nSaved {main_out}  ({len(main_df)} rows)")
    selection_df.to_csv(sel_out, index=False)
    print(f"Saved {sel_out}  ({len(selection_df)} rows)")

    plot_nc_comparison(main_df)
    print_summary(main_df, selection_df)


if __name__ == "__main__":
    main()
