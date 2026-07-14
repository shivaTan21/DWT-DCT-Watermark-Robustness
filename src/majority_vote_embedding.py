"""
Majority-vote redundant watermark embedding experiment.

Each of N=128 watermark bits is embedded redundantly across K coefficient pairs
identified as stable under Real-ESRGAN.  During extraction every copy votes
independently; the final bit is the majority vote.  This trades payload
capacity (coefficient-pair slots per block) for robustness.

Usage:
    python src/majority_vote_embedding.py              # sanity check only
    python src/majority_vote_embedding.py --experiment # full experiment
"""

import argparse
import ast
import os
import sys
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import pywt

# ── MPS autocast patch (must precede any py_real_esrgan import) ───────────────
try:
    import torch as _torch
    if (not _torch.cuda.is_available()) and _torch.backends.mps.is_available():
        _torch.cuda.amp.autocast = (
            lambda *a, **kw: _torch.amp.autocast("cpu", *a, **kw)
        )
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project imports ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from src.watermark import (
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    _dct2,
    _idct2,
    BLOCK_SIZE,
    DEFAULT_WAVELET,
)
from src.metrics import normalized_correlation, bit_error_rate
from src.attacks_traditional import jpeg_compression, gaussian_blur
from src.attacks_ai import real_esrgan_enhance, REALESRGAN_AVAILABLE

# ── constants ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_BITS = 128          # effective watermark length
WM_SHAPE = (8, 16)   # 8×16 = 128 bits
DEFAULT_ALPHA = 0.1
ABSOLUTE_MARGIN_FLOOR = 30.0  # mirrors watermark.py
SANITY_NC_THRESHOLD = 0.99
K_VALUES = [3, 5, 7]

RESULTS_DIR = os.path.join(_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
IMAGES_DIR = os.path.join(_ROOT, "data", "original_images")
FREQ_SWEEP_CSV = os.path.join(RESULTS_DIR, "frequency_sweep.csv")
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "majority_vote.csv")
ESRGAN_MODEL = os.path.join(_ROOT, "weights", "RealESRGAN_x4.pth")

ATTACKS = ["real_esrgan", "jpeg_compression_q50", "gaussian_blur_5x5"]
ATTACK_DISPLAY = {
    "real_esrgan": "Real-ESRGAN",
    "jpeg_compression_q50": "JPEG Q50",
    "gaussian_blur_5x5": "Gaussian Blur 5×5",
}

SANITY_IMAGES = ["kodim01", "kodim05", "t001"]


# ── pair selection ─────────────────────────────────────────────────────────────

def load_top_pairs(k: int, csv_path: str = FREQ_SWEEP_CSV):
    """
    Return the top-k coefficient pairs by mean Real-ESRGAN NC (best direction).

    Each entry is (pos1, pos2, canonical_label).
    Exact-duplicate pairs are silently deduplicated.
    A warning is printed if any coefficient position is shared across pairs.
    """
    df = pd.read_csv(csv_path)
    esrgan_df = df[df["attack_name"] == "real_esrgan"]

    mean_nc = esrgan_df.groupby(["coeff_pair", "direction"])["nc"].mean()
    best = mean_nc.reset_index()
    # keep best direction per pair
    best = best.loc[best.groupby("coeff_pair")["nc"].idxmax()]
    best = best.sort_values("nc", ascending=False)

    selected = []
    seen_pair_keys: set = set()

    for _, row in best.iterrows():
        if len(selected) >= k:
            break
        label = row["coeff_pair"].strip('"')
        direction = row["direction"]

        # parse canonical label "(r1,c1)/(r2,c2)" where (r1,c1) has lower index
        left, right = label.split("/")
        pos_a = tuple(ast.literal_eval(left))   # lower linear index
        pos_b = tuple(ast.literal_eval(right))  # higher linear index

        # direction A → pos1=pos_a, pos2=pos_b (standard)
        # direction B → pos1=pos_b, pos2=pos_a (reversed; bit=1 when higher-idx > lower-idx)
        pos1, pos2 = (pos_a, pos_b) if direction == "A" else (pos_b, pos_a)

        pair_key = (pos1, pos2)
        if pair_key in seen_pair_keys:
            continue  # exact duplicate — skip silently
        seen_pair_keys.add(pair_key)
        selected.append((pos1, pos2, label))

    # warn about shared positions (expected for (7,7)-anchored pairs)
    all_positions = []
    for pos1, pos2, _ in selected:
        all_positions.extend([pos1, pos2])
    shared = [p for p, cnt in Counter(all_positions).items() if cnt > 1]
    if shared:
        print(
            f"  [WARNING K={k}] Coefficient positions shared across pairs: {shared}"
        )

    return selected


# ── embedding and extraction ───────────────────────────────────────────────────

def embed_majority_vote(image: np.ndarray, watermark: np.ndarray,
                        pairs: list, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """
    Embed N watermark bits redundantly across K coefficient pairs.

    For each of the N blocks the same bit is encoded K times — once per pair —
    via successive DCT→modify→IDCT rounds on the same block.

    Position locking: when multiple pairs share a coefficient position (e.g.
    (7,7) appearing in three of the K=3 top pairs), the first pair to modify
    that position locks it.  Subsequent pairs that reference the locked position
    push their full embedding delta to the *other*, unlocked position so all K
    comparisons remain independently decodable at extraction time.  If both
    positions in a pair are already locked the pair is skipped silently.

    Parameters
    ----------
    image     : (H, W, 3) float64 RGB
    watermark : binary array, .size must be <= LL-subband block count
    pairs     : list of (pos1, pos2) tuples (length K)
    alpha     : embedding strength (same role as in watermark.py)

    Returns
    -------
    (H, W, 3) float64 RGB with watermark embedded.
    """
    bits = watermark.flatten()

    ycbcr = rgb_to_ycbcr(image.astype(np.float64))
    y = ycbcr[..., 0]

    LL, (LH, HL, HH) = pywt.dwt2(y, DEFAULT_WAVELET)
    blocks_per_row = LL.shape[1] // BLOCK_SIZE
    n_available = (LL.shape[0] // BLOCK_SIZE) * blocks_per_row
    assert bits.size <= n_available, (
        f"Watermark needs {bits.size} blocks but LL has only {n_available}."
    )

    LL_out = LL.copy()
    for idx, bit in enumerate(bits):
        br = (idx // blocks_per_row) * BLOCK_SIZE
        bc = (idx % blocks_per_row) * BLOCK_SIZE

        locked: set = set()  # positions already modified in this block

        for pos1, pos2 in pairs:
            block = LL_out[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE]
            dct_block = _dct2(block)

            c1 = dct_block[pos1]
            c2 = dct_block[pos2]
            energy = (abs(c1) + abs(c2)) / 2.0
            margin = alpha * energy + ABSOLUTE_MARGIN_FLOOR

            lock1 = pos1 in locked
            lock2 = pos2 in locked

            if bit == 1:
                deficit = margin - (c1 - c2)
                if deficit > 0:
                    if not lock1 and not lock2:
                        c1 += deficit / 2.0
                        c2 -= deficit / 2.0
                    elif not lock1:   # pos2 locked → push all delta to pos1
                        c1 += deficit
                    elif not lock2:   # pos1 locked → push all delta to pos2
                        c2 -= deficit
                    # both locked → skip silently
            else:
                deficit = margin - (c2 - c1)
                if deficit > 0:
                    if not lock1 and not lock2:
                        c2 += deficit / 2.0
                        c1 -= deficit / 2.0
                    elif not lock2:   # pos1 locked → push all delta to pos2
                        c2 += deficit
                    elif not lock1:   # pos2 locked → push all delta to pos1
                        c1 -= deficit
                    # both locked → skip silently

            if not lock1:
                dct_block[pos1] = c1
                locked.add(pos1)
            if not lock2:
                dct_block[pos2] = c2
                locked.add(pos2)

            LL_out[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE] = _idct2(dct_block)

    y_out = pywt.idwt2((LL_out, (LH, HL, HH)), DEFAULT_WAVELET)
    y_out = y_out[: y.shape[0], : y.shape[1]]

    ycbcr_out = ycbcr.copy()
    ycbcr_out[..., 0] = y_out
    return ycbcr_to_rgb(ycbcr_out)


def extract_majority_vote(image: np.ndarray, n_bits: int,
                          pairs: list) -> np.ndarray:
    """
    Extract watermark bits via majority vote across K coefficient pairs.

    A single DCT is computed per block; all K pairs read from the same DCT.
    Ties (only possible for even K) are broken by the first pair's vote.

    Parameters
    ----------
    image  : (H, W, 3) float64 or uint8 RGB
    n_bits : number of bits to extract
    pairs  : list of (pos1, pos2) tuples (length K)

    Returns
    -------
    1-D uint8 array of length n_bits.
    """
    ycbcr = rgb_to_ycbcr(image.astype(np.float64))
    y = ycbcr[..., 0]

    LL, _ = pywt.dwt2(y, DEFAULT_WAVELET)
    blocks_per_row = LL.shape[1] // BLOCK_SIZE

    bits = np.zeros(n_bits, dtype=np.uint8)
    for idx in range(n_bits):
        br = (idx // blocks_per_row) * BLOCK_SIZE
        bc = (idx % blocks_per_row) * BLOCK_SIZE
        block = LL[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE]
        dct_block = _dct2(block)

        votes = [1 if dct_block[pos1] > dct_block[pos2] else 0
                 for pos1, pos2 in pairs]

        vote_sum = sum(votes)
        half = len(votes) / 2.0
        if vote_sum > half:
            bits[idx] = 1
        elif vote_sum < half:
            bits[idx] = 0
        else:
            bits[idx] = votes[0]  # tie-break: first pair's vote

    return bits


# ── image I/O ──────────────────────────────────────────────────────────────────

def load_image(name: str) -> np.ndarray:
    """Load image by stem name (without extension), resize to 512×512, return float64 RGB."""
    path = os.path.join(IMAGES_DIR, name + ".png")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    return img.astype(np.float64)


def list_all_images() -> list:
    """Return sorted list of all image stem names in IMAGES_DIR."""
    names = []
    for fn in sorted(os.listdir(IMAGES_DIR)):
        if fn.lower().endswith(".png"):
            names.append(os.path.splitext(fn)[0])
    return names


# ── attack dispatch ────────────────────────────────────────────────────────────

def apply_attack(name: str, image: np.ndarray) -> np.ndarray:
    if name == "real_esrgan":
        if not REALESRGAN_AVAILABLE:
            raise RuntimeError("py-real-esrgan is not installed.")
        return real_esrgan_enhance(image, model_path=ESRGAN_MODEL)
    if name == "jpeg_compression_q50":
        return jpeg_compression(image, quality=50)
    if name == "gaussian_blur_5x5":
        return gaussian_blur(image, ksize=5, sigma=1.0)
    raise ValueError(f"Unknown attack: {name}")


# ── watermark generation ───────────────────────────────────────────────────────

def make_watermark(seed: int = RANDOM_SEED) -> np.ndarray:
    """Generate a reproducible 128-bit binary watermark."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=WM_SHAPE).astype(np.uint8)


# ── sanity check ──────────────────────────────────────────────────────────────

def run_sanity_check(pairs_by_k: dict, watermark: np.ndarray) -> bool:
    """
    Embed then immediately extract (no attack) for each K and each sanity image.
    Verify NC >= 0.99.  Also print effective payload capacity for each K.
    """
    print("\n=== Majority Vote Sanity Check ===")

    for k in K_VALUES:
        labels = [label for _, _, label in pairs_by_k[k]]
        print(f"K={k} pairs: [{', '.join(labels)}]")

    print()
    for k in K_VALUES:
        n_pair_mods = k * N_BITS
        print(
            f"K={k}: {N_BITS} effective bits, {n_pair_mods} total coefficient-pair"
            f" modifications ({k} per block × {N_BITS} blocks)"
        )

    print()
    all_pass = True
    for img_name in SANITY_IMAGES:
        try:
            img = load_image(img_name)
        except FileNotFoundError as exc:
            print(f"  SKIP {img_name}: {exc}")
            continue

        for k in K_VALUES:
            pairs = [(p1, p2) for p1, p2, _ in pairs_by_k[k]]
            embedded = embed_majority_vote(img, watermark, pairs)
            embedded_u8 = np.clip(embedded, 0, 255).astype(np.uint8)
            extracted = extract_majority_vote(embedded_u8.astype(np.float64),
                                              N_BITS, pairs)
            nc = normalized_correlation(watermark, extracted)
            status = "PASS" if nc >= SANITY_NC_THRESHOLD else "FAIL"
            if nc < SANITY_NC_THRESHOLD:
                all_pass = False
            print(f"{img_name} / K={k}: NC={nc:.4f} {status}")

    print()
    if all_pass:
        print("All sanity checks PASSED")
    else:
        print("WARNING: Some sanity checks FAILED")

    return all_pass


# ── main experiment ────────────────────────────────────────────────────────────

def load_standard_baseline() -> pd.DataFrame:
    """Load K=1 baseline rows from metrics.csv (alpha=0.1, variant=standard)."""
    df = pd.read_csv(METRICS_CSV)
    std = df[
        (df["alpha"] == 0.1)
        & (df["embedding_variant"] == "standard")
        & (df["attack_name"].isin(ATTACKS))
    ][["image", "attack_name", "nc", "ber"]].copy()
    std["k_value"] = 1
    return std


def run_experiment(pairs_by_k: dict, watermark: np.ndarray) -> pd.DataFrame:
    """
    Embed → attack → extract for every (K, image, attack) combination.
    Returns a DataFrame with columns: k_value, image, attack_name, nc, ber.
    """
    image_names = list_all_images()
    records = []

    for k in K_VALUES:
        pairs = [(p1, p2) for p1, p2, _ in pairs_by_k[k]]
        print(f"\n=== K={k} ({len(image_names)} images × {len(ATTACKS)} attacks) ===")

        for i, img_name in enumerate(image_names):
            img = load_image(img_name)
            embedded = embed_majority_vote(img, watermark, pairs, alpha=DEFAULT_ALPHA)
            embedded_u8 = np.clip(embedded, 0, 255).astype(np.uint8)

            row_parts = []
            for attack_name in ATTACKS:
                attacked = apply_attack(attack_name, embedded_u8.astype(np.float64))
                extracted = extract_majority_vote(attacked, N_BITS, pairs)
                nc = normalized_correlation(watermark, extracted)
                ber = bit_error_rate(watermark, extracted)
                records.append(
                    dict(k_value=k, image=img_name, attack_name=attack_name,
                         nc=nc, ber=ber)
                )
                row_parts.append(f"{attack_name[:4]}={nc:.3f}")

            print(f"  [{i+1:3d}/100] {img_name:12s}  {' | '.join(row_parts)}")

    return pd.DataFrame(records)


# ── output: CSV, plot, summary table ──────────────────────────────────────────

def save_csv(mv_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    combined = pd.concat(
        [baseline_df[["k_value", "image", "attack_name", "nc", "ber"]], mv_df],
        ignore_index=True,
    )
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults written to {OUTPUT_CSV}")


def plot_results(df: pd.DataFrame) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    k_values = [1, 3, 5, 7]
    colors = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f"]
    atk_order = ["real_esrgan", "jpeg_compression_q50", "gaussian_blur_5x5"]
    atk_labels = [ATTACK_DISPLAY[a] for a in atk_order]

    mean_nc = df.groupby(["k_value", "attack_name"])["nc"].mean()

    x = np.arange(len(atk_order))
    n_k = len(k_values)
    width = 0.18
    offsets = np.linspace(-(n_k - 1) / 2, (n_k - 1) / 2, n_k) * width

    fig, ax = plt.subplots(figsize=(10, 6))
    for ki, (k, color, offset) in enumerate(zip(k_values, colors, offsets)):
        ncs = [mean_nc.get((k, atk), np.nan) for atk in atk_order]
        ax.bar(x + offset, ncs, width, label=f"K={k}", color=color, alpha=0.9,
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Attack", fontsize=12)
    ax.set_ylabel("Mean NC", fontsize=12)
    ax.set_title("Majority-Vote Watermark Robustness\n(mean NC across 100 images)",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(atk_labels, fontsize=11)
    ax.legend(title="Redundancy", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "majority_vote_nc.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_path}")


def print_summary(df: pd.DataFrame) -> None:
    atk_cols = ["real_esrgan", "jpeg_compression_q50", "gaussian_blur_5x5"]
    mean_nc = df.groupby(["k_value", "attack_name"])["nc"].mean()

    k1_esrgan = mean_nc.get((1, "real_esrgan"), float("nan"))
    k1_jpeg = mean_nc.get((1, "jpeg_compression_q50"), float("nan"))
    k1_blur = mean_nc.get((1, "gaussian_blur_5x5"), float("nan"))

    print()
    print("K  | Real-ESRGAN NC | JPEG NC | Blur NC | vs Standard")
    print("---|----------------|---------|---------|------------")
    for k in [1, 3, 5, 7]:
        esrgan_nc = mean_nc.get((k, "real_esrgan"), float("nan"))
        jpeg_nc = mean_nc.get((k, "jpeg_compression_q50"), float("nan"))
        blur_nc = mean_nc.get((k, "gaussian_blur_5x5"), float("nan"))
        if k == 1:
            vs = "baseline"
        else:
            delta = esrgan_nc - k1_esrgan
            vs = f"{delta:+.3f}"
        print(
            f"{k}  | {esrgan_nc:.3f}          | {jpeg_nc:.3f}   | {blur_nc:.3f}   | {vs}"
        )


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", action="store_true",
        help="Run full experiment (default: sanity check only).",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    watermark = make_watermark(seed=RANDOM_SEED)

    # ── load pairs ─────────────────────────────────────────────────────────────
    print("Loading top pairs from frequency_sweep.csv …")
    pairs_by_k: dict = {}
    for k in K_VALUES:
        pairs_by_k[k] = load_top_pairs(k)

    # ── sanity check ───────────────────────────────────────────────────────────
    passed = run_sanity_check(pairs_by_k, watermark)
    if not passed:
        print("\nAborting: sanity check failed. Fix embedding before running experiment.")
        sys.exit(1)

    if not args.experiment:
        print("\nPass --experiment to run the full 100-image experiment.")
        return

    # ── full experiment ─────────────────────────────────────────────────────────
    print("\nRunning majority-vote experiment …")
    mv_df = run_experiment(pairs_by_k, watermark)

    print("\nLoading standard baseline from metrics.csv …")
    baseline_df = load_standard_baseline()

    all_df = pd.concat(
        [baseline_df[["k_value", "image", "attack_name", "nc", "ber"]], mv_df],
        ignore_index=True,
    )

    save_csv(mv_df, baseline_df)
    plot_results(all_df)
    print_summary(all_df)


if __name__ == "__main__":
    main()
