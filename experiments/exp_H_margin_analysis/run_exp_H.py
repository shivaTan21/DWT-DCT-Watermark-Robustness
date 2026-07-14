"""
Experiment H: Decision-margin analysis of watermark survival under Real-ESRGAN.

Motivation
----------
Exp F falsified the analytical prediction from Exp D: the "balanced" pair
(2,3)/(3,2), selected for minimal absolute DCT coefficient perturbation,
UNDERperformed the standard pair (4,1)/(1,4) under Real-ESRGAN
(NC 0.683 vs 0.783), while the HF pair (7,5)/(7,7) did best (0.831).

Hypothesis under test
---------------------
Blind extraction reads bit = [C1' > C2'], so survival should depend on
preserving the RELATIVE ordering (the signed decision margin M = C1 - C2),
not on minimizing the absolute perturbation of each coefficient
independently. This script measures, for every embedded block:

    BEFORE Real-ESRGAN:  C1, C2, M_before = C1 - C2
    AFTER  Real-ESRGAN:  C1', C2', M_after = C1' - C2'

and derives margin loss, margin ratio, sign flips, per-coefficient
perturbations dC1/dC2, differential perturbation (dC1 - dC2) and
common-mode perturbation (dC1 + dC2)/2.

Data sources (frozen experiments are NOT modified)
--------------------------------------------------
- standard pair: watermarked + ESRGAN images pre-computed on disk
    results/watermarked/<img>__alpha0.1.png
    results/attacked/ai_enhancement/<img>__alpha0.1__real_esrgan.png
- hf pair: same, with __optimized_positions suffix
- balanced pair: no disk files exist (Exp F computed them live and
  discarded them). We re-embed deterministically (seed 42) and re-run
  Real-ESRGAN live, replicating Exp F's patch_balanced_esrgan.py pipeline
  exactly (patches_size=64, Lanczos4 downscale) so the measured margins
  correspond to the frozen NC numbers. Outputs are cached under this
  experiment's folder so reruns are cheap.

Pipeline-confound control
-------------------------
Exp F's balanced ESRGAN used a slightly different invocation than the
disk-based standard/HF ESRGAN (patches_size 64 vs default 192; Lanczos4 vs
INTER_AREA downscale). To rule out the possibility that the balanced
deficit is an artifact of that pipeline difference, we also run the LIVE
patch-style pipeline on the standard pair's watermarked images for a
CONTROL_N-image subset and compare NC against the disk-based results
("standard_live" rows in image_summary.csv).

Outputs (in this folder)
------------------------
- block_margins.csv    one row per (image, pair, block): all raw + derived
                       block-level quantities
- image_summary.csv    one row per (image, pair): NC, BER, flip rates,
                       margin statistics
- watermarked_balanced/, esrgan_balanced/, esrgan_standard_control/
                       cached PNGs (uint8) for reproducibility
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Disable torch.cuda.amp.autocast BEFORE py_real_esrgan.model is imported
# (decorator is evaluated at class-definition time; autocast breaks on MPS).
# Same workaround as exp_F/patch_balanced_esrgan.py.
import torch


class _NoopAutocast:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __call__(self, func): return func


torch.cuda.amp.autocast = _NoopAutocast

import cv2
import numpy as np
import pandas as pd
import pywt
from PIL import Image

from src import config
from src import watermark as wm_module
from src.watermark import _dct2, rgb_to_ycbcr, BLOCK_SIZE
from src.metrics import normalized_correlation, bit_error_rate

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCK_CSV = os.path.join(OUT_DIR, "block_margins.csv")
IMAGE_CSV = os.path.join(OUT_DIR, "image_summary.csv")

WM_BAL_DIR = os.path.join(OUT_DIR, "watermarked_balanced")
ESR_BAL_DIR = os.path.join(OUT_DIR, "esrgan_balanced")
ESR_STD_CTRL_DIR = os.path.join(OUT_DIR, "esrgan_standard_control")
ESR_HF_CTRL_DIR = os.path.join(OUT_DIR, "esrgan_hf_control")

ALPHA = 0.1
SEED = 42
WM_SHAPE = config.WATERMARK_SIZE          # (32, 32) -> 1024 blocks
IMAGE_SIZE = config.IMAGE_SIZE            # (512, 512)
ESRGAN_MODEL = os.path.join(config.PROJECT_ROOT, "weights", "RealESRGAN_x4.pth")

CONTROL_N = 20   # standard-pair live-ESRGAN control subset size

# Same ESRGAN patch settings as exp_F/patch_balanced_esrgan.py (MPS JIT workaround)
PATCHES_SIZE = 64
PATCHES_PADDING = 8
PATCHES_PAD_SIZE = 8

PAIRS = {
    "standard": {"pos1": (4, 1), "pos2": (1, 4)},
    "balanced": {"pos1": (2, 3), "pos2": (3, 2)},
    "hf":       {"pos1": (7, 5), "pos2": (7, 7)},
}


# ─── I/O helpers ─────────────────────────────────────────────────────────────

def load_bgr_to_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)


def load_original(image_name):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(config.ORIGINAL_IMAGES_DIR, image_name + ext)
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    raise FileNotFoundError(f"No original image for {image_name}")


def save_rgb_u8(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def list_images():
    names = []
    for f in sorted(os.listdir(config.ORIGINAL_IMAGES_DIR)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in (".png", ".jpg", ".jpeg"):
            names.append(stem)
    return names


# ─── Embedding with custom positions (same monkeypatch pattern as Exp F) ────

def embed_custom(original, wm, pos1, pos2, alpha):
    orig1, orig2 = wm_module.COEFF_POS_1, wm_module.COEFF_POS_2
    wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = pos1, pos2
    try:
        return np.clip(wm_module.embed_watermark_rgb(original.astype(np.float64), wm, alpha=alpha), 0, 255)
    finally:
        wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = orig1, orig2


# ─── ESRGAN (live, Exp F patch-style pipeline) ───────────────────────────────

_model = None


def get_esrgan_model():
    global _model
    if _model is None:
        from py_real_esrgan.model import RealESRGAN
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        _model = RealESRGAN(device, scale=4)
        _model.load_weights(ESRGAN_MODEL, download=False)
        print(f"  ESRGAN loaded on {device}", flush=True)
    return _model


def run_esrgan_live(rgb_u8):
    model = get_esrgan_model()
    sr = model.predict(
        Image.fromarray(rgb_u8),
        patches_size=PATCHES_SIZE,
        padding=PATCHES_PADDING,
        pad_size=PATCHES_PAD_SIZE,
    )
    h, w = rgb_u8.shape[:2]
    return cv2.resize(np.array(sr), (w, h), interpolation=cv2.INTER_LANCZOS4).astype(np.float64)


def cached_esrgan(cache_dir, image_name, watermarked_rgb):
    path = os.path.join(cache_dir, f"{image_name}.png")
    img = load_bgr_to_rgb(path)
    if img is not None:
        return img
    out = run_esrgan_live(np.clip(watermarked_rgb, 0, 255).astype(np.uint8))
    save_rgb_u8(path, out)
    return out


# ─── Block-level coefficient measurement ─────────────────────────────────────

def block_coeffs(rgb_image, pos1, pos2, n_bits):
    """Return (C1, C2) arrays of length n_bits from the LL-DCT of the Y channel."""
    y = rgb_to_ycbcr(rgb_image.astype(np.float64))[..., 0]
    LL, _ = pywt.dwt2(y, wm_module.DEFAULT_WAVELET)
    blocks_per_row = LL.shape[1] // BLOCK_SIZE
    c1 = np.empty(n_bits)
    c2 = np.empty(n_bits)
    for idx in range(n_bits):
        br = (idx // blocks_per_row) * BLOCK_SIZE
        bc = (idx % blocks_per_row) * BLOCK_SIZE
        d = _dct2(LL[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE])
        c1[idx] = d[pos1]
        c2[idx] = d[pos2]
    return c1, c2


# ─── Per-(image, pair) processing ────────────────────────────────────────────

def get_before_after(image_name, pair_key, original, wm):
    """Return (before_rgb, after_rgb) for a pair, using disk data where frozen
    experiments already produced it, live ESRGAN otherwise."""
    a = ALPHA
    if pair_key == "standard":
        before = load_bgr_to_rgb(os.path.join(config.WATERMARKED_DIR, f"{image_name}__alpha{a}.png"))
        after = load_bgr_to_rgb(os.path.join(config.ATTACKED_AI_DIR, f"{image_name}__alpha{a}__real_esrgan.png"))
    elif pair_key == "hf":
        before = load_bgr_to_rgb(os.path.join(config.WATERMARKED_DIR, f"{image_name}__alpha{a}__optimized_positions.png"))
        after = load_bgr_to_rgb(os.path.join(config.ATTACKED_AI_DIR, f"{image_name}__alpha{a}__optimized_positions__real_esrgan.png"))
    elif pair_key == "balanced":
        wm_path = os.path.join(WM_BAL_DIR, f"{image_name}.png")
        before = load_bgr_to_rgb(wm_path)
        if before is None:
            before = embed_custom(original, wm, PAIRS["balanced"]["pos1"], PAIRS["balanced"]["pos2"], ALPHA)
            save_rgb_u8(wm_path, before)
            before = load_bgr_to_rgb(wm_path)  # re-read so "before" is the exact uint8 ESRGAN input
        after = cached_esrgan(ESR_BAL_DIR, image_name, before)
    elif pair_key == "standard_live":
        before = load_bgr_to_rgb(os.path.join(config.WATERMARKED_DIR, f"{image_name}__alpha{a}.png"))
        after = cached_esrgan(ESR_STD_CTRL_DIR, image_name, before) if before is not None else None
    elif pair_key == "hf_live":
        before = load_bgr_to_rgb(os.path.join(config.WATERMARKED_DIR, f"{image_name}__alpha{a}__optimized_positions.png"))
        after = cached_esrgan(ESR_HF_CTRL_DIR, image_name, before) if before is not None else None
    else:
        raise ValueError(pair_key)
    return before, after


def process(image_name, pair_key, original, wm_bits, wm_grid):
    pos_key = pair_key.removesuffix("_live")
    pos1, pos2 = PAIRS[pos_key]["pos1"], PAIRS[pos_key]["pos2"]
    before, after = get_before_after(image_name, pair_key, original, wm_grid)
    if before is None or after is None:
        print(f"  [SKIP] {image_name} {pair_key}: missing before/after image", flush=True)
        return None, None

    n_bits = wm_bits.size
    c1b, c2b = block_coeffs(before, pos1, pos2, n_bits)
    c1a, c2a = block_coeffs(after, pos1, pos2, n_bits)

    m_before = c1b - c2b
    m_after = c1a - c2a
    dc1 = c1a - c1b
    dc2 = c2a - c2b
    sign_flip = (np.sign(m_before) != np.sign(m_after)).astype(int)
    extracted = (m_after > 0).astype(np.uint8)
    bit_error = (extracted != wm_bits).astype(int)

    blocks = pd.DataFrame({
        "image": image_name,
        "pair": pair_key,
        "block_idx": np.arange(n_bits),
        "bit": wm_bits,
        "c1_before": c1b, "c2_before": c2b,
        "c1_after": c1a, "c2_after": c2a,
        "m_before": m_before, "m_after": m_after,
        "abs_m_before": np.abs(m_before), "abs_m_after": np.abs(m_after),
        "margin_loss": m_after - m_before,
        "margin_ratio": np.divide(m_after, m_before,
                                  out=np.full_like(m_after, np.nan),
                                  where=m_before != 0),
        "sign_flip": sign_flip,
        "dc1": dc1, "dc2": dc2,
        "abs_dc1": np.abs(dc1), "abs_dc2": np.abs(dc2),
        "diff_pert": dc1 - dc2,
        "common_pert": (dc1 + dc2) / 2.0,
        "bit_error": bit_error,
    })

    ext_grid = extracted.reshape(WM_SHAPE)
    summary = {
        "image": image_name,
        "pair": pair_key,
        "nc": normalized_correlation(wm_grid, ext_grid),
        "ber": bit_error_rate(wm_grid, ext_grid),
        "flip_rate": sign_flip.mean(),
        "bit_error_rate_blocks": bit_error.mean(),
        "mean_abs_m_before": np.abs(m_before).mean(),
        "min_abs_m_before": np.abs(m_before).min(),
        "median_abs_m_before": np.median(np.abs(m_before)),
        "mean_abs_m_after": np.abs(m_after).mean(),
        "mean_margin_loss_signed_to_bit": ((m_after - m_before) * np.where(wm_bits == 1, 1, -1)).mean(),
        "mean_abs_dc1": np.abs(dc1).mean(),
        "mean_abs_dc2": np.abs(dc2).mean(),
        "mean_abs_damage": (np.abs(dc1).mean() + np.abs(dc2).mean()) / 2.0,
        "mean_abs_diff_pert": np.abs(dc1 - dc2).mean(),
        "std_diff_pert": (dc1 - dc2).std(),
        "mean_abs_common_pert": np.abs((dc1 + dc2) / 2.0).mean(),
        "std_common_pert": ((dc1 + dc2) / 2.0).std(),
        "margin_var_after": m_after.var(),
    }
    return blocks, summary


def main():
    for d in (WM_BAL_DIR, ESR_BAL_DIR, ESR_STD_CTRL_DIR, ESR_HF_CTRL_DIR):
        os.makedirs(d, exist_ok=True)

    wm_grid = wm_module.generate_watermark(shape=WM_SHAPE, seed=SEED)
    wm_bits = wm_grid.flatten()
    images = list_images()
    control_images = images[:CONTROL_N]
    print(f"Images: {len(images)} | control subset: {len(control_images)}", flush=True)

    # Resume support: skip (image, pair) rows already in block CSV
    done = set()
    if os.path.exists(IMAGE_CSV):
        prev = pd.read_csv(IMAGE_CSV)
        done = set(zip(prev["image"], prev["pair"]))
        print(f"Resuming: {len(done)} (image, pair) combos already done", flush=True)

    block_frames = []
    summaries = []

    def flush_results():
        if block_frames:
            hdr = not os.path.exists(BLOCK_CSV)
            pd.concat(block_frames).to_csv(BLOCK_CSV, mode="a", header=hdr, index=False)
            block_frames.clear()
        if summaries:
            hdr = not os.path.exists(IMAGE_CSV)
            pd.DataFrame(summaries).to_csv(IMAGE_CSV, mode="a", header=hdr, index=False)
            summaries.clear()

    for i, name in enumerate(images):
        original = load_original(name)
        pair_keys = ["standard", "balanced", "hf"]
        if name in control_images:
            pair_keys.extend(["standard_live", "hf_live"])
        row_msgs = []
        for pk in pair_keys:
            if (name, pk) in done:
                continue
            blocks, summary = process(name, pk, original, wm_bits, wm_grid)
            if blocks is None:
                continue
            block_frames.append(blocks)
            summaries.append(summary)
            row_msgs.append(f"{pk}: NC={summary['nc']:.4f} flips={summary['flip_rate']:.3f}")
        if row_msgs:
            print(f"[{i+1}/{len(images)}] {name}  " + "  ".join(row_msgs), flush=True)
        if (i + 1) % 5 == 0:
            flush_results()

    flush_results()
    print("\nDone.", flush=True)

    df = pd.read_csv(IMAGE_CSV)
    print("\nMean NC by pair (should match Exp F within PNG quantization):")
    print(df.groupby("pair")[["nc", "flip_rate", "mean_abs_m_before", "mean_abs_diff_pert"]].mean().round(4))


if __name__ == "__main__":
    main()
