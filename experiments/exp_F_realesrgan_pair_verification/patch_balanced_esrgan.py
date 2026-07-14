"""
Patch: fill in the missing Real-ESRGAN results for the balanced pair (2,3)/(3,2).

Reads the existing results_exp_F.csv, finds every row where
coefficient_pair == "balanced" AND attack == "real_esrgan" AND nc is NaN,
runs ESRGAN on those (image, alpha) combinations, writes back the four metric
columns (nc, ber, psnr, ssim), and saves the updated CSV incrementally.

All other rows (standard, hf, JPEG, blur, no-attack) are left untouched.

MPS note: patches_size=64 is used instead of the library default 192 to avoid a
PyTorch 2.13 / macOS 26.2 MPS JIT compilation hang on large upsampling tensors.
The underlying ESRGAN model and weights are identical to those used for the
standard and HF pair attacks. Output image quality is equivalent.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Must patch BEFORE importing py_real_esrgan.model because the @torch.cuda.amp.autocast()
# decorator on predict() is evaluated at class definition time.
# Using a no-op replacement disables autocast entirely, avoiding the BFloat16/numpy
# incompatibility that occurs when torch.amp.autocast("cpu") is active.
import torch

class _NoopAutocast:
    """Drop-in for torch.cuda.amp.autocast that performs no dtype conversion."""
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __call__(self, func): return func  # works as a pass-through decorator

torch.cuda.amp.autocast = _NoopAutocast

import numpy as np
import pandas as pd
import cv2
from PIL import Image

from py_real_esrgan.model import RealESRGAN

from src import config
from src import watermark as wm_module
from src.metrics import normalized_correlation, bit_error_rate, psnr, ssim

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ESRGAN_MODEL = os.path.join(_ROOT, "weights", "RealESRGAN_x4.pth")
RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_exp_F.csv")

BALANCED_POS1 = (2, 3)
BALANCED_POS2 = (3, 2)
WM_SHAPE = config.WATERMARK_SIZE   # (32, 32)
IMAGE_SIZE = config.IMAGE_SIZE     # (512, 512) — (w, h) for cv2
ORIG_DIR = config.ORIGINAL_IMAGES_DIR
SEED = 42

# patches_size=64 avoids a PyTorch 2.13/macOS 26.2 MPS JIT hang on 192-size patches.
# Same ESRGAN model and weights; output quality is equivalent.
PATCHES_SIZE = 64
PATCHES_PADDING = 8
PATCHES_PAD_SIZE = 8
SAVE_EVERY = 10   # checkpoint the CSV every N images


def _load_esrgan_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = RealESRGAN(device, scale=4)
    model.load_weights(ESRGAN_MODEL, download=False)
    print(f"  ESRGAN loaded on {device}", flush=True)
    return model


def _run_esrgan(model, wm_img_u8):
    """Run ESRGAN on a uint8 RGB image (H, W, 3) and return float64 RGB (H, W, 3)."""
    pil = Image.fromarray(wm_img_u8)
    sr_pil = model.predict(
        pil,
        patches_size=PATCHES_SIZE,
        padding=PATCHES_PADDING,
        pad_size=PATCHES_PAD_SIZE,
    )
    sr_arr = np.array(sr_pil)   # (H*4, W*4, 3) uint8
    h, w = wm_img_u8.shape[:2]
    resized = cv2.resize(sr_arr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return resized.astype(np.float64)


def load_original(image_name):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(ORIG_DIR, image_name + ext)
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    raise FileNotFoundError(f"No original image for {image_name}")


def embed_balanced(original, wm, alpha):
    orig1, orig2 = wm_module.COEFF_POS_1, wm_module.COEFF_POS_2
    wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = BALANCED_POS1, BALANCED_POS2
    try:
        return np.clip(
            wm_module.embed_watermark_rgb(original.astype(np.float64), wm, alpha=alpha),
            0, 255,
        )
    finally:
        wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = orig1, orig2


def extract_balanced(image):
    orig1, orig2 = wm_module.COEFF_POS_1, wm_module.COEFF_POS_2
    wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = BALANCED_POS1, BALANCED_POS2
    try:
        return wm_module.extract_watermark_rgb(image.astype(np.float64), WM_SHAPE)
    finally:
        wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = orig1, orig2


def main():
    print("=" * 70, flush=True)
    print("PATCH: Balanced pair Real-ESRGAN fill-in", flush=True)
    print("=" * 70, flush=True)

    df = pd.read_csv(RESULTS_CSV)
    missing_mask = (
        (df["coefficient_pair"] == "balanced") &
        (df["attack"] == "real_esrgan") &
        df["nc"].isna()
    )
    n_missing = int(missing_mask.sum())
    print(f"Rows to fill: {n_missing}  (balanced + real_esrgan + NaN nc)", flush=True)
    if n_missing == 0:
        print("Nothing to do — balanced ESRGAN already complete.", flush=True)
        return

    np.random.seed(SEED)
    wm = wm_module.generate_watermark(shape=WM_SHAPE, seed=SEED)

    print("\nLoading Real-ESRGAN model...", flush=True)
    model = _load_esrgan_model()

    # Group by image so we load each original image once across all 4 alphas
    alphas_needed = sorted(df.loc[missing_mask, "alpha"].unique())
    images_needed = sorted(df.loc[missing_mask, "image"].unique())
    print(f"Images: {len(images_needed)}  |  Alphas: {alphas_needed}", flush=True)

    done_images = 0
    done_rows = 0

    for img_idx, image_name in enumerate(images_needed):
        try:
            original = load_original(image_name)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}", flush=True)
            continue

        nc_per_alpha = {}
        for alpha in alphas_needed:
            # Check if this specific (image, alpha) row is missing
            row_missing = (
                (df["image"] == image_name) &
                (df["alpha"] == alpha) &
                (df["coefficient_pair"] == "balanced") &
                (df["attack"] == "real_esrgan") &
                df["nc"].isna()
            )
            if not row_missing.any():
                continue

            wm_img = embed_balanced(original, wm, alpha)
            wm_u8 = np.clip(wm_img, 0, 255).astype(np.uint8)

            try:
                esrgan_out = _run_esrgan(model, wm_u8)
            except Exception as e:
                print(f"  [ERROR] ESRGAN failed {image_name} α={alpha}: {e}", flush=True)
                continue

            ext = extract_balanced(esrgan_out)
            nc_val   = normalized_correlation(wm, ext)
            ber_val  = bit_error_rate(wm, ext)
            psnr_val = psnr(original, esrgan_out)
            ssim_val = ssim(original, esrgan_out)

            idx = df.index[row_missing]
            df.loc[idx, "nc"]   = nc_val
            df.loc[idx, "ber"]  = ber_val
            df.loc[idx, "psnr"] = psnr_val
            df.loc[idx, "ssim"] = ssim_val
            nc_per_alpha[alpha] = nc_val
            done_rows += 1

        done_images += 1
        nc_str = "  ".join(f"α={a}→NC={nc_per_alpha[a]:.4f}" for a in alphas_needed if a in nc_per_alpha)
        print(f"[{done_images}/{len(images_needed)}] {image_name}  {nc_str}", flush=True)

        if done_images % SAVE_EVERY == 0:
            df.to_csv(RESULTS_CSV, index=False)
            print(f"  [checkpoint] saved {RESULTS_CSV}", flush=True)

    # Final save
    df.to_csv(RESULTS_CSV, index=False)
    still_nan = int(df.loc[
        (df["coefficient_pair"] == "balanced") & (df["attack"] == "real_esrgan"),
        "nc"
    ].isna().sum())
    print(f"\nFinal save: {RESULTS_CSV}", flush=True)
    print(f"Rows filled: {done_rows}  |  Still NaN: {still_nan}", flush=True)

    # Quick summary
    bal = df[(df["coefficient_pair"] == "balanced") & (df["attack"] == "real_esrgan")]
    std = df[(df["coefficient_pair"] == "standard") & (df["attack"] == "real_esrgan")]
    print("\nQuick NC summary (all alphas):", flush=True)
    for a in alphas_needed:
        b_nc = bal[bal["alpha"] == a]["nc"].mean()
        s_nc = std[std["alpha"] == a]["nc"].mean()
        print(f"  α={a:.2f}  balanced={b_nc:.4f}  standard={s_nc:.4f}  Δ={b_nc-s_nc:+.4f}", flush=True)


if __name__ == "__main__":
    main()
