"""
AI-based image enhancement "attacks".

These wrap optional third-party super-resolution / enhancement tools so the
experiment can test whether AI upscale-then-downscale enhancement acts as a
watermark-removal attack. Each wrapper degrades gracefully: if the
underlying model/library isn't installed, `get_ai_attacks()` simply omits it
rather than failing the whole run. A non-AI `fallback` enhancement (CLAHE +
denoise + unsharp mask) is always available so the AI-attack stage never
ends up empty on a machine with no GPU models installed.

To plug in an external tool that can't be called from Python (Topaz,
Photoshop Enhance, an online AI enhancer, etc.): run it manually outside
this codebase on the files in results/watermarked/, then save its output
into results/attacked/ai_enhancement/ using the
"<image>__alpha<alpha>__<attack_name>.png" naming convention used here.
run_experiment.py's extraction/metrics stage reads attacked images purely
from disk by filename, so manually-produced files are picked up exactly
like any attack implemented in this module -- just re-run the
extraction/metrics step (see README) after dropping the files in.
"""

import os

import cv2
import numpy as np

from . import config

try:
    import py_real_esrgan  # noqa: F401 — probe availability without importing model.py yet
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False

OPENCV_SR_AVAILABLE = hasattr(cv2, "dnn_superres")
ESPCN_AVAILABLE = hasattr(cv2, "dnn_superres")

ESPCN_MODEL_URL = "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb"
ESPCN_MODEL_PATH = os.path.join(config.PROJECT_ROOT, "weights", "ESPCN_x4.pb")

_real_esrgan_model_cache = {}


def _get_real_esrgan_model(model_path, scale):
    """Load (or reuse a cached) RealESRGAN model on the best available device.

    Reused across calls within a process so the (otherwise per-call) weights
    load and device transfer only happen once per (model_path, scale).
    """
    cache_key = (model_path, scale)
    if cache_key in _real_esrgan_model_cache:
        return _real_esrgan_model_cache[cache_key]

    import torch

    # py_real_esrgan/model.py decorates `predict` with @torch.cuda.amp.autocast(),
    # which fails on MPS because CUDA is not available. Patch it to the
    # device-agnostic form before model.py is imported (the decorator runs at
    # class-definition time, so the patch must precede the import).
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        torch.cuda.amp.autocast = lambda *args, **kwargs: torch.amp.autocast("cpu", *args, **kwargs)

    from py_real_esrgan.model import RealESRGAN  # imported here so patch is in place first

    weights_dir = os.path.dirname(model_path)
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = RealESRGAN(device, scale=scale)
    model.load_weights(model_path, download=True)
    _real_esrgan_model_cache[cache_key] = model
    return model


def real_esrgan_enhance(image, model_path=None, scale=4):
    """Upscale with Real-ESRGAN then resize back down to the original size."""
    if not REALESRGAN_AVAILABLE:
        raise RuntimeError("Real-ESRGAN is not installed (pip install py-real-esrgan torch).")
    if model_path is None:
        model_path = os.path.join(config.PROJECT_ROOT, "weights", "RealESRGAN_x4.pth")

    from PIL import Image

    model = _get_real_esrgan_model(model_path, scale)

    lr_image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    sr_image = model.predict(lr_image)
    resized = cv2.resize(np.array(sr_image), config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float64)


def opencv_dnn_super_resolution(image, model_path=None, model_name="edsr", scale=2):
    """Upscale with an OpenCV DNN super-resolution model then resize back down."""
    if not OPENCV_SR_AVAILABLE:
        raise RuntimeError("cv2.dnn_superres is unavailable (install opencv-contrib-python).")
    if model_path is None:
        raise ValueError("opencv_dnn_super_resolution requires model_path to a pretrained .pb model.")

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(model_name, scale)

    bgr = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    upscaled = sr.upsample(bgr)
    rgb_output = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_output, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float64)


def _ensure_espcn_weights(model_path=ESPCN_MODEL_PATH, url=ESPCN_MODEL_URL):
    """Download the ESPCN x4 weights to `model_path` if not already present."""
    if os.path.exists(model_path):
        return model_path

    import urllib.request

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(url, model_path)
    return model_path


def espcn_x4_enhance(image, model_path=ESPCN_MODEL_PATH, scale=4):
    """Upscale with OpenCV DNN super-resolution (ESPCN x4) then resize back down with Lanczos."""
    if not hasattr(cv2, "dnn_superres"):
        raise ImportError("opencv-contrib-python is required for ESPCN. Run: pip install opencv-contrib-python")

    weights_path = _ensure_espcn_weights(model_path)

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(weights_path)
    sr.setModel("espcn", scale)

    bgr = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    upscaled_bgr = sr.upsample(bgr)
    rgb_output = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_output, config.IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
    return resized.astype(np.float64)


def fallback_ai_like_enhancement(image, seed=config.RANDOM_SEED):
    """
    Placeholder "AI-like" enhancement used when no real model is installed:
    CLAHE contrast enhancement (applied to luma only, to avoid color shifts)
    + color-aware denoising + unsharp masking. This is NOT a real neural
    enhancer -- it exists so the AI-attack stage of the pipeline always has
    at least one runnable entry, and so the modular wrapper interface can
    be exercised end to end without any extra dependencies.
    """
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    contrast_bgr = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

    denoised = cv2.fastNlMeansDenoisingColored(contrast_bgr, None, h=7, hColor=7)

    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    rgb_out = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    return np.clip(rgb_out, 0, 255).astype(np.float64)


def get_ai_attacks(real_esrgan_model_path=None, opencv_sr_model_path=None,
                    opencv_sr_model_name="edsr", opencv_sr_scale=2):
    """Return {attack_name: callable(image) -> attacked_image} for every usable AI backend."""
    attacks = {
        "ai_fallback_enhancement": fallback_ai_like_enhancement,
    }

    if ESPCN_AVAILABLE:
        attacks["espcn_x4"] = espcn_x4_enhance

    if REALESRGAN_AVAILABLE:
        _esrgan_path = real_esrgan_model_path or os.path.join(config.PROJECT_ROOT, "weights", "RealESRGAN_x4.pth")
        attacks["real_esrgan"] = lambda img, _p=_esrgan_path: real_esrgan_enhance(img, model_path=_p)

    if OPENCV_SR_AVAILABLE and opencv_sr_model_path:
        attacks["opencv_dnn_sr"] = lambda img: opencv_dnn_super_resolution(
            img, model_path=opencv_sr_model_path, model_name=opencv_sr_model_name, scale=opencv_sr_scale
        )

    return attacks
