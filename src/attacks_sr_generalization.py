"""
Experiment I: additional OpenCV dnn_superres x4 models (FSRCNN, EDSR, LapSRN).

NEW module — the validated attack implementations in attacks_ai.py are NOT
modified. Each wrapper here replicates `attacks_ai.espcn_x4_enhance`
byte-for-byte in its pre/post-processing so the only varying factor across
SR attacks is the network itself:

    512x512 RGB float -> clip/uint8 -> RGB->BGR -> dnn_superres x4 (2048x2048)
    -> BGR->RGB -> cv2.resize to config.IMAGE_SIZE with INTER_LANCZOS4
    -> float64

The DnnSuperResImpl instance is cached per (model_name, path): readModel is
deterministic, so caching changes nothing numerically — it only avoids
re-parsing the 38 MB EDSR protobuf on every call.
"""

import os

import cv2
import numpy as np

from . import config

SR_GENERALIZATION_MODELS = {
    # model_name (cv2 setModel key) -> weights filename in weights/
    "fsrcnn": "FSRCNN_x4.pb",
    "edsr": "EDSR_x4.pb",
    "lapsrn": "LapSRN_x4.pb",
}

_sr_engine_cache = {}


def _get_sr_engine(model_name, model_path):
    key = (model_name, model_path)
    if key not in _sr_engine_cache:
        if not hasattr(cv2, "dnn_superres"):
            raise ImportError(
                "opencv-contrib-python is required for dnn_superres. "
                "Run: pip uninstall opencv-python && pip install opencv-contrib-python"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SR model weights not found: {model_path}")
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel(model_name, 4)
        _sr_engine_cache[key] = sr
    return _sr_engine_cache[key]


def sr_x4_enhance(image, model_name, model_path=None):
    """Upscale x4 with an OpenCV dnn_superres model, then Lanczos-resize back.

    Identical pipeline to attacks_ai.espcn_x4_enhance, parameterized by model.
    """
    if model_name not in SR_GENERALIZATION_MODELS:
        raise ValueError(f"Unknown SR model '{model_name}'. Choices: {sorted(SR_GENERALIZATION_MODELS)}")
    if model_path is None:
        model_path = os.path.join(config.PROJECT_ROOT, "weights", SR_GENERALIZATION_MODELS[model_name])

    sr = _get_sr_engine(model_name, model_path)

    bgr = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    upscaled_bgr = sr.upsample(bgr)
    rgb_output = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_output, config.IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
    return resized.astype(np.float64)


def get_sr_generalization_attacks(models=None):
    """Return {attack_name: callable} for every requested model that loads.

    A model that fails to load (missing/corrupt weights, missing contrib
    build) is logged and skipped rather than failing the whole run.
    """
    attacks = {}
    for name in (models or SR_GENERALIZATION_MODELS):
        path = os.path.join(config.PROJECT_ROOT, "weights", SR_GENERALIZATION_MODELS[name])
        try:
            _get_sr_engine(name, path)
        except Exception as exc:  # noqa: BLE001 — deliberate skip-and-continue
            print(f"[attacks_sr_generalization] SKIPPING {name}: {exc}", flush=True)
            continue
        attacks[f"{name}_x4"] = lambda img, _n=name, _p=path: sr_x4_enhance(img, _n, _p)
    return attacks
