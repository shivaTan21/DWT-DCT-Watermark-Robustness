"""
Experiment I: SR model generalization of the Exp H restoration-pressure mechanism.

Question: is the watermark-erasing mechanism found for Real-ESRGAN (Exp H)
unique to Real-ESRGAN, common to learned super-resolution, or architecture
dependent? We evaluate additional x4 SR models (FSRCNN, EDSR, LapSRN) through
the exact ESPCN wrapper pipeline (512 -> x4 SR -> 2048 -> Lanczos4 -> 512).

Frozen data reused READ-ONLY (nothing is re-embedded or re-attacked):
  - watermarked inputs: results/watermarked/<img>__alpha<a>.png
    (standard pair (4,1)/(1,4), seed 42, 32x32 watermark — identical to the
    images that produced the frozen ESPCN/Real-ESRGAN numbers)
  - originals: data/original_images/ (resized to 512x512 INTER_AREA, the
    same convention as run_experiment.py / exp_H)

Outputs (all inside this experiment folder):
  - attacked/<img>__alpha<a>__<model>_x4.png   cached attacked images
  - results_exp_I.csv                          one row per (image, alpha, model)
  - progress is saved after every image; the run is restartable (rows already
    in the CSV and cached attacked PNGs are skipped/reused).

Usage:
  run_exp_I.py --sanity              3 images, alpha 0.10, all models + runtime projection
  run_exp_I.py --models fsrcnn,lapsrn
  run_exp_I.py --models edsr         (run separately once approved)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import pandas as pd

from src import config
from src import watermark as wm_module
from src.attacks_sr_generalization import SR_GENERALIZATION_MODELS, sr_x4_enhance, _get_sr_engine
from src.metrics import normalized_correlation, bit_error_rate, psnr, ssim

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ATTACKED_DIR = os.path.join(OUT_DIR, "attacked")
RESULTS_CSV = os.path.join(OUT_DIR, "results_exp_I.csv")

ALPHAS = [0.05, 0.1, 0.2, 0.3]
SEED = 42
WM_SHAPE = config.WATERMARK_SIZE      # (32, 32)
IMAGE_SIZE = config.IMAGE_SIZE        # (512, 512)


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


def attack_cached(image_name, alpha, model_name, watermarked_rgb):
    """Run the SR attack (or reuse the cached PNG). Returns (attacked_rgb, runtime_s).

    runtime is NaN when the image came from cache. The SR output is
    integer-valued uint8 before the final float64 cast, so the PNG round-trip
    is lossless and cache reuse is numerically exact.
    """
    path = os.path.join(ATTACKED_DIR, f"{image_name}__alpha{alpha}__{model_name}_x4.png")
    cached = load_bgr_to_rgb(path)
    if cached is not None:
        return cached, float("nan")
    t0 = time.time()
    attacked = sr_x4_enhance(watermarked_rgb, model_name)
    runtime = time.time() - t0
    save_rgb_u8(path, attacked)
    return attacked, runtime


def evaluate(image_name, alpha, model_name, wm_grid, original):
    wm_path = os.path.join(config.WATERMARKED_DIR, f"{image_name}__alpha{alpha}.png")
    watermarked = load_bgr_to_rgb(wm_path)
    if watermarked is None:
        print(f"  [SKIP] missing frozen watermarked image: {wm_path}", flush=True)
        return None

    attacked, runtime = attack_cached(image_name, alpha, model_name, watermarked)
    extracted = wm_module.extract_watermark_rgb(attacked, WM_SHAPE)

    return {
        "image": image_name,
        "alpha": alpha,
        "model": model_name,
        "attack_name": f"{model_name}_x4",
        "nc": normalized_correlation(wm_grid, extracted),
        "ber": bit_error_rate(wm_grid, extracted),
        "psnr_after_attack": psnr(original, attacked),
        "ssim_after_attack": ssim(original, attacked),
        "runtime_s": runtime,
        "out_h": attacked.shape[0],
        "out_w": attacked.shape[1],
    }


def usable_models(requested):
    """Filter to models whose weights load; log and skip failures."""
    ok = []
    for name in requested:
        path = os.path.join(config.PROJECT_ROOT, "weights", SR_GENERALIZATION_MODELS[name])
        try:
            _get_sr_engine(name, path)
            ok.append(name)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL->SKIP] {name}: {exc}", flush=True)
    return ok


def sanity_check(models):
    images = list_images()[:3]
    alpha = 0.1
    wm_grid = wm_module.generate_watermark(shape=WM_SHAPE, seed=SEED)
    rows = []
    print(f"SANITY CHECK: images={images}, alpha={alpha}, models={models}\n", flush=True)
    for model_name in models:
        for image_name in images:
            original = load_original(image_name)
            r = evaluate(image_name, alpha, model_name, wm_grid, original)
            if r is None:
                continue
            bad = (np.isnan(r["nc"]) or np.isnan(r["ber"]) or np.isnan(r["psnr_after_attack"])
                   or np.isnan(r["ssim_after_attack"]) or r["out_h"] != IMAGE_SIZE[1] or r["out_w"] != IMAGE_SIZE[0])
            r["check"] = "FAIL" if bad else "ok"
            rows.append(r)
            print(f"  {model_name:8s} {image_name}  NC={r['nc']:.4f}  BER={r['ber']:.4f}  "
                  f"SSIM={r['ssim_after_attack']:.4f}  runtime={r['runtime_s']:.2f}s  "
                  f"out={r['out_h']}x{r['out_w']}  [{r['check']}]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "sanity_check.csv"), index=False)
    n_full = len(list_images()) * len(ALPHAS)
    print("\nRuntime projection for the full run (100 images x 4 alphas = "
          f"{n_full} attacks per model):", flush=True)
    for model_name in models:
        mean_rt = df[df["model"] == model_name]["runtime_s"].mean()
        print(f"  {model_name:8s} mean {mean_rt:6.2f}s/image -> projected {mean_rt * n_full / 3600:.2f} h", flush=True)


def full_run(models, results_csv=RESULTS_CSV):
    images = list_images()
    wm_grid = wm_module.generate_watermark(shape=WM_SHAPE, seed=SEED)

    done = set()
    if os.path.exists(results_csv):
        prev = pd.read_csv(results_csv)
        done = set(zip(prev["image"], prev["alpha"], prev["model"]))
        print(f"Resuming: {len(done)} rows already done", flush=True)

    for model_name in models:
        t_model = time.time()
        for i, image_name in enumerate(images):
            todo = [a for a in ALPHAS if (image_name, a, model_name) not in done]
            if not todo:
                continue
            original = load_original(image_name)
            rows = []
            for alpha in todo:
                r = evaluate(image_name, alpha, model_name, wm_grid, original)
                if r is not None:
                    rows.append(r)
            if rows:
                hdr = not os.path.exists(results_csv)
                pd.DataFrame(rows).to_csv(results_csv, mode="a", header=hdr, index=False)
            ncs = ", ".join(f"a={r['alpha']}: NC={r['nc']:.3f}" for r in rows)
            print(f"[{model_name} {i+1}/{len(images)}] {image_name}  {ncs}", flush=True)
        print(f"== {model_name} finished in {(time.time() - t_model)/60:.1f} min ==", flush=True)

    df = pd.read_csv(results_csv)
    print("\nMean NC by model x alpha:")
    print(df.pivot_table(index="model", columns="alpha", values="nc", aggfunc="mean").round(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--models", default="fsrcnn,edsr,lapsrn",
                    help="comma-separated subset of: " + ",".join(SR_GENERALIZATION_MODELS))
    ap.add_argument("--csv", default=RESULTS_CSV,
                    help="results CSV path (use a separate file when running models concurrently)")
    args = ap.parse_args()

    os.makedirs(ATTACKED_DIR, exist_ok=True)
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    models = usable_models(requested)
    if not models:
        print("No usable SR models — nothing to do.", flush=True)
        return

    if args.sanity:
        sanity_check(models)
    else:
        full_run(models, results_csv=args.csv)


if __name__ == "__main__":
    main()
