"""
Experiment J: Like-for-like rerun of the balanced pair (2,3)/(3,2)
under the CANONICAL Real-ESRGAN pipeline.

Motivation
----------
Exp F evaluated the balanced pair's Real-ESRGAN arm with a different
invocation (patch_balanced_esrgan.py: patches_size=64/padding=8/pad_size=8,
autocast fully disabled, INTER_LANCZOS4 downscale) than the standard and HF
pairs, whose attacked images were pre-computed by src/run_experiment.py via
the canonical wrapper src.attacks_ai.real_esrgan_enhance (whole-image
model.predict() with library defaults batch_size=4/patches_size=192/
padding=24/pad_size=15, INTER_AREA downscale to 512x512). The manuscript
(main_6page.tex, Table tab:pairs dagger note + Limitations) flags this and
requires a like-for-like 100-image rerun before camera-ready.

This experiment reruns ONLY the balanced pair through the identical
canonical pipeline and re-derives the standard/HF reference rows from the
same frozen disk files Exp F used, so all three arms are strictly
comparable. No existing file is modified.

Pipeline parity (balanced arm == standard/HF arms):
  originals:      data/original_images/*, cv2 BGR read, INTER_AREA resize
                  to (512,512), BGR->RGB float64            [= run_experiment.py]
  watermark:      generate_watermark((32,32), seed=42)      [data/watermark.png absent]
  embedding:      embed_watermark_rgb on Y channel, alpha in {0.05,0.1,0.2,0.3},
                  coefficient positions swapped in via module globals
  quantization:   watermarked clipped to uint8 (mirrors the lossless PNG
                  round trip the standard/HF disk files went through)
  Real-ESRGAN:    src.attacks_ai.real_esrgan_enhance, weights
                  weights/RealESRGAN_x4.pth, default predict() args,
                  INTER_AREA downscale — verified bit-identical to the
                  frozen standard/HF attacked files (see parity check)
  JPEG:           attacks_traditional.jpeg_compression(quality=50)
  Gaussian blur:  attacks_traditional.gaussian_blur(ksize=5, sigma=1.0)
  metrics:        src.metrics NC / BER / PSNR / SSIM vs the original
  statistics:     Shapiro-gated paired t / Wilcoxon signed-rank per
                  (attack, alpha), BH-FDR corrected            [= Exp F]

Outputs (all NEW files, inside this experiment directory):
  parity_check.txt              bit-identity proof of the canonical pipeline
  results_balanced_rerun.csv    per-image rows, all three pairs
  summary_balanced_rerun.csv    per (pair, alpha, attack) means/CIs
  tests_balanced_rerun.csv      paired tests, BH-FDR corrected
  balanced_vs_standard.md       report
  attacked_esrgan/              balanced-pair attacked images (audit trail)

Usage:
  python experiments/exp_J_balanced_like_for_like/run_exp_J.py            # collect (resumable) + analyze
  python experiments/exp_J_balanced_like_for_like/run_exp_J.py --analyze  # analysis only, from existing CSV
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
from scipy.stats import shapiro, ttest_rel, wilcoxon, t as t_dist, false_discovery_control

from src import config
from src import watermark as wm_module
from src.metrics import normalized_correlation, bit_error_rate, psnr, ssim
from src.attacks_traditional import jpeg_compression, gaussian_blur

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ESRGAN_OUT_DIR = os.path.join(OUT_DIR, "attacked_esrgan")

SEED = 42
WM_SHAPE = config.WATERMARK_SIZE     # (32, 32)
IMAGE_SIZE = config.IMAGE_SIZE       # (512, 512)
ESRGAN_MODEL = os.path.join(_ROOT, "weights", "RealESRGAN_x4.pth")

ORIG_DIR = config.ORIGINAL_IMAGES_DIR
WATERMARKED_DIR = config.WATERMARKED_DIR
AI_DIR = config.ATTACKED_AI_DIR

# alpha 0.10 first so the primary operating point completes earliest
ALPHAS_RUN_ORDER = [0.1, 0.05, 0.2, 0.3]
ALPHAS_SORTED = sorted(ALPHAS_RUN_ORDER)

PAIRS = {
    "standard": {"pos1": (4, 1), "pos2": (1, 4), "disk_suffix": "",
                 "label": "Standard (4,1)/(1,4)"},
    "balanced": {"pos1": (2, 3), "pos2": (3, 2), "disk_suffix": None,
                 "label": "Balanced (2,3)/(3,2)"},
    "hf":       {"pos1": (7, 5), "pos2": (7, 7), "disk_suffix": "__optimized_positions",
                 "label": "HF (7,5)/(7,7)"},
}

ATTACKS = ["none", "real_esrgan", "jpeg_q50", "gaussian_blur"]
ATTACK_DISPLAY = {
    "none": "No attack", "real_esrgan": "Real-ESRGAN",
    "jpeg_q50": "JPEG Q50", "gaussian_blur": "Gaussian Blur",
}

RESULTS_CSV = os.path.join(OUT_DIR, "results_balanced_rerun.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "summary_balanced_rerun.csv")
TESTS_CSV = os.path.join(OUT_DIR, "tests_balanced_rerun.csv")
REPORT_MD = os.path.join(OUT_DIR, "balanced_vs_standard.md")
PARITY_TXT = os.path.join(OUT_DIR, "parity_check.txt")

EXP_F_SUMMARY = os.path.join(
    _ROOT, "experiments", "exp_F_realesrgan_pair_verification", "exp_F_summary.csv")


# ─── I/O helpers (identical to Exp F / run_experiment.py conventions) ─────────

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


def _embed_custom(original, wm, pos1, pos2, alpha):
    orig1, orig2 = wm_module.COEFF_POS_1, wm_module.COEFF_POS_2
    wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = pos1, pos2
    try:
        watermarked = wm_module.embed_watermark_rgb(
            original.astype(np.float64), wm, alpha=alpha)
    finally:
        wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = orig1, orig2
    # uint8 round trip == the lossless PNG save/load the disk arms went through
    return np.clip(watermarked, 0, 255).astype(np.uint8).astype(np.float64)


def _extract_custom(image, pos1, pos2):
    orig1, orig2 = wm_module.COEFF_POS_1, wm_module.COEFF_POS_2
    wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = pos1, pos2
    try:
        return wm_module.extract_watermark_rgb(image.astype(np.float64), WM_SHAPE)
    finally:
        wm_module.COEFF_POS_1, wm_module.COEFF_POS_2 = orig1, orig2


def list_images():
    image_files = sorted(
        f for f in os.listdir(ORIG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg")))
    return [os.path.splitext(f)[0] for f in image_files]


# ─── Parity check: canonical pipeline must reproduce frozen disk files ────────

def parity_check(wm):
    """Regenerate one standard-pair and one HF-pair watermarked + attacked image
    live and require bit-identity with the frozen disk files. Proves the rerun
    path IS the pipeline that produced the standard/HF results."""
    from src.attacks_ai import real_esrgan_enhance

    lines = ["Parity check: canonical pipeline vs frozen disk files",
             f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    all_ok = True
    for pair_key in ("standard", "hf"):
        pair = PAIRS[pair_key]
        suffix = pair["disk_suffix"]
        image_name, alpha = "kodim01", 0.1

        original = load_original(image_name)
        wm_live = _embed_custom(original, wm, pair["pos1"], pair["pos2"], alpha)
        wm_disk = load_bgr_to_rgb(
            os.path.join(WATERMARKED_DIR, f"{image_name}__alpha{alpha}{suffix}.png"))
        ok_wm = np.array_equal(wm_live, wm_disk)

        esr_live = real_esrgan_enhance(wm_live, model_path=ESRGAN_MODEL)
        esr_live_u8 = np.clip(esr_live, 0, 255).astype(np.uint8).astype(np.float64)
        esr_disk = load_bgr_to_rgb(
            os.path.join(AI_DIR, f"{image_name}__alpha{alpha}{suffix}__real_esrgan.png"))
        ok_esr = np.array_equal(esr_live_u8, esr_disk)

        lines.append(f"[{pair_key}] {image_name} alpha={alpha}: "
                     f"watermarked bit-identical={ok_wm}, "
                     f"real_esrgan bit-identical={ok_esr}")
        all_ok = all_ok and ok_wm and ok_esr

    lines.append("")
    lines.append("RESULT: " + ("PASS — the live canonical pipeline is bit-identical to "
                               "the pipeline that produced the frozen standard/HF files."
                               if all_ok else "FAIL — do NOT trust the rerun; investigate."))
    with open(PARITY_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    for ln in lines:
        print("  " + ln, flush=True)
    if not all_ok:
        raise RuntimeError("Parity check failed — canonical pipeline does not "
                           "reproduce the frozen standard/HF disk files.")


# ─── Collection ────────────────────────────────────────────────────────────────

def _rows_for_pair(image_name, dataset, original, wm, alpha, pair_key, esrgan_fn):
    """Compute the four attack rows for one (image, alpha, pair)."""
    pair = PAIRS[pair_key]
    pos1, pos2 = pair["pos1"], pair["pos2"]
    suffix = pair["disk_suffix"]

    if suffix is not None:
        wm_img = load_bgr_to_rgb(
            os.path.join(WATERMARKED_DIR, f"{image_name}__alpha{alpha}{suffix}.png"))
        if wm_img is None:
            raise FileNotFoundError(f"missing watermarked disk file for {pair_key} "
                                    f"{image_name} alpha={alpha}")
    else:
        wm_img = _embed_custom(original, wm, pos1, pos2, alpha)

    rows = []

    def add(attack, attacked_img, esrgan_source=""):
        ext = _extract_custom(attacked_img, pos1, pos2)
        rows.append({
            "image": image_name, "dataset": dataset,
            "coefficient_pair": pair_key, "alpha": alpha, "attack": attack,
            "nc": normalized_correlation(wm, ext),
            "ber": bit_error_rate(wm, ext),
            "psnr": psnr(original, attacked_img),
            "ssim": ssim(original, attacked_img),
            "esrgan_source": esrgan_source,
        })

    # No attack
    add("none", wm_img)

    # Real-ESRGAN
    if suffix is not None:
        esr = load_bgr_to_rgb(
            os.path.join(AI_DIR, f"{image_name}__alpha{alpha}{suffix}__real_esrgan.png"))
        if esr is None:
            raise FileNotFoundError(f"missing ESRGAN disk file for {pair_key} "
                                    f"{image_name} alpha={alpha}")
        add("real_esrgan", esr, esrgan_source="disk_precomputed_canonical")
    else:
        esr = esrgan_fn(wm_img, model_path=ESRGAN_MODEL)
        esr_u8 = np.clip(esr, 0, 255).astype(np.uint8)
        out_png = os.path.join(
            ESRGAN_OUT_DIR, f"{image_name}__alpha{alpha}__balanced__real_esrgan.png")
        cv2.imwrite(out_png, cv2.cvtColor(esr_u8, cv2.COLOR_RGB2BGR))
        add("real_esrgan", esr_u8.astype(np.float64), esrgan_source="live_canonical")

    # JPEG Q50
    jpeg_img = np.clip(jpeg_compression(wm_img, quality=50), 0, 255)
    add("jpeg_q50", jpeg_img)

    # Gaussian blur (ksize=5, sigma=1.0 — baseline params)
    blur_img = np.clip(gaussian_blur(wm_img, ksize=5, sigma=1.0), 0, 255)
    add("gaussian_blur", blur_img)

    return rows


def collect():
    from src.attacks_ai import real_esrgan_enhance, REALESRGAN_AVAILABLE
    if not REALESRGAN_AVAILABLE:
        raise RuntimeError("py_real_esrgan required for the like-for-like rerun.")

    os.makedirs(ESRGAN_OUT_DIR, exist_ok=True)
    np.random.seed(SEED)
    wm = wm_module.generate_watermark(shape=WM_SHAPE, seed=SEED)

    if not os.path.exists(PARITY_TXT):
        print("[parity] verifying canonical pipeline against frozen disk files...",
              flush=True)
        parity_check(wm)

    image_names = list_images()
    print(f"[collect] {len(image_names)} images, alphas (run order) {ALPHAS_RUN_ORDER}",
          flush=True)

    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        done = set(zip(df["image"], df["alpha"], df["coefficient_pair"]))
        print(f"[collect] resuming: {len(df)} rows already present", flush=True)
    else:
        df = pd.DataFrame()
        done = set()

    t_start = time.time()
    n_units = 0
    for alpha in ALPHAS_RUN_ORDER:
        for img_idx, image_name in enumerate(image_names):
            todo = [pk for pk in PAIRS if (image_name, alpha, pk) not in done]
            if not todo:
                continue
            original = load_original(image_name)
            dataset = dataset_of(image_name)

            new_rows = []
            for pair_key in todo:
                new_rows.extend(_rows_for_pair(
                    image_name, dataset, original, wm, alpha, pair_key,
                    real_esrgan_enhance))

            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df.to_csv(RESULTS_CSV, index=False)
            n_units += 1

            bal_esr = [r for r in new_rows
                       if r["coefficient_pair"] == "balanced" and r["attack"] == "real_esrgan"]
            nc_str = f"bal ESRGAN NC={bal_esr[0]['nc']:.4f}" if bal_esr else "(cached)"
            elapsed = time.time() - t_start
            print(f"[alpha={alpha}] [{img_idx+1}/{len(image_names)}] {image_name}  "
                  f"{nc_str}  ({elapsed/60:.1f} min elapsed)", flush=True)

    print(f"[collect] done: {len(df)} rows in {RESULTS_CSV}", flush=True)
    return df


# ─── Statistics (identical methodology to Exp F / the paper) ─────────────────

def ci95(vals):
    n = len(vals)
    if n < 2:
        return np.nan
    se = np.std(vals, ddof=1) / np.sqrt(n)
    return float(t_dist.ppf(0.975, df=n - 1) * se)


def compute_summary(df):
    rows = []
    for (pair, alpha, attack), grp in df.groupby(["coefficient_pair", "alpha", "attack"]):
        nc_vals = grp["nc"].dropna().values
        rows.append({
            "coefficient_pair": pair, "alpha": alpha, "attack": attack,
            "n": len(nc_vals),
            "nc_mean": float(np.mean(nc_vals)) if len(nc_vals) else np.nan,
            "nc_std": float(np.std(nc_vals, ddof=1)) if len(nc_vals) > 1 else np.nan,
            "nc_ci95": ci95(nc_vals),
            "ber_mean": float(grp["ber"].mean()),
            "ber_std": float(grp["ber"].std(ddof=1)),
            "psnr_mean": float(grp["psnr"].mean()),
            "ssim_mean": float(grp["ssim"].mean()),
        })
    return pd.DataFrame(rows)


def _paired_test(diffs):
    """Shapiro-gated paired t-test / Wilcoxon signed-rank — same as Exp F."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 5:
        return "n/a", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    se_d = std_d / np.sqrt(n)
    _, p_norm = shapiro(diffs)

    if p_norm > 0.05:
        t_stat, p_raw = ttest_rel(diffs, np.zeros(n))
        ci_lo = mean_d - t_dist.ppf(0.975, n - 1) * se_d
        ci_hi = mean_d + t_dist.ppf(0.975, n - 1) * se_d
        effect = mean_d / std_d if std_d > 0 else 0.0
        return "t-test", float(t_stat), float(p_raw), mean_d, ci_lo, ci_hi, effect

    try:
        alt = "greater" if mean_d >= 0 else "less"
        stat, p_raw = wilcoxon(diffs, alternative=alt)
        from scipy.stats import norm
        z = norm.ppf(1 - p_raw / 2) * np.sign(mean_d)
        effect = float(z / np.sqrt(n))
    except Exception:
        stat, p_raw, effect = np.nan, np.nan, np.nan
    ci_lo = mean_d - 1.96 * se_d
    ci_hi = mean_d + 1.96 * se_d
    return "wilcoxon", float(stat), float(p_raw), mean_d, ci_lo, ci_hi, effect


def run_statistical_tests(df):
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
        for attack in ATTACKS:
            for alpha in ALPHAS_SORTED:
                idx = [k for k in common if k[1] == alpha and k[2] == attack]
                if not idx:
                    continue
                nc_a = sub_a.loc[idx, "nc"].values.astype(float)
                nc_b = sub_b.loc[idx, "nc"].values.astype(float)
                test, stat, p_raw, mean_d, ci_lo, ci_hi, effect = _paired_test(nc_a - nc_b)
                all_rows.append({
                    "comparison": comp_label, "pair_a": pair_a, "pair_b": pair_b,
                    "attack": attack, "alpha": alpha, "n": len(idx),
                    "mean_diff": mean_d, "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "effect_size": effect, "test": test, "stat": stat, "p_raw": p_raw,
                    "nc_mean_a": float(np.nanmean(nc_a)),
                    "nc_mean_b": float(np.nanmean(nc_b)),
                })
    tests_df = pd.DataFrame(all_rows)
    if tests_df.empty:
        return tests_df
    valid = tests_df["p_raw"].notna()
    tests_df["p_adj_bh"] = np.nan
    if valid.any():
        tests_df.loc[valid, "p_adj_bh"] = false_discovery_control(
            tests_df.loc[valid, "p_raw"].values, method="bh")
    return tests_df


# ─── Cross-check against Exp F (standard/HF arms must reproduce exactly) ─────

def crosscheck_exp_f(summary_df):
    """The standard/HF arms use the same disk files and computations as Exp F,
    so their summary numbers must match exp_F_summary.csv. Returns a dataframe
    of the comparison."""
    if not os.path.exists(EXP_F_SUMMARY):
        return None
    f_sum = pd.read_csv(EXP_F_SUMMARY)
    rows = []
    for _, r in summary_df.iterrows():
        if r["coefficient_pair"] == "balanced":
            continue
        match = f_sum[
            (f_sum["coefficient_pair"] == r["coefficient_pair"]) &
            (f_sum["alpha"] == r["alpha"]) &
            (f_sum["attack"] == r["attack"])]
        if match.empty:
            continue
        f_nc = float(match["nc_mean"].iloc[0])
        rows.append({
            "coefficient_pair": r["coefficient_pair"], "alpha": r["alpha"],
            "attack": r["attack"], "nc_mean_expJ": r["nc_mean"],
            "nc_mean_expF": f_nc, "abs_diff": abs(r["nc_mean"] - f_nc),
        })
    return pd.DataFrame(rows)


# ─── Report ───────────────────────────────────────────────────────────────────

def _sig(p):
    if p is None or np.isnan(p):
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def write_report(df, summary_df, tests_df, xcheck_df):
    L = []
    emit = L.append

    def srow(pair, attack, alpha):
        r = summary_df[(summary_df["coefficient_pair"] == pair) &
                       (summary_df["attack"] == attack) &
                       (summary_df["alpha"] == alpha)]
        return r.iloc[0] if not r.empty else None

    def trow(comp, attack, alpha):
        r = tests_df[(tests_df["comparison"] == comp) &
                     (tests_df["attack"] == attack) &
                     (tests_df["alpha"] == alpha)]
        return r.iloc[0] if not r.empty else None

    alphas_present = sorted(df["alpha"].unique())
    n_images = df["image"].nunique()

    emit("# Balanced Pair Like-for-Like Rerun (Experiment J)")
    emit("")
    emit(f"Date: {time.strftime('%Y-%m-%d')}  |  n = {n_images} images "
         f"(24 Kodak + 76 TAMPERE17)  |  α ∈ {{{', '.join(str(a) for a in alphas_present)}}}")
    emit("")
    emit("**Purpose.** Eliminate the manuscript's known limitation: in Exp F the "
         "balanced pair (2,3)/(3,2) was evaluated with a *different* Real-ESRGAN "
         "invocation (patch-based, `patches_size=64`, autocast disabled, Lanczos4 "
         "downscale) than the standard (4,1)/(1,4) and HF (7,5)/(7,7) pairs "
         "(whole-image canonical wrapper). This experiment reruns ONLY the balanced "
         "arm through the identical canonical pipeline used for the other two pairs, "
         "on all 100 images and all four embedding strengths.")
    emit("")

    # ── 1. Parameter parity verification ─────────────────────────────────────
    emit("## 1. Pipeline parity verification")
    emit("")
    emit("Every element of the balanced arm matches the pipeline that produced the "
         "standard/HF results:")
    emit("")
    emit("| Element | Standard / HF arms | Balanced rerun (this experiment) | Match |")
    emit("|---|---|---|---|")
    emit("| Attack implementation | `src.attacks_ai.real_esrgan_enhance` | same function, same module | ✓ |")
    emit("| ESRGAN weights | `weights/RealESRGAN_x4.pth`, scale 4 | same file | ✓ |")
    emit("| Invocation | whole-image `model.predict()` defaults (batch_size=4, patches_size=192, padding=24, pad_size=15) | same (no overrides) | ✓ |")
    emit("| Post-attack resize | `cv2.resize(..., config.IMAGE_SIZE, INTER_AREA)` | same | ✓ |")
    emit("| Preprocessing | originals resized to 512×512 with INTER_AREA, BGR→RGB float64 | same | ✓ |")
    emit("| Watermarked quantization | uint8 PNG round trip (run_experiment.py) | uint8 round trip before attack | ✓ |")
    emit("| Watermark bits | `generate_watermark((32,32), seed=42)` | same | ✓ |")
    emit("| Embedding | `embed_watermark_rgb`, Y channel, Haar DWT, 8×8 DCT, margin floor 30.0 | same, positions (2,3)/(3,2) | ✓ |")
    emit("| Random seed | 42 | 42 | ✓ |")
    emit("| Datasets | 24 Kodak + 76 TAMPERE17, all 100 images | same | ✓ |")
    emit("| JPEG attack | `jpeg_compression(quality=50)` | same | ✓ |")
    emit("| Blur attack | `gaussian_blur(ksize=5, sigma=1.0)` | same | ✓ |")
    emit("| Metrics | `src.metrics` NC/BER/PSNR/SSIM vs original | same | ✓ |")
    emit("| Statistics | Shapiro-gated paired t / Wilcoxon, BH-FDR (Exp F code) | same | ✓ |")
    emit("")
    if os.path.exists(PARITY_TXT):
        emit("**Bit-identity proof** (`parity_check.txt`): regenerating the standard and "
             "HF watermarked and Real-ESRGAN-attacked images live through this rerun's "
             "code path reproduces the frozen disk files **bit-for-bit**:")
        emit("")
        emit("```")
        with open(PARITY_TXT) as f:
            emit(f.read().rstrip())
        emit("```")
        emit("")
    if xcheck_df is not None and len(xcheck_df):
        max_dev = xcheck_df["abs_diff"].max()
        emit(f"**Exp F cross-check:** the standard/HF summary values recomputed here "
             f"match `exp_F_summary.csv` with max |ΔNC| = {max_dev:.2e} across all "
             f"{len(xcheck_df)} (pair, α, attack) cells"
             + (" — exact reproduction." if max_dev < 1e-9 else "."))
        emit("")

    # ── 2/3. Results ─────────────────────────────────────────────────────────
    emit("## 2. Results at α = 0.10 (primary operating point)")
    emit("")
    emit("| Attack | Metric | Standard (4,1)/(1,4) | Balanced (2,3)/(3,2) | HF (7,5)/(7,7) |")
    emit("|---|---|---|---|---|")
    for atk in ATTACKS:
        for metric, fmt in [("nc_mean", "{:.4f}"), ("ber_mean", "{:.4f}"),
                            ("psnr_mean", "{:.2f}"), ("ssim_mean", "{:.4f}")]:
            vals = []
            for pk in ("standard", "balanced", "hf"):
                r = srow(pk, atk, 0.1)
                if r is None:
                    vals.append("n/a")
                elif metric == "nc_mean":
                    vals.append(f"{r['nc_mean']:.4f} ± {r['nc_std']:.4f}")
                else:
                    vals.append(fmt.format(r[metric]))
            label = {"nc_mean": "NC (±SD)", "ber_mean": "BER",
                     "psnr_mean": "PSNR (dB)", "ssim_mean": "SSIM"}[metric]
            atk_cell = ATTACK_DISPLAY[atk] if metric == "nc_mean" else ""
            emit(f"| {atk_cell} | {label} | {vals[0]} | {vals[1]} | {vals[2]} |")
    emit("")

    emit("### Real-ESRGAN NC across all embedding strengths")
    emit("")
    emit("| α | Standard | Balanced (rerun) | Balanced (old Exp F, patch pipeline) | HF |")
    emit("|---|---|---|---|---|")
    old_f = None
    if os.path.exists(EXP_F_SUMMARY):
        old_f = pd.read_csv(EXP_F_SUMMARY)
    for alpha in alphas_present:
        cells = []
        for pk in ("standard", "balanced"):
            r = srow(pk, "real_esrgan", alpha)
            cells.append("n/a" if r is None else f"{r['nc_mean']:.4f}")
        if old_f is not None:
            m = old_f[(old_f["coefficient_pair"] == "balanced") &
                      (old_f["attack"] == "real_esrgan") & (old_f["alpha"] == alpha)]
            cells.append("n/a" if m.empty else f"{float(m['nc_mean'].iloc[0]):.4f}")
        else:
            cells.append("n/a")
        r = srow("hf", "real_esrgan", alpha)
        cells.append("n/a" if r is None else f"{r['nc_mean']:.4f}")
        emit(f"| {alpha:.2f} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    emit("")

    # ── 4. Statistical tests ─────────────────────────────────────────────────
    emit("## 3. Statistical tests (paired, BH-FDR corrected — same methodology as the paper)")
    emit("")
    for comp, title in [("balanced_vs_standard", "Balanced vs Standard"),
                        ("hf_vs_standard", "HF vs Standard"),
                        ("balanced_vs_hf", "Balanced vs HF")]:
        emit(f"### {title}")
        emit("")
        emit("| Attack | α | ΔNC (mean) | 95% CI | p_adj (BH) | sig | effect | test | n |")
        emit("|---|---|---|---|---|---|---|---|---|")
        for alpha in alphas_present:
            for atk in ATTACKS:
                r = trow(comp, atk, alpha)
                if r is None:
                    continue
                p = r["p_adj_bh"]
                emit(f"| {ATTACK_DISPLAY[atk]} | {alpha:.2f} | {r['mean_diff']:+.4f} | "
                     f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
                     f"{'n/a' if np.isnan(p) else f'{p:.2e}'} | {_sig(p)} | "
                     f"{r['effect_size']:.3f} | {r['test']} | {int(r['n'])} |")
        emit("")

    # ── 5. Conclusions ───────────────────────────────────────────────────────
    emit("## 4. Does the manuscript conclusion change?")
    emit("")
    r_bs = trow("balanced_vs_standard", "real_esrgan", 0.1)
    r_hb = trow("balanced_vs_hf", "real_esrgan", 0.1)
    s_bal = srow("balanced", "real_esrgan", 0.1)
    s_std = srow("standard", "real_esrgan", 0.1)
    s_hf = srow("hf", "real_esrgan", 0.1)
    if r_bs is not None and s_bal is not None:
        d, p = r_bs["mean_diff"], r_bs["p_adj_bh"]
        emit(f"Under the identical canonical pipeline at α=0.10, the balanced pair's "
             f"Real-ESRGAN NC is **{s_bal['nc_mean']:.4f}** vs standard "
             f"**{s_std['nc_mean']:.4f}** (ΔNC = {d:+.4f}, {_sig(p)}, "
             f"p_adj = {p:.2e}) and HF **{s_hf['nc_mean']:.4f}**.")
        emit("")
        if d > 0 and p < 0.05:
            verdict = ("The balanced pair does **not** underperform the standard pair — "
                       "it is significantly (modestly) better. The old tabulated 0.683 was a "
                       "pipeline artifact, exactly as the manuscript's n=20 control predicted.")
        elif d > 0:
            verdict = ("The balanced pair does not underperform the standard pair "
                       "(difference positive but not significant); the old tabulated "
                       "0.683 was a pipeline artifact.")
        elif p < 0.05:
            verdict = ("The balanced pair still significantly underperforms the standard "
                       "pair even under the identical pipeline — the pipeline confound "
                       "did NOT fully explain the deficit. Manuscript claims based on the "
                       "n=20 control need revisiting.")
        else:
            verdict = ("Balanced ≈ standard under the identical pipeline; the old deficit "
                       "was a pipeline artifact.")
        emit(f"**Verdict:** {verdict}")
        emit("")
        if r_hb is not None:
            emit(f"HF remains {'the best' if r_hb['mean_diff'] < 0 else 'NOT the best'} "
                 f"pair under Real-ESRGAN (balanced − HF: {r_hb['mean_diff']:+.4f}, "
                 f"{_sig(r_hb['p_adj_bh'])}).")
            emit("")

    emit("*(Sections on manuscript sentence updates are appended below; see "
         "MANUSCRIPT_UPDATES.)*")
    emit("")
    emit("---")
    emit("*Generated by experiments/exp_J_balanced_like_for_like/run_exp_J.py — "
         "no existing experiment file was modified.*")

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[report] {REPORT_MD}", flush=True)


def analyze():
    df = pd.read_csv(RESULTS_CSV)
    summary_df = compute_summary(df)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"[analyze] {SUMMARY_CSV}", flush=True)
    tests_df = run_statistical_tests(df)
    tests_df.to_csv(TESTS_CSV, index=False)
    print(f"[analyze] {TESTS_CSV}", flush=True)
    xcheck_df = crosscheck_exp_f(summary_df)
    if xcheck_df is not None:
        xcheck_df.to_csv(os.path.join(OUT_DIR, "crosscheck_exp_f.csv"), index=False)
        print(f"[analyze] Exp F cross-check max |ΔNC| = {xcheck_df['abs_diff'].max():.3e}",
              flush=True)
    write_report(df, summary_df, tests_df, xcheck_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true",
                    help="skip collection; analyze existing results CSV")
    args = ap.parse_args()
    if not args.analyze:
        collect()
    analyze()


if __name__ == "__main__":
    main()
