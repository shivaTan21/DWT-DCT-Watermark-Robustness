"""
Frequency sweep: asymmetric DCT coefficient pair robustness analysis.

Goal: determine whether coefficient-pair symmetry contributes to watermark
fragility under AI-based image enhancement, and identify pair characteristics
associated with robustness.

Pair groups tested:
  1. High-differential pairs     — diagnostic; one coeff is much less stable
  2. Low-differential jointly stable — expected robust group
  3. Reference pairs             — (4,1)/(1,4), (7,1)/(1,7), (1,2)/(2,1)

For every pair both direction orderings (A/B) are embedded and extracted.
The CSV saves all raw rows; plots/rankings use only the better direction.
"""

import sys
import os
import itertools
import time
import warnings

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from src import watermark as wm_module          # monkey-patching target
from src import config
from src.metrics import normalized_correlation, bit_error_rate
from src.attacks_traditional import jpeg_compression, gaussian_blur
from src.attacks_ai import (
    real_esrgan_enhance, REALESRGAN_AVAILABLE,
    espcn_x4_enhance, OPENCV_SR_AVAILABLE as ESPCN_AVAILABLE,
)

# ── experiment constants ───────────────────────────────────────────────────────
ALPHA = 0.1
RANDOM_SEED = 42
WM_SHAPE = config.WATERMARK_SIZE   # (32, 32)
ESRGAN_MODEL = os.path.join(_ROOT, "data", "models", "RealESRGAN_x4.pth")

ATTACKS = ["real_esrgan", "jpeg_compression_q50", "gaussian_blur_5x5", "espcn_x4"]
ATK_DISPLAY = {
    "real_esrgan":           "Real-ESRGAN",
    "jpeg_compression_q50":  "JPEG Q50",
    "gaussian_blur_5x5":     "Gaussian Blur 5×5",
    "espcn_x4":              "ESPCN x4",
}

# canonical user-readable labels for the three reference pairs
_REF_FROZENSETS = {
    frozenset([(4, 1), (1, 4)]): "(4,1)/(1,4)",
    frozenset([(7, 1), (1, 7)]): "(7,1)/(1,7)",
    frozenset([(1, 2), (2, 1)]): "(1,2)/(2,1)",
}
REFERENCE_PAIRS = [((4, 1), (1, 4)), ((7, 1), (1, 7)), ((1, 2), (2, 1))]


# ── pair labelling ─────────────────────────────────────────────────────────────
def _canonical_label(p1, p2):
    """Stable label: lower linear-index position first."""
    a, b = sorted([p1, p2], key=lambda p: p[0] * 8 + p[1])
    return f"({a[0]},{a[1]})/({b[0]},{b[1]})"


def _ref_label(p1, p2):
    """User-friendly label if this is a reference pair, else None."""
    return _REF_FROZENSETS.get(frozenset([p1, p2]))


# ── stability data ─────────────────────────────────────────────────────────────
def _load_stability():
    csv = os.path.join(config.RESULTS_DIR, "dct_stability_realesrgan.csv")
    df = pd.read_csv(csv)
    return {(int(r.coeff_row), int(r.coeff_col)): r.mean_abs_change
            for r in df.itertuples(index=False)}


# ── candidate pair selection ───────────────────────────────────────────────────
def build_candidate_pairs(stability):
    non_dc = [(r, c) for r in range(8) for c in range(8) if (r, c) != (0, 0)]

    pair_stats = []
    for p1, p2 in itertools.combinations(non_dc, 2):
        mac1, mac2 = stability[p1], stability[p2]
        diff = abs(mac1 - mac2)
        combined = mac1 + mac2
        pair_stats.append((p1, p2, diff, combined))

    # group 1: top 20 by stability_differential (high differential)
    high_diff = sorted(pair_stats, key=lambda x: x[2], reverse=True)[:20]

    # group 2: top 20 by lowest combined MAC, only where diff < 1.0
    low_diff_stable = sorted(
        (x for x in pair_stats if x[2] < 1.0),
        key=lambda x: x[3]
    )[:20]

    seen: set = set()
    candidates: list = []

    def _add(p1, p2):
        key = frozenset([p1, p2])
        if key in seen:
            return
        seen.add(key)
        a, b = sorted([p1, p2], key=lambda p: p[0] * 8 + p[1])
        candidates.append((a, b))

    for p1, p2, *_ in high_diff:
        _add(p1, p2)
    for p1, p2, *_ in low_diff_stable:
        _add(p1, p2)
    for p1, p2 in REFERENCE_PAIRS:
        _add(p1, p2)

    return candidates


# ── symmetry classification ────────────────────────────────────────────────────
def classify_symmetry(p1, p2):
    if p1 == (p2[1], p2[0]):
        return "symmetric"
    mirror_p1 = (p1[1], p1[0])
    dist = abs(mirror_p1[0] - p2[0]) + abs(mirror_p1[1] - p2[1])
    return "near-symmetric" if dist <= 2 else "asymmetric"


# ── monkey-patched embed / extract ────────────────────────────────────────────
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


# ── attack dispatch ────────────────────────────────────────────────────────────
def _apply_attack(name, image):
    if name == "real_esrgan":
        return real_esrgan_enhance(image, model_path=ESRGAN_MODEL)
    if name == "jpeg_compression_q50":
        return jpeg_compression(image, quality=50)
    if name == "gaussian_blur_5x5":
        return gaussian_blur(image, ksize=5, sigma=1.0)
    if name == "espcn_x4":
        return espcn_x4_enhance(image)
    raise ValueError(f"Unknown attack: {name}")


# ── main sweep ─────────────────────────────────────────────────────────────────
def run_sweep(candidates):
    np.random.seed(RANDOM_SEED)
    wm = wm_module.generate_watermark(shape=WM_SHAPE, seed=RANDOM_SEED)

    if not REALESRGAN_AVAILABLE:
        print("WARNING: Real-ESRGAN not installed — real_esrgan results will be NaN.")
    elif not os.path.exists(ESRGAN_MODEL):
        print(f"WARNING: model weights not found at {ESRGAN_MODEL} — real_esrgan may fail.")
    if not ESPCN_AVAILABLE:
        print("WARNING: opencv-contrib-python not installed — espcn_x4 results will be NaN.")

    image_files = sorted(
        f for f in os.listdir(config.ORIGINAL_IMAGES_DIR)
        if f.lower().endswith(".png")
    )[:20]
    print(f"Images: {len(image_files)}  |  Pairs: {len(candidates)}  |  "
          f"Directions: 2  |  Attacks: {len(ATTACKS)}")
    total_embeds = len(candidates) * 2 * len(image_files)
    total_rows   = total_embeds * len(ATTACKS)
    print(f"Total rows to generate: {total_rows}")

    records = []
    embed_count = 0
    t0 = time.time()

    for pair_idx, (p1, p2) in enumerate(candidates):
        label = _canonical_label(p1, p2)
        rl = _ref_label(p1, p2)
        tag = f"  [ref: {rl}]" if rl else ""
        print(f"\n[{pair_idx + 1:2d}/{len(candidates)}] {label}{tag}")

        for direction in ("A", "B"):
            pos1, pos2 = (p1, p2) if direction == "A" else (p2, p1)

            for img_file in image_files:
                img_path = os.path.join(config.ORIGINAL_IMAGES_DIR, img_file)
                raw = cv2.imread(img_path)
                if raw is None:
                    print(f"  [WARN] Cannot load {img_file} — skipping")
                    for atk in ATTACKS:
                        records.append(dict(coeff_pair=label, direction=direction,
                                            attack_name=atk, image=img_file,
                                            nc=np.nan, ber=np.nan))
                    continue

                img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float64)
                img = cv2.resize(img, config.IMAGE_SIZE).astype(np.float64)

                try:
                    wm_img = np.clip(_embed(img, wm, pos1, pos2), 0, 255)
                except Exception as exc:
                    print(f"  [WARN] Embed failed {label} dir={direction} {img_file}: {exc}")
                    for atk in ATTACKS:
                        records.append(dict(coeff_pair=label, direction=direction,
                                            attack_name=atk, image=img_file,
                                            nc=np.nan, ber=np.nan))
                    continue

                # soft sanity check: extract from un-attacked watermarked image
                try:
                    sanity_nc = normalized_correlation(wm, _extract(wm_img, pos1, pos2))
                    if sanity_nc < 0.99:
                        print(f"\n  [WARNING] Pair {label} dir={direction} {img_file} "
                              f"sanity NC={sanity_nc:.4f} below 0.99 — continuing anyway")
                except Exception as exc:
                    print(f"\n  [WARNING] Sanity check failed {label} dir={direction} {img_file}: {exc}")

                for atk in ATTACKS:
                    try:
                        attacked = np.clip(_apply_attack(atk, wm_img), 0, 255)
                        extracted = _extract(attacked, pos1, pos2)
                        nc  = normalized_correlation(wm, extracted)
                        ber = bit_error_rate(wm, extracted)
                    except Exception as exc:
                        print(f"  [WARN] {atk} failed {label} dir={direction} {img_file}: {exc}")
                        nc, ber = np.nan, np.nan

                    records.append(dict(coeff_pair=label, direction=direction,
                                        attack_name=atk, image=img_file,
                                        nc=nc, ber=ber))

                embed_count += 1
                elapsed = time.time() - t0
                rate    = embed_count / elapsed if elapsed > 0 else 1.0
                eta_min = (total_embeds - embed_count) / rate / 60.0
                print(f"  {direction}/{img_file}  embed {embed_count}/{total_embeds}  "
                      f"ETA {eta_min:.1f} min    ", end="\r")

    print()
    df = pd.DataFrame(records)
    out = os.path.join(config.RESULTS_DIR, "frequency_sweep.csv")
    df.to_csv(out, index=False)
    print(f"Saved {out}  ({len(df)} rows)")
    return df


# ── best-direction selection ───────────────────────────────────────────────────
def select_best_direction(df):
    """Keep rows only for the direction with higher overall mean NC per pair."""
    pivot = (df.groupby(["coeff_pair", "direction"])["nc"]
               .mean()
               .reset_index()
               .pivot(index="coeff_pair", columns="direction", values="nc")
               .fillna(-np.inf))
    best = {}
    for pair in pivot.index:
        nc_a = pivot.loc[pair, "A"] if "A" in pivot.columns else -np.inf
        nc_b = pivot.loc[pair, "B"] if "B" in pivot.columns else -np.inf
        best[pair] = "A" if nc_a >= nc_b else "B"
    mask = df.apply(lambda r: r["direction"] == best.get(r["coeff_pair"], "A"), axis=1)
    return df[mask].copy()


# ── plots ──────────────────────────────────────────────────────────────────────
def _ref_label_map(candidates):
    """canonical_label → user-friendly label for reference pairs."""
    return {
        _canonical_label(p1, p2): _ref_label(p1, p2)
        for p1, p2 in candidates
        if _ref_label(p1, p2)
    }


def plot_heatmap(df_best, candidates):
    pivot = (df_best
             .groupby(["coeff_pair", "attack_name"])["nc"]
             .mean()
             .unstack(fill_value=np.nan)
             .reindex(columns=ATTACKS))
    pivot["_overall"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_overall", ascending=False).drop(columns=["_overall"])

    rlmap = _ref_label_map(candidates)
    ytick_labels = []
    for lbl in pivot.index:
        rl = rlmap.get(lbl)
        ytick_labels.append(f"{lbl}  ★{rl}" if rl else lbl)

    n_rows = len(pivot)
    fig_h = max(10, n_rows * 0.32)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(len(ATTACKS)))
    ax.set_xticklabels([ATK_DISPLAY[a] for a in ATTACKS], fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ytick_labels, fontsize=7)
    plt.colorbar(im, ax=ax, label="Mean NC", shrink=0.6)
    ax.set_title(
        "Frequency Sweep — Mean NC by Coefficient Pair × Attack\n"
        "(sorted by overall NC ↓;  ★ = reference pair)",
        fontsize=10,
    )
    plt.tight_layout()
    out = os.path.join(config.PLOTS_DIR, "frequency_sweep_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_scatter(df_best, candidates):
    rlmap = _ref_label_map(candidates)
    mean_nc = df_best.groupby(["coeff_pair", "attack_name"])["nc"].mean().reset_index()
    esrgan = mean_nc[mean_nc["attack_name"] == "real_esrgan"].set_index("coeff_pair")["nc"]
    jpeg   = mean_nc[mean_nc["attack_name"] == "jpeg_compression_q50"].set_index("coeff_pair")["nc"]

    pairs  = sorted(set(esrgan.index) & set(jpeg.index))
    x = np.array([esrgan.get(p, np.nan) for p in pairs])
    y = np.array([jpeg.get(p, np.nan) for p in pairs])

    valid   = ~(np.isnan(x) | np.isnan(y))
    xv, yv  = x[valid], y[valid]
    pv      = [p for p, ok in zip(pairs, valid) if ok]

    # Pareto-optimal: not dominated on both axes
    pareto = np.ones(len(xv), dtype=bool)
    for i in range(len(xv)):
        for j in range(len(xv)):
            if i != j and xv[j] >= xv[i] and yv[j] >= yv[i] and (xv[j] > xv[i] or yv[j] > yv[i]):
                pareto[i] = False
                break

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xv[~pareto], yv[~pareto], alpha=0.55, c="steelblue", s=40, label="Pairs")
    ax.scatter(xv[pareto],  yv[pareto],  alpha=0.90, c="darkorange", s=80,
               marker="*", label="Pareto-optimal", zorder=4)

    for i, pair in enumerate(pv):
        rl = rlmap.get(pair)
        if rl:
            ax.scatter([xv[i]], [yv[i]], c="crimson", s=90, zorder=6)
            ax.annotate(rl, (xv[i], yv[i]), fontsize=8, color="crimson",
                        xytext=(6, 4), textcoords="offset points")

    ax.set_xlabel("Real-ESRGAN Mean NC", fontsize=11)
    ax.set_ylabel("JPEG Q50 Mean NC",    fontsize=11)
    ax.set_title("Frequency Sweep — Real-ESRGAN NC vs JPEG NC per Coefficient Pair\n"
                 "(red = reference pairs;  ★ = Pareto-optimal)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(config.PLOTS_DIR, "frequency_sweep_scatter.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def symmetry_analysis(df_best, candidates):
    pair_to_sym = {_canonical_label(p1, p2): classify_symmetry(p1, p2)
                   for p1, p2 in candidates}
    dfb = df_best.copy()
    dfb["symmetry"] = dfb["coeff_pair"].map(pair_to_sym)

    categories = ["symmetric", "near-symmetric", "asymmetric"]
    records = []
    for cat in categories:
        sub = dfb[dfb["symmetry"] == cat]
        for atk in ATTACKS:
            atk_sub = sub[sub["attack_name"] == atk]
            records.append(dict(
                category=cat, attack_name=atk,
                mean_nc=float(atk_sub["nc"].mean()) if len(atk_sub) else np.nan,
                mean_ber=float(atk_sub["ber"].mean()) if len(atk_sub) else np.nan,
                sample_count=len(atk_sub),
            ))

    sym_df = pd.DataFrame(records)
    out_csv = os.path.join(config.RESULTS_DIR, "symmetry_summary.csv")
    sym_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # grouped bar chart
    pivot = (sym_df.pivot(index="category", columns="attack_name", values="mean_nc")
                   .reindex(index=categories, columns=ATTACKS))
    n_cats = len(categories)
    width  = 0.22
    x      = np.arange(len(ATTACKS))
    offsets = np.linspace(0, width * (n_cats - 1), n_cats)
    colors = ["#27ae60", "#2980b9", "#c0392b"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (cat, color) in enumerate(zip(categories, colors)):
        vals = [pivot.loc[cat, a] if cat in pivot.index else np.nan for a in ATTACKS]
        ax.bar(x + offsets[i], vals, width, label=cat.capitalize(),
               color=color, alpha=0.82, edgecolor="white")

    ax.set_xticks(x + offsets[n_cats // 2])
    ax.set_xticklabels([ATK_DISPLAY[a] for a in ATTACKS])
    ax.set_ylabel("Mean NC")
    ax.set_title("Watermark Robustness by Symmetry Category")
    ax.legend()
    ax.set_ylim(-0.3, 1.1)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(config.PLOTS_DIR, "symmetry_analysis.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")

    return sym_df


# ── console summary ────────────────────────────────────────────────────────────
def print_summary(df_best, sym_df):
    sep = "=" * 62

    # top 5 by overall NC
    overall_nc = (df_best.groupby("coeff_pair")["nc"]
                         .mean().sort_values(ascending=False))
    print(f"\n{sep}")
    print("Top 5 coefficient pairs by overall mean NC")
    print(sep)
    for pair, nc in overall_nc.head(5).items():
        print(f"  {pair}: {nc:.4f}")

    # top 5 by ESRGAN NC
    esrgan_nc = (df_best[df_best["attack_name"] == "real_esrgan"]
                 .groupby("coeff_pair")["nc"].mean()
                 .sort_values(ascending=False))
    print(f"\n{sep}")
    print("Top 5 coefficient pairs by Real-ESRGAN NC")
    print(sep)
    for pair, nc in esrgan_nc.head(5).items():
        print(f"  {pair}: {nc:.4f}")

    # symmetry comparison table
    print(f"\n{sep}")
    print("Symmetry category comparison")
    print(sep)
    hdr = (f"{'Category':<20} | {'Real-ESRGAN NC':>16} | "
           f"{'JPEG NC':>9} | {'Blur NC':>9} | {'ESPCN NC':>9} | {'Overall NC':>10}")
    print(hdr)
    print("-" * len(hdr))

    nc_by_cat: dict = {}
    for cat in ["symmetric", "near-symmetric", "asymmetric"]:
        rows = sym_df[sym_df["category"] == cat]
        nc_map = {r.attack_name: r.mean_nc for r in rows.itertuples(index=False)}
        e  = nc_map.get("real_esrgan",          np.nan)
        j  = nc_map.get("jpeg_compression_q50", np.nan)
        b  = nc_map.get("gaussian_blur_5x5",    np.nan)
        es = nc_map.get("espcn_x4",             np.nan)
        ov = float(np.nanmean([e, j, b, es]))
        nc_by_cat[cat] = ov
        print(f"  {cat:<18} | {e:>16.4f} | {j:>9.4f} | {b:>9.4f} | {es:>9.4f} | {ov:>10.4f}")

    sym_nc  = nc_by_cat.get("symmetric",      np.nan)
    asym_nc = nc_by_cat.get("asymmetric",     np.nan)
    near_nc = nc_by_cat.get("near-symmetric", np.nan)

    print()
    answer = "YES" if asym_nc > sym_nc else "NO"
    print(f"Do asymmetric pairs outperform symmetric pairs on average? {answer}")
    print(f"  Asymmetric mean NC:    {asym_nc:.4f}")
    print(f"  Symmetric mean NC:     {sym_nc:.4f}")
    print(f"  Near-symmetric mean NC:{near_nc:.4f}")

    print(f"\n{sep}")
    print("Recommendation")
    print(sep)
    if asym_nc > sym_nc:
        print("  Is symmetry harmful?    YES — asymmetric pairs outperform symmetric on average.")
        print("  Is symmetry beneficial? NO")
    elif sym_nc > asym_nc:
        print("  Is symmetry harmful?    NO")
        print("  Is symmetry beneficial? YES — symmetric pairs outperform asymmetric on average.")
    else:
        print("  Is symmetry harmful?    UNCLEAR")
        print("  Is symmetry beneficial? UNCLEAR")

    best_overall = nc_by_cat.get(
        max(nc_by_cat, key=lambda k: nc_by_cat[k]), np.nan
    )
    print(f"\n  Is stability differential more predictive than symmetry?")
    print(f"    → Compare low-differential jointly-stable pairs (expected robust group)")
    print(f"      against high-differential pairs (diagnostic group) in the heatmap.")
    print(f"      See results/plots/frequency_sweep_heatmap.png  and  symmetry_analysis.png.")


# ── entry point ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    stability  = _load_stability()
    candidates = build_candidate_pairs(stability)

    print(f"Candidate pairs: {len(candidates)}")
    for cat_name, fn in [
        ("symmetric",      lambda p1, p2: classify_symmetry(p1, p2) == "symmetric"),
        ("near-symmetric", lambda p1, p2: classify_symmetry(p1, p2) == "near-symmetric"),
        ("asymmetric",     lambda p1, p2: classify_symmetry(p1, p2) == "asymmetric"),
    ]:
        n = sum(1 for p1, p2 in candidates if fn(p1, p2))
        print(f"  {cat_name}: {n}")

    df      = run_sweep(candidates)
    df_best = select_best_direction(df)

    plot_heatmap(df_best, candidates)
    plot_scatter(df_best, candidates)
    sym_df = symmetry_analysis(df_best, candidates)
    print_summary(df_best, sym_df)


if __name__ == "__main__":
    main()
