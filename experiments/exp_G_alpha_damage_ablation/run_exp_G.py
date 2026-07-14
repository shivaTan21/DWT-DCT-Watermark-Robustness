"""
Experiment G: Alpha Stability of DCT Damage Profiles

Goal: Determine whether the Real-ESRGAN/JPEG DCT perturbation relationship
is stable across watermark embedding strength (α).

Uses: standard coefficient pair (4,1)/(1,4) only.
Reads pre-computed attacked files from results/attacked/ for all four alpha values.
Does NOT re-run ESRGAN; uses existing results/attacked/ai_enhancement/ disk files.

Primary question:
  Does the Real-ESRGAN/JPEG anti-complementarity (Spearman ρ = −0.907 at α=0.1)
  remain stable across α ∈ {0.05, 0.10, 0.20, 0.30}?

Interpretation:
  - If ρ remains strongly negative and consistent → pattern is robust to α.
  - If ρ changes substantially → trade-off is α-dependent.
  - Correlational evidence only; no causal claims.

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
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr, pearsonr

from src import config
from src.watermark import BLOCK_SIZE, DEFAULT_WAVELET, _dct2, rgb_to_ycbcr, COEFF_POS_1, COEFF_POS_2

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ALPHAS = [0.05, 0.10, 0.20, 0.30]
STANDARD_POS = [(4, 1), (1, 4)]   # standard embedding positions to track

WATERMARKED_DIR = config.WATERMARKED_DIR
AI_DIR = config.ATTACKED_AI_DIR
TRAD_DIR = config.ATTACKED_TRADITIONAL_DIR
ORIG_DIR = config.ORIGINAL_IMAGES_DIR

DAMAGE_CSV = os.path.join(OUT_DIR, "dct_damage_profiles.csv")
CORR_CSV = os.path.join(OUT_DIR, "alpha_profile_correlations.csv")
RANK_CSV = os.path.join(OUT_DIR, "alpha_rank_stability.csv")


# ─── DCT analysis helpers (mirror analyze_dct_stability.py convention) ────────

def _load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)


def _y_ll_dct_blocks(rgb_image):
    """Y channel → single-level Haar DWT → LL subband → list of 8×8 DCT blocks."""
    y = rgb_to_ycbcr(rgb_image)[..., 0]
    LL, _ = pywt.dwt2(y, DEFAULT_WAVELET)
    rows, cols = LL.shape
    blocks = []
    for br in range(0, rows - BLOCK_SIZE + 1, BLOCK_SIZE):
        for bc in range(0, cols - BLOCK_SIZE + 1, BLOCK_SIZE):
            blocks.append(_dct2(LL[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE]))
    return blocks


def compute_damage_profile(wm_path, atk_path):
    """
    Compute per-block |watermarked_DCT - attacked_DCT| and aggregate to 8×8 stats.
    Returns (mean_abs 8×8, std_abs 8×8, median_abs 8×8, mean_rel 8×8).
    """
    wm_img = _load_rgb(wm_path)
    atk_img = _load_rgb(atk_path)
    if wm_img is None or atk_img is None:
        return None

    wm_blocks = _y_ll_dct_blocks(wm_img)
    atk_blocks = _y_ll_dct_blocks(atk_img)
    if len(wm_blocks) != len(atk_blocks):
        return None

    diffs = np.stack([np.abs(w - a) for w, a in zip(wm_blocks, atk_blocks)])  # (N, 8, 8)
    wm_abs = np.stack([np.abs(w) for w in wm_blocks])                          # (N, 8, 8)

    mean_abs = diffs.mean(axis=0)
    std_abs = diffs.std(axis=0, ddof=1) if len(diffs) > 1 else np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    median_abs = np.median(diffs, axis=0)

    # Mean relative change: |diff| / (|watermarked_coeff| + ε)
    rel = diffs / (wm_abs + 1e-6)
    mean_rel = rel.mean(axis=0)

    return mean_abs, std_abs, median_abs, mean_rel


def find_image_stems(alpha):
    """List image stems that have watermarked + both attacked files for given alpha."""
    stems = []
    for f in sorted(os.listdir(WATERMARKED_DIR)):
        if not f.endswith(f"__alpha{alpha}.png"):
            continue
        if "__" in f.replace(f"__alpha{alpha}.png", ""):
            continue   # skip variant-suffixed files (stable_positions, optimized_positions)
        stem = f.replace(f"__alpha{alpha}.png", "")
        esrgan_path = os.path.join(AI_DIR, f"{stem}__alpha{alpha}__real_esrgan.png")
        jpeg_path = os.path.join(TRAD_DIR, f"{stem}__alpha{alpha}__jpeg_compression_q50.png")
        wm_path = os.path.join(WATERMARKED_DIR, f)
        if os.path.exists(esrgan_path) and os.path.exists(jpeg_path):
            stems.append((stem, wm_path, esrgan_path, jpeg_path))
    return stems


# ─── Profile computation ──────────────────────────────────────────────────────

def build_damage_profiles():
    """
    For each (alpha, attack), compute 8×8 mean/std/median/rel damage matrices
    across all 100 images. Returns a DataFrame and a dict of numpy matrices.
    """
    profile_records = []
    profile_matrices = {}   # (alpha, attack) → mean_abs 8×8

    for alpha in ALPHAS:
        stems = find_image_stems(alpha)
        print(f"  α={alpha}: {len(stems)} image pairs")
        if not stems:
            print(f"  [WARN] No files found for α={alpha}. Skipping.")
            continue

        for attack_key, atk_idx in [("real_esrgan", 2), ("jpeg_q50", 3)]:
            all_mean_abs = []
            all_std_abs = []
            all_median_abs = []
            all_mean_rel = []

            for stem, wm_path, esrgan_path, jpeg_path in stems:
                atk_path = esrgan_path if attack_key == "real_esrgan" else jpeg_path
                result = compute_damage_profile(wm_path, atk_path)
                if result is None:
                    continue
                mean_abs, std_abs, median_abs, mean_rel = result
                all_mean_abs.append(mean_abs)
                all_std_abs.append(std_abs)
                all_median_abs.append(median_abs)
                all_mean_rel.append(mean_rel)

            if not all_mean_abs:
                continue

            agg_mean = np.mean(all_mean_abs, axis=0)   # (8, 8)
            agg_std = np.mean(all_std_abs, axis=0)
            agg_median = np.median(all_median_abs, axis=0)
            agg_rel = np.mean(all_mean_rel, axis=0)

            profile_matrices[(alpha, attack_key)] = agg_mean

            for r in range(BLOCK_SIZE):
                for c in range(BLOCK_SIZE):
                    profile_records.append({
                        "alpha": alpha,
                        "attack": attack_key,
                        "coeff_row": r,
                        "coeff_col": c,
                        "mean_abs_change": float(agg_mean[r, c]),
                        "std_abs_change": float(agg_std[r, c]),
                        "median_abs_change": float(agg_median[r, c]),
                        "mean_rel_change": float(agg_rel[r, c]),
                    })

    return pd.DataFrame(profile_records), profile_matrices


# ─── Correlation analysis ─────────────────────────────────────────────────────

def isotropy_r(mat):
    """Pearson r between mat[i,j] and mat[j,i] for all i≠j."""
    ij, ji = [], []
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if i != j:
                ij.append(mat[i, j])
                ji.append(mat[j, i])
    if len(ij) < 3:
        return np.nan
    r, _ = pearsonr(ij, ji)
    return float(r)


def rank_of_position(mat, pos):
    """Rank of mat[pos] among all 64 values (rank 1 = largest damage)."""
    flat = mat.flatten()
    val = mat[pos]
    return int(np.sum(flat >= val))   # number of positions >= this value


def build_correlations(profile_matrices):
    """Per-alpha: Spearman/Pearson correlation between ESRGAN and JPEG damage profiles,
    isotropy for each attack, and standard position tracking."""
    rows = []
    for alpha in ALPHAS:
        esrgan_mat = profile_matrices.get((alpha, "real_esrgan"))
        jpeg_mat = profile_matrices.get((alpha, "jpeg_q50"))
        if esrgan_mat is None or jpeg_mat is None:
            continue

        flat_e = esrgan_mat.flatten()
        flat_j = jpeg_mat.flatten()

        rho, p_rho = spearmanr(flat_e, flat_j)
        r_p, p_r = pearsonr(flat_e, flat_j)

        iso_e = isotropy_r(esrgan_mat)
        iso_j = isotropy_r(jpeg_mat)

        row = {
            "alpha": alpha,
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "pearson_r": float(r_p),
            "pearson_p": float(p_r),
            "esrgan_isotropy_r": iso_e,
            "jpeg_isotropy_r": iso_j,
        }
        for pos in STANDARD_POS:
            tag = f"pos_{pos[0]}_{pos[1]}"
            row[f"damage_esrgan_{tag}"] = float(esrgan_mat[pos])
            row[f"damage_jpeg_{tag}"] = float(jpeg_mat[pos])
            row[f"rank_esrgan_{tag}"] = rank_of_position(esrgan_mat, pos)
            row[f"rank_jpeg_{tag}"] = rank_of_position(jpeg_mat, pos)

        rows.append(row)
    return pd.DataFrame(rows)


def top_k_positions(mat, k=10):
    """Return the k positions with highest mean damage (as set of (row,col) tuples)."""
    flat = mat.flatten()
    idx = np.argsort(flat)[::-1][:k]
    return set(zip(idx // BLOCK_SIZE, idx % BLOCK_SIZE))


def build_rank_stability(profile_matrices):
    """Cross-alpha rank stability for each attack."""
    rows = []
    for attack in ["real_esrgan", "jpeg_q50"]:
        mats = {alpha: profile_matrices.get((alpha, attack)) for alpha in ALPHAS}
        for i, a1 in enumerate(ALPHAS):
            for a2 in ALPHAS[i+1:]:
                if mats[a1] is None or mats[a2] is None:
                    continue
                flat1 = mats[a1].flatten()
                flat2 = mats[a2].flatten()
                rho, p = spearmanr(flat1, flat2)
                top1 = top_k_positions(mats[a1], k=10)
                top2 = top_k_positions(mats[a2], k=10)
                jaccard = len(top1 & top2) / len(top1 | top2) if top1 | top2 else np.nan
                rows.append({
                    "attack": attack,
                    "alpha1": a1, "alpha2": a2,
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "jaccard_top10": float(jaccard),
                })
    return pd.DataFrame(rows)


# ─── Figures ─────────────────────────────────────────────────────────────────

def fig_heatmaps(profile_matrices, corr_df):
    """4 rows × 2 cols: ESRGAN and JPEG heatmaps side by side for each alpha."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 22))

    # Shared color scales within each attack
    all_e = [profile_matrices.get((a, "real_esrgan")) for a in ALPHAS
             if profile_matrices.get((a, "real_esrgan")) is not None]
    all_j = [profile_matrices.get((a, "jpeg_q50")) for a in ALPHAS
             if profile_matrices.get((a, "jpeg_q50")) is not None]
    vmax_e = max(m.max() for m in all_e) if all_e else 1
    vmax_j = max(m.max() for m in all_j) if all_j else 1

    for row_idx, alpha in enumerate(ALPHAS):
        e_mat = profile_matrices.get((alpha, "real_esrgan"))
        j_mat = profile_matrices.get((alpha, "jpeg_q50"))

        corr_row = corr_df[corr_df["alpha"] == alpha]
        rho = float(corr_row["spearman_rho"].iloc[0]) if not corr_row.empty else np.nan

        for col_idx, (mat, vmax, attack_label, cmap) in enumerate([
            (e_mat, vmax_e, "Real-ESRGAN", "YlOrRd"),
            (j_mat, vmax_j, "JPEG Q50", "Blues"),
        ]):
            ax = axes[row_idx, col_idx]
            if mat is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            plt.colorbar(im, ax=ax, label="mean |Δcoeff|", shrink=0.85)

            # Mark standard embedding positions
            for pos in STANDARD_POS:
                ax.add_patch(plt.Rectangle(
                    (pos[1] - 0.5, pos[0] - 0.5), 1, 1,
                    fill=False, edgecolor="lime", linewidth=2.5,
                ))
                ax.text(pos[1], pos[0], f"P{STANDARD_POS.index(pos)+1}",
                        ha="center", va="center", fontsize=7,
                        color="lime", fontweight="bold")

            ax.set_xticks(range(BLOCK_SIZE))
            ax.set_yticks(range(BLOCK_SIZE))
            ax.set_xlabel("DCT col freq", fontsize=8)
            ax.set_ylabel("DCT row freq", fontsize=8)
            rho_str = f"  (ρ={rho:.3f})" if col_idx == 1 and not np.isnan(rho) else ""
            ax.set_title(f"α={alpha} — {attack_label}{rho_str}", fontsize=10)

    fig.suptitle(
        "Experiment G — DCT Damage Profiles: Real-ESRGAN vs JPEG Q50 across α\n"
        "Green boxes = standard embedding positions (4,1) and (1,4)",
        fontsize=13,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "exp_G_heatmaps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def fig_attack_correlation_vs_alpha(corr_df):
    """ESRGAN/JPEG Spearman ρ vs alpha."""
    fig, ax = plt.subplots(figsize=(8, 5))
    alphas = corr_df["alpha"].values
    rhos = corr_df["spearman_rho"].values
    ax.plot(alphas, rhos, "o-", color="#e74c3c", linewidth=2.5, markersize=10, zorder=5)

    for a, r in zip(alphas, rhos):
        ax.annotate(f"ρ={r:.3f}", (a, r), xytext=(6, 6),
                    textcoords="offset points", fontsize=10)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(-0.9, color="orange", linestyle=":", linewidth=1.5,
               label="Reference: prior α=0.1 finding (ρ=−0.907)")
    ax.set_xlabel("Embedding strength α", fontsize=12)
    ax.set_ylabel("Spearman ρ (ESRGAN vs JPEG damage profiles)", fontsize=12)
    ax.set_title(
        "Experiment G — ESRGAN/JPEG Damage Anti-Complementarity vs α\n"
        "Negative ρ = ESRGAN damages positions JPEG does not (and vice versa)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.35)
    ax.set_ylim(-1.1, 0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "exp_G_attack_correlation_vs_alpha.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def fig_profile_stability(rank_df):
    """Heatmaps of profile correlations across alpha values for each attack."""
    attacks = ["real_esrgan", "jpeg_q50"]
    attack_labels = {"real_esrgan": "Real-ESRGAN", "jpeg_q50": "JPEG Q50"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, attack in zip(axes, attacks):
        mat = np.full((len(ALPHAS), len(ALPHAS)), np.nan)
        sub = rank_df[rank_df["attack"] == attack]
        for _, row in sub.iterrows():
            i = ALPHAS.index(row["alpha1"])
            j = ALPHAS.index(row["alpha2"])
            mat[i, j] = row["spearman_rho"]
            mat[j, i] = row["spearman_rho"]
        np.fill_diagonal(mat, 1.0)

        im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.85)
        ax.set_xticks(range(len(ALPHAS)))
        ax.set_yticks(range(len(ALPHAS)))
        ax.set_xticklabels([f"α={a}" for a in ALPHAS], fontsize=9)
        ax.set_yticklabels([f"α={a}" for a in ALPHAS], fontsize=9)
        ax.set_title(f"{attack_labels[attack]}\nProfile Rank Stability (Spearman ρ)", fontsize=11)

        for i in range(len(ALPHAS)):
            for j in range(len(ALPHAS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=9, color="black")

    fig.suptitle(
        "Experiment G — Cross-Alpha Profile Stability\n"
        "ρ close to 1 = same positions damaged regardless of α",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "exp_G_profile_stability.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def fig_standard_position_damage(profile_matrices):
    """Damage at (4,1) and (1,4) across alpha for ESRGAN and JPEG."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pos in zip(axes, STANDARD_POS):
        alphas_e, dmg_e = [], []
        alphas_j, dmg_j = [], []
        for alpha in ALPHAS:
            e_mat = profile_matrices.get((alpha, "real_esrgan"))
            j_mat = profile_matrices.get((alpha, "jpeg_q50"))
            if e_mat is not None:
                alphas_e.append(alpha)
                dmg_e.append(e_mat[pos])
            if j_mat is not None:
                alphas_j.append(alpha)
                dmg_j.append(j_mat[pos])

        ax.plot(alphas_e, dmg_e, "o-", color="#e74c3c", linewidth=2.5,
                markersize=9, label="Real-ESRGAN")
        ax.plot(alphas_j, dmg_j, "s--", color="#3498db", linewidth=2.5,
                markersize=9, label="JPEG Q50")

        for a, d in zip(alphas_e, dmg_e):
            ax.annotate(f"{d:.2f}", (a, d), xytext=(5, 5),
                        textcoords="offset points", fontsize=9, color="#e74c3c")
        for a, d in zip(alphas_j, dmg_j):
            ax.annotate(f"{d:.2f}", (a, d), xytext=(5, -14),
                        textcoords="offset points", fontsize=9, color="#3498db")

        ax.set_xlabel("Embedding strength α", fontsize=12)
        ax.set_ylabel("Mean |Δ DCT coeff|", fontsize=12)
        ax.set_title(f"Damage at standard position ({pos[0]},{pos[1]})\nvs α", fontsize=11)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 0.35)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Experiment G — Standard Embedding Position Damage vs α\n"
        "Standard pair: (4,1) and (1,4)",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "exp_G_standard_position_damage.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(corr_df, rank_df, profile_matrices):
    lines = []
    def emit(*args):
        lines.append(" ".join(str(a) for a in args))

    emit("# Experiment G: Alpha Stability of DCT Damage Profiles — Report")
    emit()
    emit(f"Date: 2026-07-13  |  Standard pair (4,1)/(1,4)  |  n=100 images per alpha")
    emit()
    emit("## 1. Overview")
    emit()
    emit("This experiment tests whether the ESRGAN/JPEG DCT damage anti-complementarity")
    emit("(Spearman ρ = −0.907 previously measured at α=0.10) holds across")
    emit("α ∈ {0.05, 0.10, 0.20, 0.30}.")
    emit()
    emit("All analysis uses pre-computed disk files from `results/attacked/`.")
    emit("No new ESRGAN inference is required.")
    emit()

    emit("## 2. ESRGAN/JPEG Correlation by Alpha")
    emit()
    emit("| α    | Spearman ρ | p-value    | Pearson r | ESRGAN iso r | JPEG iso r |")
    emit("|------|------------|------------|-----------|--------------|------------|")
    for _, row in corr_df.iterrows():
        emit(
            f"| {row['alpha']:.2f} | {row['spearman_rho']:+.4f}     | "
            f"{row['spearman_p']:.3e} | {row['pearson_r']:+.4f}    | "
            f"{row['esrgan_isotropy_r']:.4f}       | {row['jpeg_isotropy_r']:.4f}      |"
        )
    emit()

    if not corr_df.empty:
        rhos = corr_df["spearman_rho"].values
        rho_range = float(np.max(rhos) - np.min(rhos))
        all_negative = all(r < 0 for r in rhos)
        emit("### Stability Assessment")
        emit()
        if all_negative:
            emit(f"All ρ values are negative (range: {rhos.min():.3f} to {rhos.max():.3f}).")
            emit(f"ρ range across α: {rho_range:.4f}")
            if rho_range < 0.05:
                emit("**Conclusion: The anti-complementarity pattern is highly stable across α.**")
                emit("The relative perturbation relationship between ESRGAN and JPEG is not")
                emit("meaningfully dependent on embedding strength within this range.")
            elif rho_range < 0.15:
                emit("**Conclusion: The anti-complementarity pattern is moderately stable across α.**")
                emit("Some alpha-dependence exists but the sign and general magnitude are preserved.")
            else:
                emit("**Conclusion: The anti-complementarity is alpha-dependent.**")
                emit("The ρ values vary substantially across α; the trade-off cannot be treated")
                emit("as alpha-invariant without qualification.")
        else:
            emit("**ρ values are not consistently negative — anti-complementarity is NOT stable.**")
            emit(f"Values: {[f'{r:.3f}' for r in rhos]}")

    emit()
    emit("## 3. Standard Position Damage")
    emit()
    emit("| α    | ESRGAN (4,1) | ESRGAN (1,4) | Rank(4,1) | Rank(1,4) | JPEG (4,1) | JPEG (1,4) |")
    emit("|------|--------------|--------------|-----------|-----------|------------|------------|")
    for _, row in corr_df.iterrows():
        alpha = row["alpha"]
        emit(
            f"| {alpha:.2f} | "
            f"{row['damage_esrgan_pos_4_1']:.3f}        | "
            f"{row['damage_esrgan_pos_1_4']:.3f}        | "
            f"{row['rank_esrgan_pos_4_1']}         | "
            f"{row['rank_esrgan_pos_1_4']}         | "
            f"{row['damage_jpeg_pos_4_1']:.3f}       | "
            f"{row['damage_jpeg_pos_1_4']:.3f}       |"
        )
    emit()
    emit("(Rank 1 = most damaged position among all 64 DCT coefficients.)")
    emit()

    emit("## 4. Cross-Alpha Rank Stability")
    emit()
    for attack in ["real_esrgan", "jpeg_q50"]:
        attack_label = "Real-ESRGAN" if attack == "real_esrgan" else "JPEG Q50"
        emit(f"### {attack_label}")
        emit()
        emit("| α1   | α2   | Spearman ρ | p-value    | Jaccard top-10 |")
        emit("|------|------|------------|------------|----------------|")
        sub = rank_df[rank_df["attack"] == attack]
        for _, row in sub.iterrows():
            emit(
                f"| {row['alpha1']:.2f} | {row['alpha2']:.2f} | "
                f"{row['spearman_rho']:+.4f}     | {row['spearman_p']:.3e} | "
                f"{row['jaccard_top10']:.3f}           |"
            )
        emit()

    emit("## 5. Primary Question — Answer")
    emit()
    emit("**Does the Real-ESRGAN/JPEG anti-complementarity remain stable across α?**")
    emit()
    if not corr_df.empty:
        rhos = corr_df["spearman_rho"].values
        all_neg = all(r < 0 for r in rhos)
        rho_range = float(np.max(rhos) - np.min(rhos))

        if all_neg and rho_range < 0.10:
            emit("**YES — the anti-complementarity is robust to embedding strength.**")
            emit()
            emit(f"Spearman ρ ranges from {rhos.min():.3f} to {rhos.max():.3f} across α.")
            emit("The sign is consistently negative, and the absolute magnitude shows")
            emit(f"only {rho_range:.4f} variation. The relative DCT perturbation pattern")
            emit("is determined by the attack (ESRGAN vs JPEG) rather than the embedding strength.")
            emit()
            emit("This supports the paper claim that the ESRGAN/JPEG anti-complementarity")
            emit("is a property of the attacks themselves, not an artifact of a specific α.")
        elif all_neg:
            emit("**PARTIALLY YES — anti-complementarity holds but magnitude varies with α.**")
            emit()
            emit(f"All ρ values are negative (range {rhos.min():.3f} to {rhos.max():.3f}),")
            emit(f"indicating consistent anti-complementarity, but ρ range = {rho_range:.4f}.")
            emit("Paper claims should acknowledge alpha-dependence in magnitude.")
        else:
            emit("**NO — the anti-complementarity is alpha-dependent.**")
            emit()
            emit(f"ρ values: {[f'{r:.3f}' for r in rhos]} — some values are non-negative.")
            emit("The claim of robust anti-complementarity cannot be made unconditionally.")

    emit()
    emit("## 6. Interpretation Notes")
    emit()
    emit("- Results are correlational. No causal mechanism is claimed from these correlations alone.")
    emit("- 'Stable' means: relative ranking of coefficient positions is preserved,")
    emit("  not that absolute damage magnitudes are identical across α.")
    emit("- Isotropy (how symmetric ESRGAN damage is) is reported per α but not claimed")
    emit("  as 'intrinsic' to ESRGAN without further evidence.")
    emit()
    emit("---")
    emit("*Report generated by experiments/exp_G_alpha_damage_ablation/run_exp_G.py*")

    report_path = os.path.join(OUT_DIR, "EXP_G_REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report: {report_path}")
    return report_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXPERIMENT G: ALPHA STABILITY OF DCT DAMAGE PROFILES")
    print("=" * 70)

    if (os.path.exists(DAMAGE_CSV) and
            os.path.exists(CORR_CSV) and
            os.path.exists(RANK_CSV)):
        print("[cache] Loading existing CSV outputs...")
        damage_df = pd.read_csv(DAMAGE_CSV)
        corr_df = pd.read_csv(CORR_CSV)
        rank_df = pd.read_csv(RANK_CSV)
        # Rebuild profile_matrices from damage_df
        profile_matrices = {}
        for (alpha, attack), grp in damage_df.groupby(["alpha", "attack"]):
            mat = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
            for _, row in grp.iterrows():
                mat[int(row["coeff_row"]), int(row["coeff_col"])] = row["mean_abs_change"]
            profile_matrices[(alpha, attack)] = mat
    else:
        print("\n[1/4] Computing DCT damage profiles across alpha values...")
        damage_df, profile_matrices = build_damage_profiles()
        damage_df.to_csv(DAMAGE_CSV, index=False)
        print(f"  Saved {DAMAGE_CSV}  ({len(damage_df)} rows)")

        print("\n[2/4] Computing per-alpha ESRGAN/JPEG correlations...")
        corr_df = build_correlations(profile_matrices)
        corr_df.to_csv(CORR_CSV, index=False)
        print(f"  Saved {CORR_CSV}")

        print("\n[3/4] Computing cross-alpha rank stability...")
        rank_df = build_rank_stability(profile_matrices)
        rank_df.to_csv(RANK_CSV, index=False)
        print(f"  Saved {RANK_CSV}")

    print("\nGenerating figures...")
    fig_heatmaps(profile_matrices, corr_df)
    fig_attack_correlation_vs_alpha(corr_df)
    fig_profile_stability(rank_df)
    fig_standard_position_damage(profile_matrices)

    print("\nWriting report...")
    write_report(corr_df, rank_df, profile_matrices)

    print("\n" + "=" * 70)
    print("Done. Output in:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
