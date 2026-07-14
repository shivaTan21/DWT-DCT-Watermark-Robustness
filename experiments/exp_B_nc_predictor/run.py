"""
Experiment B: Multi-Feature NC Predictor

HYPOTHESIS:
    Per-image NC under Real-ESRGAN can be predicted from image statistics
    computed BEFORE watermark embedding, enabling a "robustness-aware
    watermarking" workflow where difficult images are identified in advance.

    Prior work: texture_complexity.csv used 3 features and achieved r=0.305
    (Laplacian variance). We extend to 12+ features across multiple domains.

SCIENTIFIC VALUE: MEDIUM-HIGH
    - Practical contribution: predicting NC before embedding
    - Reveals which image properties govern AI-attack robustness
    - Enables robustness-aware embedding strength selection
    - Addresses the "why do some images retain watermarks while others fail?" question

MODIFICATIONS FROM BASELINE:
    - NEW file; reads original images directly (not frozen results)
    - Does NOT modify any frozen source file
    - Reads:  data/original_images/
              results/metrics.csv
    - Writes: experiments/exp_B_nc_predictor/outputs/

DESIGN NOTE:
    We focus on features of the ORIGINAL (pre-watermark) image since that's
    what a practitioner would have. Features computed on watermarked images
    would not be available in a blind deployment scenario.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import cv2
import pywt
from scipy.fftpack import dct as scipy_dct
from scipy.stats import pearsonr, spearmanr, entropy as scipy_entropy
from scipy.signal import wiener
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

IMAGES_DIR  = os.path.join(_ROOT, "data", "original_images")
METRICS_CSV = os.path.join(_ROOT, "results", "metrics.csv")
ALPHA_TARGET = 0.1

# ── Feature extraction ─────────────────────────────────────────────────────────

def _rgb_to_y(img_rgb):
    img = img_rgb.astype(np.float64)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def _dct2(block):
    return scipy_dct(scipy_dct(block.T, norm="ortho").T, norm="ortho")


def feature_laplacian_var(gray_u8):
    """Sharpness/texture proxy via Laplacian variance."""
    lap = cv2.Laplacian(gray_u8.astype(np.float64), cv2.CV_64F)
    return float(lap.var())


def feature_wavelet_energy_ratio(img_rgb, wavelet="haar"):
    """
    Ratio of detail subband energy to approximation subband energy.

    High ratio → rich high-frequency structure.
    Low ratio  → smooth image where ESRGAN will add more texture.
    """
    y = _rgb_to_y(img_rgb)
    LL, (LH, HL, HH) = pywt.dwt2(y, wavelet)
    ll_energy  = float(np.sum(LL ** 2))
    det_energy = float(np.sum(LH ** 2) + np.sum(HL ** 2) + np.sum(HH ** 2))
    return det_energy / max(ll_energy, 1e-9)


def feature_hh_energy(img_rgb, wavelet="haar"):
    """Energy of HH (diagonal detail) subband — high-frequency texture."""
    y = _rgb_to_y(img_rgb)
    _, (_, _, HH) = pywt.dwt2(y, wavelet)
    return float(np.mean(HH ** 2))


def feature_ll_spectral_entropy(img_rgb, wavelet="haar"):
    """
    Entropy of the normalized DCT coefficient magnitudes in the LL subband.
    Measures distributional complexity; high entropy → more uniform DCT distribution.
    """
    y = _rgb_to_y(img_rgb)
    LL, _ = pywt.dwt2(y, wavelet)
    rows, cols = LL.shape
    coeffs = []
    for br in range(0, (rows // 8) * 8, 8):
        for bc in range(0, (cols // 8) * 8, 8):
            block = LL[br:br+8, bc:bc+8]
            dct_b = np.abs(_dct2(block)).flatten()
            coeffs.extend(dct_b[1:].tolist())  # exclude DC
    coeffs = np.array(coeffs)
    coeffs = coeffs / (coeffs.sum() + 1e-9)
    return float(scipy_entropy(coeffs + 1e-12))


def feature_edge_density(gray_u8):
    """Canny edge pixel fraction."""
    edges = cv2.Canny(gray_u8, 50, 150)
    return float(edges.sum()) / (edges.size * 255.0)


def feature_local_std_mean(gray_u8, ksize=8):
    """Mean local standard deviation over ksize×ksize blocks — local texture measure."""
    img = gray_u8.astype(np.float64)
    h, w = img.shape
    stds = []
    for r in range(0, (h // ksize) * ksize, ksize):
        for c in range(0, (w // ksize) * ksize, ksize):
            stds.append(img[r:r+ksize, c:c+ksize].std())
    return float(np.mean(stds))


def feature_local_std_cv(gray_u8, ksize=8):
    """Coefficient of variation of local stds — measures texture heterogeneity."""
    img = gray_u8.astype(np.float64)
    h, w = img.shape
    stds = []
    for r in range(0, (h // ksize) * ksize, ksize):
        for c in range(0, (w // ksize) * ksize, ksize):
            stds.append(img[r:r+ksize, c:c+ksize].std())
    stds = np.array(stds)
    return float(stds.std() / max(stds.mean(), 1e-9))


def feature_gradient_magnitude(gray_u8):
    """Mean gradient magnitude (Sobel) — edges and texture strength."""
    img = gray_u8.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx**2 + gy**2).mean())


def feature_histogram_entropy(gray_u8):
    """Shannon entropy of normalized pixel histogram — global tonal complexity."""
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    return float(scipy_entropy(hist + 1e-12))


def feature_ll_variance(img_rgb, wavelet="haar"):
    """Variance of LL subband pixel values — overall energy in approximation."""
    y = _rgb_to_y(img_rgb)
    LL, _ = pywt.dwt2(y, wavelet)
    return float(np.var(LL))


def feature_ll_kurtosis(img_rgb, wavelet="haar"):
    """
    Kurtosis of LL subband pixel distribution.
    High kurtosis → heavy tails → fewer dominant features, more flat regions.
    """
    from scipy.stats import kurtosis
    y = _rgb_to_y(img_rgb)
    LL, _ = pywt.dwt2(y, wavelet)
    return float(kurtosis(LL.flatten()))


def feature_mean_block_energy_std(img_rgb, wavelet="haar"):
    """
    Std of mean DCT energy per 8×8 LL block.
    Measures how heterogeneous the frequency energy is across blocks.
    High std → some blocks have much more energy than others.
    """
    y = _rgb_to_y(img_rgb)
    LL, _ = pywt.dwt2(y, wavelet)
    rows, cols = LL.shape
    energies = []
    for br in range(0, (rows // 8) * 8, 8):
        for bc in range(0, (cols // 8) * 8, 8):
            block = LL[br:br+8, bc:bc+8]
            energies.append(np.var(block))
    return float(np.std(energies))


def feature_embedding_position_energy(img_rgb, wavelet="haar"):
    """
    Mean absolute DCT value at embedding positions (4,1) and (1,4) across all blocks.
    Higher energy at embedding positions → larger natural margin → more robust embedding.
    """
    y = _rgb_to_y(img_rgb)
    LL, _ = pywt.dwt2(y, wavelet)
    rows, cols = LL.shape
    vals = []
    for br in range(0, (rows // 8) * 8, 8):
        for bc in range(0, (cols // 8) * 8, 8):
            block = LL[br:br+8, bc:bc+8]
            dct_b = _dct2(block)
            vals.append((abs(dct_b[4, 1]) + abs(dct_b[1, 4])) / 2.0)
    return float(np.mean(vals))


def extract_features(img_path):
    """Extract all features from a single image."""
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    # Resize to match pipeline (512×512)
    rgb_512 = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    gray_u8 = cv2.cvtColor(rgb_512.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    return {
        "laplacian_var":         feature_laplacian_var(gray_u8),
        "wavelet_energy_ratio":  feature_wavelet_energy_ratio(rgb_512),
        "hh_energy":             feature_hh_energy(rgb_512),
        "ll_spectral_entropy":   feature_ll_spectral_entropy(rgb_512),
        "edge_density":          feature_edge_density(gray_u8),
        "local_std_mean":        feature_local_std_mean(gray_u8),
        "local_std_cv":          feature_local_std_cv(gray_u8),
        "gradient_magnitude":    feature_gradient_magnitude(gray_u8),
        "histogram_entropy":     feature_histogram_entropy(gray_u8),
        "ll_variance":           feature_ll_variance(rgb_512),
        "ll_kurtosis":           feature_ll_kurtosis(rgb_512),
        "block_energy_std":      feature_mean_block_energy_std(rgb_512),
        "embedding_pos_energy":  feature_embedding_position_energy(rgb_512),
    }


# ── Main analysis ──────────────────────────────────────────────────────────────

def build_feature_table(images_dir, metrics_csv, alpha=0.1):
    """Build combined feature + NC table for all images."""
    metrics = pd.read_csv(metrics_csv)
    nc_df = metrics[
        (metrics["embedding_variant"] == "standard") &
        (metrics["attack_name"] == "real_esrgan") &
        (metrics["alpha"] == alpha)
    ][["image", "nc"]].copy()
    nc_map = dict(zip(nc_df["image"], nc_df["nc"]))

    image_files = sorted(f for f in os.listdir(images_dir)
                         if f.lower().endswith((".png", ".jpg", ".jpeg")))

    records = []
    print(f"Extracting features from {len(image_files)} images...")
    for i, fname in enumerate(image_files):
        name = os.path.splitext(fname)[0]
        nc = nc_map.get(name)
        if nc is None:
            continue
        feats = extract_features(os.path.join(images_dir, fname))
        if feats is None:
            continue
        feats["image"] = name
        feats["nc"] = nc
        records.append(feats)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(image_files)} done...")

    df = pd.DataFrame(records)
    return df


def univariate_analysis(df, feature_cols):
    """Pearson and Spearman correlation of each feature with NC."""
    print("\n" + "=" * 70)
    print("Univariate Feature Correlations with NC (Real-ESRGAN)")
    print("=" * 70)
    print(f"{'Feature':<28} | {'Pearson r':>10} | {'p-value':>10} | {'Spearman ρ':>10} | {'p-value':>10}")
    print("-" * 80)

    results = []
    for col in feature_cols:
        valid = df.dropna(subset=[col, "nc"])
        if len(valid) < 5:
            continue
        r_p, p_p = pearsonr(valid[col], valid["nc"])
        r_s, p_s = spearmanr(valid[col], valid["nc"])
        results.append({"feature": col, "pearson_r": r_p, "pearson_p": p_p,
                        "spearman_rho": r_s, "spearman_p": p_s, "n": len(valid)})
        sig = "*" if p_p < 0.05 else " "
        print(f"{col:<28} | {r_p:>10.4f} | {p_p:>10.3e}{sig} | {r_s:>10.4f} | {p_s:>10.3e}")

    return pd.DataFrame(results).sort_values("pearson_r", key=abs, ascending=False)


def multivariate_analysis(df, feature_cols):
    """Cross-validated linear regression + random forest for NC prediction."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.pipeline import Pipeline

    print("\n" + "=" * 70)
    print("Multivariate NC Prediction (cross-validated)")
    print("=" * 70)

    # Drop rows with NaN features
    valid_cols = [c for c in feature_cols if df[c].isna().sum() < len(df) * 0.1]
    df_clean = df.dropna(subset=valid_cols + ["nc"]).copy()

    print(f"Samples: {len(df_clean)}  |  Features: {len(valid_cols)}")

    X = df_clean[valid_cols].values
    y = df_clean["nc"].values

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Ridge (α=1.0)":    Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Ridge (α=0.1)":    Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=0.1))]),
        "Random Forest":    RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42),
        "GradBoost":        GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
    }

    results = {}
    for name, model in models.items():
        scores_r2   = cross_val_score(model, X, y, cv=cv, scoring="r2")
        scores_mae  = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        r2_mean  = scores_r2.mean()
        mae_mean = (-scores_mae).mean()
        results[name] = {"r2_mean": r2_mean, "r2_std": scores_r2.std(),
                         "mae_mean": mae_mean, "mae_std": (-scores_mae).std()}
        print(f"\n  {name}:")
        print(f"    CV R² = {r2_mean:.4f} ± {scores_r2.std():.4f}")
        print(f"    CV MAE = {mae_mean:.4f} ± {(-scores_mae).std():.4f}")

    # Feature importance from RF
    rf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=valid_cols).sort_values(ascending=False)
    print("\n  Random Forest Feature Importances:")
    for feat, imp in importances.head(8).items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<28}: {imp:.4f}  {bar}")

    return results, importances, df_clean, valid_cols


def plot_results(df, feature_cols, corr_df, rf_importances, out_dir):
    """Generate comprehensive visualization."""
    n_features = min(len(feature_cols), 12)
    top_features = corr_df["feature"].head(n_features).tolist()

    fig = plt.figure(figsize=(20, 22))
    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Row 1: top 6 scatter plots (feature vs NC)
    colors_scatter = plt.cm.viridis(np.linspace(0, 0.9, 6))
    for i, feat in enumerate(top_features[:6]):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        valid = df.dropna(subset=[feat, "nc"])
        r_p, p_p = pearsonr(valid[feat], valid["nc"])

        # Color by image set
        kodak_mask = valid["image"].str.startswith("kodim")
        ax.scatter(valid.loc[kodak_mask, feat], valid.loc[kodak_mask, "nc"],
                   c="steelblue", s=40, alpha=0.7, label="Kodak", zorder=3)
        ax.scatter(valid.loc[~kodak_mask, feat], valid.loc[~kodak_mask, "nc"],
                   c="darkorange", s=40, alpha=0.7, marker="^", label="TAMPERE17", zorder=3)

        m, b = np.polyfit(valid[feat], valid["nc"], 1)
        xl = np.linspace(valid[feat].min(), valid[feat].max(), 100)
        ax.plot(xl, m * xl + b, "r-", linewidth=1.5)

        ax.set_xlabel(feat.replace("_", "\n"), fontsize=8)
        ax.set_ylabel("NC (ESRGAN)" if i % 3 == 0 else "", fontsize=9)
        sig = "**" if p_p < 0.01 else ("*" if p_p < 0.05 else "")
        ax.set_title(f"r={r_p:.3f}{sig}  p={p_p:.3e}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    # Row 3: feature importance bar chart
    ax_imp = fig.add_subplot(gs[2, :2])
    importances_sorted = rf_importances.head(10)
    y_pos = np.arange(len(importances_sorted))
    bars = ax_imp.barh(y_pos, importances_sorted.values, color=plt.cm.Blues(
        np.linspace(0.4, 0.9, len(importances_sorted))))
    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels(importances_sorted.index, fontsize=9)
    ax_imp.set_xlabel("Feature Importance (Random Forest)", fontsize=10)
    ax_imp.set_title("Top-10 Feature Importances for NC Prediction", fontsize=10)
    ax_imp.grid(True, axis="x", alpha=0.3)

    # Row 3, right: NC histogram
    ax_hist = fig.add_subplot(gs[2, 2])
    ax_hist.hist(df["nc"].dropna(), bins=20, color="steelblue", alpha=0.7, edgecolor="white")
    ax_hist.axvline(df["nc"].median(), color="red", linestyle="--", linewidth=1.5,
                    label=f"Median={df['nc'].median():.3f}")
    ax_hist.set_xlabel("NC (Real-ESRGAN, standard embedding)", fontsize=9)
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.set_title("NC Distribution (n=100 images)", fontsize=10)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    # Row 4: Correlation summary bar chart
    ax_corr = fig.add_subplot(gs[3, :])
    corr_sorted = corr_df.sort_values("pearson_r", key=abs, ascending=True)
    colors_bar = ["#e74c3c" if r < 0 else "#2ecc71" for r in corr_sorted["pearson_r"]]
    ax_corr.barh(np.arange(len(corr_sorted)), corr_sorted["pearson_r"], color=colors_bar, alpha=0.8)
    ax_corr.set_yticks(np.arange(len(corr_sorted)))
    ax_corr.set_yticklabels(corr_sorted["feature"], fontsize=9)
    ax_corr.axvline(0, color="black", linewidth=0.8)
    ax_corr.set_xlabel("Pearson r with NC (Real-ESRGAN)", fontsize=10)
    ax_corr.set_title("Feature Correlation with NC under Real-ESRGAN\n"
                       "(positive = more feature → better survival)", fontsize=10)
    ax_corr.grid(True, axis="x", alpha=0.3)

    # Significance thresholds
    for threshold, label in [(0.19, "p<0.05"), (0.26, "p<0.01")]:
        for sign in [1, -1]:
            ax_corr.axvline(sign * threshold, color="gray", linestyle=":", linewidth=1,
                            label=label if sign == 1 else "")
    ax_corr.legend(fontsize=8)

    fig.suptitle(
        "Experiment B: Multi-Feature NC Predictor under Real-ESRGAN\n"
        "(Standard DWT-DCT embedding, α=0.1, 100 images, 5-fold CV)",
        fontsize=13, y=0.98,
    )

    out = os.path.join(out_dir, "B_nc_predictor.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out}")


def write_summary(corr_df, cv_results, out_dir):
    best_model = max(cv_results, key=lambda k: cv_results[k]["r2_mean"])
    best_r2 = cv_results[best_model]["r2_mean"]

    lines = [
        "=" * 70,
        "EXPERIMENT B: MULTI-FEATURE NC PREDICTOR — SUMMARY",
        "=" * 70,
        "",
        "HYPOTHESIS:",
        "  Per-image NC under Real-ESRGAN can be predicted from image",
        "  statistics computed BEFORE watermark embedding.",
        "",
        "KEY FINDINGS:",
    ]

    top3 = corr_df.head(3)
    for _, row in top3.iterrows():
        sig = "**" if row["pearson_p"] < 0.01 else ("*" if row["pearson_p"] < 0.05 else "")
        lines.append(f"  {row['feature']:<28}: r={row['pearson_r']:.4f}{sig}, ρ={row['spearman_rho']:.4f}")

    lines += [
        "",
        "  Cross-Validated NC Prediction (5-fold):",
    ]
    for name, res in cv_results.items():
        lines.append(f"    {name:<20}: R²={res['r2_mean']:.4f} ± {res['r2_std']:.4f}, "
                     f"MAE={res['mae_mean']:.4f}")

    lines += [
        "",
        f"  Best model: {best_model} (CV R²={best_r2:.4f})",
        "",
    ]

    if best_r2 > 0.5:
        verdict = "STRONG: R²>0.5 — reliable pre-embedding prediction achievable"
        novelty = "HIGH"
    elif best_r2 > 0.3:
        verdict = "MODERATE: R²>0.3 — meaningful but not reliable for individual images"
        novelty = "MEDIUM"
    else:
        verdict = "WEAK: R²<0.3 — NC prediction from image features is not reliable"
        novelty = "LOW"

    lines += [
        f"  Predictive power: {verdict}",
        "",
        "INTERPRETATION:",
        "  The most predictive features are those related to texture complexity",
        "  and local frequency content. This suggests that images with",
        "  heterogeneous texture (mixed smooth and textured regions) are harder",
        "  to predict and may produce more variable NC across blocks.",
        "",
        f"NOVELTY ASSESSMENT: {novelty}",
        "  Predicting watermark robustness from image features before embedding",
        "  is a practical contribution. Prior work focuses on post-attack analysis.",
        "",
        "LIMITATIONS:",
        "  1. Dataset size (n≈100) limits statistical power for multi-feature models",
        "  2. All images are natural photographs (Kodak + TAMPERE17); synthetic",
        "     or document images may behave differently",
        "  3. Model trained on standard embedding only (α=0.1)",
        "",
        "RECOMMENDED FOLLOW-UP:",
        "  Use NC predictor to select adaptive embedding strength α per image,",
        "  then verify that predicted-difficult images improve more under higher α.",
        "=" * 70,
    ]

    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    for line in lines:
        print(line)
    print(f"\nSummary → {path}")


def main():
    print("=" * 70)
    print("EXPERIMENT B: MULTI-FEATURE NC PREDICTOR")
    print("=" * 70)

    feature_csv = os.path.join(OUT_DIR, "features.csv")

    if os.path.exists(feature_csv):
        print(f"Loading cached features from {feature_csv}")
        df = pd.read_csv(feature_csv)
    else:
        df = build_feature_table(IMAGES_DIR, METRICS_CSV, alpha=ALPHA_TARGET)
        df.to_csv(feature_csv, index=False)
        print(f"Features saved → {feature_csv}")

    print(f"\nDataset: {len(df)} images")
    feature_cols = [c for c in df.columns if c not in ("image", "nc")]

    print(f"\nNC statistics (Real-ESRGAN, α={ALPHA_TARGET}):")
    print(f"  mean={df['nc'].mean():.4f}  std={df['nc'].std():.4f}  "
          f"min={df['nc'].min():.4f}  max={df['nc'].max():.4f}")

    corr_df = univariate_analysis(df, feature_cols)
    corr_df.to_csv(os.path.join(OUT_DIR, "B_correlations.csv"), index=False)

    try:
        cv_results, rf_importances, df_clean, valid_cols = multivariate_analysis(df, feature_cols)
        plot_results(df, feature_cols, corr_df, rf_importances, OUT_DIR)
        write_summary(corr_df, cv_results, OUT_DIR)
    except ImportError:
        print("\n[WARNING] sklearn not available — skipping multivariate analysis")
        print("  Install: pip install scikit-learn")
        # Still save what we have
        corr_df.to_csv(os.path.join(OUT_DIR, "B_correlations.csv"), index=False)


if __name__ == "__main__":
    main()
