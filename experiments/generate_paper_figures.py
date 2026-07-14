"""
Generate publication-quality summary figures for IEEE WIFS 2026 paper.

Combines results from all experiments into 3 key figures:
  Fig 1: DCT perturbation profiles (ESRGAN vs JPEG) + isotropy scatter
  Fig 2: ESRGAN-JPEG anti-complementarity + Pareto frontier
  Fig 3: NC prediction features + pair comparison summary

Run from project root:
    python experiments/generate_paper_figures.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ESRGAN_STAB_CSV = os.path.join(_ROOT, "results", "dct_stability_realesrgan.csv")
JPEG_STAB_CSV   = os.path.join(_ROOT, "results", "dct_stability_jpeg.csv")
FREQ_SWEEP_CSV  = os.path.join(_ROOT, "results", "frequency_sweep.csv")
TEXTURE_CSV     = os.path.join(_ROOT, "results", "texture_complexity.csv")
EXP_B_CSV       = os.path.join(_ROOT, "experiments", "exp_B_nc_predictor", "outputs", "features.csv")
EXP_D_CSV       = os.path.join(_ROOT, "experiments", "exp_D_exhaustive_optimization", "outputs", "D_all_pairs.csv")

JPEG_Q50 = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99],
], dtype=np.float64)


def load_damage_matrix(csv_path):
    df = pd.read_csv(csv_path)
    mat = np.zeros((8, 8))
    for _, row in df.iterrows():
        mat[int(row["coeff_row"]), int(row["coeff_col"])] = row["mean_abs_change"]
    return mat


# ── Figure 1: DCT Damage Profiles ────────────────────────────────────────────

def figure1_dct_profiles(esrgan_mat, jpeg_mat):
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # Panel A: ESRGAN damage heatmap
    ax_a = fig.add_subplot(gs[0])
    disp = esrgan_mat.copy(); disp[0,0] = np.nan
    im = ax_a.imshow(disp, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax_a, label="Mean |Δcoeff|", shrink=0.85)
    ax_a.set_title("(a) Real-ESRGAN\nDCT Perturbation", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("Freq (col)", fontsize=10); ax_a.set_ylabel("Freq (row)", fontsize=10)
    ax_a.set_xticks(range(8)); ax_a.set_yticks(range(8))
    for pos in [(4,1),(1,4)]:
        ax_a.add_patch(plt.Rectangle((pos[1]-.5,pos[0]-.5),1,1,fill=False,edgecolor="blue",linewidth=2.5))
    ax_a.text(1, 4.4, "Std", fontsize=7, color="blue", ha="center")
    ax_a.text(4, 1.4, "Std", fontsize=7, color="blue", ha="center")

    # Panel B: JPEG heatmap
    ax_b = fig.add_subplot(gs[1])
    disp_j = jpeg_mat.copy(); disp_j[0,0] = np.nan
    im_b = ax_b.imshow(disp_j, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im_b, ax=ax_b, label="Mean |Δcoeff|", shrink=0.85)
    ax_b.set_title("(b) JPEG Q50\nDCT Perturbation", fontsize=12, fontweight="bold")
    ax_b.set_xlabel("Freq (col)", fontsize=10)
    ax_b.set_xticks(range(8)); ax_b.set_yticks(range(8))

    # Panel C: Isotropy scatter
    ax_c = fig.add_subplot(gs[2])
    vals_ij_e, vals_ji_e = [], []
    vals_ij_j, vals_ji_j = [], []
    for i in range(8):
        for j in range(8):
            if i != j:
                vals_ij_e.append(esrgan_mat[i,j]); vals_ji_e.append(esrgan_mat[j,i])
                vals_ij_j.append(jpeg_mat[i,j]);   vals_ji_j.append(jpeg_mat[j,i])

    r_e, _ = pearsonr(vals_ij_e, vals_ji_e)
    r_j, _ = pearsonr(vals_ij_j, vals_ji_j)

    ax_c.scatter(vals_ij_e, vals_ji_e, s=30, alpha=0.7, c="steelblue", label=f"ESRGAN (r={r_e:.3f})")
    ax_c.scatter(vals_ij_j, vals_ji_j, s=30, alpha=0.7, c="darkorange", marker="^", label=f"JPEG (r={r_j:.3f})")
    mn = min(vals_ij_e + vals_ij_j + vals_ji_e + vals_ji_j)
    mx = max(vals_ij_e + vals_ij_j + vals_ji_e + vals_ji_j)
    ax_c.plot([mn,mx],[mn,mx],"k--",linewidth=1.2,label="y=x (isotropy)")
    ax_c.set_xlabel("M[i,j]", fontsize=10); ax_c.set_ylabel("M[j,i]", fontsize=10)
    ax_c.set_title("(c) Perturbation Isotropy\n(symmetric positions)", fontsize=12, fontweight="bold")
    ax_c.legend(fontsize=9); ax_c.grid(True, alpha=0.3)

    # Panel D: Radial profile
    ax_d = fig.add_subplot(gs[3])
    radii = [np.sqrt(i**2+j**2) for i in range(8) for j in range(8) if (i,j)!=(0,0)]
    e_v = [esrgan_mat[i,j] for i in range(8) for j in range(8) if (i,j)!=(0,0)]
    j_v = [jpeg_mat[i,j]   for i in range(8) for j in range(8) if (i,j)!=(0,0)]

    ax_d.scatter(radii, e_v, s=30, alpha=0.6, c="steelblue", label="Real-ESRGAN")
    ax_d.scatter(radii, j_v, s=30, alpha=0.6, c="darkorange", marker="^", label="JPEG Q50")

    # Mark (4,1)/(1,4)
    for pos in [(4,1),(1,4)]:
        r_pos = np.sqrt(pos[0]**2+pos[1]**2)
        ax_d.scatter([r_pos],[esrgan_mat[pos]],s=120,c="crimson",zorder=6,marker="*")
    ax_d.annotate("(4,1)/(1,4)\n[standard]",
                  xy=(np.sqrt(17),10.2), fontsize=8, color="crimson",
                  xytext=(3.8,11.5), arrowprops=dict(arrowstyle="->",color="crimson"))

    ax_d.set_xlabel("DCT frequency radius √(i²+j²)", fontsize=10)
    ax_d.set_ylabel("Mean |Δcoeff|", fontsize=10)
    ax_d.set_title("(d) Damage vs Frequency Radius\n(★ = standard embedding)", fontsize=12, fontweight="bold")
    ax_d.legend(fontsize=9); ax_d.grid(True, alpha=0.3)

    fig.suptitle(
        "Fig. 1: DCT Perturbation Profiles in the DWT LL Subband — Real-ESRGAN vs JPEG Q50",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "FIGURE1_dct_profiles.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved → {out}")


# ── Figure 2: Anti-complementarity + Pareto frontier ─────────────────────────

def figure2_tradeoff(esrgan_mat):
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # Panel A: Anti-complementarity (position-level)
    ax_a = fig.add_subplot(gs[0])
    e_flat = esrgan_mat.flatten()
    q_flat = JPEG_Q50.flatten()
    r_vals = [np.sqrt(i**2+j**2) for i in range(8) for j in range(8)]

    sc = ax_a.scatter(e_flat, q_flat, c=r_vals, cmap="viridis", s=50, alpha=0.8)
    plt.colorbar(sc, ax=ax_a, label="Freq radius √(i²+j²)", shrink=0.85)

    r_aq, _ = spearmanr(e_flat, q_flat)
    for pos, label in [((4,1),"(4,1)"),((1,4),"(1,4)")]:
        ax_a.scatter([esrgan_mat[pos]],[JPEG_Q50[pos]],c="red",s=120,marker="D",zorder=6)
        ax_a.annotate(label, (esrgan_mat[pos],JPEG_Q50[pos]),
                      xytext=(5,5),textcoords="offset points",fontsize=8,color="red")

    # Optimal zone annotation
    ax_a.add_patch(plt.Rectangle((3,0),5,65,fill=True,facecolor="palegreen",alpha=0.25,
                                  edgecolor="darkgreen",linewidth=1.5,linestyle="--"))
    ax_a.text(4, 55, "Optimal zone:\nLow E-dmg,\nLow Q-step", fontsize=8, color="darkgreen")

    ax_a.set_xlabel("ESRGAN mean |Δcoeff|", fontsize=11)
    ax_a.set_ylabel("JPEG Q50 quantization step", fontsize=11)
    ax_a.set_title(f"(a) Position-Level Anti-Complementarity\nSpearman ρ={r_aq:.3f} (p<10⁻²⁵)",
                   fontsize=11, fontweight="bold")
    ax_a.legend(["DCT positions", "Standard (4,1)/(1,4)"],
                fontsize=8, handler_map={})
    ax_a.grid(True, alpha=0.3)

    # Panel B: Pair-level ESRGAN vs JPEG NC (from frequency_sweep)
    ax_b = fig.add_subplot(gs[1])
    if os.path.exists(FREQ_SWEEP_CSV):
        sweep = pd.read_csv(FREQ_SWEEP_CSV)
        pivot = (sweep.groupby(["coeff_pair","direction"])["nc"].mean().reset_index()
                 .pivot(index="coeff_pair",columns="direction",values="nc").fillna(-np.inf))
        best_dir = {p: "A" if pivot.loc[p,"A"]>=pivot.loc[p,"B"] else "B" for p in pivot.index
                    if "A" in pivot.columns and "B" in pivot.columns}
        sweep_best = sweep[sweep.apply(lambda r: r["direction"]==best_dir.get(r["coeff_pair"],"A"),axis=1)]

        nc_e = sweep_best[sweep_best["attack_name"]=="real_esrgan"].groupby("coeff_pair")["nc"].mean()
        nc_j = sweep_best[sweep_best["attack_name"]=="jpeg_compression_q50"].groupby("coeff_pair")["nc"].mean()
        common = nc_e.index.intersection(nc_j.index)

        r_ej, _ = spearmanr(nc_e[common], nc_j[common])

        # Symmetry coloring
        def sym_class(label):
            parts = label.replace("(","").replace(")","").split("/")
            p1 = tuple(int(x) for x in parts[0].split(","))
            p2 = tuple(int(x) for x in parts[1].split(","))
            if p1 == (p2[1],p2[0]): return "symmetric"
            mirror = (p1[1],p1[0])
            return "near-symmetric" if abs(mirror[0]-p2[0])+abs(mirror[1]-p2[1])<=2 else "asymmetric"

        colors_map = {"symmetric":"#27ae60","near-symmetric":"#2980b9","asymmetric":"#c0392b"}
        for label in common:
            sc_cls = sym_class(label)
            ax_b.scatter([nc_e[label]],[nc_j[label]],
                         c=colors_map[sc_cls],s=50,alpha=0.8,zorder=3)

        # Pareto front
        xe, xj = nc_e[common].values, nc_j[common].values
        pareto = np.ones(len(common),dtype=bool)
        for i in range(len(common)):
            for jj in range(len(common)):
                if i!=jj and xe[jj]>=xe[i] and xj[jj]>=xj[i] and (xe[jj]>xe[i] or xj[jj]>xj[i]):
                    pareto[i]=False; break
        ax_b.scatter(xe[pareto],xj[pareto],c="gold",s=100,marker="*",zorder=5,
                     label="Pareto-optimal")

        # Mark standard pair
        std_lbl = "(1,4)/(4,1)"
        std_lbl_alt = "(4,1)/(1,4)"
        for sl in [std_lbl, std_lbl_alt]:
            if sl in common:
                ax_b.scatter([nc_e[sl]],[nc_j[sl]],c="red",s=120,marker="D",zorder=6)
                ax_b.annotate("Standard", (nc_e[sl],nc_j[sl]),
                              xytext=(5,-12),textcoords="offset points",fontsize=8,color="red")

        from matplotlib.patches import Patch
        legend_elems = [
            Patch(facecolor="#27ae60", label="Symmetric"),
            Patch(facecolor="#2980b9", label="Near-symmetric"),
            Patch(facecolor="#c0392b", label="Asymmetric"),
            plt.scatter([],[],c="gold",s=80,marker="*",label="Pareto"),
        ]
        ax_b.legend(handles=legend_elems, fontsize=7)
        ax_b.set_xlabel("NC under Real-ESRGAN", fontsize=11)
        ax_b.set_ylabel("NC under JPEG Q50", fontsize=11)
        ax_b.set_title(f"(b) ESRGAN-JPEG NC Trade-Off\nSpearman ρ={r_ej:.3f}",
                       fontsize=11, fontweight="bold")
        ax_b.grid(True, alpha=0.3)

    # Panel C: Analytical prediction — all 1953 pairs
    ax_c = fig.add_subplot(gs[2])
    if os.path.exists(EXP_D_CSV):
        all_pairs = pd.read_csv(EXP_D_CSV)
        colors_map = {"symmetric":"#27ae60","near-symmetric":"#2980b9","asymmetric":"#c0392b"}
        for cls, alpha, size in [("asymmetric",0.1,8),("near-symmetric",0.4,20),("symmetric",0.9,40)]:
            sub = all_pairs[all_pairs["symmetry_class"]==cls]
            ax_c.scatter(sub["pred_esrgan_nc"],sub["pred_jpeg_nc"],
                         c=colors_map[cls],s=size,alpha=alpha,
                         label=f"{cls} (n={len(sub)})")

        # Highlight Pareto
        par = all_pairs[all_pairs["pareto_optimal"]==True]
        ax_c.scatter(par["pred_esrgan_nc"],par["pred_jpeg_nc"],
                     c="gold",s=80,marker="*",zorder=5,label=f"Pareto (n={len(par)})")

        # Highlight top balanced
        top1 = all_pairs.nlargest(1,"pred_combined")
        ax_c.scatter(top1["pred_esrgan_nc"],top1["pred_jpeg_nc"],
                     c="crimson",s=150,marker="D",zorder=7,label="Best balanced (2,3)/(3,2)")

        ax_c.set_xlabel("Predicted ESRGAN NC (norm. stability score)", fontsize=11)
        ax_c.set_ylabel("Predicted JPEG NC (norm. Q-step score)", fontsize=11)
        ax_c.set_title(f"(c) Analytical Frontier — All 1953 Pairs\n(colored by symmetry class)",
                       fontsize=11, fontweight="bold")
        ax_c.legend(fontsize=7); ax_c.grid(True, alpha=0.2)

    fig.suptitle(
        "Fig. 2: ESRGAN-JPEG Anti-Complementarity and Embedding Position Trade-Off",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "FIGURE2_tradeoff.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 saved → {out}")


# ── Figure 3: NC Predictor + Pair Summary ─────────────────────────────────────

def figure3_predictor_and_pairs():
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # Panel A: ll_spectral_entropy vs NC
    if os.path.exists(EXP_B_CSV):
        bdf = pd.read_csv(EXP_B_CSV)
        ax_a = fig.add_subplot(gs[0])
        r_e, p_e = pearsonr(bdf["ll_spectral_entropy"].dropna(), bdf["nc"].dropna())
        kodak = bdf["image"].str.startswith("kodim")
        ax_a.scatter(bdf.loc[kodak,"ll_spectral_entropy"], bdf.loc[kodak,"nc"],
                     c="steelblue", s=50, alpha=0.8, label="Kodak (n=24)")
        ax_a.scatter(bdf.loc[~kodak,"ll_spectral_entropy"], bdf.loc[~kodak,"nc"],
                     c="darkorange", s=50, alpha=0.8, marker="^", label="TAMPERE17 (n=76)")
        m, b = np.polyfit(bdf["ll_spectral_entropy"].dropna(), bdf["nc"].dropna(), 1)
        xl = np.linspace(bdf["ll_spectral_entropy"].min(), bdf["ll_spectral_entropy"].max(), 100)
        ax_a.plot(xl, m*xl+b, "r-", linewidth=2, label=f"Fit (r={r_e:.3f})")
        ax_a.set_xlabel("LL Spectral Entropy", fontsize=11)
        ax_a.set_ylabel("NC (Real-ESRGAN, α=0.1)", fontsize=11)
        ax_a.set_title(f"(a) Best NC Predictor: LL Spectral Entropy\nr={r_e:.3f}, p<10⁻⁸",
                       fontsize=11, fontweight="bold")
        ax_a.legend(fontsize=9); ax_a.grid(True, alpha=0.3)

    # Panel B: Feature correlation bar
    ax_b = fig.add_subplot(gs[1])
    features = {
        "LL Spectral Entropy": 0.533,
        "Edge Density": 0.459,
        "HH Energy": 0.413,
        "Block Energy Std": -0.393,
        "Local Std CV": -0.477,
        "Wavelet E. Ratio": 0.238,
        "Gradient Magnitude": 0.291,
        "Laplacian Variance": 0.303,
        "Embed. Pos. Energy": 0.265,
    }
    feats = list(features.keys())
    vals  = list(features.values())
    colors = ["#2ecc71" if v>0 else "#e74c3c" for v in vals]
    y_pos = np.arange(len(feats))
    ax_b.barh(y_pos, vals, color=colors, alpha=0.85)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(feats, fontsize=9)
    ax_b.axvline(0, color="black", linewidth=0.8)
    ax_b.axvline(0.19, color="gray", linestyle=":", linewidth=1)
    ax_b.axvline(-0.19, color="gray", linestyle=":", linewidth=1)
    ax_b.set_xlabel("Pearson r with NC (Real-ESRGAN)", fontsize=11)
    ax_b.set_title("(b) Feature Correlations with NC\n(green=positive, red=negative; p<0.05 line shown)",
                   fontsize=11, fontweight="bold")
    ax_b.grid(True, axis="x", alpha=0.3)

    # Panel C: Pair comparison bar (JPEG + blur from Exp E)
    ax_c = fig.add_subplot(gs[2])
    pairs = ["Standard\n(4,1)/(1,4)", "Balanced\n(2,3)/(3,2)", "High-Freq\n(7,5)/(7,7)", "Symmetric\n(1,2)/(2,1)"]
    jpeg_nc = [0.9900, 0.9965, 0.8822, 0.9947]
    blur_nc = [0.9279, 0.9549, 0.9799, 0.9370]
    esrgan_nc = [0.795, None, 0.864, 0.793]  # from existing data; None = not yet measured

    x = np.arange(len(pairs))
    width = 0.28

    bars_j = ax_c.bar(x - width, jpeg_nc, width, label="JPEG Q50", color="darkorange", alpha=0.85)
    bars_b = ax_c.bar(x, blur_nc, width, label="Gaussian Blur", color="steelblue", alpha=0.85)

    # ESRGAN from existing data
    esrgan_x = [xi for xi, v in zip(x, esrgan_nc) if v is not None]
    esrgan_y = [v for v in esrgan_nc if v is not None]
    ax_c.bar([xi + width for xi in esrgan_x], esrgan_y, width,
             label="Real-ESRGAN*", color="#e74c3c", alpha=0.85)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(pairs, fontsize=8)
    ax_c.set_ylabel("Mean NC", fontsize=11)
    ax_c.set_ylim(0.7, 1.05)
    ax_c.set_title("(c) Candidate Pair Comparison\n(*ESRGAN: from prior experiments)",
                   fontsize=11, fontweight="bold")
    ax_c.legend(fontsize=9)
    ax_c.grid(True, axis="y", alpha=0.3)
    ax_c.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="NC=0.9")

    # Mark standard for reference
    ax_c.axhline(jpeg_nc[0], color="orange", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_c.axhline(blur_nc[0], color="steelblue", linestyle=":", linewidth=0.8, alpha=0.5)

    fig.suptitle(
        "Fig. 3: NC Predictability and Candidate Pair Verification",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "FIGURE3_predictor_pairs.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure 3 saved → {out}")


def main():
    print("Generating publication-quality paper figures...")
    esrgan_mat = load_damage_matrix(ESRGAN_STAB_CSV)
    jpeg_mat   = load_damage_matrix(JPEG_STAB_CSV)

    figure1_dct_profiles(esrgan_mat, jpeg_mat)
    figure2_tradeoff(esrgan_mat)
    figure3_predictor_and_pairs()

    print("\nAll figures generated:")
    for f in ["FIGURE1_dct_profiles.png", "FIGURE2_tradeoff.png", "FIGURE3_predictor_pairs.png"]:
        out = os.path.join(OUT_DIR, f)
        if os.path.exists(out):
            size_kb = os.path.getsize(out) // 1024
            print(f"  {f}: {size_kb} KB")


if __name__ == "__main__":
    main()
