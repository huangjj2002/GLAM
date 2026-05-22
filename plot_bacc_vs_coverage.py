"""
Generate BACC vs Coverage plot (selective prediction curve).

EDL line   -> sort by Dirichlet uncertainty (K/S), ascending = most confident first
Baseline   -> sort by binary entropy of GLAM pred_score, ascending = most confident first
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EDL_CSV = os.path.join(
    BASE_DIR,
    "outputs/edl_kfold_ft_20260516_175609/per_model_predictions/fold0_edl_predictions.csv",
)
GLAM_CSV = os.path.join(
    BASE_DIR,
    "outputs/glam_lp_fold0_5ep_no_sampler/per_model_predictions/fold0_predictions.csv",
)
OUTPUT_DIR = os.path.join(BASE_DIR, "validation_comprehensive_analysis")


def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    edl = pd.read_csv(EDL_CSV)
    glam = pd.read_csv(GLAM_CSV)

    edl_val = edl[edl["fold"] == 0].copy()
    glam_sub = glam[["patient_id", "image_id", "pred_score", "pred_class"]].copy()
    glam_sub.rename(columns={"pred_score": "glam_pred_score",
                              "pred_class": "glam_pred_class"}, inplace=True)

    df = edl_val.merge(glam_sub, on=["patient_id", "image_id"], how="left")
    print(f"Validation rows: {len(df)}")

    y_true = df["cancer"].values.astype(int)
    n = len(y_true)

    edl_pred = df["pred_class"].values.astype(int)
    glam_pred = df["glam_pred_class"].values.astype(int)

    edl_uncertainty = df["uncertainty"].values
    glam_entropy = binary_entropy(df["glam_pred_score"].values)

    coverages = np.arange(0.05, 1.05, 0.05)

    # EDL: sort by uncertainty ascending (most confident first)
    edl_order = np.argsort(edl_uncertainty)
    # Baseline: sort by entropy ascending (most confident first)
    glam_order = np.argsort(glam_entropy)

    edl_baccs = []
    glam_baccs = []
    for cov in coverages:
        k = max(1, int(cov * n))
        # EDL
        idx_e = edl_order[:k]
        if len(np.unique(y_true[idx_e])) < 2:
            edl_baccs.append(float("nan"))
        else:
            edl_baccs.append(float(balanced_accuracy_score(y_true[idx_e], edl_pred[idx_e])))
        # Baseline
        idx_g = glam_order[:k]
        if len(np.unique(y_true[idx_g])) < 2:
            glam_baccs.append(float("nan"))
        else:
            glam_baccs.append(float(balanced_accuracy_score(y_true[idx_g], glam_pred[idx_g])))

    edl_baccs = np.array(edl_baccs)
    glam_baccs = np.array(glam_baccs)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(coverages, edl_baccs, color='#1f77b4', linewidth=2,
            label='EDL (uncertainty=K/S)')
    ax.plot(coverages, glam_baccs, color='#ff7f0e', linewidth=2,
            label='Baseline (entropy)')

    ax.set_xlabel('Coverage (fraction of samples retained)', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('BACC vs Coverage (EDL uncertainty vs Baseline entropy)',
                 fontsize=14, pad=10)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.grid(True, linestyle='-', alpha=0.7, color='lightgray')
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bacc_vs_coverage.png"), dpi=200)
    plt.close(fig)
    print("Saved bacc_vs_coverage.png")

    # Print table
    print("\nCoverage | EDL BACC | Baseline BACC")
    print("-" * 40)
    for i, cov in enumerate(coverages):
        print(f"  {cov:.2f}   |  {edl_baccs[i]:.4f}  |   {glam_baccs[i]:.4f}")


if __name__ == "__main__":
    main()
