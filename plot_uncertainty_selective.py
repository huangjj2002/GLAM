"""
Generate Accuracy vs Coverage plots (selective prediction curves)
matching the reference style from uncertainty1.png / uncertainty2.png.

EDL line   -> sort by Dirichlet uncertainty (K/S), ascending = most confident first
Baseline   -> sort by binary entropy of GLAM pred_score, ascending = most confident first
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
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
    """H(p) = -p*log(p) - (1-p)*log(1-p), clips for numerical safety."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def selective_prediction(y_true, confidence, coverages):
    """
    Sort samples by confidence (ascending = most confident first).
    For each coverage level, keep the top-K most confident samples
    and compute accuracy.
    """
    n = len(y_true)
    order = np.argsort(confidence)  # lowest uncertainty/entropy first
    y_sorted = y_true[order]

    accs = []
    for cov in coverages:
        k = max(1, int(cov * n))
        acc = float(y_sorted[:k].mean())  # accuracy = fraction correct
        # Actually for binary classification accuracy = (TP+TN)/N,
        # but since we just want "fraction of correct predictions":
        acc = float(accuracy_simple(y_sorted[:k]))
        accs.append(acc)
    return np.array(accs)


def accuracy_simple(y_true_subset):
    """1 - mean absolute error from 0.5 threshold; same as accuracy when y in {0,1}."""
    # Actually for selective prediction we need pred vs true.
    # In the reference plot accuracy = fraction correct using pred_class vs true label.
    # We'll pass pred_correct array (1=correct, 0=wrong) instead.
    return float(y_true_subset.mean())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load & merge (fold-0 validation only)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Compute "correct prediction" arrays
    # ------------------------------------------------------------------
    # EDL: pred_class vs cancer
    edl_correct = (df["pred_class"].values == y_true).astype(float)
    # GLAM baseline: glam_pred_class vs cancer
    glam_correct = (df["glam_pred_class"].values == y_true).astype(float)

    # Confidence measures (lower = more confident)
    edl_uncertainty = df["uncertainty"].values                       # K/S
    glam_entropy = binary_entropy(df["glam_pred_score"].values)     # entropy

    # ------------------------------------------------------------------
    # Sort by confidence & compute accuracy at each coverage
    # ------------------------------------------------------------------
    coverages = np.arange(0.05, 1.05, 0.05)
    n = len(y_true)

    # EDL: sorted by uncertainty ascending
    edl_order = np.argsort(edl_uncertainty)
    edl_correct_sorted = edl_correct[edl_order]

    # Baseline: sorted by entropy ascending
    glam_order = np.argsort(glam_entropy)
    glam_correct_sorted = glam_correct[glam_order]

    edl_accs = []
    glam_accs = []
    for cov in coverages:
        k = max(1, int(cov * n))
        edl_accs.append(float(edl_correct_sorted[:k].mean()))
        glam_accs.append(float(glam_correct_sorted[:k].mean()))

    edl_accs = np.array(edl_accs)
    glam_accs = np.array(glam_accs)

    # ------------------------------------------------------------------
    # Plot 1: standard Accuracy vs Coverage
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(coverages, edl_accs, color='#1f77b4', linewidth=2,
            label='EDL (uncertainty=K/S)')
    ax.plot(coverages, glam_accs, color='#ff7f0e', linewidth=2,
            label='Baseline (entropy)')

    ax.set_xlabel('Coverage (fraction of samples retained)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Coverage (EDL uncertainty vs Baseline entropy)',
                 fontsize=14, pad=10)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.70, 1.00)
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.set_yticks(np.arange(0.70, 1.01, 0.05))
    ax.grid(True, linestyle='-', alpha=0.7, color='lightgray')
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "uncertainty1_accuracy_vs_coverage.png"),
                dpi=200)
    plt.close(fig)
    print("Saved uncertainty1_accuracy_vs_coverage.png")

    # ------------------------------------------------------------------
    # Plot 2: same data, slightly different y-range to highlight differences
    # (zoom into the meaningful range)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(coverages, edl_accs, color='#1f77b4', linewidth=2,
            label='EDL (uncertainty=K/S)')
    ax.plot(coverages, glam_accs, color='#ff7f0e', linewidth=2,
            label='Baseline (entropy)')

    ax.set_xlabel('Coverage (fraction of samples retained)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Coverage (EDL uncertainty vs Baseline entropy)',
                 fontsize=14, pad=10)
    ax.set_xlim(0.0, 1.0)
    # Auto y-range with some padding
    y_min = max(0.0, min(edl_accs.min(), glam_accs.min()) - 0.02)
    y_max = min(1.0, max(edl_accs.max(), glam_accs.max()) + 0.02)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "uncertainty2_accuracy_vs_coverage_zoom.png"),
                dpi=200)
    plt.close(fig)
    print("Saved uncertainty2_accuracy_vs_coverage_zoom.png")

    # ------------------------------------------------------------------
    # Print table for reference
    # ------------------------------------------------------------------
    print("\nCoverage | EDL Acc  | Baseline Acc")
    print("-" * 38)
    for i, cov in enumerate(coverages):
        print(f"  {cov:.2f}   |  {edl_accs[i]:.4f}  |   {glam_accs[i]:.4f}")


if __name__ == "__main__":
    main()
