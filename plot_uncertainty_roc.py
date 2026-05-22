"""
Generate ROC curves for uncertainty measures as error detectors,
matching the reference style from uncertainty.png.

4 uncertainty measures compared:
1. Predictive Entropy (EDL Dirichlet mean entropy)
2. Aleatoric Uncertainty (expected data uncertainty under Dirichlet)
3. Epistemic Uncertainty (K/S from EDL)
4. Baseline Entropy (binary entropy of GLAM pred_score)

Positive class = "misclassified" (pred_class != true label)
Score = uncertainty measure value (higher = more likely misclassified)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.special import digamma

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
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def dirichlet_predictive_entropy(alpha):
    """H[p(y|x)] = -sum_k (alpha_k/S) * log(alpha_k/S)"""
    S = alpha.sum(axis=1, keepdims=True)
    mean = alpha / S
    mean = np.clip(mean, 1e-12, None)
    return np.sum(-mean * np.log(mean), axis=1)


def dirichlet_aleatoric(alpha):
    """
    Aleatoric = E_{Dir(alpha)}[H[p(y|theta)]]
              = sum_k (alpha_k/S) * [psi(S+1) - psi(alpha_k+1)]
    """
    S = alpha.sum(axis=1, keepdims=True)
    weights = alpha / S
    expected_entropy = np.sum(
        weights * (digamma(S + 1) - digamma(alpha + 1)), axis=1
    )
    return expected_entropy


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load & merge
    # ------------------------------------------------------------------
    print("Loading data...")
    edl = pd.read_csv(EDL_CSV)
    glam = pd.read_csv(GLAM_CSV)

    edl_val = edl[edl["fold"] == 0].copy()
    glam_sub = glam[["patient_id", "image_id", "pred_score"]].copy()
    glam_sub.rename(columns={"pred_score": "glam_pred_score"}, inplace=True)

    df = edl_val.merge(glam_sub, on=["patient_id", "image_id"], how="left")
    print(f"Validation rows: {len(df)}")

    y_true = df["cancer"].values.astype(int)

    # ------------------------------------------------------------------
    # Misclassification labels (positive = error)
    # ------------------------------------------------------------------
    misclass_edl = (df["pred_class"].values != y_true).astype(int)
    misclass_glam = ((df["glam_pred_score"].values >= 0.5).astype(int) != y_true).astype(int)

    # ------------------------------------------------------------------
    # Compute 4 uncertainty measures
    # ------------------------------------------------------------------
    alpha = df[["alpha_0", "alpha_1"]].values  # shape (N, 2)

    # 1. Predictive Entropy
    pred_entropy = dirichlet_predictive_entropy(alpha)

    # 2. Aleatoric Uncertainty
    aleatoric = dirichlet_aleatoric(alpha)

    # 3. Epistemic Uncertainty (K/S)
    epistemic = df["uncertainty"].values

    # 4. Baseline Entropy (GLAM binary entropy)
    baseline_entropy = binary_entropy(df["glam_pred_score"].values)

    # ------------------------------------------------------------------
    # ROC for EDL errors (using EDL misclassification as positive)
    # ------------------------------------------------------------------
    measures_edl = {
        "Predictive Entropy": (pred_entropy, '#1f77b4'),
        "Aleatoric": (aleatoric, '#ff7f0e'),
        "Epistemic (K/S)": (epistemic, '#2ca02c'),
    }

    # Baseline entropy detects GLAM errors
    measures_baseline = {
        "Baseline Entropy": (baseline_entropy, '#d62728'),
    }

    # ------------------------------------------------------------------
    # Compute ROC curves & AUC
    # ------------------------------------------------------------------
    roc_data = []

    for name, (scores, color) in measures_edl.items():
        fpr, tpr, _ = roc_curve(misclass_edl, scores)
        auc = roc_auc_score(misclass_edl, scores)
        roc_data.append((name, fpr, tpr, auc, color))
        print(f"  {name:<25s}  AUC = {auc:.4f}  (detecting EDL errors)")

    # Baseline: detect GLAM errors using baseline entropy
    for name, (scores, color) in measures_baseline.items():
        fpr, tpr, _ = roc_curve(misclass_glam, scores)
        auc = roc_auc_score(misclass_glam, scores)
        roc_data.append((name, fpr, tpr, auc, color))
        print(f"  {name:<25s}  AUC = {auc:.4f}  (detecting Baseline errors)")

    # Also compute baseline entropy for EDL errors for a fair comparison
    fpr_bl_edl, tpr_bl_edl, _ = roc_curve(misclass_edl, baseline_entropy)
    auc_bl_edl = roc_auc_score(misclass_edl, baseline_entropy)
    print(f"\n  Baseline Entropy (EDL err) AUC = {auc_bl_edl:.4f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, fpr, tpr, auc, color in roc_data:
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC={auc:.4f})')

    ax.plot([0, 1], [0, 1], color='#7f7f7f', linestyle='--', linewidth=1.5)

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (detecting errors)', fontsize=12)
    ax.set_title('Uncertainty Measures as Error Detectors (ROC)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.grid(True, linestyle='-', alpha=0.3, color='lightgray')
    ax.legend(loc='lower right', frameon=True, fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "uncertainty_roc_error_detectors.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved {out_path}")

    # ------------------------------------------------------------------
    # Also plot: all 4 measures detecting EDL errors (fair comparison)
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for name, (scores, color) in {**measures_edl, "Baseline Entropy": (baseline_entropy, '#d62728')}.items():
        fpr, tpr, _ = roc_curve(misclass_edl, scores)
        auc = roc_auc_score(misclass_edl, scores)
        ax2.plot(fpr, tpr, color=color, linewidth=2,
                 label=f'{name} (AUC={auc:.4f})')
        print(f"  [EDL errors] {name:<25s}  AUC = {auc:.4f}")

    ax2.plot([0, 1], [0, 1], color='#7f7f7f', linestyle='--', linewidth=1.5)

    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate (detecting errors)', fontsize=12)
    ax2.set_title('Uncertainty Measures as Error Detectors (ROC)',
                 fontsize=14, fontweight='bold')
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax2.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax2.grid(True, linestyle='-', alpha=0.3, color='lightgray')
    ax2.legend(loc='lower right', frameon=True, fontsize=10)

    fig2.tight_layout()
    out_path2 = os.path.join(OUTPUT_DIR, "uncertainty_roc_fair_comparison.png")
    fig2.savefig(out_path2, dpi=200)
    plt.close(fig2)
    print(f"Saved {out_path2}")


if __name__ == "__main__":
    main()
