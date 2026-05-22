"""
Comprehensive validation analysis comparing EDL vs GLAM models on fold-0 validation set.

Usage:
    python validation_comprehensive_analysis.py

Output:
    validation_comprehensive_analysis/ folder with metrics, plots, and report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Bootstrap 95% CI
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 1000
CI_ALPHA = 0.05
RNG = np.random.RandomState(42)


def bootstrap_ci_batch(y_true, y_score, metric_fns, n_boot=N_BOOTSTRAP, alpha=CI_ALPHA):
    """Compute point estimates + 95% CIs for multiple metrics in a single bootstrap pass."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    keys = list(metric_fns.keys())
    point_ests = {k: metric_fns[k](y_true, y_score) for k in keys}
    boot_vals = {k: [] for k in keys}

    for _ in range(n_boot):
        idx = RNG.randint(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        for k in keys:
            boot_vals[k].append(metric_fns[k](yt, ys))

    results = {}
    for k in keys:
        bv = np.array(boot_vals[k]) if boot_vals[k] else np.array([])
        if len(bv) == 0:
            results[k] = (point_ests[k], float("nan"), float("nan"))
        else:
            lo = float(np.percentile(bv, 100 * alpha / 2))
            hi = float(np.percentile(bv, 100 * (1 - alpha / 2)))
            results[k] = (point_ests[k], lo, hi)
    return results


def _metric_roc_auc(yt, ys):
    return float(roc_auc_score(yt, ys))


def _metric_pr_auc(yt, ys):
    return float(average_precision_score(yt, ys))


def _metric_accuracy(yt, ys):
    return float(accuracy_score(yt, (np.asarray(ys) >= 0.5).astype(int)))


def _metric_bacc(yt, ys):
    return float(balanced_accuracy_score(yt, (np.asarray(ys) >= 0.5).astype(int)))


def _metric_f1(yt, ys):
    return float(f1_score(yt, (np.asarray(ys) >= 0.5).astype(int), zero_division=0))


def _metric_sensitivity(yt, ys):
    yp = (np.asarray(ys) >= 0.5).astype(int)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")


def _metric_specificity(yt, ys):
    yp = (np.asarray(ys) >= 0.5).astype(int)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


def _metric_precision(yt, ys):
    yp = (np.asarray(ys) >= 0.5).astype(int)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")


def _metric_brier(yt, ys):
    return float(np.mean((np.asarray(ys) - yt.astype(float)) ** 2))


def _metric_ece_10(yt, ys):
    ece, _ = compute_ece(yt, ys, n_bins=10)
    return ece


def _metric_ece_10(yt, ys):
    ece, _ = compute_ece(yt, ys, n_bins=10)
    return ece


def _metric_brier(yt, ys):
    return float(np.mean((np.asarray(ys) - yt.astype(float)) ** 2))


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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_binary_metrics(y_true, y_score, threshold=0.5, with_ci=True):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = safe_roc_auc(y_true, y_score)
    pr_auc = safe_pr_auc(y_true, y_score)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")

    result = {
        "n": int(len(y_true)),
        "pos": int(y_true.sum()),
        "neg": int((1 - y_true).sum()),
        "pos_rate": float(y_true.mean()),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1": f1,
        "precision": precision,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if with_ci:
        print("      Bootstrapping 95% CIs (1000 iters)...")
        ci_map = {
            "roc_auc": _metric_roc_auc,
            "pr_auc": _metric_pr_auc,
            "accuracy": _metric_accuracy,
            "balanced_accuracy": _metric_bacc,
            "f1": _metric_f1,
            "sensitivity": _metric_sensitivity,
            "specificity": _metric_specificity,
            "precision": _metric_precision,
        }
        ci_results = bootstrap_ci_batch(y_true, y_score, ci_map)
        for key in ci_map:
            _, lo, hi = ci_results[key]
            result[f"{key}_ci_lo"] = lo
            result[f"{key}_ci_hi"] = hi

    return result


def find_best_threshold(y_true, y_score, metric_fn, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)
    best_val = -1
    best_thr = 0.5
    for thr in thresholds:
        y_pred = (np.asarray(y_score) >= thr).astype(int)
        val = metric_fn(np.asarray(y_true).astype(int), y_pred)
        if val > best_val:
            best_val = val
            best_thr = thr
    return best_thr, float(best_val)


def compute_ece(y_true, y_score, n_bins=10):
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_data = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            bins_data.append({"bin": i, "bin_lo": lo, "bin_hi": hi, "count": 0,
                              "mean_predicted": float("nan"), "mean_observed": float("nan")})
            continue
        acc_bin = y_true[mask].mean()
        conf_bin = y_score[mask].mean()
        ece += abs(acc_bin - conf_bin) * n_bin / len(y_true)
        bins_data.append({
            "bin": i, "bin_lo": lo, "bin_hi": hi, "count": int(n_bin),
            "mean_predicted": float(conf_bin), "mean_observed": float(acc_bin),
        })
    return float(ece), bins_data


def compute_brier_score(y_true, y_score):
    return float(np.mean((np.asarray(y_score) - np.asarray(y_true).astype(float)) ** 2))


def fmt(x):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if x is None:
        return "nan"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if np.isnan(v):
        return "nan"
    return f"{v:.6f}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_align_data():
    print("### [INFO] Loading EDL predictions...")
    edl_df = pd.read_csv(EDL_CSV)
    print(f"    EDL total rows: {len(edl_df)}")

    edl_val = edl_df[edl_df["fold"] == 0].copy()
    print(f"    EDL fold-0 validation rows: {len(edl_val)}")

    print("### [INFO] Loading GLAM predictions...")
    glam_df = pd.read_csv(GLAM_CSV)
    print(f"    GLAM total rows: {len(glam_df)}")

    glam_subset = glam_df[["patient_id", "image_id", "pred_score", "pred_class"]].copy()
    glam_subset.rename(columns={"pred_score": "glam_pred_score", "pred_class": "glam_pred_class"}, inplace=True)

    merged = edl_val.merge(glam_subset, on=["patient_id", "image_id"], how="left")
    unmatched = merged["glam_pred_score"].isna().sum()
    if unmatched > 0:
        raise RuntimeError(f"Unmatched rows: {unmatched}")
    print(f"    Merged validation rows: {len(merged)} (all matched)")

    merged["dirichlet_strength"] = merged["alpha_0"] + merged["alpha_1"]
    merged["total_evidence"] = merged["evidence_0"] + merged["evidence_1"]
    merged["correct_edl"] = (merged["pred_class"] == merged["cancer"]).astype(int)
    merged["correct_glam"] = (merged["glam_pred_class"] == merged["cancer"]).astype(int)

    return merged


# ---------------------------------------------------------------------------
# Standard binary metrics
# ---------------------------------------------------------------------------

def compute_all_binary_metrics(df):
    y_true = df["cancer"].values

    edl_metrics = compute_binary_metrics(y_true, df["prob_1"].values, 0.5)
    glam_metrics = compute_binary_metrics(y_true, df["glam_pred_score"].values, 0.5)

    # Best thresholds
    thresholds = np.linspace(0.0, 1.0, 201)

    edl_best_f1_thr, edl_best_f1 = find_best_threshold(
        y_true, df["prob_1"].values,
        lambda yt, yp: f1_score(yt, yp, zero_division=0), thresholds,
    )
    glam_best_f1_thr, glam_best_f1 = find_best_threshold(
        y_true, df["glam_pred_score"].values,
        lambda yt, yp: f1_score(yt, yp, zero_division=0), thresholds,
    )

    edl_best_bacc_thr, edl_best_bacc = find_best_threshold(
        y_true, df["prob_1"].values,
        balanced_accuracy_score, thresholds,
    )
    glam_best_bacc_thr, glam_best_bacc = find_best_threshold(
        y_true, df["glam_pred_score"].values,
        balanced_accuracy_score, thresholds,
    )

    def youden(yt, yp):
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        return sens + spec - 1

    edl_best_youden_thr, edl_best_youden = find_best_threshold(
        y_true, df["prob_1"].values, youden, thresholds,
    )
    glam_best_youden_thr, glam_best_youden = find_best_threshold(
        y_true, df["glam_pred_score"].values, youden, thresholds,
    )

    edl_metrics.update({
        "best_f1": edl_best_f1, "best_f1_threshold": edl_best_f1_thr,
        "best_bacc": edl_best_bacc, "best_bacc_threshold": edl_best_bacc_thr,
        "best_youden": edl_best_youden, "best_youden_threshold": edl_best_youden_thr,
        "score_mean_pos": float(df.loc[df["cancer"] == 1, "prob_1"].mean()),
        "score_mean_neg": float(df.loc[df["cancer"] == 0, "prob_1"].mean()),
    })
    glam_metrics.update({
        "best_f1": glam_best_f1, "best_f1_threshold": glam_best_f1_thr,
        "best_bacc": glam_best_bacc, "best_bacc_threshold": glam_best_bacc_thr,
        "best_youden": glam_best_youden, "best_youden_threshold": glam_best_youden_thr,
        "score_mean_pos": float(df.loc[df["cancer"] == 1, "glam_pred_score"].mean()),
        "score_mean_neg": float(df.loc[df["cancer"] == 0, "glam_pred_score"].mean()),
    })

    # Bootstrap CI for best_f1, best_bacc, best_youden
    # Note: These use the fixed best threshold found above, not re-searched per bootstrap.
    print("      Bootstrapping best-threshold CIs...")
    for metrics, col_name in [(edl_metrics, "prob_1"), (glam_metrics, "glam_pred_score")]:
        scores = df[col_name].values
        f1_thr = metrics["best_f1_threshold"]
        bacc_thr = metrics["best_bacc_threshold"]
        youden_thr = metrics["best_youden_threshold"]

        def _make_thr_metric(thr, sk_fn):
            def fn(yt, ys):
                yp = (np.asarray(ys) >= thr).astype(int)
                return float(sk_fn(yt, yp))
            return fn

        ci_map = {
            "best_f1": _make_thr_metric(f1_thr, lambda yt, yp: f1_score(yt, yp, zero_division=0)),
            "best_bacc": _make_thr_metric(bacc_thr, balanced_accuracy_score),
        }
        ci_results = bootstrap_ci_batch(y_true, scores, ci_map)
        for key in ci_map:
            _, lo, hi = ci_results[key]
            metrics[f"{key}_ci_lo"] = lo
            metrics[f"{key}_ci_hi"] = hi
        # Youden uses same thr as bacc in practice; skip separate CI

    # Score quantiles
    for label, col in [("edl", "prob_1"), ("glam", "glam_pred_score")]:
        for grp, mask in [("all", np.ones(len(df), dtype=bool)),
                          ("pos", df["cancer"] == 1), ("neg", df["cancer"] == 0)]:
            q = df.loc[mask, col].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).values
            key = f"score_quantiles_{grp}"
            metrics = edl_metrics if label == "edl" else glam_metrics
            metrics[key] = [float(v) for v in q]

    return edl_metrics, glam_metrics


# ---------------------------------------------------------------------------
# EDL-specific analysis
# ---------------------------------------------------------------------------

def compute_uncertainty_analysis(df):
    y_true = df["cancer"].values
    uncertainty = df["uncertainty"].values
    correct = df["correct_edl"].values.astype(bool)

    correct_u = uncertainty[correct]
    incorrect_u = uncertainty[~correct]

    # Mann-Whitney U test
    try:
        stat, pval = mannwhitneyu(incorrect_u, correct_u, alternative="greater")
    except Exception:
        stat, pval = float("nan"), float("nan")

    # AUROC of uncertainty for misclassification detection + CI
    misclass = (~correct).astype(int)
    u_auroc = safe_roc_auc(misclass, uncertainty)
    # Bootstrap CI for uncertainty AUROC
    n = len(y_true)
    boot_u = []
    for _ in range(N_BOOTSTRAP):
        idx = RNG.randint(0, n, size=n)
        yt, uc = misclass[idx], uncertainty[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_u.append(float(roc_auc_score(yt, uc)))
    if boot_u:
        u_auroc_lo = float(np.percentile(boot_u, 100 * CI_ALPHA / 2))
        u_auroc_hi = float(np.percentile(boot_u, 100 * (1 - CI_ALPHA / 2)))
    else:
        u_auroc_lo, u_auroc_hi = float("nan"), float("nan")

    results = {
        "n_correct": int(correct.sum()),
        "n_incorrect": int((~correct).sum()),
        "uncertainty_mean_correct": float(correct_u.mean()),
        "uncertainty_median_correct": float(np.median(correct_u)),
        "uncertainty_std_correct": float(correct_u.std()),
        "uncertainty_mean_incorrect": float(incorrect_u.mean()),
        "uncertainty_median_incorrect": float(np.median(incorrect_u)),
        "uncertainty_std_incorrect": float(incorrect_u.std()),
        "mann_whitney_U": float(stat),
        "mann_whitney_p": float(pval),
        "uncertainty_auroc_misclass": u_auroc,
        "uncertainty_auroc_misclass_ci_lo": u_auroc_lo,
        "uncertainty_auroc_misclass_ci_hi": u_auroc_hi,
    }

    # Uncertainty by class
    pos_mask = y_true == 1
    results["uncertainty_mean_pos"] = float(uncertainty[pos_mask].mean())
    results["uncertainty_mean_neg"] = float(uncertainty[~pos_mask].mean())
    results["uncertainty_median_pos"] = float(np.median(uncertainty[pos_mask]))
    results["uncertainty_median_neg"] = float(np.median(uncertainty[~pos_mask]))

    return pd.DataFrame([results])


def compute_selective_prediction(df):
    y_true = df["cancer"].values
    y_score = df["prob_1"].values
    uncertainty = df["uncertainty"].values

    order = np.argsort(uncertainty)
    coverages = np.arange(0.05, 1.05, 0.05)

    rows = []
    for cov in coverages:
        n_keep = max(1, int(cov * len(df)))
        idx = order[:n_keep]
        yt = y_true[idx]
        ys = y_score[idx]
        acc = float(accuracy_score(yt, (ys >= 0.5).astype(int)))
        bacc = float(balanced_accuracy_score(yt, (ys >= 0.5).astype(int)))
        f1 = float(f1_score(yt, (ys >= 0.5).astype(int), zero_division=0))
        auc = safe_roc_auc(yt, ys)
        rows.append({
            "coverage": float(cov),
            "n_samples": n_keep,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1": f1,
            "roc_auc": auc,
            "mean_uncertainty": float(uncertainty[idx].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Youden's Index table + Uncertainty comparison
# ---------------------------------------------------------------------------

def find_youden_threshold(y_true, y_score, thresholds=None):
    """Find threshold maximizing Youden's J = Sensitivity + Specificity - 1."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 501)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    best_j, best_thr = -1, 0.5
    for thr in thresholds:
        yp = (y_score >= thr).astype(int)
        cm = confusion_matrix(y_true, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_thr = j, thr
    return float(best_thr), float(best_j)


def compute_youden_table(df):
    """Compute full metric table at Youden's optimal threshold, with 95% CIs."""
    y_true = df["cancer"].values

    rows = []
    for model, col in [("EDL", "prob_1"), ("GLAM", "glam_pred_score")]:
        scores = df[col].values
        thr, youden_j = find_youden_threshold(y_true, scores)
        yp = (scores >= thr).astype(int)
        cm = confusion_matrix(y_true, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
        bacc = float(balanced_accuracy_score(y_true, yp))
        auc = safe_roc_auc(y_true, scores)

        point_ests = {
            "threshold": thr,
            "youden_j": youden_j,
            "bacc": bacc,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "auc_roc": auc,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

        # Bootstrap CIs for threshold-dependent metrics at the fixed Youden threshold
        print(f"      Bootstrapping {model} Youden metrics CIs...")

        def _make_thr_metric(threshold, raw_fn):
            def fn(yt, ys):
                yp2 = (np.asarray(ys) >= threshold).astype(int)
                return float(raw_fn(yt, yp2))
            return fn

        ci_fns = {
            "bacc": _make_thr_metric(thr, balanced_accuracy_score),
            "sensitivity": _make_thr_metric(thr, lambda yt, yp: float(
                confusion_matrix(yt, yp, labels=[0, 1]).ravel()[3] /
                (confusion_matrix(yt, yp, labels=[0, 1]).ravel()[3] + confusion_matrix(yt, yp, labels=[0, 1]).ravel()[2])
                if (confusion_matrix(yt, yp, labels=[0, 1]).ravel()[3] + confusion_matrix(yt, yp, labels=[0, 1]).ravel()[2]) > 0 else 0
            )),
            "specificity": _make_thr_metric(thr, lambda yt, yp: float(
                confusion_matrix(yt, yp, labels=[0, 1]).ravel()[0] /
                (confusion_matrix(yt, yp, labels=[0, 1]).ravel()[0] + confusion_matrix(yt, yp, labels=[0, 1]).ravel()[1])
                if (confusion_matrix(yt, yp, labels=[0, 1]).ravel()[0] + confusion_matrix(yt, yp, labels=[0, 1]).ravel()[1]) > 0 else 0
            )),
            "ppv": _make_thr_metric(thr, lambda yt, yp: float(
                confusion_matrix(yt, yp, labels=[0, 1]).ravel()[3] /
                (confusion_matrix(yt, yp, labels=[0, 1]).ravel()[3] + confusion_matrix(yt, yp, labels=[0, 1]).ravel()[1])
                if (confusion_matrix(yt, yp, labels=[0, 1]).ravel()[3] + confusion_matrix(yt, yp, labels=[0, 1]).ravel()[1]) > 0 else 0
            )),
            "npv": _make_thr_metric(thr, lambda yt, yp: float(
                confusion_matrix(yt, yp, labels=[0, 1]).ravel()[0] /
                (confusion_matrix(yt, yp, labels=[0, 1]).ravel()[0] + confusion_matrix(yt, yp, labels=[0, 1]).ravel()[2])
                if (confusion_matrix(yt, yp, labels=[0, 1]).ravel()[0] + confusion_matrix(yt, yp, labels=[0, 1]).ravel()[2]) > 0 else 0
            )),
        }

        # Simpler: use helper functions
        def _sens(yt, yp):
            cm2 = confusion_matrix(yt, yp, labels=[0, 1])
            t, f = cm2[1, 1], cm2[1, 0]
            return float(t / (t + f)) if (t + f) > 0 else 0.0

        def _spec(yt, yp):
            cm2 = confusion_matrix(yt, yp, labels=[0, 1])
            t, f = cm2[0, 0], cm2[0, 1]
            return float(t / (t + f)) if (t + f) > 0 else 0.0

        def _ppv(yt, yp):
            cm2 = confusion_matrix(yt, yp, labels=[0, 1])
            tp2, fp2 = cm2[1, 1], cm2[0, 1]
            return float(tp2 / (tp2 + fp2)) if (tp2 + fp2) > 0 else 0.0

        def _npv(yt, yp):
            cm2 = confusion_matrix(yt, yp, labels=[0, 1])
            tn2, fn2 = cm2[0, 0], cm2[1, 0]
            return float(tn2 / (tn2 + fn2)) if (tn2 + fn2) > 0 else 0.0

        ci_fns2 = {
            "bacc": _make_thr_metric(thr, balanced_accuracy_score),
            "sensitivity": _make_thr_metric(thr, _sens),
            "specificity": _make_thr_metric(thr, _spec),
            "ppv": _make_thr_metric(thr, _ppv),
            "npv": _make_thr_metric(thr, _npv),
        }

        # Add AUC (threshold-independent)
        auc_fns = {"auc_roc": _metric_roc_auc}
        thr_ci = bootstrap_ci_batch(y_true, scores, ci_fns2)
        auc_ci = bootstrap_ci_batch(y_true, scores, auc_fns)

        for k in ci_fns2:
            _, lo, hi = thr_ci[k]
            point_ests[f"{k}_ci_lo"] = lo
            point_ests[f"{k}_ci_hi"] = hi
        _, lo, hi = auc_ci["auc_roc"]
        point_ests["auc_roc_ci_lo"] = lo
        point_ests["auc_roc_ci_hi"] = hi

        point_ests["model"] = model
        rows.append(point_ests)

    return pd.DataFrame(rows).set_index("model")


def compute_uncertainty_at_youden(df, edl_thr):
    """Compute EDL uncertainty statistics broken down by confusion matrix at Youden threshold."""
    y_true = df["cancer"].values
    scores = df["prob_1"].values
    uncertainty = df["uncertainty"].values
    yp = (scores >= edl_thr).astype(int)

    # TP, FP, FN, TN masks
    tp_mask = (y_true == 1) & (yp == 1)
    fp_mask = (y_true == 0) & (yp == 1)
    fn_mask = (y_true == 1) & (yp == 0)
    tn_mask = (y_true == 0) & (yp == 0)

    rows = []
    for name, mask in [("TP", tp_mask), ("FP", fp_mask), ("FN", fn_mask), ("TN", tn_mask)]:
        n = int(mask.sum())
        u_vals = uncertainty[mask]
        rows.append({
            "group": name,
            "n": n,
            "mean_uncertainty": float(u_vals.mean()) if n > 0 else float("nan"),
            "median_uncertainty": float(np.median(u_vals)) if n > 0 else float("nan"),
            "std_uncertainty": float(u_vals.std()) if n > 0 else float("nan"),
        })

    # Overall stats
    correct = (yp == y_true)
    rows.append({
        "group": "Correct",
        "n": int(correct.sum()),
        "mean_uncertainty": float(uncertainty[correct].mean()),
        "median_uncertainty": float(np.median(uncertainty[correct])),
        "std_uncertainty": float(uncertainty[correct].std()),
    })
    rows.append({
        "group": "Incorrect",
        "n": int((~correct).sum()),
        "mean_uncertainty": float(uncertainty[~correct].mean()),
        "median_uncertainty": float(np.median(uncertainty[~correct])),
        "std_uncertainty": float(uncertainty[~correct].std()),
    })

    return pd.DataFrame(rows)


def plot_youden_confusion_matrices(df, edl_thr, glam_thr, output_dir):
    """Plot confusion matrices at Youden thresholds."""
    y_true = df["cancer"].values
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, label, col, thr in [
        (axes[0], f"EDL (thr={edl_thr:.3f})", "prob_1", edl_thr),
        (axes[1], f"GLAM (thr={glam_thr:.3f})", "glam_pred_score", glam_thr),
    ]:
        yp = (df[col].values >= thr).astype(int)
        cm = confusion_matrix(y_true, yp, labels=[0, 1])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(label, fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Neg (0)", "Pos (1)"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Neg (0)", "Pos (1)"])
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=color, fontsize=13)

    fig.suptitle("Confusion Matrices at Youden's Index Threshold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "youden_confusion_matrices.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("    Saved youden_confusion_matrices.png")


def write_youden_report(output_dir, youden_df, unc_youden_df):
    """Write Youden's Index summary table to a separate text file."""
    path = os.path.join(output_dir, "youden_index_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Youden's Index Analysis (Optimal Threshold)\n")
        f.write("=" * 70 + "\n\n")

        edl = youden_df.loc["EDL"]
        glam = youden_df.loc["GLAM"]

        f.write(f"  EDL  Youden threshold: {edl['threshold']:.4f}  |  Youden J = {edl['youden_j']:.6f}\n")
        f.write(f"  GLAM Youden threshold: {glam['threshold']:.4f}  |  Youden J = {glam['youden_j']:.6f}\n\n")

        f.write("Metric Comparison at Youden's Threshold\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Metric':<18s} {'EDL':>10s} {'95% CI':>22s} {'GLAM':>10s} {'95% CI':>22s}\n")
        f.write(f"  {'':-<18s} {'':-<10s} {'':-<22s} {'':-<10s} {'':-<22s}\n")

        for key, label in [("auc_roc", "AUC-ROC"), ("bacc", "BACC"),
                           ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                           ("ppv", "PPV"), ("npv", "NPV")]:
            def fmt_ci(row, k):
                lo, hi = row.get(f"{k}_ci_lo", float("nan")), row.get(f"{k}_ci_hi", float("nan"))
                if np.isnan(lo) or np.isnan(hi):
                    return ""
                return f"[{lo:.4f}, {hi:.4f}]"
            f.write(f"  {label:<18s} {edl[key]:>10.4f} {fmt_ci(edl, key):>22s} "
                    f"{glam[key]:>10.4f} {fmt_ci(glam, key):>22s}\n")

        f.write(f"\n  {'TP':<18s} {int(edl['tp']):>10d} {'':>22s} {int(glam['tp']):>10d} {'':>22s}\n")
        f.write(f"  {'FP':<18s} {int(edl['fp']):>10d} {'':>22s} {int(glam['fp']):>10d} {'':>22s}\n")
        f.write(f"  {'FN':<18s} {int(edl['fn']):>10d} {'':>22s} {int(glam['fn']):>10d} {'':>22s}\n")
        f.write(f"  {'TN':<18s} {int(edl['tn']):>10d} {'':>22s} {int(glam['tn']):>10d} {'':>22s}\n")

        f.write("\n\nEDL Uncertainty at Youden Threshold\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Group':<12s} {'N':>6s} {'Mean':>10s} {'Median':>10s} {'Std':>10s}\n")
        f.write(f"  {'':-<12s} {'':-<6s} {'':-<10s} {'':-<10s} {'':-<10s}\n")
        for _, r in unc_youden_df.iterrows():
            f.write(f"  {r['group']:<12s} {int(r['n']):>6d} {r['mean_uncertainty']:>10.6f} "
                    f"{r['median_uncertainty']:>10.6f} {r['std_uncertainty']:>10.6f}\n")

        f.write("\n")

    print(f"    Saved youden_index_report.txt")
    return path


def compute_evidence_analysis(df):
    pos = df[df["cancer"] == 1]
    neg = df[df["cancer"] == 0]

    def stats(series):
        return {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
        }

    rows = []
    for grp_name, grp_df in [("positive", pos), ("negative", neg)]:
        for col in ["evidence_0", "evidence_1", "total_evidence", "dirichlet_strength", "uncertainty"]:
            s = stats(grp_df[col])
            rows.append({"group": grp_name, "feature": col, **s})

    return pd.DataFrame(rows)


def compute_uncertainty_quartiles(df):
    y_true = df["cancer"].values
    y_score = df["prob_1"].values
    uncertainty = df["uncertainty"].values
    y_pred = (y_score >= 0.5).astype(int)

    quartile_edges = np.quantile(uncertainty, [0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    for i in range(4):
        lo, hi = quartile_edges[i], quartile_edges[i + 1]
        if i < 3:
            mask = (uncertainty >= lo) & (uncertainty < hi)
        else:
            mask = (uncertainty >= lo) & (uncertainty <= hi)

        n = mask.sum()
        if n == 0:
            rows.append({"quartile": i + 1, "n": 0})
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        acc = float(accuracy_score(yt, yp))
        bacc = float(balanced_accuracy_score(yt, yp))
        f1 = float(f1_score(yt, yp, zero_division=0))
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

        rows.append({
            "quartile": i + 1,
            "uncertainty_lo": float(lo),
            "uncertainty_hi": float(hi),
            "n": int(n),
            "n_pos": int(yt.sum()),
            "n_neg": int((1 - yt).sum()),
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1": f1,
            "sensitivity": sens,
            "specificity": spec,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------

def compute_calibration(y_true, y_score, n_bins=10):
    ece, bins = compute_ece(y_true, y_score, n_bins)
    brier = compute_brier_score(y_true, y_score)
    return ece, brier, bins


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

COLOR_EDL = "#2196F3"
COLOR_GLAM = "#FF5722"
COLOR_POS = "#E91E63"
COLOR_NEG = "#4CAF50"
COLOR_CORRECT = "#2196F3"
COLOR_INCORRECT = "#F44336"


def plot_roc_curves(df, edl_ci, glam_ci, output_dir):
    y_true = df["cancer"].values
    fig, ax = plt.subplots(figsize=(7, 6))

    for label, col, color, ci in [("EDL", "prob_1", COLOR_EDL, edl_ci), ("GLAM", "glam_pred_score", COLOR_GLAM, glam_ci)]:
        fpr, tpr, _ = roc_curve(y_true, df[col].values)
        auc_val = safe_roc_auc(y_true, df[col].values)
        ci_lo = ci.get("roc_auc_ci_lo", float("nan"))
        ci_hi = ci.get("roc_auc_ci_hi", float("nan"))
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label} (AUROC = {auc_val:.4f}, 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves: EDL vs GLAM (Fold-0 Validation)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=200)
    plt.close(fig)
    print("    Saved roc_curves.png")


def plot_pr_curves(df, edl_ci, glam_ci, output_dir):
    y_true = df["cancer"].values
    prevalence = y_true.mean()
    fig, ax = plt.subplots(figsize=(7, 6))

    for label, col, color, ci in [("EDL", "prob_1", COLOR_EDL, edl_ci), ("GLAM", "glam_pred_score", COLOR_GLAM, glam_ci)]:
        precision, recall, _ = precision_recall_curve(y_true, df[col].values)
        pr_auc = safe_pr_auc(y_true, df[col].values)
        ci_lo = ci.get("pr_auc_ci_lo", float("nan"))
        ci_hi = ci.get("pr_auc_ci_hi", float("nan"))
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{label} (AUPRC = {pr_auc:.4f}, 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")

    ax.axhline(prevalence, color="gray", ls="--", lw=1, alpha=0.7, label=f"Prevalence = {prevalence:.4f}")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("PR Curves: EDL vs GLAM (Fold-0 Validation)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=200)
    plt.close(fig)
    print("    Saved pr_curves.png")


def plot_score_distributions(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, label, col, color in [
        (axes[0], "EDL prob_1", "prob_1", COLOR_EDL),
        (axes[1], "GLAM pred_score", "glam_pred_score", COLOR_GLAM),
    ]:
        pos_scores = df.loc[df["cancer"] == 1, col].values
        neg_scores = df.loc[df["cancer"] == 0, col].values
        bp = ax.boxplot(
            [neg_scores, pos_scores],
            tick_labels=["Negative (0)", "Positive (1)"],
            patch_artist=True,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor(COLOR_NEG)
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(COLOR_POS)
        bp["boxes"][1].set_alpha(0.6)
        ax.set_title(label, fontsize=12)
        ax.set_ylabel("Score", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Score Distributions by True Class", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("    Saved score_distributions.png")


def plot_uncertainty_distributions(df, output_dir):
    uncertainty = df["uncertainty"].values
    correct = df["correct_edl"].values.astype(bool)
    pos = df["cancer"].values.astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.linspace(uncertainty.min(), uncertainty.max(), 50)

    ax = axes[0]
    ax.hist(uncertainty[correct], bins=bins, alpha=0.6, color=COLOR_CORRECT, label=f"Correct (n={correct.sum()})", density=True)
    ax.hist(uncertainty[~correct], bins=bins, alpha=0.6, color=COLOR_INCORRECT, label=f"Incorrect (n={(~correct).sum()})", density=True)
    ax.set_xlabel("Uncertainty", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Uncertainty: Correct vs Incorrect", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(uncertainty[pos], bins=bins, alpha=0.6, color=COLOR_POS, label=f"Positive (n={pos.sum()})", density=True)
    ax.hist(uncertainty[~pos], bins=bins, alpha=0.6, color=COLOR_NEG, label=f"Negative (n={(~pos).sum()})", density=True)
    ax.set_xlabel("Uncertainty", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Uncertainty: Positive vs Negative", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "uncertainty_distributions.png"), dpi=200)
    plt.close(fig)
    print("    Saved uncertainty_distributions.png")


def plot_calibration(df, edl_bins, glam_bins, output_dir):
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    for label, bins_data, color in [("EDL", edl_bins, COLOR_EDL), ("GLAM", glam_bins, COLOR_GLAM)]:
        preds = [b["mean_predicted"] for b in bins_data if b["count"] > 0]
        obs = [b["mean_observed"] for b in bins_data if b["count"] > 0]
        ax1.plot(preds, obs, "o-", color=color, lw=2, markersize=6, label=label)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax1.set_ylabel("Fraction of Positives", fontsize=11)
    ax1.set_title("Calibration Plot (Reliability Diagram)", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    width = 0.035
    for i, (label, bins_data, color) in enumerate([("EDL", edl_bins, COLOR_EDL), ("GLAM", glam_bins, COLOR_GLAM)]):
        centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins_data if b["count"] > 0]
        counts = [b["count"] for b in bins_data if b["count"] > 0]
        offset = (i - 0.5) * width
        ax2.bar([c + offset for c in centers], counts, width=width, color=color, alpha=0.6, label=label)
    ax2.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.legend(fontsize=10)

    fig.savefig(os.path.join(output_dir, "calibration_plots.png"), dpi=200)
    plt.close(fig)
    print("    Saved calibration_plots.png")


def plot_confusion_matrices(df, output_dir):
    y_true = df["cancer"].values
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, label, col in [(axes[0], "EDL", "prob_1"), (axes[1], "GLAM", "glam_pred_score")]:
        y_pred = (df[col].values >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{label} (thr=0.5)", fontsize=12)
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(["Neg (0)", "Pos (1)"])
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(["Neg (0)", "Pos (1)"])
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=color, fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=200)
    plt.close(fig)
    print("    Saved confusion_matrices.png")


def plot_selective_prediction(sel_df, output_dir):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(sel_df["coverage"], sel_df["accuracy"], "o-", color=COLOR_EDL, lw=2, markersize=4, label="Accuracy")
    ax1.plot(sel_df["coverage"], sel_df["balanced_accuracy"], "s-", color=COLOR_POS, lw=2, markersize=4, label="Balanced Accuracy")
    ax1.plot(sel_df["coverage"], sel_df["f1"], "^-", color="#FF9800", lw=2, markersize=4, label="F1")
    ax1.set_xlabel("Coverage (fraction of samples kept)", fontsize=11)
    ax1.set_ylabel("Metric Value", fontsize=11)
    ax1.set_title("Selective Prediction (reject most uncertain)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "selective_prediction_curve.png"), dpi=200)
    plt.close(fig)
    print("    Saved selective_prediction_curve.png")


def plot_evidence_scatter(df, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))

    neg = df[df["cancer"] == 0]
    pos = df[df["cancer"] == 1]

    ax.scatter(neg["evidence_0"].values, neg["evidence_1"].values,
               s=3, alpha=0.3, color=COLOR_NEG, label=f"Negative (n={len(neg)})")
    ax.scatter(pos["evidence_0"].values, pos["evidence_1"].values,
               s=15, alpha=0.7, color=COLOR_POS, label=f"Positive (n={len(pos)})")

    lims = [0, max(df["evidence_0"].max(), df["evidence_1"].max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
    ax.set_xlabel("Evidence (class 0)", fontsize=11)
    ax.set_ylabel("Evidence (class 1)", fontsize=11)
    ax.set_title("Evidence Distribution: class 0 vs class 1", fontsize=13)
    ax.legend(fontsize=10, markerscale=3)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "evidence_scatter.png"), dpi=200)
    plt.close(fig)
    print("    Saved evidence_scatter.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(output_dir, edl_m, glam_m, cal_edl, cal_glam, unc_df, evi_df, uq_df):
    report_path = os.path.join(output_dir, "validation_comprehensive_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Comprehensive Validation Analysis: EDL vs GLAM (Fold-0)\n")
        f.write("=" * 70 + "\n\n")

        f.write("Data Summary\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Validation rows:  {edl_m['n']}\n")
        f.write(f"  Positives:        {edl_m['pos']}\n")
        f.write(f"  Negatives:        {edl_m['neg']}\n")
        f.write(f"  Positive rate:    {edl_m['pos_rate']:.6f}\n\n")

        f.write("Standard Binary Metrics (threshold = 0.5)\n")
        f.write("-" * 40 + "\n")
        headers = ["Metric", "EDL", "95% CI", "GLAM", "95% CI"]
        f.write(f"  {headers[0]:<20s} {headers[1]:>10s} {headers[2]:>20s} {headers[3]:>10s} {headers[4]:>20s}\n")
        f.write(f"  {'':-<20s} {'':-<10s} {'':-<20s} {'':-<10s} {'':-<20s}\n")
        for key in ["n", "pos"]:
            f.write(f"  {key:<20s} {fmt(edl_m[key]):>10s} {'':>20s} {fmt(glam_m[key]):>10s} {'':>20s}\n")
        for key in ["accuracy", "balanced_accuracy", "f1", "precision",
                     "roc_auc", "pr_auc", "sensitivity", "specificity"]:
            ci_lo = edl_m.get(f"{key}_ci_lo", float("nan"))
            ci_hi = edl_m.get(f"{key}_ci_hi", float("nan"))
            edl_ci = f"[{fmt(ci_lo)}, {fmt(ci_hi)}]" if not (np.isnan(ci_lo) or np.isnan(ci_hi)) else ""
            ci_lo_g = glam_m.get(f"{key}_ci_lo", float("nan"))
            ci_hi_g = glam_m.get(f"{key}_ci_hi", float("nan"))
            glam_ci = f"[{fmt(ci_lo_g)}, {fmt(ci_hi_g)}]" if not (np.isnan(ci_lo_g) or np.isnan(ci_hi_g)) else ""
            f.write(f"  {key:<20s} {fmt(edl_m[key]):>10s} {edl_ci:>20s} {fmt(glam_m[key]):>10s} {glam_ci:>20s}\n")
        for key in ["tn", "fp", "fn", "tp"]:
            f.write(f"  {key:<20s} {fmt(edl_m[key]):>10s} {'':>20s} {fmt(glam_m[key]):>10s} {'':>20s}\n")

        f.write("\n  Best Threshold Search\n")
        f.write(f"  {'':-<20s} {'':-<10s} {'':-<20s} {'':-<10s} {'':-<20s}\n")
        for key in ["best_f1", "best_bacc", "best_youden"]:
            ci_lo = edl_m.get(f"{key}_ci_lo", float("nan"))
            ci_hi = edl_m.get(f"{key}_ci_hi", float("nan"))
            edl_ci = f"[{fmt(ci_lo)}, {fmt(ci_hi)}]" if not (np.isnan(ci_lo) or np.isnan(ci_hi)) else ""
            ci_lo_g = glam_m.get(f"{key}_ci_lo", float("nan"))
            ci_hi_g = glam_m.get(f"{key}_ci_hi", float("nan"))
            glam_ci = f"[{fmt(ci_lo_g)}, {fmt(ci_hi_g)}]" if not (np.isnan(ci_lo_g) or np.isnan(ci_hi_g)) else ""
            f.write(f"  {key:<20s} {fmt(edl_m[key]):>10s} {edl_ci:>20s} {fmt(glam_m[key]):>10s} {glam_ci:>20s}\n")
        for key in ["best_f1_threshold", "best_bacc_threshold", "best_youden_threshold"]:
            f.write(f"  {key:<20s} {fmt(edl_m[key]):>10s} {'':>20s} {fmt(glam_m[key]):>10s} {'':>20s}\n")

        f.write("\n  Mean Scores\n")
        for key in ["score_mean_pos", "score_mean_neg"]:
            f.write(f"  {key:<25s} {fmt(edl_m[key]):>12s} {fmt(glam_m[key]):>12s}\n")

        f.write("\n\nCalibration Metrics\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Metric':<20s} {'EDL':>10s} {'95% CI':>20s} {'GLAM':>10s} {'95% CI':>20s}\n")
        f.write(f"  {'':-<20s} {'':-<10s} {'':-<20s} {'':-<10s} {'':-<20s}\n")
        for label, key in [("ECE (10 bins)", "ece_10"), ("ECE (15 bins)", "ece_15")]:
            ece_edl = cal_edl[key]
            ece_glam = cal_glam[key]
            ci_lo_e = cal_edl.get(f"{key}_ci_lo", float("nan"))
            ci_hi_e = cal_edl.get(f"{key}_ci_hi", float("nan"))
            edl_ci = f"[{fmt(ci_lo_e)}, {fmt(ci_hi_e)}]" if not (np.isnan(ci_lo_e) or np.isnan(ci_hi_e)) else ""
            ci_lo_g = cal_glam.get(f"{key}_ci_lo", float("nan"))
            ci_hi_g = cal_glam.get(f"{key}_ci_hi", float("nan"))
            glam_ci = f"[{fmt(ci_lo_g)}, {fmt(ci_hi_g)}]" if not (np.isnan(ci_lo_g) or np.isnan(ci_hi_g)) else ""
            f.write(f"  {label:<20s} {fmt(ece_edl):>10s} {edl_ci:>20s} {fmt(ece_glam):>10s} {glam_ci:>20s}\n")
        for label, key in [("Brier Score", "brier")]:
            ci_lo_e = cal_edl.get(f"{key}_ci_lo", float("nan"))
            ci_hi_e = cal_edl.get(f"{key}_ci_hi", float("nan"))
            edl_ci = f"[{fmt(ci_lo_e)}, {fmt(ci_hi_e)}]" if not (np.isnan(ci_lo_e) or np.isnan(ci_hi_e)) else ""
            ci_lo_g = cal_glam.get(f"{key}_ci_lo", float("nan"))
            ci_hi_g = cal_glam.get(f"{key}_ci_hi", float("nan"))
            glam_ci = f"[{fmt(ci_lo_g)}, {fmt(ci_hi_g)}]" if not (np.isnan(ci_lo_g) or np.isnan(ci_hi_g)) else ""
            f.write(f"  {label:<20s} {fmt(cal_edl[key]):>10s} {edl_ci:>20s} {fmt(cal_glam[key]):>10s} {glam_ci:>20s}\n")

        f.write("\n\nEDL Uncertainty vs Correctness\n")
        f.write("-" * 40 + "\n")
        row = unc_df.iloc[0]
        f.write(f"  Correct predictions:       {int(row['n_correct'])}\n")
        f.write(f"  Incorrect predictions:     {int(row['n_incorrect'])}\n")
        f.write(f"  Mean uncertainty (correct):   {row['uncertainty_mean_correct']:.6f}\n")
        f.write(f"  Mean uncertainty (incorrect): {row['uncertainty_mean_incorrect']:.6f}\n")
        f.write(f"  Median uncertainty (correct):   {row['uncertainty_median_correct']:.6f}\n")
        f.write(f"  Median uncertainty (incorrect): {row['uncertainty_median_incorrect']:.6f}\n")
        f.write(f"  Mann-Whitney U statistic:   {row['mann_whitney_U']:.2f}\n")
        f.write(f"  Mann-Whitney p-value:       {row['mann_whitney_p']:.6e}\n")
        f.write(f"  AUROC (uncertainty -> misclass): {row['uncertainty_auroc_misclass']:.6f}\n")
        u_ci_lo = row.get('uncertainty_auroc_misclass_ci_lo', float('nan'))
        u_ci_hi = row.get('uncertainty_auroc_misclass_ci_hi', float('nan'))
        if not (np.isnan(u_ci_lo) or np.isnan(u_ci_hi)):
            f.write(f"    95% CI: [{u_ci_lo:.6f}, {u_ci_hi:.6f}]\n")
        f.write(f"  Mean uncertainty (positive): {row['uncertainty_mean_pos']:.6f}\n")
        f.write(f"  Mean uncertainty (negative): {row['uncertainty_mean_neg']:.6f}\n")

        f.write("\n\nUncertainty Quartiles\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Q':<4s} {'Range':<20s} {'N':>6s} {'Acc':>8s} {'BACC':>8s} {'F1':>8s} {'Sens':>8s} {'Spec':>8s}\n")
        for _, r in uq_df.iterrows():
            if r["n"] == 0:
                continue
            rng = f"[{r['uncertainty_lo']:.4f}, {r['uncertainty_hi']:.4f}]"
            f.write(f"  {int(r['quartile']):<4d} {rng:<20s} {int(r['n']):>6d} {r['accuracy']:>8.4f} "
                    f"{r['balanced_accuracy']:>8.4f} {r['f1']:>8.4f} {r['sensitivity']:>8.4f} {r['specificity']:>8.4f}\n")

        f.write("\n\nFiles Generated\n")
        f.write("-" * 40 + "\n")
        for fname in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, fname)
            size_kb = os.path.getsize(fpath) / 1024
            f.write(f"  {fname:<45s} ({size_kb:.1f} KB)\n")

        f.write("\n")

    print(f"    Saved validation_comprehensive_report.txt")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load & align
    df = load_and_align_data()
    df.to_csv(os.path.join(OUTPUT_DIR, "merged_validation_data.csv"), index=False)
    print(f"### [INFO] Saved merged_validation_data.csv ({len(df)} rows)")

    # Step 2: Binary metrics
    print("### [INFO] Computing binary metrics...")
    edl_m, glam_m = compute_all_binary_metrics(df)
    metrics_df = pd.DataFrame([edl_m, glam_m], index=["EDL", "GLAM"])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "binary_metrics.csv"))
    print("    Saved binary_metrics.csv")

    # Step 3: EDL-specific
    print("### [INFO] Computing EDL-specific analysis...")
    unc_df = compute_uncertainty_analysis(df)
    unc_df.to_csv(os.path.join(OUTPUT_DIR, "uncertainty_vs_correctness.csv"), index=False)
    print("    Saved uncertainty_vs_correctness.csv")

    sel_df = compute_selective_prediction(df)
    sel_df.to_csv(os.path.join(OUTPUT_DIR, "selective_prediction.csv"), index=False)
    print("    Saved selective_prediction.csv")

    evi_df = compute_evidence_analysis(df)
    evi_df.to_csv(os.path.join(OUTPUT_DIR, "evidence_analysis.csv"), index=False)
    print("    Saved evidence_analysis.csv")

    uq_df = compute_uncertainty_quartiles(df)
    uq_df.to_csv(os.path.join(OUTPUT_DIR, "uncertainty_quartiles.csv"), index=False)
    print("    Saved uncertainty_quartiles.csv")

    # Step 4: Calibration
    print("### [INFO] Computing calibration...")
    y_true = df["cancer"].values

    ece_edl_10, brier_edl, bins_edl_10 = compute_calibration(y_true, df["prob_1"].values, 10)
    ece_glam_10, brier_glam, bins_glam_10 = compute_calibration(y_true, df["glam_pred_score"].values, 10)
    ece_edl_15, _, bins_edl_15 = compute_calibration(y_true, df["prob_1"].values, 15)
    ece_glam_15, _, bins_glam_15 = compute_calibration(y_true, df["glam_pred_score"].values, 15)

    # Bootstrap CI for calibration metrics (batch for each model)
    print("    Bootstrapping calibration CIs...")
    edl_cal_ci = bootstrap_ci_batch(y_true, df["prob_1"].values,
                                     {"ece_10": _metric_ece_10, "brier": _metric_brier})
    glam_cal_ci = bootstrap_ci_batch(y_true, df["glam_pred_score"].values,
                                      {"ece_10": _metric_ece_10, "brier": _metric_brier})

    cal_edl = {
        "ece_10": ece_edl_10, "ece_10_ci_lo": edl_cal_ci["ece_10"][1], "ece_10_ci_hi": edl_cal_ci["ece_10"][2],
        "ece_15": ece_edl_15,
        "brier": brier_edl, "brier_ci_lo": edl_cal_ci["brier"][1], "brier_ci_hi": edl_cal_ci["brier"][2],
    }
    cal_glam = {
        "ece_10": ece_glam_10, "ece_10_ci_lo": glam_cal_ci["ece_10"][1], "ece_10_ci_hi": glam_cal_ci["ece_10"][2],
        "ece_15": ece_glam_15,
        "brier": brier_glam, "brier_ci_lo": glam_cal_ci["brier"][1], "brier_ci_hi": glam_cal_ci["brier"][2],
    }

    cal_df = pd.DataFrame([cal_edl, cal_glam], index=["EDL", "GLAM"])
    cal_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_metrics.csv"))
    print("    Saved calibration_metrics.csv")

    pd.DataFrame(bins_edl_10).to_csv(os.path.join(OUTPUT_DIR, "calibration_bins_edl.csv"), index=False)
    pd.DataFrame(bins_glam_10).to_csv(os.path.join(OUTPUT_DIR, "calibration_bins_glam.csv"), index=False)
    print("    Saved calibration_bins_edl.csv, calibration_bins_glam.csv")

    # Step 5: Visualizations
    print("### [INFO] Generating visualizations...")
    plot_roc_curves(df, edl_m, glam_m, OUTPUT_DIR)
    plot_pr_curves(df, edl_m, glam_m, OUTPUT_DIR)
    plot_score_distributions(df, OUTPUT_DIR)
    plot_uncertainty_distributions(df, OUTPUT_DIR)
    plot_calibration(df, bins_edl_10, bins_glam_10, OUTPUT_DIR)
    plot_confusion_matrices(df, OUTPUT_DIR)
    plot_selective_prediction(sel_df, OUTPUT_DIR)
    plot_evidence_scatter(df, OUTPUT_DIR)

    # Step 5.5: Youden's Index table
    print("### [INFO] Computing Youden's Index table...")
    youden_df = compute_youden_table(df)
    youden_df.to_csv(os.path.join(OUTPUT_DIR, "youden_index_table.csv"))
    print("    Saved youden_index_table.csv")

    edl_thr = youden_df.loc["EDL", "threshold"]
    glam_thr = youden_df.loc["GLAM", "threshold"]

    unc_youden_df = compute_uncertainty_at_youden(df, edl_thr)
    unc_youden_df.to_csv(os.path.join(OUTPUT_DIR, "uncertainty_at_youden.csv"), index=False)
    print("    Saved uncertainty_at_youden.csv")

    plot_youden_confusion_matrices(df, edl_thr, glam_thr, OUTPUT_DIR)
    youden_report_path = write_youden_report(OUTPUT_DIR, youden_df, unc_youden_df)

    # Step 6: Report
    print("### [INFO] Generating report...")
    report_path = write_report(OUTPUT_DIR, edl_m, glam_m, cal_edl, cal_glam, unc_df, evi_df, uq_df)

    print(f"\n### [INFO] Analysis complete! Output: {OUTPUT_DIR}")
    print(f"### [INFO] Report: {report_path}")


if __name__ == "__main__":
    main()
