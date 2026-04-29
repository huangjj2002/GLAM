import argparse
import os
import pickle
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold


def normalize_path(path: str) -> str:
    return os.path.normpath(str(path))


def build_row_path(row, img_root, path_pattern, patient_col, image_col):
    pid = str(row[patient_col])
    iid = str(row[image_col])
    rel = path_pattern.format(pid=pid, iid=iid)
    return normalize_path(os.path.join(img_root, rel))


def load_fold_prediction(ckpt_path: str):
    ckpt_dir = os.path.dirname(ckpt_path)
    pred_dir = os.path.join(ckpt_dir, "predictions")
    score_path = os.path.join(pred_dir, "scores.npy")
    label_path = os.path.join(pred_dir, "labels.npy")
    paths_path = os.path.join(pred_dir, "path.pickle")
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"Missing prediction file: {score_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing prediction file: {label_path}")
    if not os.path.exists(paths_path):
        raise FileNotFoundError(f"Missing prediction file: {paths_path}")

    scores = np.load(score_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    with open(paths_path, "rb") as f:
        paths = pickle.load(f)

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    paths = [normalize_path(p) for p in list(paths)]

    if scores.ndim == 2 and scores.shape[1] > 1:
        pos_scores = scores[:, 1]
    else:
        pos_scores = scores.reshape(-1)

    if labels.ndim == 2 and labels.shape[1] > 1:
        binary_labels = labels.argmax(axis=1)
    else:
        binary_labels = labels.reshape(-1).astype(int)

    if not (len(paths) == len(pos_scores) == len(binary_labels)):
        raise RuntimeError(
            f"Length mismatch for ckpt={ckpt_path}: "
            f"len(paths)={len(paths)}, len(scores)={len(pos_scores)}, len(labels)={len(binary_labels)}"
        )
    return paths, pos_scores.astype(float), binary_labels.astype(int)


def align_scores_to_df(df_paths, pred_paths, pred_scores):
    if len(df_paths) == len(pred_paths) and np.all(df_paths == np.asarray(pred_paths)):
        return pred_scores

    path_to_idx = defaultdict(deque)
    for i, p in enumerate(pred_paths):
        path_to_idx[p].append(i)

    aligned = np.full(len(df_paths), np.nan, dtype=float)
    missing = 0
    for i, p in enumerate(df_paths):
        if not path_to_idx[p]:
            missing += 1
            continue
        pred_i = path_to_idx[p].popleft()
        aligned[i] = float(pred_scores[pred_i])

    if missing > 0 or np.isnan(aligned).any():
        raise RuntimeError(
            f"Failed to align predictions to input CSV rows. missing={missing}, "
            f"nan_count={int(np.isnan(aligned).sum())}"
        )
    return aligned


def safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_binary_metrics(y_true, y_score, threshold=0.5):
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

    return {
        "n": int(len(y_true)),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def split_metrics(df, y_true, y_score, split_col, threshold):
    overall = compute_binary_metrics(y_true, y_score, threshold)
    per_split = {}
    if split_col in df.columns:
        split_series = df[split_col].astype(str)
        for split_name in sorted(split_series.unique()):
            mask = split_series == split_name
            per_split[str(split_name)] = compute_binary_metrics(
                y_true[mask.to_numpy()],
                y_score[mask.to_numpy()],
                threshold,
            )
    else:
        per_split["all"] = overall
    return overall, per_split


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


def build_output_df(df, patient_col, image_col, split_col, label_col, cohort_col):
    output_cols = [patient_col, image_col]
    if split_col in df.columns:
        output_cols.append(split_col)
    output_cols.append(label_col)
    output_cols.append(cohort_col)
    return df[output_cols].copy()


def write_report(path, ckpt_paths, fold_metrics, ensemble_metrics):
    with open(path, "w", encoding="utf-8") as f:
        f.write("K-fold full-data evaluation report\n")
        f.write("==================================\n\n")
        f.write("Checkpoints:\n")
        for i, ckpt in enumerate(ckpt_paths):
            f.write(f"  fold_{i}: {ckpt}\n")

        f.write("\nPer-model metrics:\n")
        for fold_idx in sorted(fold_metrics.keys()):
            block = fold_metrics[fold_idx]
            f.write(f"\n  [fold_{fold_idx}]\n")
            f.write("    Overall:\n")
            for k, v in block["overall"].items():
                f.write(f"      {k}: {fmt(v)}\n")
            f.write("    Per-split:\n")
            for split_name, metrics in block["per_split"].items():
                f.write(f"      [{split_name}]\n")
                for k, v in metrics.items():
                    f.write(f"        {k}: {fmt(v)}\n")

        f.write("\nEnsemble metrics:\n")
        f.write("  Overall:\n")
        for k, v in ensemble_metrics["overall"].items():
            f.write(f"    {k}: {fmt(v)}\n")
        f.write("  Per-split:\n")
        for split_name, metrics in ensemble_metrics["per_split"].items():
            f.write(f"    [{split_name}]\n")
            for k, v in metrics.items():
                f.write(f"      {k}: {fmt(v)}\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--path_pattern", type=str, default="{pid}/{iid}")
    parser.add_argument("--patient_col", type=str, default="patient_id")
    parser.add_argument("--image_col", type=str, default="image_id")
    parser.add_argument("--label_col", type=str, default="cancer")
    parser.add_argument("--split_col", type=str, default="split")
    parser.add_argument("--cohort_col", type=str, default="cohert_num")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ckpt_paths", nargs="+", required=True)
    parser.add_argument("--output_report", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--per_model_output_dir", type=str, required=True)
    parser.add_argument("--test_split_value", type=str, default="test",
                        help="Value in split_col that identifies test samples (fold=-1)")
    parser.add_argument("--train_split_value", type=str, default="training",
                        help="Value in split_col that identifies training samples")
    parser.add_argument("--k_fold", type=int, default=5,
                        help="Number of folds (must match training)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(args.per_model_output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    required_cols = [args.patient_col, args.image_col, args.label_col]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in input CSV: {args.csv_path}")

    cohort_col = args.cohort_col
    if cohort_col not in df.columns:
        if cohort_col == "cohert_num" and "cohort_num" in df.columns:
            cohort_col = "cohort_num"
        else:
            raise KeyError(
                f"Cohort column '{args.cohort_col}' not found in input CSV. "
                f"Available columns: {list(df.columns)}"
            )

    df = df.copy()
    df["_path"] = df.apply(
        lambda row: build_row_path(
            row,
            args.img_root,
            args.path_pattern,
            args.patient_col,
            args.image_col,
        ),
        axis=1,
    )
    df_paths = df["_path"].to_numpy()
    y_true_csv = df[args.label_col].astype(int).to_numpy()

    # Identify test-set rows so we can mark their fold as -1
    is_test = None
    if args.split_col in df.columns:
        is_test = df[args.split_col].astype(str) == args.test_split_value

    # Assign each training sample to its fold (matching the KFold split used in training)
    # Test samples will get fold = -1
    fold_assignment = pd.Series(-1, index=df.index)
    if args.split_col in df.columns and args.k_fold > 1:
        is_train = df[args.split_col].astype(str) == args.train_split_value
        train_df = df.loc[is_train]
        uniq_pids = train_df[args.patient_col].astype(str).unique()
        kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=42)
        splits = list(kf.split(uniq_pids))
        for fold_i, (_, val_idx) in enumerate(splits):
            val_pids = set(uniq_pids[val_idx])
            mask = is_train & df[args.patient_col].astype(str).isin(val_pids)
            fold_assignment.loc[mask] = fold_i

    fold_scores = []
    fold_metrics = {}
    ref_labels = None

    base_out_df = build_output_df(
        df,
        args.patient_col,
        args.image_col,
        args.split_col,
        args.label_col,
        cohort_col,
    )

    for fold_idx, ckpt in enumerate(args.ckpt_paths):
        pred_paths, pred_scores, pred_labels = load_fold_prediction(ckpt)
        aligned_scores = align_scores_to_df(df_paths, pred_paths, pred_scores)
        aligned_labels = align_scores_to_df(df_paths, pred_paths, pred_labels).astype(int)

        if ref_labels is None:
            ref_labels = aligned_labels
        elif not np.array_equal(ref_labels, aligned_labels):
            raise RuntimeError(
                f"Label mismatch detected in predictions for fold={fold_idx}. "
                "Please ensure all fold evaluations used the same full dataset."
            )

        fold_scores.append(aligned_scores)
        fold_overall, fold_per_split = split_metrics(
            df,
            y_true_csv,
            aligned_scores,
            args.split_col,
            args.threshold,
        )
        fold_metrics[fold_idx] = {"overall": fold_overall, "per_split": fold_per_split}

        fold_pred_class = (aligned_scores >= args.threshold).astype(int)
        fold_out = base_out_df.copy()
        fold_out["fold"] = fold_assignment
        fold_out["pred_score"] = aligned_scores
        fold_out["pred_class"] = fold_pred_class
        fold_csv_path = os.path.join(
            args.per_model_output_dir, f"fold{fold_idx}_predictions.csv"
        )
        fold_out.to_csv(fold_csv_path, index=False)
        print(f"[INFO] Fold-{fold_idx} prediction CSV saved to: {fold_csv_path}")

    if ref_labels is not None and not np.array_equal(y_true_csv, ref_labels):
        print(
            "[WARN] Labels from input CSV and saved prediction labels differ; using labels from input CSV for metric reporting."
        )

    score_mat = np.stack(fold_scores, axis=1)
    ensemble_score = score_mat.mean(axis=1)
    ensemble_pred_class = (ensemble_score >= args.threshold).astype(int)

    ensemble_overall, ensemble_per_split = split_metrics(
        df,
        y_true_csv,
        ensemble_score,
        args.split_col,
        args.threshold,
    )
    ensemble_metrics = {"overall": ensemble_overall, "per_split": ensemble_per_split}

    ensemble_out = base_out_df.copy()
    ensemble_out["fold"] = fold_assignment
    ensemble_out["pred_score"] = ensemble_score
    ensemble_out["pred_class"] = ensemble_pred_class
    ensemble_out.to_csv(args.output_csv, index=False)

    write_report(args.output_report, args.ckpt_paths, fold_metrics, ensemble_metrics)

    print(f"[INFO] Report saved to: {args.output_report}")
    print(f"[INFO] Ensemble prediction CSV saved to: {args.output_csv}")
    print(f"[INFO] Per-model prediction CSV dir: {args.per_model_output_dir}")


if __name__ == "__main__":
    main()
