#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EDL Training & Evaluation Script
=================================

Three-stage pipeline (mirrors run.sh):
  Stage 1 — K-fold training with EDL model
  Stage 2 — Full-data inference with each fold model (outputs evidence, uncertainty)
  Stage 3 — Aggregate CSVs with fold, evidence, alpha, prob, uncertainty columns

Usage:
  python edl_train.py \
    --pretrained_model pretrain_model/last.ckpt \
    --csv_path data.csv \
    --img_root /path/to/images \
    --path_pattern '{pid}/{iid}' \
    --k_fold 5 \
    --max_epochs 25

Original project files are NOT modified.
"""

import argparse
import datetime
import os
import sys
import pickle
from collections import Counter, defaultdict, deque
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from dataset.data_module import DataModule
from dataset.mammo_eval_dataset import RSNAMammo
from dataset.pretrain_embed_dataset import multimodal_collate_fn
from dataset.transforms import RSNAMammoTransform

from edl_model import EDLModel
from edl_loss import assert_kl_sanity
from split_utils import build_output_split_series, build_split_metadata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MinEpochEarlyStopping(EarlyStopping):
    def __init__(self, min_epochs=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_epochs = int(min_epochs or 0)

    def _run_early_stopping_check(self, trainer):
        if self.min_epochs > 0 and (trainer.current_epoch + 1) < self.min_epochs:
            return
        return super()._run_early_stopping_check(trainer)


class MinEpochModelCheckpoint(ModelCheckpoint):
    def __init__(self, min_epochs=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_epochs = int(min_epochs or 0)

    def _should_skip_saving_checkpoint(self, trainer):
        if super()._should_skip_saving_checkpoint(trainer):
            return True
        return self.min_epochs > 0 and (trainer.current_epoch + 1) < self.min_epochs


# ===========================================================================
# Utility helpers
# ===========================================================================

def normalize_path(path: str) -> str:
    return os.path.normpath(str(path))


def build_row_path(row, img_root, path_pattern, patient_col, image_col):
    pid = str(row[patient_col])
    iid = str(row[image_col])
    rel = path_pattern.format(pid=pid, iid=iid)
    return normalize_path(os.path.join(img_root, rel))


# ===========================================================================
# Stage 1: K-fold training
# ===========================================================================

def train_one_fold(args_dict, fold, ckpt_dir):
    """Train EDL model for a single fold and return the best checkpoint path."""
    args_dict = dict(args_dict)
    args_dict["fold"] = fold

    print(f"\n{'='*60}")
    print(f"  Stage 1: Training fold {fold}")
    print(f"{'='*60}\n")

    datamodule = _build_datamodule(args_dict)
    _maybe_set_auto_pos_weight(args_dict, datamodule)
    model = EDLModel(**args_dict)

    monitor = args_dict.get("monitor_metric", "val_AUROC")
    mode = "max"
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        MinEpochModelCheckpoint(
            monitor=monitor, dirpath=ckpt_dir,
            save_last=True, mode=mode, save_top_k=2,
            min_epochs=args_dict.get("early_stop_min_epochs", 0),
        ),
    ]
    if args_dict.get("early_stop", False):
        callbacks.append(MinEpochEarlyStopping(
            monitor=monitor,
            patience=args_dict.get("early_stop_patience", 5),
            min_delta=args_dict.get("early_stop_min_delta", 0.0),
            mode=mode, verbose=True,
            min_epochs=args_dict.get("early_stop_min_epochs", 0),
        ))
        print(
            f"### [EDL] Early stopping enabled: monitor={monitor}, mode={mode}, "
            f"patience={args_dict.get('early_stop_patience', 5)}, "
            f"min_epochs={args_dict.get('early_stop_min_epochs', 0)}"
        )

    logger_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(logger_dir, exist_ok=True)
    extension = args_dict.get("experiment_name", f"edl_fold{fold}")
    wandb_logger = WandbLogger(project="GLAM_EDL", save_dir=logger_dir, name=extension)

    trainer = Trainer(
        accelerator="gpu",
        strategy=args_dict.get("strategy", "auto"),
        devices=args_dict.get("devices", 1),
        precision=args_dict.get("precision", "32"),
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=args_dict.get("max_epochs", 25),
        deterministic=False,
        accumulate_grad_batches=args_dict.get("accumulate_grad_batches", 1),
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)

    best_ckpt_yaml = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_yaml)

    best_ckpt_path = callbacks[1].best_model_path
    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        best_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    print(f"### [EDL] Fold {fold} best ckpt: {best_ckpt_path}")
    return best_ckpt_path


# ===========================================================================
# Stage 2: Full-data inference
# ===========================================================================

def test_full_data(args_dict, fold, ckpt_path):
    """Load a trained EDL model and run inference on ALL data."""
    print(f"\n{'='*60}")
    print(f"  Stage 2: Full-data inference fold {fold}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}\n")

    test_args = dict(args_dict)
    test_args["k_fold"] = 0
    test_args["fold"] = 0
    test_args["rsna_use_all_data"] = True

    datamodule = _build_datamodule(test_args, split="test")
    model = EDLModel.load_from_checkpoint(
        ckpt_path, map_location="cpu", strict=False, **test_args
    )
    model.eval()

    trainer = Trainer(
        accelerator="gpu",
        precision=test_args.get("precision", "32"),
        devices=1, max_epochs=1,
        deterministic=False, inference_mode=True,
        enable_progress_bar=True,
    )

    model.all_scores = None
    model.all_labels = None
    model.all_evidence = None
    model.all_probs = None
    model.all_uncertainty = None
    model.img_path = []

    trainer.test(model, datamodule=datamodule)

    result = {
        "paths": model.img_path,
        "evidence": model.all_evidence,
        "probs": model.all_probs,
        "scores": model.all_scores,
        "labels": model.all_labels,
        "uncertainty": model.all_uncertainty,
    }
    print(f"### [EDL] Fold {fold} inference done. Samples: {len(result['paths'])}")
    return result


# ===========================================================================
# Stage 3: CSV generation
# ===========================================================================

def generate_csvs(
    df, img_root, path_pattern, patient_col, image_col, label_col,
    split_col, cohort_col, split_source, train_split_value, test_split_value,
    train_cohorts, test_cohorts, output_split_col,
    val_split, val_fraction, val_max_fraction, val_min_positive_patients, val_random_state,
    k_fold, fold_results, fold_ids, output_dir,
):
    """Generate per-fold and ensemble CSVs with evidence, alpha, prob, uncertainty."""
    print(f"\n{'='*60}")
    print(f"  Stage 3: Generating CSVs")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()
    df["_path"] = df.apply(
        lambda row: build_row_path(row, img_root, path_pattern, patient_col, image_col), axis=1,
    )
    df_paths = df["_path"].to_numpy()
    y_true_csv = df[label_col].astype(int).to_numpy()

    split_metadata = build_split_metadata(
        df,
        patient_col=patient_col,
        label_col=label_col,
        split_source=split_source,
        split_col=split_col,
        train_split_value=train_split_value,
        test_split_value=test_split_value,
        cohort_col=cohort_col,
        train_cohorts=train_cohorts,
        test_cohorts=test_cohorts,
        k_fold=k_fold,
        val_split=val_split,
        val_fraction=val_fraction,
        val_max_fraction=val_max_fraction,
        val_min_positive_patients=val_min_positive_patients,
        val_random_state=val_random_state,
    )
    fold_assignment = split_metadata["fold_assignment"]
    base_split = split_metadata["base_split"]
    actual_cohort_col = split_metadata["actual_cohort_col"]

    base_cols = [patient_col, image_col, label_col]
    if actual_cohort_col is not None:
        base_cols.append(actual_cohort_col)

    fold_prob_arrays = []

    for result_idx, result in enumerate(fold_results):
        fold_idx = fold_ids[result_idx]
        pred_paths = [normalize_path(p) for p in result["paths"]]
        pred_evidence = result["evidence"]
        pred_probs = result["probs"]
        pred_uncertainty = result["uncertainty"]
        pred_labels = result["labels"]

        aligned = _align_to_df(df_paths, pred_paths, pred_evidence, pred_probs, pred_uncertainty, pred_labels)
        fold_prob_arrays.append(aligned["probs"])

        fold_out = df[base_cols].copy()
        current_fold = fold_idx if k_fold > 1 else None
        fold_out[output_split_col] = build_output_split_series(
            base_split,
            fold_assignment,
            current_fold=current_fold,
        ).values
        fold_out["fold"] = fold_assignment.values
        fold_out["evidence_0"] = aligned["evidence"][:, 0]
        fold_out["evidence_1"] = aligned["evidence"][:, 1]
        fold_out["alpha_0"] = aligned["evidence"][:, 0] + 1.0
        fold_out["alpha_1"] = aligned["evidence"][:, 1] + 1.0
        fold_out["prob_0"] = aligned["probs"][:, 0]
        fold_out["prob_1"] = aligned["probs"][:, 1]
        fold_out["uncertainty"] = aligned["uncertainty"]
        fold_out["pred_class"] = aligned["probs"].argmax(axis=-1)
        fold_out["label"] = y_true_csv

        fold_csv_path = os.path.join(output_dir, f"fold{fold_idx}_edl_predictions.csv")
        fold_out.to_csv(fold_csv_path, index=False)
        print(f"[INFO] Fold-{fold_idx} EDL prediction CSV saved to: {fold_csv_path}")

        pos_scores = aligned["probs"][:, 1]
        idx_pred = (pos_scores >= 0.5).astype(int)
        try:
            fold_auc = roc_auc_score(y_true_csv, pos_scores)
            fold_acc = accuracy_score(y_true_csv, idx_pred)
            fold_bacc = balanced_accuracy_score(y_true_csv, idx_pred)
            fold_f1 = f1_score(y_true_csv, idx_pred)
            print(f"  Fold-{fold_idx}: AUC={fold_auc:.4f}, ACC={fold_acc:.4f}, BACC={fold_bacc:.4f}, F1={fold_f1:.4f}")
        except Exception as e:
            print(f"  Fold-{fold_idx}: Metric computation failed: {e}")

    # Ensemble
    ensemble_probs = np.stack(fold_prob_arrays, axis=0).mean(axis=0)

    ensemble_out = df[base_cols].copy()
    ensemble_out[output_split_col] = base_split.values
    ensemble_out["fold"] = fold_assignment.values
    ensemble_out["prob_0"] = ensemble_probs[:, 0]
    ensemble_out["prob_1"] = ensemble_probs[:, 1]
    ensemble_out["pred_class"] = ensemble_probs.argmax(axis=-1)
    ensemble_out["label"] = y_true_csv

    # Ensemble uncertainty & evidence (average across folds)
    fold_unc_arrays = []
    fold_ev_arrays = []
    for result_idx, result in enumerate(fold_results):
        fold_idx = fold_ids[result_idx]
        pred_paths = [normalize_path(p) for p in result["paths"]]
        path_to_idx = defaultdict(deque)
        for i, p in enumerate(pred_paths):
            path_to_idx[p].append(i)
        aligned_unc = np.full(len(df_paths), np.nan, dtype=float)
        aligned_ev = np.full((len(df_paths), result["evidence"].shape[-1]), np.nan, dtype=float)
        for i, p in enumerate(df_paths):
            if path_to_idx[p]:
                j = path_to_idx[p].popleft()
                aligned_unc[i] = float(result["uncertainty"][j])
                aligned_ev[i] = result["evidence"][j]
        fold_unc_arrays.append(aligned_unc)
        fold_ev_arrays.append(aligned_ev)

    ensemble_uncertainty = np.stack(fold_unc_arrays, axis=0).mean(axis=0)
    ensemble_evidence = np.stack(fold_ev_arrays, axis=0).mean(axis=0)

    ensemble_out["uncertainty"] = ensemble_uncertainty
    ensemble_out["evidence_0"] = ensemble_evidence[:, 0]
    ensemble_out["evidence_1"] = ensemble_evidence[:, 1]
    ensemble_out["alpha_0"] = ensemble_evidence[:, 0] + 1.0
    ensemble_out["alpha_1"] = ensemble_evidence[:, 1] + 1.0

    ensemble_csv_path = os.path.join(output_dir, "ensemble_edl_predictions.csv")
    ensemble_out.to_csv(ensemble_csv_path, index=False)
    print(f"\n[INFO] Ensemble EDL prediction CSV saved to: {ensemble_csv_path}")

    pos_scores = ensemble_probs[:, 1]
    idx_pred = (pos_scores >= 0.5).astype(int)
    try:
        ens_auc = roc_auc_score(y_true_csv, pos_scores)
        ens_acc = accuracy_score(y_true_csv, idx_pred)
        ens_bacc = balanced_accuracy_score(y_true_csv, idx_pred)
        ens_f1 = f1_score(y_true_csv, idx_pred)
        print(f"  Ensemble: AUC={ens_auc:.4f}, ACC={ens_acc:.4f}, BACC={ens_bacc:.4f}, F1={ens_f1:.4f}")
    except Exception as e:
        print(f"  Ensemble: Metric computation failed: {e}")

    # Report
    report_path = os.path.join(output_dir, "edl_eval_report.txt")
    _write_report(
        report_path,
        fold_results,
        ensemble_probs,
        y_true_csv,
        fold_ids=fold_ids,
        ensemble_uncertainty=ensemble_uncertainty,
        ensemble_evidence=ensemble_evidence,
    )
    print(f"[INFO] Report saved to: {report_path}")

    return ensemble_csv_path


def _align_to_df(df_paths, pred_paths, evidence, probs, uncertainty, labels):
    """Align prediction arrays to the dataframe path order."""
    if len(df_paths) == len(pred_paths) and np.all(df_paths == np.asarray(pred_paths)):
        return {
            "evidence": np.asarray(evidence),
            "probs": np.asarray(probs),
            "uncertainty": np.asarray(uncertainty),
            "labels": np.asarray(labels),
        }

    path_to_idx = defaultdict(deque)
    for i, p in enumerate(pred_paths):
        path_to_idx[p].append(i)

    n = len(df_paths)
    ev = np.asarray(evidence)
    pr = np.asarray(probs)
    unc = np.asarray(uncertainty)
    lb = np.asarray(labels)

    aligned_evidence = np.full((n, ev.shape[-1]), np.nan, dtype=float)
    aligned_probs = np.full((n, pr.shape[-1]), np.nan, dtype=float)
    aligned_unc = np.full(n, np.nan, dtype=float)
    aligned_labels = np.full((n, lb.shape[-1]), np.nan, dtype=float)

    missing = 0
    for i, p in enumerate(df_paths):
        if not path_to_idx[p]:
            missing += 1
            continue
        j = path_to_idx[p].popleft()
        aligned_evidence[i] = ev[j]
        aligned_probs[i] = pr[j]
        aligned_unc[i] = unc[j]
        aligned_labels[i] = lb[j]

    if missing > 0:
        raise RuntimeError(f"Alignment failed: {missing} paths not found in predictions.")
    return {
        "evidence": aligned_evidence,
        "probs": aligned_probs,
        "uncertainty": aligned_unc,
        "labels": aligned_labels,
    }


def _write_distribution_diagnostics(f, probs, uncertainty=None, evidence=None, prefix=""):
    pos_scores = np.asarray(probs)[:, 1]
    prob_q = np.nanquantile(pos_scores, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    f.write(f"  {prefix}prob_1 quantiles [0,1,5,50,95,99,100]%: ")
    f.write(", ".join(f"{v:.6f}" for v in prob_q) + "\n")
    if uncertainty is not None:
        unc = np.asarray(uncertainty, dtype=float)
        unc_q = np.nanquantile(unc, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
        f.write(f"  {prefix}uncertainty quantiles [0,1,5,50,95,99,100]%: ")
        f.write(", ".join(f"{v:.6f}" for v in unc_q) + "\n")
    if evidence is not None:
        ev = np.asarray(evidence, dtype=float)
        strength = ev.sum(axis=1) + ev.shape[1]
        strength_q = np.nanquantile(strength, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
        f.write(f"  {prefix}sum(alpha) quantiles [0,1,5,50,95,99,100]%: ")
        f.write(", ".join(f"{v:.6f}" for v in strength_q) + "\n")
    if float(np.nanmax(pos_scores) - np.nanmin(pos_scores)) < 1e-3:
        f.write(f"  {prefix}WARNING: prob_1 range is very narrow; predictions may be collapsed.\n")
    if uncertainty is not None:
        unc = np.asarray(uncertainty, dtype=float)
        if float(np.nanmax(unc) - np.nanmin(unc)) < 1e-3:
            f.write(f"  {prefix}WARNING: uncertainty range is very narrow; uncertainty may be collapsed.\n")


def _write_report(
    path,
    fold_results,
    ensemble_probs,
    y_true,
    fold_ids=None,
    ensemble_uncertainty=None,
    ensemble_evidence=None,
):
    """Write a text report with per-fold and ensemble metrics."""
    if fold_ids is None:
        fold_ids = list(range(len(fold_results)))

    with open(path, "w", encoding="utf-8") as f:
        f.write("EDL K-fold Full-data Evaluation Report\n")
        f.write("=" * 40 + "\n\n")

        for result_idx, result in enumerate(fold_results):
            fold_idx = fold_ids[result_idx]
            probs = result["probs"]
            pos_scores = probs[:, 1]
            idx_pred = (pos_scores >= 0.5).astype(int)
            try:
                auc = roc_auc_score(y_true, pos_scores)
                acc = accuracy_score(y_true, idx_pred)
                bacc = balanced_accuracy_score(y_true, idx_pred)
                f1 = f1_score(y_true, idx_pred)
            except Exception:
                auc = acc = bacc = f1 = float("nan")

            f.write(f"[fold_{fold_idx}]\n")
            f.write(f"  AUC:      {auc:.6f}\n")
            f.write(f"  Accuracy: {acc:.6f}\n")
            f.write(f"  BACC:     {bacc:.6f}\n")
            f.write(f"  F1:       {f1:.6f}\n")
            f.write(f"  Mean uncertainty: {result['uncertainty'].mean():.6f}\n")
            _write_distribution_diagnostics(
                f,
                probs,
                uncertainty=result.get("uncertainty"),
                evidence=result.get("evidence"),
            )
            f.write("\n")

        ens_pos = ensemble_probs[:, 1]
        ens_pred = (ens_pos >= 0.5).astype(int)
        try:
            ens_auc = roc_auc_score(y_true, ens_pos)
            ens_acc = accuracy_score(y_true, ens_pred)
            ens_bacc = balanced_accuracy_score(y_true, ens_pred)
            ens_f1 = f1_score(y_true, ens_pred)
        except Exception:
            ens_auc = ens_acc = ens_bacc = ens_f1 = float("nan")

        f.write("[Ensemble]\n")
        f.write(f"  AUC:      {ens_auc:.6f}\n")
        f.write(f"  Accuracy: {ens_acc:.6f}\n")
        f.write(f"  BACC:     {ens_bacc:.6f}\n")
        f.write(f"  F1:       {ens_f1:.6f}\n")
        _write_distribution_diagnostics(
            f,
            ensemble_probs,
            uncertainty=ensemble_uncertainty,
            evidence=ensemble_evidence,
        )
        f.write("\n")


# ===========================================================================
# DataModule builder
# ===========================================================================

def _build_datamodule(args_dict, split=None):
    """Build DataModule from args dict."""
    datamodule = DataModule(
        RSNAMammo,
        multimodal_collate_fn,
        RSNAMammoTransform,
        data_pct=1.0,
        batch_size=args_dict.get("batch_size", 32),
        num_workers=args_dict.get("num_workers", 0),
        img_size=args_dict.get("img_size", 336),
        crop_size=args_dict.get("crop_size", 336),
        llm_type="bert",
        train_split="train",
        valid_split="valid",
        test_split="test",
        k_fold=args_dict.get("k_fold", 0),
        fold=args_dict.get("fold", 0),
        rsna_csv_path=args_dict.get("rsna_csv_path", None),
        rsna_img_root=args_dict.get("rsna_img_root", None),
        rsna_path_pattern=args_dict.get("rsna_path_pattern", None),
        rsna_patient_col=args_dict.get("rsna_patient_col", "patient_id"),
        rsna_image_col=args_dict.get("rsna_image_col", "image_id"),
        rsna_label_col=args_dict.get("rsna_label_col", "cancer"),
        rsna_split_col=args_dict.get("rsna_split_col", "split"),
        rsna_train_split_value=args_dict.get("rsna_train_split_value", "training"),
        rsna_test_split_value=args_dict.get("rsna_test_split_value", "test"),
        rsna_split_source=args_dict.get("split_source", "cohort"),
        rsna_cohort_col=args_dict.get("cohort_col", "cohert_num"),
        rsna_train_cohorts=args_dict.get("train_cohorts", "1-8"),
        rsna_test_cohorts=args_dict.get("test_cohorts", "9-10"),
        rsna_val_split=args_dict.get("rsna_val_split", False),
        rsna_val_fraction=args_dict.get("rsna_val_fraction", 0.15),
        rsna_val_max_fraction=args_dict.get("rsna_val_max_fraction", 0.25),
        rsna_val_min_positive_patients=args_dict.get("rsna_val_min_positive_patients", 3),
        rsna_val_random_state=args_dict.get("rsna_val_random_state", 42),
        rsna_use_all_data=args_dict.get("rsna_use_all_data", False),
        balance_training=args_dict.get("balance_training", False),
        pred_only=False,
    )
    return datamodule


def _maybe_set_auto_pos_weight(args_dict, datamodule):
    """Auto-compute positive-class weight as neg/pos for the current fold."""
    if args_dict.get("pos_weight", None) is not None:
        print(f"### [EDL] Using configured pos_weight={float(args_dict['pos_weight']):.6f}")
        return

    try:
        train_ds = datamodule.train_dataloader().dataset
        labels = np.asarray(getattr(train_ds, "labels", []))
        if labels.size == 0:
            raise RuntimeError("train dataset has no labels")

        num_pos = int((labels == 1).sum())
        num_neg = int((labels == 0).sum())
        if num_pos == 0:
            raise RuntimeError("no positive samples in training split")

        args_dict["pos_weight"] = float(num_neg / num_pos)
        print(
            f"### [EDL] Auto pos_weight (neg/pos) for fold={args_dict.get('fold', 0)}: "
            f"neg={num_neg}, pos={num_pos}, pos_weight={args_dict['pos_weight']:.6f}"
        )
    except Exception as exc:
        args_dict["pos_weight"] = None
        print(f"### [EDL][Warning] Failed to auto-compute pos_weight: {exc}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="EDL Training & Evaluation")

    # Required paths
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--path_pattern", type=str, default="{pid}/{iid}")
    parser.add_argument("--pretrained_model", type=str, required=True)

    # Column names in CSV
    parser.add_argument("--patient_col", type=str, default="patient_id")
    parser.add_argument("--image_col", type=str, default="image_id")
    parser.add_argument("--label_col", type=str, default="cancer")
    parser.add_argument("--split_col", type=str, default="split")
    parser.add_argument("--split_source", type=str, default="cohort", choices=["split", "cohort"])
    parser.add_argument("--cohort_col", type=str, default="cohert_num")
    parser.add_argument("--train_split_value", type=str, default="training")
    parser.add_argument("--test_split_value", type=str, default="test")
    parser.add_argument("--train_cohorts", type=str, default="1-8")
    parser.add_argument("--test_cohorts", type=str, default="9-10")
    parser.add_argument("--output_split_col", type=str, default="split")
    parser.add_argument("--rsna_val_split", action="store_true")
    parser.add_argument("--rsna_val_fraction", type=float, default=0.15)
    parser.add_argument("--rsna_val_max_fraction", type=float, default=0.25)
    parser.add_argument("--rsna_val_min_positive_patients", type=int, default=3)
    parser.add_argument("--rsna_val_random_state", type=int, default=42)

    # Model
    parser.add_argument("--img_encoder", type=str, default="dinov2_vitb14_reg")
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--linear_proj", action="store_true")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--evidence_type", type=str, default="softplus", choices=["relu", "exp", "softplus"])
    parser.add_argument("--edl_loss_type", type=str, default="digamma", choices=["digamma", "log", "mse"])
    parser.add_argument("--annealing_step", type=int, default=10)
    parser.add_argument("--lambda_kl", type=float, default=0.1)
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Positive-class weight for EDL binary loss. If omitted, auto-computed as neg/pos per fold.",
    )
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze the GLAM backbone and train only the EDL head.")

    # Training
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument(
        "--folds_to_run",
        type=str,
        default=None,
        help=(
            "Comma-separated fold indices to train/evaluate while keeping "
            "--k_fold as the split definition, e.g. '0' or '0,2,4'. "
            "Default: run all folds."
        ),
    )
    parser.add_argument("--max_epochs", type=int, default=25)
    parser.add_argument("--warm_up", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--balance_training", action="store_true")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Early stopping
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--early_stop_min_epochs", type=int, default=0)
    parser.add_argument("--monitor_metric", type=str, default="val_AUROC")

    # Image size
    parser.add_argument("--img_size", type=int, default=336)
    parser.add_argument("--crop_size", type=int, default=336)

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="edl_kfold")
    parser.add_argument("--seed", type=int, default=42)

    # pretrained_encoder (for GLAM backbone init)
    parser.add_argument("--pretrained_encoder", type=str, default=None)

    return parser.parse_args()


def _parse_folds_to_run(folds_to_run, k_fold):
    """Parse and validate a comma-separated fold list."""
    if k_fold == 0:
        if folds_to_run is None or str(folds_to_run).strip() in ["", "0"]:
            return [0]
        if str(folds_to_run).strip() != "0":
            raise ValueError("--folds_to_run must be '0' when --k_fold is 0.")
        return [0]
    if k_fold < 2:
        raise ValueError("--k_fold must be 0 or >= 2 for EDL training.")

    if folds_to_run is None or str(folds_to_run).strip() == "":
        return list(range(k_fold))

    folds = []
    seen = set()
    for raw_item in str(folds_to_run).split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            fold = int(item)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --folds_to_run value '{item}'. Use comma-separated integers."
            ) from exc
        if not 0 <= fold < k_fold:
            raise ValueError(f"Fold index {fold} is outside valid range [0, {k_fold - 1}].")
        if fold not in seen:
            folds.append(fold)
            seen.add(fold)

    if not folds:
        raise ValueError("--folds_to_run did not contain any valid fold index.")
    return folds


def main():
    args = parse_args()

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    assert_kl_sanity()
    print("### [EDL] KL sanity check passed: Dir(1)||Dir(1)=0 and non-uniform KL>0")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        output_dir = os.path.join(BASE_DIR, "outputs", f"edl_{args.experiment_name}_{timestamp}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_dir = os.path.join(output_dir, "logs")
    per_model_dir = os.path.join(output_dir, "per_model_predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(per_model_dir, exist_ok=True)

    # Print config
    print(f"\n{'='*60}")
    print(f"  EDL Training Configuration")
    print(f"{'='*60}")
    print(f"  CSV:                  {args.csv_path}")
    print(f"  IMG_ROOT:             {args.img_root}")
    print(f"  PATH_PATTERN:         {args.path_pattern}")
    print(f"  PRETRAINED_MODEL:     {args.pretrained_model}")
    print(f"  K_FOLD:               {args.k_fold}")
    folds_to_run = _parse_folds_to_run(args.folds_to_run, args.k_fold)
    print(f"  FOLDS_TO_RUN:         {','.join(str(f) for f in folds_to_run)}")
    print(f"  MAX_EPOCHS:           {args.max_epochs}")
    print(f"  BATCH_SIZE:           {args.batch_size}")
    print(f"  LEARNING_RATE:        {args.learning_rate}")
    print(f"  EDL_LOSS_TYPE:        {args.edl_loss_type}")
    print(f"  EVIDENCE_TYPE:        {args.evidence_type}")
    print(f"  FREEZE_BACKBONE:      {args.freeze_backbone}")
    print(f"  ANNEALING_STEP:       {args.annealing_step}")
    print(f"  LAMBDA_KL:            {args.lambda_kl}")
    print(f"  POS_WEIGHT:           {args.pos_weight if args.pos_weight is not None else 'auto per fold'}")
    print(f"  MONITOR_METRIC:       {args.monitor_metric}")
    print(f"  EARLY_STOP_MIN_EPOCHS:{args.early_stop_min_epochs}")
    print(f"  VAL_SPLIT:            {args.rsna_val_split}")
    print(f"  VAL_FRACTION:         {args.rsna_val_fraction}")
    print(f"  VAL_MAX_FRACTION:     {args.rsna_val_max_fraction}")
    print(f"  VAL_MIN_POS_PATIENTS: {args.rsna_val_min_positive_patients}")
    print(f"  OUTPUT_DIR:           {output_dir}")
    print(f"{'='*60}\n")

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    if not os.path.isdir(args.img_root):
        raise FileNotFoundError(f"IMG_ROOT not found: {args.img_root}")
    if not os.path.exists(args.pretrained_model):
        raise FileNotFoundError(f"Pretrained model not found: {args.pretrained_model}")

    args_dict = vars(args)
    args_dict["rsna_mammo"] = True
    args_dict["img_cls_ft"] = True
    args_dict["llm_type"] = "bert"
    args_dict["weighted_binary"] = True
    args_dict["train_by_epoch"] = True
    args_dict["rsna_csv_path"] = args.csv_path
    args_dict["rsna_img_root"] = args.img_root
    args_dict["rsna_path_pattern"] = args.path_pattern
    args_dict["rsna_patient_col"] = args.patient_col
    args_dict["rsna_image_col"] = args.image_col
    args_dict["rsna_label_col"] = args.label_col
    args_dict["rsna_split_col"] = args.split_col
    args_dict["rsna_train_split_value"] = args.train_split_value
    args_dict["rsna_test_split_value"] = args.test_split_value
    args_dict["split_source"] = args.split_source
    args_dict["cohort_col"] = args.cohort_col
    args_dict["train_cohorts"] = args.train_cohorts
    args_dict["test_cohorts"] = args.test_cohorts
    args_dict["rsna_val_split"] = args.rsna_val_split
    args_dict["rsna_val_fraction"] = args.rsna_val_fraction
    args_dict["rsna_val_max_fraction"] = args.rsna_val_max_fraction
    args_dict["rsna_val_min_positive_patients"] = args.rsna_val_min_positive_patients
    args_dict["rsna_val_random_state"] = args.rsna_val_random_state

    # =========================================================================
    # Stage 1: K-fold training
    # =========================================================================
    print(f"\n{'#'*60}")
    print(f"  Stage 1/3: {args.k_fold}-fold EDL training")
    print(f"  Running folds: {','.join(str(f) for f in folds_to_run)}")
    print(f"{'#'*60}\n")

    best_ckpts = {}
    for fold in folds_to_run:
        fold_ckpt_dir = os.path.join(results_dir, f"fold{fold}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)
        args_dict["experiment_name"] = f"{args.experiment_name}_fold{fold}"
        best_ckpt = train_one_fold(args_dict, fold, fold_ckpt_dir)
        best_ckpts[fold] = best_ckpt

    # =========================================================================
    # Stage 2: Full-data inference
    # =========================================================================
    print(f"\n{'#'*60}")
    print(f"  Stage 2/3: Full-data inference with {len(folds_to_run)} model(s)")
    print(f"{'#'*60}\n")

    fold_results = []
    for fold in folds_to_run:
        args_dict["experiment_name"] = f"{args.experiment_name}_fold{fold}"
        result = test_full_data(args_dict, fold, best_ckpts[fold])
        fold_results.append(result)

    # =========================================================================
    # Stage 3: Generate CSVs
    # =========================================================================
    df = pd.read_csv(args.csv_path)
    generate_csvs(
        df=df, img_root=args.img_root, path_pattern=args.path_pattern,
        patient_col=args.patient_col, image_col=args.image_col,
        label_col=args.label_col, split_col=args.split_col,
        cohort_col=args.cohort_col, split_source=args.split_source,
        train_split_value=args.train_split_value,
        test_split_value=args.test_split_value,
        train_cohorts=args.train_cohorts,
        test_cohorts=args.test_cohorts,
        output_split_col=args.output_split_col,
        val_split=args.rsna_val_split,
        val_fraction=args.rsna_val_fraction,
        val_max_fraction=args.rsna_val_max_fraction,
        val_min_positive_patients=args.rsna_val_min_positive_patients,
        val_random_state=args.rsna_val_random_state,
        k_fold=args.k_fold, fold_results=fold_results,
        fold_ids=folds_to_run,
        output_dir=per_model_dir,
    )

    print(f"\n{'='*60}")
    print(f"  All done!")
    print(f"  Output dir: {output_dir}")
    print(f"  Per-model CSVs: {per_model_dir}/")
    print(f"  Ensemble CSV: {per_model_dir}/ensemble_edl_predictions.csv")
    print(f"  Report: {per_model_dir}/edl_eval_report.txt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
