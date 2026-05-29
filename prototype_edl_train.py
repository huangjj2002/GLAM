#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prototype + EDL Training & Evaluation Script.

This is a third, independent entrypoint next to train.py and edl_train.py. It
reuses the EDL data pipeline, split logic, losses, metrics, and CSV generation,
but initializes and trains a class-wise prototype EDL head.
"""

import argparse
import datetime
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from dataset.mammo_eval_dataset import RSNAMammo
from dataset.pretrain_embed_dataset import multimodal_collate_fn
from dataset.transforms import RSNAMammoTransform
from edl_loss import assert_kl_sanity
from edl_train import (
    MinEpochEarlyStopping,
    MinEpochModelCheckpoint,
    TextProgressPrinter,
    _build_datamodule,
    _maybe_set_auto_pos_weight,
    _parse_folds_to_run,
    build_row_path,
    generate_csvs as generate_edl_csvs,
    normalize_path,
)
from prototype_edl_model import PrototypeEDLModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_one_fold(args_dict, fold, ckpt_dir):
    """Train Prototype + EDL model for a single fold."""
    args_dict = dict(args_dict)
    args_dict["fold"] = fold

    print(f"\n{'='*60}")
    print(f"  Stage 1: Training Prototype EDL fold {fold}")
    print(f"{'='*60}\n")

    datamodule = _build_datamodule(args_dict)
    _maybe_set_auto_pos_weight(args_dict, datamodule)
    model = PrototypeEDLModel(**args_dict)

    if args_dict.get("prototype_init", "kmeans") == "kmeans":
        init_loader = _build_prototype_init_loader(args_dict)
        init_device = torch.device(
            "cuda:0"
            if torch.cuda.is_available() and int(args_dict.get("devices", 1) or 0) > 0
            else "cpu"
        )
        print(
            "### [PrototypeEDL] Initializing prototypes from current-fold train "
            f"embeddings on {init_device}"
        )
        model.to(init_device)
        model.initialize_prototypes_from_loader(
            init_loader,
            device=init_device,
            random_state=int(args_dict.get("seed", 42)) + int(fold),
        )
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("### [PrototypeEDL] Using random prototype initialization")

    monitor = args_dict.get("monitor_metric", "val_AUROC")
    mode = "max"
    checkpoint_callback = MinEpochModelCheckpoint(
        monitor=monitor,
        dirpath=ckpt_dir,
        save_last=True,
        mode=mode,
        save_top_k=2,
        min_epochs=args_dict.get("early_stop_min_epochs", 0),
    )
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        checkpoint_callback,
    ]
    if not args_dict.get("no_progress_bar", False):
        callbacks.append(TextProgressPrinter(args_dict.get("progress_print_interval", 50)))
    if args_dict.get("early_stop", False):
        callbacks.append(
            MinEpochEarlyStopping(
                monitor=monitor,
                patience=args_dict.get("early_stop_patience", 5),
                min_delta=args_dict.get("early_stop_min_delta", 0.0),
                mode=mode,
                verbose=True,
                min_epochs=args_dict.get("early_stop_min_epochs", 0),
            )
        )
        print(
            f"### [PrototypeEDL] Early stopping enabled: monitor={monitor}, mode={mode}, "
            f"patience={args_dict.get('early_stop_patience', 5)}, "
            f"min_epochs={args_dict.get('early_stop_min_epochs', 0)}"
        )

    logger_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(logger_dir, exist_ok=True)
    extension = args_dict.get("experiment_name", f"prototype_edl_fold{fold}")
    wandb_logger = WandbLogger(project="GLAM_PROTOTYPE_EDL", save_dir=logger_dir, name=extension)

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
        enable_progress_bar=not args_dict.get("no_progress_bar", False),
    )

    trainer.fit(model, datamodule=datamodule)

    best_ckpt_yaml = os.path.join(ckpt_dir, "best_ckpts.yaml")
    checkpoint_callback.to_yaml(filepath=best_ckpt_yaml)

    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        best_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    print(f"### [PrototypeEDL] Fold {fold} best ckpt: {best_ckpt_path}")
    return best_ckpt_path


def test_full_data(args_dict, fold, ckpt_path):
    """Load a trained Prototype EDL model and run inference on all data."""
    print(f"\n{'='*60}")
    print(f"  Stage 2: Full-data Prototype EDL inference fold {fold}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}\n")

    test_args = dict(args_dict)
    test_args["k_fold"] = 0
    test_args["fold"] = 0
    test_args["rsna_use_all_data"] = True

    datamodule = _build_datamodule(test_args, split="test")
    model = PrototypeEDLModel.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=False,
        **test_args,
    )
    model.eval()

    trainer = Trainer(
        accelerator="gpu",
        precision=test_args.get("precision", "32"),
        devices=1,
        max_epochs=1,
        deterministic=False,
        inference_mode=True,
        enable_progress_bar=not test_args.get("no_progress_bar", False),
    )

    model.all_scores = None
    model.all_labels = None
    model.all_evidence = None
    model.all_probs = None
    model.all_uncertainty = None
    model.all_proto_top_idx = None
    model.all_proto_top_distance = None
    model.all_proto_top_similarity = None
    model.all_proto_top_evidence = None
    model.img_path = []

    trainer.test(model, datamodule=datamodule)

    result = {
        "paths": model.img_path,
        "evidence": model.all_evidence,
        "probs": model.all_probs,
        "scores": model.all_scores,
        "labels": model.all_labels,
        "uncertainty": model.all_uncertainty,
        "proto_top_idx": model.all_proto_top_idx,
        "proto_top_distance": model.all_proto_top_distance,
        "proto_top_similarity": model.all_proto_top_similarity,
        "proto_top_evidence": model.all_proto_top_evidence,
    }
    print(f"### [PrototypeEDL] Fold {fold} inference done. Samples: {len(result['paths'])}")
    return result


def generate_csvs_with_prototypes(
    df,
    img_root,
    path_pattern,
    patient_col,
    image_col,
    label_col,
    split_col,
    cohort_col,
    split_source,
    train_split_value,
    test_split_value,
    train_cohorts,
    test_cohorts,
    output_split_col,
    val_split,
    val_fraction,
    val_max_fraction,
    val_min_positive_patients,
    val_random_state,
    k_fold,
    fold_results,
    fold_ids,
    output_dir,
):
    """Generate EDL-compatible CSVs and append per-fold prototype explanation columns."""
    ensemble_csv = generate_edl_csvs(
        df=df,
        img_root=img_root,
        path_pattern=path_pattern,
        patient_col=patient_col,
        image_col=image_col,
        label_col=label_col,
        split_col=split_col,
        cohort_col=cohort_col,
        split_source=split_source,
        train_split_value=train_split_value,
        test_split_value=test_split_value,
        train_cohorts=train_cohorts,
        test_cohorts=test_cohorts,
        output_split_col=output_split_col,
        val_split=val_split,
        val_fraction=val_fraction,
        val_max_fraction=val_max_fraction,
        val_min_positive_patients=val_min_positive_patients,
        val_random_state=val_random_state,
        k_fold=k_fold,
        fold_results=fold_results,
        fold_ids=fold_ids,
        output_dir=output_dir,
    )

    _append_prototype_columns_to_fold_csvs(
        df=df,
        img_root=img_root,
        path_pattern=path_pattern,
        patient_col=patient_col,
        image_col=image_col,
        fold_results=fold_results,
        fold_ids=fold_ids,
        output_dir=output_dir,
    )
    return ensemble_csv


def _append_prototype_columns_to_fold_csvs(
    df,
    img_root,
    path_pattern,
    patient_col,
    image_col,
    fold_results,
    fold_ids,
    output_dir,
):
    df = df.copy()
    df["_path"] = df.apply(
        lambda row: build_row_path(row, img_root, path_pattern, patient_col, image_col),
        axis=1,
    )
    df_paths = df["_path"].to_numpy()

    for result_idx, result in enumerate(fold_results):
        fold_idx = fold_ids[result_idx]
        fold_csv_path = os.path.join(output_dir, f"fold{fold_idx}_edl_predictions.csv")
        if not os.path.exists(fold_csv_path):
            print(f"### [PrototypeEDL][Warning] Missing fold CSV: {fold_csv_path}")
            continue

        proto_idx = result.get("proto_top_idx")
        proto_distance = result.get("proto_top_distance")
        proto_similarity = result.get("proto_top_similarity")
        proto_evidence = result.get("proto_top_evidence")
        if proto_idx is None:
            print(f"### [PrototypeEDL][Warning] No prototype outputs for fold {fold_idx}")
            continue

        pred_paths = [normalize_path(p) for p in result["paths"]]
        aligned_idx = _align_proto_array(df_paths, pred_paths, proto_idx)
        aligned_distance = _align_proto_array(df_paths, pred_paths, proto_distance)
        aligned_similarity = _align_proto_array(df_paths, pred_paths, proto_similarity)
        aligned_evidence = _align_proto_array(df_paths, pred_paths, proto_evidence)

        out_df = pd.read_csv(fold_csv_path)
        topk = aligned_idx.shape[-1]
        num_classes = aligned_idx.shape[1]
        for class_idx in range(num_classes):
            for rank in range(topk):
                prefix = f"proto_c{class_idx}_top{rank + 1}"
                out_df[f"{prefix}_idx"] = aligned_idx[:, class_idx, rank].astype(int)
                out_df[f"{prefix}_distance"] = aligned_distance[:, class_idx, rank]
                out_df[f"{prefix}_similarity"] = aligned_similarity[:, class_idx, rank]
                out_df[f"{prefix}_evidence"] = aligned_evidence[:, class_idx, rank]

        out_df.to_csv(fold_csv_path, index=False)
        print(f"[INFO] Prototype columns appended to: {fold_csv_path}")


def _align_proto_array(df_paths, pred_paths, values):
    values = np.asarray(values)
    if len(df_paths) == len(pred_paths) and np.all(df_paths == np.asarray(pred_paths)):
        return values

    path_to_idx = defaultdict(deque)
    for i, path in enumerate(pred_paths):
        path_to_idx[path].append(i)

    aligned_shape = (len(df_paths),) + values.shape[1:]
    aligned = np.empty(aligned_shape, dtype=values.dtype)
    if np.issubdtype(values.dtype, np.floating):
        aligned.fill(np.nan)
    else:
        aligned.fill(-1)
    missing = 0
    for i, path in enumerate(df_paths):
        if not path_to_idx[path]:
            missing += 1
            continue
        pred_i = path_to_idx[path].popleft()
        aligned[i] = values[pred_i]

    if missing > 0:
        raise RuntimeError(f"Prototype alignment failed: {missing} paths not found.")
    return aligned


def _build_prototype_init_loader(args_dict):
    """Build a deterministic train-split loader for KMeans prototype initialization."""
    transform = RSNAMammoTransform(
        False,
        args_dict.get("img_size", 336),
        args_dict.get("crop_size", 336),
    )
    dataset = RSNAMammo(
        split="train",
        transform=transform,
        data_pct=1.0,
        llm_type="bert",
        k_fold=args_dict.get("k_fold", 0),
        fold=args_dict.get("fold", 0),
        csv_path=args_dict.get("rsna_csv_path", None),
        img_root=args_dict.get("rsna_img_root", None),
        path_pattern=args_dict.get("rsna_path_pattern", None),
        patient_col=args_dict.get("rsna_patient_col", "patient_id"),
        image_col=args_dict.get("rsna_image_col", "image_id"),
        label_col=args_dict.get("rsna_label_col", "cancer"),
        split_col=args_dict.get("rsna_split_col", "split"),
        train_split_value=args_dict.get("rsna_train_split_value", "training"),
        test_split_value=args_dict.get("rsna_test_split_value", "test"),
        split_source=args_dict.get("split_source", "cohort"),
        cohort_col=args_dict.get("cohort_col", "cohert_num"),
        train_cohorts=args_dict.get("train_cohorts", "1-8"),
        test_cohorts=args_dict.get("test_cohorts", "9-10"),
        val_split=args_dict.get("rsna_val_split", False),
        val_fraction=args_dict.get("rsna_val_fraction", 0.15),
        val_max_fraction=args_dict.get("rsna_val_max_fraction", 0.25),
        val_min_positive_patients=args_dict.get("rsna_val_min_positive_patients", 3),
        val_random_state=args_dict.get("rsna_val_random_state", 42),
        use_all_data=False,
    )
    batch_size = int(
        args_dict.get("prototype_init_batch_size", 0)
        or args_dict.get("batch_size", 32)
    )
    num_workers = int(
        args_dict.get("prototype_init_num_workers", 0)
        or args_dict.get("num_workers", 0)
    )
    return DataLoader(
        dataset,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Prototype + EDL Training & Evaluation")

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--path_pattern", type=str, default="{pid}/{iid}")
    parser.add_argument("--pretrained_model", type=str, required=True)

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

    parser.add_argument("--img_encoder", type=str, default="dinov2_vitb14_reg")
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--linear_proj", action="store_true")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--evidence_type", type=str, default="softplus", choices=["relu", "exp", "softplus"])
    parser.add_argument("--edl_loss_type", type=str, default="digamma", choices=["digamma", "log", "mse"])
    parser.add_argument("--annealing_step", type=int, default=10)
    parser.add_argument("--lambda_kl", type=float, default=0.1)
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--edl_proto_k", type=int, default=4)
    parser.add_argument("--edl_proto_topk", type=int, default=3)
    parser.add_argument("--prototype_temperature", type=float, default=1.0)
    parser.add_argument("--prototype_init", type=str, default="kmeans", choices=["kmeans", "random"])
    parser.add_argument("--no_prototype_normalize", action="store_true")
    parser.add_argument(
        "--prototype_init_max_samples_per_class",
        type=int,
        default=0,
        help="Optional cap for KMeans embeddings per class. 0 means use all current-fold train samples.",
    )
    parser.add_argument(
        "--prototype_init_batch_size",
        type=int,
        default=0,
        help="Batch size for prototype init embedding extraction. 0 reuses --batch_size.",
    )
    parser.add_argument(
        "--prototype_init_num_workers",
        type=int,
        default=0,
        help="Workers for prototype init embedding extraction. 0 reuses --num_workers.",
    )

    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--folds_to_run", type=str, default=None)
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

    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--early_stop_min_epochs", type=int, default=0)
    parser.add_argument("--monitor_metric", type=str, default="val_AUROC")
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument("--progress_print_interval", type=int, default=50)

    parser.add_argument("--img_size", type=int, default=336)
    parser.add_argument("--crop_size", type=int, default=336)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="prototype_edl_kfold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained_encoder", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.prototype_normalize = not args.no_prototype_normalize

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    assert_kl_sanity()
    print("### [PrototypeEDL] KL sanity check passed")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        output_dir = os.path.join(BASE_DIR, "outputs", f"prototype_edl_{args.experiment_name}_{timestamp}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_dir = os.path.join(output_dir, "logs")
    per_model_dir = os.path.join(output_dir, "per_model_predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(per_model_dir, exist_ok=True)

    folds_to_run = _parse_folds_to_run(args.folds_to_run, args.k_fold)

    print(f"\n{'='*60}")
    print("  Prototype EDL Training Configuration")
    print(f"{'='*60}")
    print(f"  CSV:                  {args.csv_path}")
    print(f"  IMG_ROOT:             {args.img_root}")
    print(f"  PATH_PATTERN:         {args.path_pattern}")
    print(f"  PRETRAINED_MODEL:     {args.pretrained_model}")
    print(f"  K_FOLD:               {args.k_fold}")
    print(f"  FOLDS_TO_RUN:         {','.join(str(f) for f in folds_to_run)}")
    print(f"  MAX_EPOCHS:           {args.max_epochs}")
    print(f"  BATCH_SIZE:           {args.batch_size}")
    print(f"  LEARNING_RATE:        {args.learning_rate}")
    print(f"  EDL_LOSS_TYPE:        {args.edl_loss_type}")
    print(f"  EVIDENCE_TYPE:        {args.evidence_type}")
    print(f"  FREEZE_BACKBONE:      {args.freeze_backbone}")
    print(f"  EDL_PROTO_K:          {args.edl_proto_k}")
    print(f"  EDL_PROTO_TOPK:       {args.edl_proto_topk}")
    print(f"  PROTOTYPE_INIT:       {args.prototype_init}")
    print(f"  PROTOTYPE_TEMP:       {args.prototype_temperature}")
    print(f"  PROTOTYPE_NORMALIZE:  {args.prototype_normalize}")
    print(f"  ANNEALING_STEP:       {args.annealing_step}")
    print(f"  LAMBDA_KL:            {args.lambda_kl}")
    print(f"  POS_WEIGHT:           {args.pos_weight if args.pos_weight is not None else 'auto per fold'}")
    print(f"  MONITOR_METRIC:       {args.monitor_metric}")
    print(f"  VAL_SPLIT:            {args.rsna_val_split}")
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

    print(f"\n{'#'*60}")
    print(f"  Stage 1/3: {args.k_fold}-fold Prototype EDL training")
    print(f"  Running folds: {','.join(str(f) for f in folds_to_run)}")
    print(f"{'#'*60}\n")

    best_ckpts = {}
    for fold in folds_to_run:
        fold_ckpt_dir = os.path.join(results_dir, f"fold{fold}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)
        args_dict["experiment_name"] = f"{args.experiment_name}_fold{fold}"
        best_ckpts[fold] = train_one_fold(args_dict, fold, fold_ckpt_dir)

    print(f"\n{'#'*60}")
    print(f"  Stage 2/3: Full-data inference with {len(folds_to_run)} model(s)")
    print(f"{'#'*60}\n")

    fold_results = []
    for fold in folds_to_run:
        args_dict["experiment_name"] = f"{args.experiment_name}_fold{fold}"
        fold_results.append(test_full_data(args_dict, fold, best_ckpts[fold]))

    print(f"\n{'#'*60}")
    print("  Stage 3/3: Generate Prototype EDL CSVs")
    print(f"{'#'*60}\n")

    df = pd.read_csv(args.csv_path)
    generate_csvs_with_prototypes(
        df=df,
        img_root=args.img_root,
        path_pattern=args.path_pattern,
        patient_col=args.patient_col,
        image_col=args.image_col,
        label_col=args.label_col,
        split_col=args.split_col,
        cohort_col=args.cohort_col,
        split_source=args.split_source,
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
        k_fold=args.k_fold,
        fold_results=fold_results,
        fold_ids=folds_to_run,
        output_dir=per_model_dir,
    )

    print(f"\n{'='*60}")
    print("  All done!")
    print(f"  Output dir: {output_dir}")
    print(f"  Per-model CSVs: {per_model_dir}/")
    print(f"  Ensemble CSV: {per_model_dir}/ensemble_edl_predictions.csv")
    print(f"  Report: {per_model_dir}/edl_eval_report.txt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
