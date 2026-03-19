import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything

from dataset.data_module import DataModule
from dataset.mammo_eval_dataset import RSNAMammo, VinDr
from dataset.pretrain_embed_dataset import EmbedPretrainingDataset, multimodal_collate_fn
from dataset.transforms import Moco2Transform, RSNAMammoTransform, SimCLRFixedTransform, SimCLRTransform
from model import GLAM


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export per-fold predictions to CSV for train/valid/test splits."
    )
    parser.add_argument("--base_exp_name", type=str, required=True)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold_indices", type=int, nargs="*", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--score_col", type=str, default="pred_score")
    parser.add_argument("--class_col", type=str, default="pred_class")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
    )
    parser.add_argument(
        "--ckpt_select",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Use best checkpoint from best_ckpts.yaml or last.ckpt.",
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default=None,
        help="Checkpoint root. Defaults to <repo>/logs/ckpts/GLAM.",
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--llm_type", type=str, default="gpt")
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--vindr", action="store_true")
    parser.add_argument("--rsna_mammo", action="store_true")
    parser = GLAM.add_model_specific_args(parser)
    return parser


def resolve_dataset(args):
    if args.embed:
        return EmbedPretrainingDataset
    if args.vindr:
        return VinDr
    if args.rsna_mammo:
        return RSNAMammo
    return EmbedPretrainingDataset


def resolve_transform(args):
    if args.slip:
        if args.patch_contrast or args.fixed_view:
            return SimCLRFixedTransform
        return SimCLRTransform
    if args.rsna_trans:
        return RSNAMammoTransform
    if args.patch_contrast or args.fixed_view:
        return SimCLRFixedTransform
    return Moco2Transform


def build_datamodule(args, dataset, transform_obj, split_name):
    return DataModule(
        dataset,
        multimodal_collate_fn,
        transform_obj,
        args.data_pct,
        args.batch_size,
        args.num_workers,
        llm_type=args.llm_type,
        train_split=args.train_split,
        valid_split=args.valid_split,
        test_split=split_name,
        train_sub_set=args.train_sub_set,
        structural_cap=args.structural_cap,
        simple_cap=args.simple_cap,
        natural_cap=args.natural_cap,
        instance_test_cap=args.instance_test_cap,
        inter_side=args.inter_side,
        inter_view=args.inter_view,
        balanced_test=args.balanced_test,
        slip=args.slip or getattr(args, "moco", False),
        balance_training=args.balance_training,
        pred_density=args.pred_density,
        img_size=args.img_size,
        crop_size=args.crop_size,
        raw_caption=args.raw_caption,
        load_jpg=args.load_jpg,
        mask_ratio=args.mask_ratio,
        mask_meta=args.mask_meta,
        aug_orig_img=args.aug_orig_img,
        balance_ratio=args.balance_ratio,
        less_train_neg=args.less_train_neg,
        test_data_pct=args.test_data_pct,
        bootstrap_test=args.bootstrap_test,
        aug_text=args.aug_text,
        heavy_aug=args.heavy_aug,
        pred_only=True,
        prob_diff_dcm=args.prob_diff_dcm,
        extra_cap=args.extra_cap,
        extract_train=args.extract_train,
        screen_only=args.screen_only,
        aligned_mlo=args.aligned_mlo,
        align_orientation=args.align_orientation,
        remove_text=args.remove_text,
        fixed_view=args.fixed_view,
        pred_mass=args.pred_mass,
        pred_calc=args.pred_calc,
        k_fold=args.k_fold,
        fold=args.fold,
        rsna_csv_path=args.rsna_csv_path,
        rsna_img_root=args.rsna_img_root,
        rsna_path_pattern=args.rsna_path_pattern,
        rsna_patient_col=args.rsna_patient_col,
        rsna_image_col=args.rsna_image_col,
        rsna_label_col=args.rsna_label_col,
        rsna_split_col=args.rsna_split_col,
        rsna_train_split_value=args.rsna_train_split_value,
        rsna_test_split_value=args.rsna_test_split_value,
    )


def select_checkpoint(ckpt_dir, ckpt_select):
    best_yaml = ckpt_dir / "best_ckpts.yaml"
    if ckpt_select == "best" and best_yaml.is_file() and best_yaml.stat().st_size > 0:
        best_score = None
        best_path = None
        for line in best_yaml.read_text(encoding="utf-8").splitlines():
            if ": " not in line:
                continue
            path_str, score_str = line.rsplit(": ", 1)
            try:
                score = float(score_str.strip())
            except ValueError:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_path = Path(path_str.strip())
        if best_path is not None and best_path.is_file():
            return best_path
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.is_file():
        return last_ckpt
    return None


def find_fold_checkpoint_dirs(ckpt_root, base_exp_name, fold_indices):
    ckpt_root = Path(ckpt_root)
    fold_dirs = {}
    for fold in fold_indices:
        pattern = f"*{base_exp_name}_fold{fold}"
        matches = sorted(
            ckpt_root.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            fold_dirs[fold] = matches[0]
    return fold_dirs


def score_and_class_from_predictions(all_scores, args):
    scores = np.asarray(all_scores)
    if scores.ndim == 2 and scores.shape[1] == 1:
        pos_scores = scores[:, 0]
        pred_class = (pos_scores >= args.threshold).astype(int)
        return pos_scores, pred_class
    if scores.ndim == 1:
        pos_scores = scores
        pred_class = (pos_scores >= args.threshold).astype(int)
        return pos_scores, pred_class
    if scores.ndim == 2 and scores.shape[1] == 2:
        pos_scores = scores[:, 1]
        pred_class = np.argmax(scores, axis=1)
        return pos_scores, pred_class
    pos_scores = np.max(scores, axis=1)
    pred_class = np.argmax(scores, axis=1)
    return pos_scores, pred_class


def export_split_predictions(model, datamodule, split_name, output_csv, args):
    trainer = Trainer(
        accelerator=args.accelerator,
        precision=args.precision,
        devices=1,
        fast_dev_run=args.dev,
        max_epochs=1,
        deterministic=args.deterministic,
        inference_mode=True,
        enable_progress_bar=(not args.no_progress_bar),
        logger=False,
    )
    model.all_scores = None
    model.all_labels = None
    model.img_path = []
    trainer.test(model, datamodule=datamodule)

    test_loader = datamodule.test_dataloader()
    dataset = test_loader.dataset
    df = dataset.df.reset_index(drop=True).copy()
    scores, pred_class = score_and_class_from_predictions(model.all_scores, args)

    if len(df) != len(scores):
        raise RuntimeError(
            f"Split={split_name} row count mismatch: df={len(df)} preds={len(scores)}"
        )

    # Keep a single split indicator column.
    if "split" not in df.columns:
        df["split"] = split_name
    df[args.score_col] = scores
    df[args.class_col] = pred_class
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.eval = True
    args.pred_only = True
    args.save_prediction = False
    args.data_pct = 1.0
    args.accumulate_grad_batches = 1
    args.strategy = None
    args.devices = 1
    args.grad_ckpt = False
    args.train_sub_set = False
    args.deterministic = False

    if args.ckpt_root is None:
        args.ckpt_root = str(Path(__file__).resolve().parent / "logs" / "ckpts" / "GLAM")
    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).resolve().parent / f"{args.base_exp_name}_fold_predictions"
        )

    seed_everything(args.seed)
    dataset = resolve_dataset(args)
    transform_obj = resolve_transform(args)
    fold_indices = (
        args.fold_indices if args.fold_indices is not None else list(range(args.num_folds))
    )
    fold_dirs = find_fold_checkpoint_dirs(args.ckpt_root, args.base_exp_name, fold_indices)

    if not fold_dirs:
        raise FileNotFoundError(
            f"No checkpoint directories found under {args.ckpt_root} for base_exp_name={args.base_exp_name}"
        )

    output_root = Path(args.output_dir)
    print(f"### Output dir: {output_root}")

    for fold in fold_indices:
        ckpt_dir = fold_dirs.get(fold)
        if ckpt_dir is None:
            print(f"### [WARN] fold {fold}: checkpoint directory not found, skipping")
            continue

        ckpt_path = select_checkpoint(ckpt_dir, args.ckpt_select)
        if ckpt_path is None:
            print(f"### [WARN] fold {fold}: usable checkpoint not found in {ckpt_dir}, skipping")
            continue

        print(f"### fold {fold}: using checkpoint {ckpt_path}")
        args.fold = fold
        args.pretrained_model = str(ckpt_path)
        model = GLAM.load_from_checkpoint(
            str(ckpt_path), map_location="cpu", strict=False, **vars(args)
        )
        model.eval()

        run_dir = output_root / f"run{fold}"
        merged_parts = []
        for split_name in args.splits:
            datamodule = build_datamodule(args, dataset, transform_obj, split_name)
            output_csv = run_dir / f"{split_name}_predictions.csv"
            print(f"### fold {fold}: exporting {split_name} -> {output_csv}")
            split_df = export_split_predictions(model, datamodule, split_name, output_csv, args)
            merged_parts.append(split_df)

        if merged_parts:
            merged_csv = run_dir / "all_predictions.csv"
            merged_df = pd.concat(merged_parts, ignore_index=True)
            merged_df.to_csv(merged_csv, index=False)
            print(f"### fold {fold}: merged all splits -> {merged_csv}")


if __name__ == "__main__":
    main()
