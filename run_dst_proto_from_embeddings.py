#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train DST-Prototype directly from cached GLAM embeddings.

Expected input bundle:
  embeddings.npy
  metadata.csv
"""

import argparse

from edl_proto_models import PrototypeDSTHead, PrototypeDSTNLLLoss
from embedding_proto_common import run_cached_prototype_experiment


def add_common_args(parser):
    parser.add_argument("--embedding-dir", default="outputs/origin_embeddings/origin_last_all")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings.npy. Overrides --embedding-dir.")
    parser.add_argument("--metadata", default=None, help="Path to metadata.csv. Overrides --embedding-dir.")
    parser.add_argument("--output-dir", default="outputs/embedding_dst_proto")
    parser.add_argument("--label", default="label")

    parser.add_argument("--prototypes-per-class", "--prototypes_per_class", type=int, default=4)
    parser.add_argument("--prototype-topk", "--prototype_topk", type=int, default=3)
    parser.add_argument("--prototype-init", "--prototype_init", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-normalize", "--no_normalize", action="store_true")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--best-metric", "--best_metric", choices=["bacc", "auc", "loss"], default="loss")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--proto-attract-weight", "--proto_attract_weight", type=float, default=0.0)
    parser.add_argument("--proto-separation-weight", "--proto_separation_weight", type=float, default=0.0)
    parser.add_argument("--proto-diversity-weight", "--proto_diversity_weight", type=float, default=0.0)
    parser.add_argument("--proto-loss-weight", "--proto_loss_weight", type=float, default=1.0)
    parser.add_argument("--proto-margin", "--proto_margin", type=float, default=1.0)
    parser.add_argument("--proto-balance-classes", "--proto_balance_classes", choices=["y", "n"], default="y")

    parser.add_argument("--split-mode", "--split_mode", choices=["auto", "metadata", "cohort", "fold"], default="auto")
    parser.add_argument("--split-col", "--split_col", default="split")
    parser.add_argument("--fold-col", "--fold_col", default="fold")
    parser.add_argument("--val-fold", "--val_fold", type=int, default=0)
    parser.add_argument("--cohort-col", "--cohort_col", default="cohort")
    parser.add_argument("--train-cohorts", "--train_cohorts", default="1-8")
    parser.add_argument("--test-cohorts", "--test_cohorts", default="9-10")
    parser.add_argument("--holdout-val-percent", "--holdout_val_percent", type=float, default=20.0)
    parser.add_argument("--group-cols", "--group_cols", default="patient_id")
    parser.add_argument("--score-agg", "--score_agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--n-folds", "--n_folds", type=int, default=5)
    parser.add_argument("--fold-start", "--fold_start", type=int, default=0)
    parser.add_argument("--cv-group-col", "--cv_group_col", default="patient_id")

    parser.add_argument("--max-samples", "--max_samples", type=int, default=None)
    parser.add_argument("--max-train-samples", "--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu-id", "--gpu_id", type=int, default=None)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=0)
    parser.add_argument("--run-id", "--run_id", default=None)
    parser.add_argument("--no-timestamp", "--no_timestamp", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(description="Train DST-Prototype from cached GLAM embeddings.")
    add_common_args(parser)
    parser.add_argument("--dst-gamma-init", "--dst_gamma_init", type=float, default=1.0)
    parser.add_argument("--dst-alpha-init", "--dst_alpha_init", type=float, default=0.0)
    parser.add_argument(
        "--temperature",
        default=None,
        help="Backward-compatible distance temperature. Positive value maps to gamma_init=1/temperature.",
    )
    parser.add_argument("--class-weight-mode", "--class_weight_mode", choices=["none", "inverse"], default="inverse")
    return parser


def resolve_gamma_init(args):
    if args.temperature is None or str(args.temperature).lower() == "auto":
        return float(args.dst_gamma_init)
    temperature = float(args.temperature)
    if temperature <= 0:
        raise ValueError("--temperature must be positive or 'auto'.")
    return 1.0 / temperature


def main():
    args = build_parser().parse_args()
    args._requested_output_dir = args.output_dir
    gamma_init = resolve_gamma_init(args)

    def build_model(feature_dim):
        return PrototypeDSTHead(
            in_features=feature_dim,
            num_classes=2,
            prototypes_per_class=args.prototypes_per_class,
            topk=args.prototype_topk,
            normalize=not args.no_normalize,
            gamma_init=gamma_init,
            alpha_init=args.dst_alpha_init,
            dropout=args.dropout,
        )

    def build_criterion(_y_train, class_weight_info):
        class_weights = None
        if args.class_weight_mode == "inverse":
            class_weights = class_weight_info.get("class_weights")
            if class_weights is not None:
                print(f"[DST] class_weights={class_weights}")
        return PrototypeDSTNLLLoss(class_weights=class_weights)

    run_cached_prototype_experiment(
        args,
        build_model=build_model,
        build_criterion=build_criterion,
        method="DST-Prototype",
        output_prefix="dst_proto",
        is_dst=True,
    )


if __name__ == "__main__":
    main()
