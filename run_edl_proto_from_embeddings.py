#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train EDL-Prototype directly from cached GLAM embeddings.

Expected input bundle:
  embeddings.npy
  metadata.csv
"""

import argparse

from embedding_proto_common import (
    CachedPrototypeEDLHead,
    EDLPrototypeLoss,
    run_cached_prototype_experiment,
)


def add_common_args(parser):
    parser.add_argument("--embedding-dir", default="outputs/origin_embeddings/origin_last_all")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings.npy. Overrides --embedding-dir.")
    parser.add_argument("--metadata", default=None, help="Path to metadata.csv. Overrides --embedding-dir.")
    parser.add_argument("--output-dir", default="outputs/embedding_edl_proto")
    parser.add_argument("--label", default="label")

    parser.add_argument("--prototypes-per-class", "--prototypes_per_class", type=int, default=4)
    parser.add_argument("--prototype-topk", "--prototype_topk", type=int, default=3)
    parser.add_argument("--prototype-init", "--prototype_init", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument("--prototype-temperature", "--prototype_temperature", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-normalize", "--no_normalize", action="store_true")

    parser.add_argument("--epochs", type=int, default=25)
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
    parser = argparse.ArgumentParser(description="Train EDL-Prototype from cached GLAM embeddings.")
    add_common_args(parser)
    parser.add_argument("--evidence-type", "--evidence_type", choices=["relu", "exp", "softplus"], default="softplus")
    parser.add_argument("--edl-loss-type", "--edl_loss_type", choices=["digamma", "log", "mse"], default="digamma")
    parser.add_argument("--annealing-step", "--annealing_step", type=int, default=10)
    parser.add_argument("--lambda-kl", "--lambda_kl", type=float, default=0.1)
    parser.add_argument("--pos-weight", "--pos_weight", type=float, default=None)
    parser.add_argument("--no-auto-pos-weight", "--no_auto_pos_weight", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args._requested_output_dir = args.output_dir

    def build_model(feature_dim):
        return CachedPrototypeEDLHead(
            in_features=feature_dim,
            num_classes=2,
            prototypes_per_class=args.prototypes_per_class,
            topk=args.prototype_topk,
            temperature=args.prototype_temperature,
            normalize=not args.no_normalize,
            dropout=args.dropout,
            evidence_type=args.evidence_type,
        )

    def build_criterion(_y_train, class_weight_info):
        pos_weight = args.pos_weight
        if pos_weight is None and not args.no_auto_pos_weight:
            pos_weight = class_weight_info.get("pos_weight")
            if pos_weight is not None:
                print(f"[EDL] auto pos_weight={pos_weight:.6f}")
        return EDLPrototypeLoss(
            loss_type=args.edl_loss_type,
            annealing_step=args.annealing_step,
            lambda_kl=args.lambda_kl,
            pos_weight=pos_weight,
        )

    run_cached_prototype_experiment(
        args,
        build_model=build_model,
        build_criterion=build_criterion,
        method="EDL-Prototype",
        output_prefix="edl_proto",
        is_dst=False,
    )


if __name__ == "__main__":
    main()
