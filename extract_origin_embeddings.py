import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GLAM


DEFAULT_EXTRA_METADATA_COLS = [
    "cancer",
    "Cancer",
    "label",
    "target",
    "split",
    "cohert_num",
    "cohort_num",
    "laterality",
    "view",
    "view_position",
    "ViewPosition",
    "ImageLateralityFinal",
    "site_id",
    "machine_id",
    "implant",
    "density",
    "breast_density",
    "BIRADS",
    "birads",
    "asses",
]


def option_was_provided(argv, *names):
    for arg in argv:
        for name in names:
            if arg == name or arg.startswith(name + "="):
                return True
    return False


def str_or_none(value):
    if value is None:
        return None
    value = str(value)
    if value.strip() == "":
        return None
    return value


def normalize_path(path):
    return os.path.normpath(str(path))


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Export origin/GLAM image encoder embeddings to "
            "embeddings.npy + metadata.csv + manifest.json."
        )
    )

    # Small train.py-compatible surface not provided by GLAM.add_model_specific_args.
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--llm_type", type=str, default="bert")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--vindr", action="store_true")
    parser.add_argument("--rsna_mammo", action="store_true", default=True)

    parser = GLAM.add_model_specific_args(parser)

    # Export-specific args.
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "valid", "val", "test", "all"],
        help="Dataset split to export. 'all' disables RSNA CSV split filtering.",
    )
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_patch", action="store_true")
    parser.add_argument("--save_logits", dest="save_logits", action="store_true", default=True)
    parser.add_argument("--no_save_logits", dest="save_logits", action="store_false")
    parser.add_argument(
        "--metadata_cols",
        type=str,
        default="",
        help="Comma-separated extra source CSV columns to copy into metadata.csv.",
    )

    # Friendly aliases matching the examples in the plan.
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--img_root", type=str, default=None)
    parser.add_argument("--path_pattern", type=str, default=None)
    parser.add_argument("--patient_col", type=str, default=None)
    parser.add_argument("--image_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)

    # Negative switches for origin defaults that are store_true in GLAM's parser.
    parser.add_argument("--no_linear_proj", action="store_true")
    parser.add_argument("--no_img_cls_ft", action="store_true")
    parser.add_argument("--no_weighted_binary", action="store_true")

    return parser


def apply_origin_defaults(args, argv):
    if args.csv_path is not None and not option_was_provided(argv, "--rsna_csv_path"):
        args.rsna_csv_path = args.csv_path
    if args.img_root is not None and not option_was_provided(argv, "--rsna_img_root"):
        args.rsna_img_root = args.img_root
    if args.path_pattern is not None and not option_was_provided(argv, "--rsna_path_pattern"):
        args.rsna_path_pattern = args.path_pattern
    if args.patient_col is not None and not option_was_provided(argv, "--rsna_patient_col"):
        args.rsna_patient_col = args.patient_col
    if args.image_col is not None and not option_was_provided(argv, "--rsna_image_col"):
        args.rsna_image_col = args.image_col
    if args.label_col is not None and not option_was_provided(argv, "--rsna_label_col"):
        args.rsna_label_col = args.label_col

    if not option_was_provided(argv, "--emb_dim"):
        args.emb_dim = 512
    if not option_was_provided(argv, "--img_size"):
        args.img_size = 336
    if not option_was_provided(argv, "--crop_size"):
        args.crop_size = 336
    if not option_was_provided(argv, "--batch_size"):
        args.batch_size = 32
    if not option_was_provided(argv, "--num_workers"):
        args.num_workers = 4
    if not option_was_provided(argv, "--devices"):
        args.devices = 1
    if not option_was_provided(argv, "--accelerator"):
        args.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if not option_was_provided(argv, "--strategy"):
        args.strategy = "auto"
    if not option_was_provided(argv, "--num_classes"):
        args.num_classes = 1

    if args.no_linear_proj:
        args.linear_proj = False
    elif not option_was_provided(argv, "--linear_proj"):
        args.linear_proj = True

    if args.no_img_cls_ft:
        args.img_cls_ft = False
    elif not option_was_provided(argv, "--img_cls_ft"):
        args.img_cls_ft = True

    if args.no_weighted_binary:
        args.weighted_binary = False
    elif not option_was_provided(argv, "--weighted_binary") and args.num_classes == 1:
        args.weighted_binary = True

    if args.split == "val":
        args.split = "valid"
    if args.split == "all":
        args.rsna_use_all_data = True

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("outputs", "origin_embeddings", f"origin_embeddings_{stamp}")

    # This is an inference-only script.
    args.dev = False
    args.grad_ckpt = False
    args.train_sub_set = False
    args.data_pct = 1.0

    return args


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")
    return device


def build_dataset(args):
    from dataset.mammo_eval_dataset import RSNAMammo
    from dataset.transforms import Moco2Transform, RSNAMammoTransform

    dataset_split = "test" if args.split == "all" else args.split
    transform_cls = RSNAMammoTransform if args.rsna_trans else Moco2Transform
    transform = transform_cls(
        False,
        args.img_size,
        args.crop_size,
        args.align_orientation,
        args.remove_text,
    )

    dataset_kwargs = {
        "k_fold": args.k_fold,
        "fold": args.fold,
        "csv_path": args.rsna_csv_path,
        "img_root": args.rsna_img_root,
        "path_pattern": args.rsna_path_pattern,
        "patient_col": args.rsna_patient_col,
        "image_col": args.rsna_image_col,
        "label_col": args.rsna_label_col,
        "split_col": args.rsna_split_col,
        "train_split_value": args.rsna_train_split_value,
        "test_split_value": args.rsna_test_split_value,
        "split_source": args.split_source,
        "cohort_col": args.rsna_cohort_col,
        "train_cohorts": args.train_cohorts,
        "test_cohorts": args.test_cohorts,
        "val_split": args.rsna_val_split,
        "val_fraction": args.rsna_val_fraction,
        "val_max_fraction": args.rsna_val_max_fraction,
        "val_min_positive_patients": args.rsna_val_min_positive_patients,
        "val_random_state": args.rsna_val_random_state,
        "use_all_data": args.rsna_use_all_data,
        "split": dataset_split,
        "transform": transform,
        "data_pct": args.data_pct,
        "llm_type": args.llm_type,
        "simple_cap": args.simple_cap,
        "train_sub_set": args.train_sub_set,
        "structural_cap": args.structural_cap,
        "natural_cap": args.natural_cap,
        "instance_test_cap": args.instance_test_cap,
        "inter_side": args.inter_side,
        "inter_view": args.inter_view,
        "balanced_test": args.balanced_test,
        "slip": args.slip,
        "pred_density": args.pred_density,
        # Keep the same call shape as DataModule for parity with current training/eval code.
        "imsize": args.img_size,
        "raw_caption": args.raw_caption,
        "load_jpg": args.load_jpg,
        "aug_orig_img": args.aug_orig_img,
        "test_data_pct": args.test_data_pct,
        "bootstrap_test": args.bootstrap_test,
        "pred_only": args.pred_only,
        "max_words": args.max_words,
        "prob_diff_dcm": args.prob_diff_dcm,
        "screen_only": args.screen_only,
        "aligned_mlo": args.aligned_mlo,
        "fixed_view": args.fixed_view,
        "pred_mass": args.pred_mass,
        "pred_calc": args.pred_calc,
    }
    return RSNAMammo(**dataset_kwargs)


def build_loader(args, dataset, device):
    from dataset.pretrain_embed_dataset import multimodal_collate_fn

    return DataLoader(
        dataset,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        drop_last=False,
        collate_fn=multimodal_collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def load_model(args, device):
    if args.pretrained_model is not None:
        print(f"### Loading origin classifier checkpoint: {args.pretrained_model}", flush=True)
        model = GLAM.load_from_checkpoint(
            args.pretrained_model,
            map_location="cpu",
            strict=False,
            **vars(args),
        )
        if args.ema and not args.ema_no_load:
            model.set_ema_encoder()
    else:
        if args.pretrained_encoder is not None:
            print(f"### Loading encoder checkpoint: {args.pretrained_encoder}", flush=True)
        else:
            print("### Warning: no checkpoint provided; exporting randomly initialized features.", flush=True)
        model = GLAM(**vars(args))

    model.eval()
    model.to(device)
    return model


def get_actual_cohort_col(df, requested_col):
    if requested_col in df.columns:
        return requested_col
    if requested_col == "cohert_num" and "cohort_num" in df.columns:
        return "cohort_num"
    return None


def metadata_value(row, col):
    value = row[col]
    if pd.isna(value):
        return ""
    return value


def build_metadata_record(args, dataset, row_id, path, label, logit, probability):
    df = dataset.df.reset_index(drop=True)
    row = df.iloc[row_id]

    patient_col = getattr(dataset, "_patient_col", args.rsna_patient_col)
    image_col = getattr(dataset, "_image_col", args.rsna_image_col)
    label_col = getattr(dataset, "_label_col", args.rsna_label_col)
    cohort_col = get_actual_cohort_col(df, args.rsna_cohort_col)

    if args.split == "all" and args.rsna_split_col in row.index:
        split_value = metadata_value(row, args.rsna_split_col)
    else:
        split_value = args.split

    record = {
        "row_id": row_id,
        "image_path": normalize_path(path),
        "patient_id": metadata_value(row, patient_col) if patient_col in row.index else "",
        "image_id": metadata_value(row, image_col) if image_col in row.index else "",
        "label": int(label),
        "split": split_value,
        "cohort": metadata_value(row, cohort_col) if cohort_col else "",
        "logit": "" if logit is None else float(logit),
        "probability_1": "" if probability is None else float(probability),
    }

    requested_cols = [c.strip() for c in args.metadata_cols.split(",") if c.strip()]
    for col in DEFAULT_EXTRA_METADATA_COLS + requested_cols:
        if col not in row.index:
            continue
        if col in {patient_col, image_col, label_col, args.rsna_split_col, cohort_col}:
            continue
        if col in record:
            continue
        record[col] = metadata_value(row, col)

    # Preserve the raw label column when its name is informative and not already present.
    if label_col in row.index and label_col not in {"label"}:
        source_label_key = f"source_{label_col}"
        if source_label_key not in record:
            record[source_label_key] = metadata_value(row, label_col)

    return record


def classifier_outputs(model, features, args):
    if not args.save_logits or args.pretrained_model is None:
        return None, None
    head = getattr(model.img_encoder_q, "global_embed", None)
    if head is None:
        return None, None

    logits = head(features)
    if logits.ndim == 1 or logits.shape[-1] == 1 or args.weighted_binary:
        logits_1 = logits.reshape(-1)
        probs_1 = torch.sigmoid(logits_1)
    else:
        logits_1 = logits[:, 1] if logits.shape[-1] > 1 else logits.reshape(-1)
        probs_1 = torch.softmax(logits, dim=-1)[:, 1] if logits.shape[-1] > 1 else torch.sigmoid(logits_1)
    return logits_1.detach().float().cpu().numpy(), probs_1.detach().float().cpu().numpy()


def extract(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "embeddings.npy"
    patch_path = output_dir / "patch_embeddings.npy"
    metadata_path = output_dir / "metadata.csv"
    manifest_path = output_dir / "manifest.json"

    device = resolve_device(args.device)
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    print(f"### Export directory: {output_dir}", flush=True)
    print(f"### Device: {device}", flush=True)

    dataset = build_dataset(args)
    loader = build_loader(args, dataset, device)
    model = load_model(args, device)

    max_rows = len(dataset)
    if args.max_batches and args.max_batches > 0:
        max_rows = min(max_rows, args.max_batches * args.batch_size)
    if max_rows == 0:
        raise RuntimeError("No rows to export after dataset filtering.")

    embeddings_mmap = None
    patch_mmap = None
    metadata_records = []
    written = 0

    if args.save_logits and args.pretrained_model is None:
        print(
            "### Warning: --save_logits is enabled, but --pretrained_model was not provided; "
            "logit/probability_1 will be empty.",
            flush=True,
        )

    total_batches = len(loader)
    if args.max_batches and args.max_batches > 0:
        total_batches = min(total_batches, args.max_batches)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, total=total_batches, desc="Extracting")):
            if args.max_batches and args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            images = batch["imgs"].to(device, non_blocking=True)
            img_feat_q, patch_feat_q, img_full = model.img_encoder_q(images)
            features = img_full.mean(dim=1) if args.pool_feat else img_feat_q
            logits, probs = classifier_outputs(model, features, args)

            feature_np = features.detach().float().cpu().numpy().astype(np_dtype, copy=False)
            batch_size, feature_dim = feature_np.shape
            if embeddings_mmap is None:
                embeddings_mmap = np.lib.format.open_memmap(
                    embedding_path,
                    mode="w+",
                    dtype=np_dtype,
                    shape=(max_rows, feature_dim),
                )

            end = written + batch_size
            if end > max_rows:
                keep = max_rows - written
                feature_np = feature_np[:keep]
                patch_feat_q = patch_feat_q[:keep]
                batch_size = keep
                end = max_rows
                if logits is not None:
                    logits = logits[:keep]
                    probs = probs[:keep]

            embeddings_mmap[written:end] = feature_np

            if args.save_patch:
                patch_np = patch_feat_q.detach().float().cpu().numpy().astype(np_dtype, copy=False)
                if patch_mmap is None:
                    patch_mmap = np.lib.format.open_memmap(
                        patch_path,
                        mode="w+",
                        dtype=np_dtype,
                        shape=(max_rows, patch_np.shape[1], patch_np.shape[2]),
                    )
                patch_mmap[written:end] = patch_np[:batch_size]

            batch_paths = np.asarray(batch["path"]).reshape(-1).tolist()
            for i in range(batch_size):
                row_id = written + i
                label = dataset.labels[row_id]
                path = batch_paths[i]
                expected_path = dataset.filenames[row_id]
                if normalize_path(path) != normalize_path(expected_path):
                    raise RuntimeError(
                        "DataLoader order mismatch: "
                        f"row_id={row_id}, batch_path={path}, expected_path={expected_path}"
                    )
                metadata_records.append(
                    build_metadata_record(
                        args,
                        dataset,
                        row_id,
                        path,
                        label,
                        None if logits is None else logits[i],
                        None if probs is None else probs[i],
                    )
                )

            written = end
            if written >= max_rows:
                break

    if embeddings_mmap is None:
        raise RuntimeError("Embedding extraction produced no batches.")
    embeddings_mmap.flush()
    if patch_mmap is not None:
        patch_mmap.flush()

    metadata = pd.DataFrame(metadata_records)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    manifest = {
        "embedding_file": str(embedding_path),
        "metadata_file": str(metadata_path),
        "patch_embedding_file": str(patch_path) if args.save_patch else None,
        "embedding_shape": [int(written), int(embeddings_mmap.shape[1])],
        "embedding_dtype": args.dtype,
        "feature_name": "img_full.mean(dim=1)" if args.pool_feat else "img_feat_q",
        "checkpoint_path": str_or_none(args.pretrained_model),
        "encoder_checkpoint_path": str_or_none(args.pretrained_encoder),
        "csv_path": str_or_none(args.rsna_csv_path),
        "img_root": str_or_none(args.rsna_img_root),
        "path_pattern": str_or_none(args.rsna_path_pattern),
        "model_args": json_safe(vars(args)),
        "metadata_columns": metadata.columns.tolist(),
        "label_definition": {
            "label": "Binary cancer label from the active RSNAMammo dataset split.",
            "source_label_column": getattr(dataset, "_label_col", args.rsna_label_col),
            "probability_1": "Positive-class probability from origin classifier head when --pretrained_model is used.",
        },
        "num_rows": int(written),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(manifest), f, indent=2, ensure_ascii=False)

    print("### Saved:", flush=True)
    print(f"  embeddings: {embedding_path}", flush=True)
    print(f"  metadata  : {metadata_path}", flush=True)
    print(f"  manifest  : {manifest_path}", flush=True)
    if args.save_patch:
        print(f"  patches   : {patch_path}", flush=True)
    print(f"### Exported rows: {written}", flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    args = apply_origin_defaults(args, argv)
    extract(args)


if __name__ == "__main__":
    main()
