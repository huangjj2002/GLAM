import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from edl_loss import (
    compute_dirichlet_prob,
    compute_uncertainty,
    edl_digamma_loss,
    edl_log_loss,
    edl_mse_loss,
    get_evidence,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_run_id(run_id=None):
    if run_id is None or str(run_id).strip() == "":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id).strip())
    return run_id.strip("_") or datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamped_output_dir(output_dir, args):
    base_dir = resolve_path(output_dir)
    if getattr(args, "no_timestamp", False):
        return base_dir, None
    run_id = resolve_run_id(getattr(args, "run_id", None))
    return base_dir.with_name(f"{base_dir.name}_run_{run_id}"), run_id


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(args):
    if getattr(args, "gpu_id", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(args.device)


def parse_int_set(spec):
    values = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.update(range(int(start), int(end) + 1))
        else:
            values.add(int(part))
    return values


def parse_group_cols(group_cols):
    cols = [col.strip() for col in str(group_cols).split(",") if col.strip()]
    return cols or ["patient_id"]


def first_existing_column(df, candidates):
    for col in candidates:
        if col and col in df.columns:
            return col
    return None


def normalize_features(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def load_embedding_bundle(args):
    embedding_dir = resolve_path(args.embedding_dir)
    embedding_path = resolve_path(args.embeddings) if args.embeddings else embedding_dir / "embeddings.npy"
    metadata_path = resolve_path(args.metadata) if args.metadata else embedding_dir / "metadata.csv"

    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embedding_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    embeddings = np.load(embedding_path, mmap_mode=None)
    meta = pd.read_csv(metadata_path).reset_index(drop=True)
    if len(meta) != len(embeddings):
        raise ValueError(f"metadata rows ({len(meta)}) != embedding rows ({len(embeddings)})")
    if args.label not in meta.columns:
        raise ValueError(f"metadata is missing label column: {args.label}")

    labels = meta[args.label].astype(int).to_numpy()
    features = np.asarray(embeddings, dtype=np.float32)
    if not getattr(args, "no_normalize", False):
        features = normalize_features(features).astype(np.float32, copy=False)

    return features, meta, labels, embedding_path, metadata_path


def stratified_holdout_indices(indices, labels, val_fraction, seed):
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices, dtype=int)
    labels = np.asarray(labels).astype(int)
    val_indices = []
    for class_id in np.unique(labels):
        class_indices = indices[labels == class_id].copy()
        if len(class_indices) == 0:
            continue
        rng.shuffle(class_indices)
        n_val = int(round(len(class_indices) * val_fraction))
        if len(class_indices) > 1:
            n_val = min(max(n_val, 1), len(class_indices) - 1)
        else:
            n_val = 0
        val_indices.extend(class_indices[:n_val].tolist())
    if not val_indices and len(indices) > 1:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        val_indices = shuffled[:n_val].tolist()
    return np.asarray(val_indices, dtype=int)


def assign_splits(meta, labels, args):
    split = pd.Series("train", index=meta.index, dtype="object")

    split_col = first_existing_column(meta, [args.split_col, "split", "dst_split"])
    use_metadata = args.split_mode in {"auto", "metadata"} and split_col is not None
    if use_metadata:
        raw = meta[split_col].astype(str).str.strip().str.lower()
        split.loc[raw.isin(["val", "valid", "validation"])] = "val"
        split.loc[raw == "test"] = "test"
        split.loc[raw.isin(["train", "training"])] = "train"

    fold_col = first_existing_column(meta, [args.fold_col, "fold"])
    use_fold = args.split_mode in {"auto", "fold"} and fold_col is not None
    if use_fold:
        fold = pd.to_numeric(meta[fold_col], errors="coerce")
        split.loc[fold == -1] = "test"
        if (split == "val").sum() == 0:
            split.loc[(fold == int(args.val_fold)) & (split == "train")] = "val"

    cohort_col = first_existing_column(
        meta,
        [args.cohort_col, "cohort", "cohort_num", "cohert_num", "source_cohert_num"],
    )
    use_cohort = args.split_mode in {"auto", "cohort"} and cohort_col is not None
    if use_cohort and (split == "test").sum() == 0:
        cohort = pd.to_numeric(meta[cohort_col], errors="coerce")
        train_cohorts = parse_int_set(args.train_cohorts)
        test_cohorts = parse_int_set(args.test_cohorts)
        split.loc[cohort.isin(test_cohorts)] = "test"
        split.loc[cohort.isin(train_cohorts)] = "train"

    if (split == "val").sum() == 0 and args.holdout_val_percent > 0:
        train_idx = np.flatnonzero(split.values == "train")
        if len(train_idx) > 1:
            val_idx = stratified_holdout_indices(
                train_idx,
                labels[train_idx],
                args.holdout_val_percent / 100.0,
                args.seed,
            )
            split.iloc[val_idx] = "val"

    return split, {
        "split_col": split_col,
        "fold_col": fold_col,
        "cohort_col": cohort_col,
    }


def infer_test_mask(meta, args):
    test_mask = pd.Series(False, index=meta.index)
    split_col = first_existing_column(meta, [args.split_col, "split", "dst_split"])
    if split_col is not None:
        raw = meta[split_col].astype(str).str.strip().str.lower()
        test_mask |= raw == "test"

    fold_col = first_existing_column(meta, [args.fold_col, "fold"])
    if fold_col is not None:
        fold = pd.to_numeric(meta[fold_col], errors="coerce")
        test_mask |= fold == -1

    cohort_col = first_existing_column(
        meta,
        [args.cohort_col, "cohort", "cohort_num", "cohert_num", "source_cohert_num"],
    )
    if cohort_col is not None:
        cohort = pd.to_numeric(meta[cohort_col], errors="coerce")
        test_mask |= cohort.isin(parse_int_set(args.test_cohorts))

    return test_mask


def assign_cv_splits(meta, labels, args, fold):
    group_col = first_existing_column(meta, [args.cv_group_col, "patient_id"])
    if group_col is None:
        raise ValueError("Patient-safe CV requires a group column such as patient_id.")

    split = pd.Series("train", index=meta.index, dtype="object")
    test_mask = infer_test_mask(meta, args)
    if test_mask.any():
        test_groups = meta.loc[test_mask, group_col].drop_duplicates()
        test_mask = meta[group_col].isin(test_groups)
    split.loc[test_mask] = "test"

    train_val_mask = ~test_mask
    rel_fold = int(fold) - int(args.fold_start)
    n_folds = int(args.n_folds)
    if rel_fold < 0 or rel_fold >= n_folds:
        raise ValueError(f"CV fold {fold} is outside [{args.fold_start}, {args.fold_start + n_folds - 1}].")

    pool_idx = np.flatnonzero(train_val_mask.values)
    grouped = (
        pd.DataFrame(
            {
                "group": meta.iloc[pool_idx][group_col].to_numpy(),
                "label": labels[pool_idx],
            }
        )
        .groupby("group", dropna=False)["label"]
        .max()
    )
    group_values = grouped.index.to_numpy()
    group_labels = grouped.to_numpy(dtype=int)
    if len(group_values) < n_folds:
        raise ValueError(f"Cannot generate {n_folds} folds from only {len(group_values)} groups.")

    rng = np.random.default_rng(args.seed)
    val_groups = []
    for class_id in np.unique(group_labels):
        class_groups = group_values[group_labels == class_id].copy()
        rng.shuffle(class_groups)
        val_groups.extend(np.array_split(class_groups, n_folds)[rel_fold].tolist())
    if not val_groups:
        raise ValueError("Generated CV validation split is empty.")

    split.loc[train_val_mask & meta[group_col].isin(set(val_groups))] = "val"
    return split, {"cv_group_col": group_col, "cv_fold": int(fold)}


def stratified_limit_positions(positions, labels, max_count, seed):
    positions = np.asarray(positions, dtype=int)
    if max_count is None or len(positions) <= max_count:
        return positions
    if max_count <= 0:
        raise ValueError("--max-train-samples must be positive.")

    rng = np.random.default_rng(seed)
    selected = []
    labels = np.asarray(labels).astype(int)
    for class_id in np.unique(labels[positions]):
        class_positions = positions[labels[positions] == class_id]
        quota = int(round(len(class_positions) * max_count / len(positions)))
        quota = min(len(class_positions), max(1, quota))
        selected.extend(rng.choice(class_positions, size=quota, replace=False).tolist())

    selected = np.asarray(sorted(set(selected)), dtype=int)
    if len(selected) > max_count:
        selected = np.sort(rng.choice(selected, size=max_count, replace=False))
    elif len(selected) < max_count:
        remaining = np.setdiff1d(positions, selected, assume_unique=False)
        fill = min(len(remaining), max_count - len(selected))
        if fill > 0:
            selected = np.sort(np.concatenate([selected, rng.choice(remaining, size=fill, replace=False)]))
    return selected


def simple_kmeans(x, k, seed, max_iter=100):
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if len(x) < k:
        raise ValueError("simple_kmeans requires len(x) >= k.")
    init_idx = rng.choice(len(x), size=k, replace=False)
    centers = x[init_idx].copy()
    labels = np.full(len(x), -1, dtype=np.int64)
    for _ in range(max_iter):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.any():
                centers[cluster_id] = x[mask].mean(axis=0)
            else:
                centers[cluster_id] = x[rng.integers(0, len(x))]
    return centers.astype(np.float32)


def classwise_prototypes(x_train, y_train, prototypes_per_class, seed):
    centers = []
    for class_id in [0, 1]:
        class_x = np.asarray(x_train[y_train == class_id], dtype=np.float32)
        if len(class_x) == 0:
            raise ValueError(f"No training rows for class {class_id}; cannot initialize prototypes.")
        k = int(prototypes_per_class)
        if k == 1:
            class_centers = class_x.mean(axis=0, keepdims=True)
        elif len(class_x) >= k:
            class_centers = simple_kmeans(class_x, k, seed + class_id)
        else:
            repeat_count = int(np.ceil(k / len(class_x)))
            class_centers = np.tile(class_x, (repeat_count, 1))[:k]
            print(f"[WARN] class {class_id} has only {len(class_x)} rows; repeated prototypes to K={k}.")
        centers.append(class_centers.astype(np.float32))
    return np.stack(centers, axis=0).astype(np.float32)


class EmbeddingDataset(Dataset):
    def __init__(self, features, labels, positions):
        self.features = features
        self.labels = np.asarray(labels).astype(np.int64)
        self.positions = np.asarray(positions, dtype=np.int64)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        row = int(self.positions[index])
        feature = torch.from_numpy(np.asarray(self.features[row], dtype=np.float32))
        label = int(self.labels[row])
        return feature, label, row


def make_loader(features, labels, positions, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        EmbeddingDataset(features, labels, positions),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if shuffle else None,
    )


class CachedPrototypeEDLHead(nn.Module):
    def __init__(
        self,
        in_features,
        num_classes=2,
        prototypes_per_class=4,
        topk=3,
        temperature=1.0,
        normalize=True,
        dropout=0.0,
        evidence_type="softplus",
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("CachedPrototypeEDLHead currently expects binary output.")
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be >= 1.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.evidence_type = evidence_type
        self.drop = nn.Dropout(p=float(dropout))

        prototypes = torch.empty(self.num_classes, self.prototypes_per_class, self.in_features)
        nn.init.normal_(prototypes, mean=0.0, std=0.02)
        self.prototypes = nn.Parameter(prototypes)

    def initialize_prototypes(self, prototypes):
        prototypes = torch.as_tensor(prototypes, dtype=self.prototypes.dtype)
        expected = (self.num_classes, self.prototypes_per_class, self.in_features)
        if tuple(prototypes.shape) != expected:
            raise ValueError(f"Expected prototypes with shape {expected}, got {tuple(prototypes.shape)}.")
        with torch.no_grad():
            self.prototypes.copy_(prototypes.to(device=self.prototypes.device, dtype=self.prototypes.dtype))

    def forward(self, features):
        features = self.drop(features.float())
        if self.normalize:
            features = F.normalize(features, dim=-1)
            prototypes = F.normalize(self.prototypes, dim=-1)
        else:
            prototypes = self.prototypes

        diff = features[:, None, None, :] - prototypes[None, :, :, :]
        distances = torch.sum(diff * diff, dim=-1)
        similarities = -distances / self.temperature
        class_logits = torch.logsumexp(similarities, dim=-1)

        evidence = get_evidence(class_logits, self.evidence_type)
        alpha = evidence + 1.0
        prob = compute_dirichlet_prob(evidence)
        uncertainty = compute_uncertainty(evidence)
        proto_evidence = get_evidence(similarities, self.evidence_type)

        out = {
            "logits": class_logits,
            "evidence": evidence,
            "alpha": alpha,
            "prob": prob,
            "uncertainty": uncertainty,
            "prototype_distances": distances,
            "prototype_similarity": similarities,
            "prototype_evidence": proto_evidence,
        }

        if self.topk > 0:
            topk = min(self.topk, self.prototypes_per_class)
            top_evidence, top_idx = torch.topk(proto_evidence, k=topk, dim=-1)
            out.update(
                {
                    "topk_proto_idx": top_idx,
                    "topk_proto_evidence": top_evidence,
                    "topk_proto_similarity": torch.gather(similarities, dim=-1, index=top_idx),
                    "topk_proto_distances": torch.gather(distances, dim=-1, index=top_idx),
                }
            )
        return out


class EDLPrototypeLoss(nn.Module):
    def __init__(
        self,
        loss_type="digamma",
        annealing_step=10,
        lambda_kl=0.1,
        pos_weight=None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.annealing_step = int(annealing_step)
        self.lambda_kl = float(lambda_kl)
        self.pos_weight = None if pos_weight is None else float(pos_weight)

    def forward(self, head_output, labels, epoch=0):
        labels = labels.long()
        target = F.one_hot(labels, num_classes=2).float().to(head_output["evidence"].device)
        kwargs = {
            "epoch_num": int(epoch),
            "annealing_step": self.annealing_step,
            "lambda_kl": self.lambda_kl,
            "pos_weight": self.pos_weight,
        }
        if self.loss_type == "digamma":
            return edl_digamma_loss(head_output["evidence"], target, **kwargs)
        if self.loss_type == "log":
            return edl_log_loss(head_output["evidence"], target, **kwargs)
        if self.loss_type == "mse":
            return edl_mse_loss(head_output["evidence"], target, **kwargs)
        raise ValueError(f"Unknown EDL loss type: {self.loss_type}")


class LossAdapter(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, head_output, labels, epoch=0):
        try:
            return self.criterion(head_output, labels, epoch=epoch)
        except TypeError:
            return self.criterion(head_output, labels)


def compute_inverse_class_weights(y_train):
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    info = {"n_neg": n_neg, "n_pos": n_pos, "pos_weight": None, "class_weights": None}
    if n_neg <= 0 or n_pos <= 0:
        return None, info
    pos_weight = n_neg / max(n_pos, 1)
    class_weights = np.array([1.0, pos_weight], dtype=np.float32)
    info["pos_weight"] = float(pos_weight)
    info["class_weights"] = [float(class_weights[0]), float(class_weights[1])]
    return class_weights, info


def _class_balanced_mean(values, labels, num_classes, balance_classes):
    if not balance_classes:
        return values.mean()

    means = []
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        if class_mask.any():
            means.append(values[class_mask].mean())
    if not means:
        return values.mean()
    return torch.stack(means).mean()


def _prototype_diversity_loss(model, margin):
    prototypes = getattr(model, "prototypes", None)
    if prototypes is None:
        return None
    if bool(getattr(model, "normalize", False)):
        prototypes = F.normalize(prototypes, p=2, dim=-1)

    losses = []
    for class_idx in range(prototypes.shape[0]):
        if prototypes.shape[1] < 2:
            continue
        pairwise_distance = torch.pdist(prototypes[class_idx], p=2)
        losses.append(F.relu(float(margin) - pairwise_distance).pow(2).mean())
    if not losses:
        return prototypes.new_zeros(())
    return torch.stack(losses).mean()


def prototype_regularization_loss(model, head_output, labels, args):
    distances = head_output.get("prototype_distances")
    if distances is None:
        zero = labels.new_zeros((), dtype=torch.float32)
        return zero, {
            "proto_loss": 0.0,
            "proto_loss_raw": 0.0,
            "proto_attract_loss": 0.0,
            "proto_separation_loss": 0.0,
            "proto_diversity_loss": 0.0,
        }

    labels = labels.long()
    distances = distances.clamp_min(0.0)
    num_classes = int(distances.shape[1])
    margin = float(getattr(args, "proto_margin", 1.0))
    balance_classes = str(getattr(args, "proto_balance_classes", "y")).lower() == "y"
    batch_indices = torch.arange(labels.shape[0], device=labels.device)

    own_distances = distances[batch_indices, labels]
    nearest_own = own_distances.min(dim=-1).values.clamp_min(1e-12).sqrt()
    attract_loss = _class_balanced_mean(nearest_own, labels, num_classes, balance_classes)

    other_mask = F.one_hot(labels, num_classes=num_classes).bool().unsqueeze(-1)
    other_distances = distances.masked_fill(other_mask, float("inf")).flatten(start_dim=1)
    nearest_other = other_distances.min(dim=1).values.clamp_min(1e-12).sqrt()
    separation_loss = _class_balanced_mean(
        F.relu(margin - nearest_other).pow(2),
        labels,
        num_classes,
        balance_classes,
    )

    diversity_loss = _prototype_diversity_loss(model, margin)
    if diversity_loss is None:
        diversity_loss = distances.new_zeros(())

    raw = (
        float(getattr(args, "proto_attract_weight", 0.0)) * attract_loss
        + float(getattr(args, "proto_separation_weight", 0.0)) * separation_loss
        + float(getattr(args, "proto_diversity_weight", 0.0)) * diversity_loss
    )
    total = float(getattr(args, "proto_loss_weight", 1.0)) * raw
    return total, {
        "proto_loss": float(total.detach().cpu()),
        "proto_loss_raw": float(raw.detach().cpu()),
        "proto_attract_loss": float(attract_loss.detach().cpu()),
        "proto_separation_loss": float(separation_loss.detach().cpu()),
        "proto_diversity_loss": float(diversity_loss.detach().cpu()),
    }


def compute_total_loss(model, head_output, labels, criterion, args, epoch):
    task_loss = criterion(head_output, labels, epoch=epoch)
    proto_loss, proto_stats = prototype_regularization_loss(model, head_output, labels, args)
    total_loss = task_loss + proto_loss
    stats = {
        "total_loss": float(total_loss.detach().cpu()),
        "task_loss": float(task_loss.detach().cpu()),
    }
    stats.update(proto_stats)
    return total_loss, stats


def train_one_epoch(model, loader, criterion, optimizer, device, args, epoch):
    model.train()
    total_loss = 0.0
    total_n = 0
    totals = {
        "task_loss": 0.0,
        "proto_loss": 0.0,
        "proto_loss_raw": 0.0,
        "proto_attract_loss": 0.0,
        "proto_separation_loss": 0.0,
        "proto_diversity_loss": 0.0,
    }
    for features, labels, _ in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        head_output = model(features)
        loss, stats = compute_total_loss(model, head_output, labels, criterion, args=args, epoch=epoch)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_n += batch_size
        for key in totals:
            totals[key] += stats[key] * batch_size
    avg_stats = {key: value / max(total_n, 1) for key, value in totals.items()}
    return total_loss / max(total_n, 1), avg_stats


def predict_arrays(model, features, labels, positions, criterion, args, device, epoch=0):
    loader = make_loader(
        features,
        labels,
        positions,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model.eval()
    row_ids = []
    output_chunks = {}
    total_loss = 0.0
    total_n = 0
    totals = {
        "task_loss": 0.0,
        "proto_loss": 0.0,
        "proto_loss_raw": 0.0,
        "proto_attract_loss": 0.0,
        "proto_separation_loss": 0.0,
        "proto_diversity_loss": 0.0,
    }

    with torch.no_grad():
        for batch_features, batch_labels, batch_rows in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            head_output = model(batch_features)
            loss, stats = compute_total_loss(model, head_output, batch_labels, criterion, args=args, epoch=epoch)
            batch_size = int(batch_labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_n += batch_size
            for key in totals:
                totals[key] += stats[key] * batch_size
            row_ids.append(batch_rows.numpy())
            for key, value in head_output.items():
                if torch.is_tensor(value):
                    output_chunks.setdefault(key, []).append(value.detach().cpu().numpy())

    if not row_ids:
        return np.asarray([], dtype=int), {}, float("nan"), {}

    row_ids = np.concatenate(row_ids).astype(int)
    order = np.argsort(row_ids)
    row_ids = row_ids[order]
    outputs = {}
    for key, chunks in output_chunks.items():
        outputs[key] = np.concatenate(chunks, axis=0)[order]
    avg_stats = {key: value / max(total_n, 1) for key, value in totals.items()}
    return row_ids, outputs, total_loss / max(total_n, 1), avg_stats


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def roc_auc_numpy(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(score)
    pos_rank_sum = ranks[y_true == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(y_true, score, threshold):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    y_pred = (score >= threshold).astype(int)
    auc = roc_auc_numpy(y_true, score)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    bacc = np.nanmean([sensitivity, specificity])
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "auc": auc,
        "bacc_at_threshold": float(bacc),
        "sensitivity_at_threshold": float(sensitivity),
        "specificity_at_threshold": float(specificity),
        "pred_pos_at_threshold": int(y_pred.sum()),
        "threshold": float(threshold),
    }


def grouped_frame(pred_df, label_col, score_col, group_cols, score_agg):
    missing = [col for col in group_cols if col not in pred_df.columns]
    if missing:
        return None
    return (
        pred_df.groupby(group_cols, dropna=False)
        .agg({label_col: "max", score_col: score_agg})
        .reset_index()
    )


def evaluate_predictions(pred_df, label_col, args):
    rows = []
    group_cols = parse_group_cols(args.group_cols)
    for split_name in ["train", "val", "test", "all"]:
        split_df = pred_df if split_name == "all" else pred_df[pred_df["split"] == split_name]
        if len(split_df) == 0:
            continue
        image_metrics = binary_metrics(split_df[label_col], split_df["prob_1"], args.threshold)
        image_metrics.update({"split": split_name, "grain": "image", "score_agg": ""})
        rows.append(image_metrics)

        grouped = grouped_frame(split_df, label_col, "prob_1", group_cols, args.score_agg)
        if grouped is not None and len(grouped) > 0:
            group_metrics = binary_metrics(grouped[label_col], grouped["prob_1"], args.threshold)
            group_metrics.update(
                {
                    "split": split_name,
                    "grain": ",".join(group_cols),
                    "score_agg": args.score_agg,
                }
            )
            rows.append(group_metrics)
    return pd.DataFrame(rows)


def select_metric(metrics_df, eval_split, args, eval_loss):
    if args.best_metric == "loss":
        return -float(eval_loss), {
            "split": eval_split,
            "grain": "loss",
            "val_loss": float(eval_loss),
        }
    if metrics_df.empty:
        return float("-inf"), {}

    group_grain = ",".join(parse_group_cols(args.group_cols))
    split_rows = metrics_df[metrics_df["split"] == eval_split]
    if split_rows.empty:
        split_rows = metrics_df[metrics_df["split"] == "all"]
    group_rows = split_rows[split_rows["grain"] == group_grain]
    metric_rows = group_rows if not group_rows.empty else split_rows
    if metric_rows.empty:
        return float("-inf"), {}
    metric_row = metric_rows.iloc[0].to_dict()
    key = "auc" if args.best_metric == "auc" else "bacc_at_threshold"
    value = float(metric_row.get(key, float("nan")))
    if not np.isfinite(value):
        value = float("-inf")
    return value, metric_row


def base_output_frame(meta, label_col, split_values, fold_value=None):
    base_cols = []
    for col in ["row_id", "patient_id", "image_id", "image_path", "cohort", "cohort_num", "cohert_num"]:
        if col in meta.columns and col not in base_cols:
            base_cols.append(col)
    if label_col in meta.columns and label_col not in base_cols:
        base_cols.append(label_col)
    out = meta[base_cols].copy()
    out["split"] = np.asarray(split_values)
    out["fold"] = -1 if fold_value is None else int(fold_value)
    if label_col != "label":
        out["label"] = meta[label_col].astype(int).to_numpy()
    return out


def attach_prediction_aliases(out, args):
    group_cols = parse_group_cols(args.group_cols)
    out["pred_class"] = (out["prob_1"] >= args.threshold).astype(int)
    out["prediction_score"] = out["prob_1"]
    out["pred_score"] = out["prob_1"]
    out["image_prediction_prob"] = out["prob_1"]
    if all(col in out.columns for col in group_cols):
        out["patient_prediction_prob"] = out.groupby(group_cols, dropna=False)["image_prediction_prob"].transform(args.score_agg)
        out["prediction_prob"] = out["patient_prediction_prob"]
    else:
        out["patient_prediction_prob"] = out["image_prediction_prob"]
        out["prediction_prob"] = out["image_prediction_prob"]
    out["prediction_label"] = (out["prediction_prob"] >= args.threshold).astype(int)
    out["prediction_group_cols"] = ",".join(group_cols)
    out["prediction_score_agg"] = args.score_agg
    out["prediction_threshold"] = float(args.threshold)
    return out


def add_topk_columns(out, outputs, is_dst=False):
    top_idx = outputs.get("topk_proto_idx")
    if top_idx is None:
        return out
    top_dist = outputs.get("topk_proto_distances")
    top_sim = outputs.get("topk_proto_similarity")
    top_ev = outputs.get("topk_proto_evidence")
    top_mass = outputs.get("topk_proto_mass")
    num_classes = top_idx.shape[1]
    topk = top_idx.shape[2]
    for class_idx in range(num_classes):
        for rank in range(topk):
            prefix = f"proto_c{class_idx}_top{rank + 1}"
            out[f"{prefix}_idx"] = top_idx[:, class_idx, rank].astype(int)
            if top_dist is not None:
                out[f"{prefix}_distance"] = top_dist[:, class_idx, rank]
            if top_sim is not None:
                out[f"{prefix}_similarity"] = top_sim[:, class_idx, rank]
            if top_ev is not None:
                out[f"{prefix}_evidence"] = top_ev[:, class_idx, rank]
            if is_dst:
                mass_values = top_mass[:, class_idx, rank] if top_mass is not None else top_ev[:, class_idx, rank]
                out[f"{prefix}_mass"] = mass_values
    return out


def outputs_to_frame(meta, label_col, split_values, outputs, args, fold_value=None, is_dst=False):
    out = base_output_frame(meta, label_col, split_values, fold_value=fold_value)
    prob = outputs["prob"]
    out["prob_0"] = prob[:, 0]
    out["prob_1"] = prob[:, 1]

    if is_dst:
        mass = outputs["dst_mass"]
        out["dst_mass_0"] = mass[:, 0]
        out["dst_mass_1"] = mass[:, 1]
        out["dst_mass_omega"] = mass[:, 2]
        out["evidence_0"] = mass[:, 0]
        out["evidence_1"] = mass[:, 1]
        out["alpha_0"] = mass[:, 0] + 1.0
        out["alpha_1"] = mass[:, 1] + 1.0
        out["uncertainty"] = mass[:, 2]
    else:
        evidence = outputs["evidence"]
        out["evidence_0"] = evidence[:, 0]
        out["evidence_1"] = evidence[:, 1]
        out["alpha_0"] = evidence[:, 0] + 1.0
        out["alpha_1"] = evidence[:, 1] + 1.0
        out["uncertainty"] = outputs["uncertainty"]

    out = attach_prediction_aliases(out, args)
    out = add_topk_columns(out, outputs, is_dst=is_dst)
    return out


def prepare_training_positions(meta, labels, split_values, args):
    if args.max_samples is not None and args.max_samples > 0 and len(meta) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        keep = []
        split_label = pd.Series(split_values.astype(str)) + "|" + pd.Series(labels.astype(str))
        all_positions = np.arange(len(meta), dtype=int)
        for _, group_pos in pd.Series(all_positions).groupby(split_label, sort=False):
            group_values = group_pos.to_numpy(dtype=int)
            quota = int(round(len(group_values) * args.max_samples / len(meta)))
            quota = min(len(group_values), max(1, quota))
            keep.extend(rng.choice(group_values, size=quota, replace=False).tolist())
        keep = np.asarray(sorted(set(keep)), dtype=int)
    else:
        keep = np.arange(len(meta), dtype=int)

    train_positions = np.intersect1d(np.flatnonzero(split_values == "train"), keep)
    val_positions = np.intersect1d(np.flatnonzero(split_values == "val"), keep)
    test_positions = np.intersect1d(np.flatnonzero(split_values == "test"), keep)
    train_positions = stratified_limit_positions(
        train_positions,
        labels,
        args.max_train_samples,
        args.seed,
    )
    if len(train_positions) == 0:
        raise ValueError("No training rows after split assignment.")
    if len(np.unique(labels[train_positions])) < 2:
        raise ValueError("Training split must contain both classes.")
    if len(val_positions) > 0:
        eval_positions = val_positions
        eval_split = "val"
    elif len(test_positions) > 0:
        eval_positions = test_positions
        eval_split = "test"
        print("[WARN] No validation rows; early stopping will use test rows.")
    else:
        eval_positions = train_positions
        eval_split = "train"
        print("[WARN] No validation/test rows; early stopping will use train rows.")
    return train_positions, eval_positions, eval_split, keep


def save_history(history, output_dir, prefix):
    history_path = output_dir / f"{prefix}_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    return history_path


def _plot_loss_curve(plot_df, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.6, 7.68), dpi=100)
    ax.plot(
        plot_df["epoch"],
        plot_df["train_eval_loss"],
        color="#1f77b4",
        linewidth=2.0,
        label="train loss",
    )
    ax.plot(
        plot_df["epoch"],
        plot_df["eval_loss"],
        color="#d62728",
        linewidth=2.0,
        label="val loss",
    )
    ax.set_title(title, fontsize=20, pad=10)
    ax.set_xlabel("epoch", fontsize=16)
    ax.set_ylabel("loss", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=14, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_loss_curve(history, output_dir, prefix, title_prefix, prototypes_per_class):
    history_df = pd.DataFrame(history)
    if history_df.empty or "epoch" not in history_df.columns:
        return None, []
    if "train_eval_loss" not in history_df.columns or "eval_loss" not in history_df.columns:
        return None, []

    try:
        import matplotlib  # noqa: F401
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib; loss curve was not saved: {exc}")
        return None, []

    safe_title = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(title_prefix)).strip("_")
    curve_dir = output_dir / "loss" / f"{safe_title}_k_{int(prototypes_per_class)}"
    curve_dir.mkdir(parents=True, exist_ok=True)

    curve_files = []
    needed = ["fold", "epoch", "train_eval_loss", "eval_loss"]
    history_df = history_df[needed].copy()
    for col in needed:
        history_df[col] = pd.to_numeric(history_df[col], errors="coerce")
    history_df = history_df.dropna(subset=["fold", "epoch", "train_eval_loss", "eval_loss"])

    for fold, fold_df in history_df.groupby("fold", sort=True):
        plot_df = fold_df.sort_values("epoch")
        if plot_df.empty:
            continue
        fold_int = int(fold)
        path = curve_dir / f"fold_{fold_int}.png"
        title = f"{title_prefix} k={int(prototypes_per_class)} - fold {fold_int}"
        _plot_loss_curve(plot_df, path, title)
        curve_files.append(path)

    loss_curve_path = output_dir / f"{prefix}_loss_curve.png"
    mean_df = (
        history_df.groupby("epoch", as_index=False)[["train_eval_loss", "eval_loss"]]
        .mean()
        .sort_values("epoch")
    )
    if not mean_df.empty:
        _plot_loss_curve(
            mean_df,
            loss_curve_path,
            f"{title_prefix} k={int(prototypes_per_class)} - mean",
        )
    else:
        loss_curve_path = None

    return loss_curve_path, curve_files


def write_report(path, metrics_df, manifest):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{manifest['method']} cached-feature report\n")
        f.write(f"created_at: {manifest['created_at']}\n")
        f.write(f"best_metric: {manifest['best_metric_name']}={manifest['best_metric_value']:.6f}\n")
        f.write(f"best_epoch: {manifest['best_epoch']}\n")
        f.write(f"threshold: {manifest['threshold']}\n\n")
        if metrics_df.empty:
            f.write("No metrics available.\n")
        else:
            f.write(metrics_df.to_string(index=False))
            f.write("\n")


def run_cached_prototype_experiment(args, build_model, build_criterion, method, output_prefix, is_dst=False):
    set_seed(args.seed)
    device = choose_device(args)
    output_dir, run_id = timestamped_output_dir(args.output_dir, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir = output_dir / "per_model_predictions"
    per_model_dir.mkdir(parents=True, exist_ok=True)

    features, meta, labels, embedding_path, metadata_path = load_embedding_bundle(args)
    original_num_rows = int(len(meta))

    if int(args.n_folds) <= 1:
        split_values, split_info = assign_splits(meta, labels, args)
        folds = [(int(args.val_fold), split_values.to_numpy(), split_info)]
    else:
        folds = []
        for fold in range(int(args.fold_start), int(args.fold_start) + int(args.n_folds)):
            split_values, split_info = assign_cv_splits(meta, labels, args, fold)
            folds.append((fold, split_values.to_numpy(), split_info))

    all_metrics = []
    fold_frames = []
    history_all = []
    best_overall = {"metric": float("-inf"), "loss": float("inf"), "epoch": -1, "fold": None}

    for fold, split_values, split_info in folds:
        print(f"\n{'=' * 60}")
        print(f"  {method} cached-feature training fold {fold}")
        print(f"{'=' * 60}")
        train_positions, eval_positions, eval_split, sampled_positions = prepare_training_positions(
            meta,
            labels,
            split_values,
            args,
        )
        y_train = labels[train_positions]
        feature_dim = int(features.shape[1])
        model = build_model(feature_dim).to(device)

        if args.prototype_init == "kmeans":
            prototypes = classwise_prototypes(
                features[train_positions],
                labels[train_positions],
                args.prototypes_per_class,
                args.seed + int(fold),
            )
            model.initialize_prototypes(torch.as_tensor(prototypes, dtype=torch.float32).to(device))
            prototype_initialized_from = "train_embedding_kmeans"
        else:
            prototype_initialized_from = "random"

        _, class_weight_info = compute_inverse_class_weights(y_train)
        criterion = LossAdapter(build_criterion(y_train, class_weight_info)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_loader = make_loader(
            features,
            labels,
            train_positions,
            args.batch_size,
            shuffle=True,
            seed=args.seed + int(fold),
            num_workers=args.num_workers,
        )

        best_metric = float("-inf")
        best_epoch = -1
        best_state = None
        no_improve = 0
        history = []

        print(f"[device] {device}")
        print(f"[split counts] {pd.Series(split_values).value_counts().to_dict()}")
        print(f"[train rows] {len(train_positions)}  [eval rows] {len(eval_positions)} ({eval_split})")

        for epoch in range(int(args.epochs)):
            train_step_loss, train_step_stats = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                args,
                epoch,
            )
            _, _, train_eval_loss, train_eval_stats = predict_arrays(
                model,
                features,
                labels,
                train_positions,
                criterion,
                args,
                device,
                epoch=epoch,
            )
            eval_row_ids, eval_outputs, eval_loss, eval_stats = predict_arrays(
                model,
                features,
                labels,
                eval_positions,
                criterion,
                args,
                device,
                epoch=epoch,
            )
            eval_meta = meta.iloc[eval_row_ids].reset_index(drop=True)
            eval_frame = outputs_to_frame(
                eval_meta,
                args.label,
                split_values[eval_row_ids],
                eval_outputs,
                args,
                fold_value=fold,
                is_dst=is_dst,
            )
            metrics_df = evaluate_predictions(eval_frame, "label", args)
            metric_value, metric_row = select_metric(metrics_df, eval_split, args, eval_loss)
            improved = metric_value > best_metric + 1e-8
            if improved:
                best_metric = metric_value
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            history_row = {
                "fold": int(fold),
                "epoch": epoch + 1,
                "train_loss": float(train_eval_loss),
                "train_step_loss": float(train_step_loss),
                "train_eval_loss": float(train_eval_loss),
                "eval_split": eval_split,
                "eval_loss": float(eval_loss),
                "best_metric_name": args.best_metric,
                "eval_metric": float(metric_value),
                "best_metric": float(best_metric),
                "improved": bool(improved),
            }
            for key, value in train_step_stats.items():
                history_row[f"train_step_{key}"] = value
            for key, value in train_eval_stats.items():
                history_row[f"train_eval_{key}"] = value
            for key, value in eval_stats.items():
                history_row[f"{eval_split}_{key}"] = value
            for key in [
                "grain",
                "auc",
                "bacc_at_threshold",
                "sensitivity_at_threshold",
                "specificity_at_threshold",
                "pred_pos_at_threshold",
                "val_loss",
            ]:
                if key in metric_row:
                    history_row[f"eval_{key}"] = metric_row[key]
            history.append(history_row)
            best_text = f"best_val_loss={-best_metric:.4f}" if args.best_metric == "loss" else f"best={best_metric:.4f}"
            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"train_step_loss={train_step_loss:.4f} train_eval_loss={train_eval_loss:.4f} "
                f"{eval_split}_loss={eval_loss:.4f} "
                f"metric={metric_value:.4f} {best_text}"
            )

            if args.patience > 0 and no_improve >= args.patience:
                print(f"Early stopping after {epoch + 1} epochs (patience={args.patience}).")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        all_positions = np.arange(len(meta), dtype=int)
        row_ids, outputs, _, _ = predict_arrays(
            model,
            features,
            labels,
            all_positions,
            criterion,
            args,
            device,
            epoch=max(best_epoch, 0),
        )
        pred_frame = outputs_to_frame(
            meta.iloc[row_ids].reset_index(drop=True),
            args.label,
            split_values[row_ids],
            outputs,
            args,
            fold_value=fold,
            is_dst=is_dst,
        )
        fold_csv = per_model_dir / f"fold{fold}_edl_predictions.csv"
        pred_frame.to_csv(fold_csv, index=False)
        print(f"[INFO] Fold prediction CSV saved to: {fold_csv}")

        fold_metric_df = evaluate_predictions(pred_frame, "label", args)
        fold_metric_df.insert(0, "model", f"fold{fold}")
        all_metrics.append(fold_metric_df)
        fold_frames.append(pred_frame)
        history_all.extend(history)

        if best_metric > best_overall["metric"]:
            best_loss = -float(best_metric) if args.best_metric == "loss" else float("nan")
            best_overall = {
                "metric": float(best_metric),
                "loss": best_loss,
                "epoch": int(best_epoch),
                "fold": int(fold),
            }

        ckpt_path = output_dir / f"{output_prefix}_fold{fold}_best.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "method": method,
                "feature_dim": feature_dim,
                "fold": int(fold),
                "best_epoch": int(best_epoch) + 1,
                "best_metric": float(best_metric),
                "best_val_loss": -float(best_metric) if args.best_metric == "loss" else None,
                "best_metric_name": args.best_metric,
                "args": json_safe(vars(args)),
                "class_weight_info": class_weight_info,
                "prototype_initialized_from": prototype_initialized_from,
                "split_info": split_info,
            },
            ckpt_path,
        )

    ensemble_frame = fold_frames[0].copy()
    proto_cols = [col for col in ensemble_frame.columns if col.startswith("proto_")]
    if proto_cols:
        ensemble_frame = ensemble_frame.drop(columns=proto_cols)
    if len(folds) > 1:
        split_stack = np.stack([split_values for _, split_values, _ in folds], axis=0)
        ensemble_split = np.where((split_stack == "test").any(axis=0), "test", "train")
        ensemble_frame["split"] = ensemble_split
    ensemble_frame["fold"] = -1
    prob_stack = np.stack([frame[["prob_0", "prob_1"]].to_numpy(dtype=float) for frame in fold_frames], axis=0)
    ensemble_prob = prob_stack.mean(axis=0)
    ensemble_frame["prob_0"] = ensemble_prob[:, 0]
    ensemble_frame["prob_1"] = ensemble_prob[:, 1]

    for col in ["evidence_0", "evidence_1", "alpha_0", "alpha_1", "uncertainty", "dst_mass_0", "dst_mass_1", "dst_mass_omega"]:
        if all(col in frame.columns for frame in fold_frames):
            values = np.stack([frame[col].to_numpy(dtype=float) for frame in fold_frames], axis=0).mean(axis=0)
            ensemble_frame[col] = values
    if not is_dst and {"evidence_0", "evidence_1"}.issubset(ensemble_frame.columns):
        ensemble_frame["alpha_0"] = ensemble_frame["evidence_0"] + 1.0
        ensemble_frame["alpha_1"] = ensemble_frame["evidence_1"] + 1.0
    if is_dst and {"dst_mass_0", "dst_mass_1", "dst_mass_omega"}.issubset(ensemble_frame.columns):
        ensemble_frame["evidence_0"] = ensemble_frame["dst_mass_0"]
        ensemble_frame["evidence_1"] = ensemble_frame["dst_mass_1"]
        ensemble_frame["alpha_0"] = ensemble_frame["dst_mass_0"] + 1.0
        ensemble_frame["alpha_1"] = ensemble_frame["dst_mass_1"] + 1.0
        ensemble_frame["uncertainty"] = ensemble_frame["dst_mass_omega"]

    ensemble_frame = attach_prediction_aliases(ensemble_frame, args)
    ensemble_csv = per_model_dir / "ensemble_edl_predictions.csv"
    ensemble_frame.to_csv(ensemble_csv, index=False)

    ensemble_metrics = evaluate_predictions(ensemble_frame, "label", args)
    ensemble_metrics.insert(0, "model", "ensemble")
    all_metrics.append(ensemble_metrics)
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    metrics_path = output_dir / f"{output_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    history_path = save_history(history_all, output_dir, output_prefix)
    curve_title_prefix = "DST" if is_dst else "EDL-Prototype"
    loss_curve_path, loss_curve_files = save_loss_curve(
        history_all,
        output_dir,
        output_prefix,
        curve_title_prefix,
        args.prototypes_per_class,
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "embedding_path": str(embedding_path),
        "metadata_path": str(metadata_path),
        "requested_output_dir": getattr(args, "_requested_output_dir", str(output_dir)),
        "output_dir": str(output_dir),
        "run_id": run_id,
        "num_rows": int(len(meta)),
        "original_num_rows": original_num_rows,
        "feature_dim": int(features.shape[1]),
        "label": args.label,
        "device": args.device,
        "gpu_id": args.gpu_id,
        "effective_device": str(device),
        "n_folds": int(args.n_folds),
        "prototypes_per_class": int(args.prototypes_per_class),
        "prototype_topk": int(args.prototype_topk),
        "prototype_init": args.prototype_init,
        "proto_attract_weight": float(getattr(args, "proto_attract_weight", 0.0)),
        "proto_separation_weight": float(getattr(args, "proto_separation_weight", 0.0)),
        "proto_diversity_weight": float(getattr(args, "proto_diversity_weight", 0.0)),
        "proto_loss_weight": float(getattr(args, "proto_loss_weight", 1.0)),
        "proto_margin": float(getattr(args, "proto_margin", 1.0)),
        "proto_balance_classes": getattr(args, "proto_balance_classes", "y"),
        "threshold": float(args.threshold),
        "normalize": not bool(args.no_normalize),
        "group_cols": parse_group_cols(args.group_cols),
        "score_agg": args.score_agg,
        "best_metric_name": "val_loss" if args.best_metric == "loss" else args.best_metric,
        "best_metric_value": float(best_overall["loss"] if args.best_metric == "loss" else best_overall["metric"]),
        "best_metric_larger_is_better_value": float(best_overall["metric"]),
        "best_epoch": int(best_overall["epoch"]) + 1,
        "best_fold": best_overall["fold"],
        "prediction_file": str(ensemble_csv),
        "metrics_file": str(metrics_path),
        "history_file": str(history_path),
        "loss_curve_file": str(loss_curve_path) if loss_curve_path is not None else None,
        "loss_curve_files": [str(path) for path in loss_curve_files],
        "args": json_safe(vars(args)),
    }
    manifest_path = output_dir / f"{output_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")

    report_path = per_model_dir / "edl_eval_report.txt"
    write_report(report_path, metrics_df, manifest)

    print("\nDone.")
    print(f"  predictions: {ensemble_csv}")
    print(f"  metrics:     {metrics_path}")
    print(f"  history:     {history_path}")
    if loss_curve_path is not None:
        print(f"  loss curve:  {loss_curve_path}")
    print(f"  manifest:    {manifest_path}")
    print(f"  report:      {report_path}")
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))
    return metrics_df, manifest
