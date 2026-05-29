"""
Prototype + EDL model.

This module keeps the existing GLAM backbone and EDL training behaviour, but
replaces the linear EDL head with a class-wise prototype head.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from edl_loss import compute_dirichlet_prob, compute_uncertainty, get_evidence
from edl_model import EDLModel


class PrototypeEDLHead(nn.Module):
    """Class-wise prototype head that turns distances into class logits."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 2,
        prototypes_per_class: int = 4,
        temperature: float = 1.0,
        normalize: bool = True,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("PrototypeEDLHead currently expects binary classification.")
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be >= 1.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

        prototypes = torch.empty(
            self.num_classes,
            self.prototypes_per_class,
            self.in_features,
        )
        nn.init.normal_(prototypes, mean=0.0, std=0.02)
        self.prototypes = nn.Parameter(prototypes)

    def forward(self, features: torch.Tensor):
        if self.normalize:
            features = F.normalize(features, dim=-1)
            prototypes = F.normalize(self.prototypes, dim=-1)
        else:
            prototypes = self.prototypes

        diff = features[:, None, None, :] - prototypes[None, :, :, :]
        distances = torch.sum(diff * diff, dim=-1)
        similarities = -distances / self.temperature
        class_logits = torch.logsumexp(similarities, dim=-1)
        return class_logits, distances, similarities


class PrototypeEDLModel(EDLModel):
    """EDLModel variant whose evidence comes from class-wise prototypes."""

    def __init__(
        self,
        edl_proto_k: int = 4,
        edl_proto_topk: int = 3,
        prototype_temperature: float = 1.0,
        prototype_normalize: bool = True,
        prototype_init: str = "kmeans",
        prototype_init_max_samples_per_class: int = 0,
        prototype_init_batch_size: int = 0,
        prototype_init_num_workers: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        feature_dim = self.glam.img_encoder_q.feature_dim
        self.edl_head = PrototypeEDLHead(
            in_features=feature_dim,
            num_classes=self.num_classes,
            prototypes_per_class=edl_proto_k,
            temperature=prototype_temperature,
            normalize=prototype_normalize,
        )
        self.edl_proto_topk = int(edl_proto_topk)
        self.prototype_init = str(prototype_init)
        self.prototype_init_max_samples_per_class = int(prototype_init_max_samples_per_class or 0)
        self.prototype_init_batch_size = int(prototype_init_batch_size or 0)
        self.prototype_init_num_workers = int(prototype_init_num_workers or 0)

        self.hparams["edl_proto_k"] = int(edl_proto_k)
        self.hparams["edl_proto_topk"] = int(edl_proto_topk)
        self.hparams["prototype_temperature"] = float(prototype_temperature)
        self.hparams["prototype_normalize"] = bool(prototype_normalize)
        self.hparams["prototype_init"] = str(prototype_init)
        self.hparams["prototype_init_max_samples_per_class"] = int(
            prototype_init_max_samples_per_class or 0
        )
        self.hparams["prototype_init_batch_size"] = int(prototype_init_batch_size or 0)
        self.hparams["prototype_init_num_workers"] = int(prototype_init_num_workers or 0)

        if getattr(self.hparams, "freeze_backbone", False):
            self._freeze_backbone_for_edl()

        self.all_proto_top_idx = None
        self.all_proto_top_distance = None
        self.all_proto_top_similarity = None
        self.all_proto_top_evidence = None
        self._last_proto_distances = None
        self._last_proto_similarities = None

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"### [PrototypeEDL] prototypes_per_class = {edl_proto_k}")
        print(f"### [PrototypeEDL] prototype_temperature = {prototype_temperature}")
        print(f"### [PrototypeEDL] prototype_normalize = {prototype_normalize}")
        print(f"### [PrototypeEDL] Total params: {total_params:,}")
        print(f"### [PrototypeEDL] Trainable params: {trainable_params:,}")

    def _extract_img_features(self, images: torch.Tensor) -> torch.Tensor:
        if getattr(self.hparams, "freeze_backbone", False):
            with torch.no_grad():
                img_feat_q, _, img_full = self.glam.img_encoder_q(images)
                if self.glam.hparams.pool_feat:
                    img_feat_q = img_full.mean(dim=1)
            return img_feat_q

        img_feat_q, _, img_full = self.glam.img_encoder_q(images)
        if self.glam.hparams.pool_feat:
            img_feat_q = img_full.mean(dim=1)
        return img_feat_q

    def forward_edl(self, images: torch.Tensor):
        features = self._extract_img_features(images)
        logits, distances, similarities = self.edl_head(features)

        evidence = get_evidence(logits, self.hparams.evidence_type)
        alpha = evidence + 1.0
        prob = compute_dirichlet_prob(evidence)
        uncertainty = compute_uncertainty(evidence)

        self._last_proto_distances = distances
        self._last_proto_similarities = similarities
        return logits, evidence, alpha, prob, uncertainty

    def _shared_step(self, batch, batch_idx, split="train"):
        output = super()._shared_step(batch, batch_idx, split=split)
        if split == "test":
            self._append_latest_proto_outputs()
        return output

    def _append_latest_proto_outputs(self):
        if self._last_proto_distances is None or self._last_proto_similarities is None:
            return

        topk = min(self.edl_proto_topk, self.edl_head.prototypes_per_class)
        if topk <= 0:
            return

        distances = self._last_proto_distances.detach()
        similarities = self._last_proto_similarities.detach()
        top_distance, top_idx = torch.topk(distances, k=topk, dim=-1, largest=False)
        top_similarity = torch.gather(similarities, dim=-1, index=top_idx)
        proto_evidence = get_evidence(similarities, self.hparams.evidence_type)
        top_evidence = torch.gather(proto_evidence, dim=-1, index=top_idx)

        arrays = {
            "all_proto_top_idx": top_idx.cpu().numpy().astype(np.int64),
            "all_proto_top_distance": top_distance.float().cpu().numpy(),
            "all_proto_top_similarity": top_similarity.float().cpu().numpy(),
            "all_proto_top_evidence": top_evidence.float().cpu().numpy(),
        }
        for attr, value in arrays.items():
            current = getattr(self, attr)
            if current is None:
                setattr(self, attr, value)
            else:
                setattr(self, attr, np.concatenate([current, value], axis=0))

    @torch.no_grad()
    def initialize_prototypes_from_loader(
        self,
        dataloader,
        device=None,
        random_state: int = 42,
    ):
        """Initialize prototypes with per-class KMeans on current-fold train embeddings."""
        if device is None:
            device = next(self.parameters()).device

        was_training = self.training
        self.eval()

        embeddings = []
        labels = []
        for batch in dataloader:
            images = batch["imgs"].to(device, non_blocking=True)
            batch_labels = batch["multi_hot_label"]
            if batch_labels.shape[-1] == 1:
                batch_targets = batch_labels.reshape(-1).long()
            else:
                batch_targets = batch_labels.argmax(dim=-1).long()

            features = self._extract_img_features(images)
            if self.edl_head.normalize:
                features = F.normalize(features, dim=-1)
            embeddings.append(features.detach().float().cpu().numpy())
            labels.append(batch_targets.detach().cpu().numpy())

        if was_training:
            self.train()

        if not embeddings:
            raise RuntimeError("Prototype initialization failed: no training embeddings collected.")

        embedding_array = np.concatenate(embeddings, axis=0)
        label_array = np.concatenate(labels, axis=0).astype(int)
        centers = self._compute_kmeans_centers(
            embedding_array,
            label_array,
            random_state=random_state,
        )

        centers_tensor = torch.as_tensor(
            centers,
            dtype=self.edl_head.prototypes.dtype,
            device=self.edl_head.prototypes.device,
        )
        self.edl_head.prototypes.data.copy_(centers_tensor)
        print(
            "### [PrototypeEDL] KMeans prototype init complete: "
            f"embeddings={len(embedding_array)}, shape={tuple(centers_tensor.shape)}"
        )

    def _compute_kmeans_centers(self, embeddings, labels, random_state: int = 42):
        k = self.edl_head.prototypes_per_class
        centers = []

        for class_idx in range(self.num_classes):
            class_embeddings = embeddings[labels == class_idx]
            max_samples = self.prototype_init_max_samples_per_class
            if max_samples > 0 and len(class_embeddings) > max_samples:
                rng = np.random.default_rng(random_state + class_idx)
                keep = rng.choice(len(class_embeddings), size=max_samples, replace=False)
                class_embeddings = class_embeddings[keep]

            if len(class_embeddings) == 0:
                print(
                    "### [PrototypeEDL][Warning] "
                    f"class {class_idx} has no train embeddings; using global mean."
                )
                class_centers = np.repeat(embeddings.mean(axis=0, keepdims=True), k, axis=0)
            elif len(class_embeddings) < k:
                print(
                    "### [PrototypeEDL][Warning] "
                    f"class {class_idx} has only {len(class_embeddings)} samples for K={k}; "
                    "repeating available embeddings."
                )
                repeat_idx = np.arange(k) % len(class_embeddings)
                class_centers = class_embeddings[repeat_idx]
            else:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                class_centers = kmeans.fit(class_embeddings).cluster_centers_

            centers.append(class_centers.astype(np.float32))
            print(
                "### [PrototypeEDL] "
                f"class={class_idx} prototype init samples={len(class_embeddings)}"
            )

        return np.stack(centers, axis=0)
