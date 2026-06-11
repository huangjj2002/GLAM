import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dst_pytorch import Dempster_Shafer_Module, DistanceActivation_layer


def dst_activation_init_gamma(gamma_init):
    """Map distance-decay rate to DistanceActivation_layer eta init."""
    return math.sqrt(max(float(gamma_init), 1e-6))


def dst_activation_init_alpha(alpha_init):
    """Pass through DistanceActivation_layer xi.weight init."""
    return float(alpha_init)


def pignistic(mass, n_class):
    """Convert normalized DST mass to pignistic probabilities."""
    class_mass = mass[..., :n_class]
    omega = mass[..., n_class]
    probs = class_mass + (1.0 / n_class) * omega.unsqueeze(-1)
    return probs, omega


class PrototypeDSTNLLLoss(nn.Module):
    """NLL on pignistic probabilities, matching the reference DST loss."""

    def __init__(self, class_weights=None, eps=1e-10):
        super().__init__()
        self.eps = float(eps)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights, dtype=torch.float32),
            )

    def forward(self, head_output, targets):
        probs = head_output["prob"].clamp(min=self.eps, max=1.0)
        target_indices = targets.long().to(probs.device)
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=probs.device, dtype=probs.dtype)
        return F.nll_loss(torch.log(probs), target_indices, weight=weight)


class PrototypeDSTHead(nn.Module):
    """Prototype head built on Dempster-Shafer evidence combination."""

    def __init__(
        self,
        in_features,
        num_classes=2,
        prototypes_per_class=4,
        topk=3,
        normalize=True,
        gamma_init=1.0,
        alpha_init=0.0,
        dropout=0.0,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("PrototypeDSTHead currently expects binary output.")
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be >= 1.")

        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.n_prototypes = self.num_classes * self.prototypes_per_class
        self.topk = int(topk)
        self.normalize = bool(normalize)
        self.gamma_init = float(gamma_init)
        self.alpha_init = float(alpha_init)
        self.drop = nn.Dropout(p=dropout)

        init_gamma = dst_activation_init_gamma(self.gamma_init)
        init_alpha = dst_activation_init_alpha(self.alpha_init)
        self.ds_module = Dempster_Shafer_Module(
            n_feature_maps=self.in_features,
            n_classes=self.num_classes,
            n_prototypes=self.n_prototypes,
        )
        self.ds_module.ds1_activate = DistanceActivation_layer(
            n_prototypes=self.n_prototypes,
            init_alpha=init_alpha,
            init_gamma=init_gamma,
        )
        self.reset_parameters()

    @property
    def prototypes(self):
        return self.ds_module.ds1.w.view(
            self.num_classes,
            self.prototypes_per_class,
            self.in_features,
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ds_module.ds1.w)

    def initialize_prototypes(self, prototypes):
        prototypes = torch.as_tensor(prototypes, dtype=self.ds_module.ds1.w.dtype)
        expected = (self.num_classes, self.prototypes_per_class, self.in_features)
        if tuple(prototypes.shape) != expected:
            raise ValueError(f"Expected prototypes with shape {expected}, got {tuple(prototypes.shape)}.")
        with torch.no_grad():
            self.ds_module.ds1.w.copy_(
                prototypes.reshape(self.n_prototypes, self.in_features).to(
                    device=self.ds_module.ds1.w.device,
                    dtype=self.ds_module.ds1.w.dtype,
                )
            )

    def _compute_distances(self, x):
        prototypes = self.ds_module.ds1.w
        if self.normalize:
            x = F.normalize(x, dim=-1)
            prototypes = F.normalize(prototypes, dim=-1)
        return (x[:, None, :] - prototypes[None, :, :]).pow(2).sum(dim=-1)

    def _reshape_prototypes(self, tensor):
        batch_size = tensor.shape[0]
        return tensor.view(batch_size, self.num_classes, self.prototypes_per_class)

    def forward(self, x):
        x = self.drop(x.float())
        distances = self._compute_distances(x)
        ed_ac = self.ds_module.ds1_activate(distances)
        mass_prototypes = self.ds_module.ds2(ed_ac)
        mass_prototypes_omega = self.ds_module.ds2_omega(mass_prototypes)
        mass_dempster = self.ds_module.ds3_dempster(mass_prototypes_omega)
        mass = self.ds_module.ds3_normalize(mass_dempster)

        prob, uncertainty = pignistic(mass, self.num_classes)

        prototype_mass = torch.zeros(
            x.shape[0],
            self.num_classes,
            self.prototypes_per_class,
            device=x.device,
            dtype=x.dtype,
        )
        for class_idx in range(self.num_classes):
            start = class_idx * self.prototypes_per_class
            end = start + self.prototypes_per_class
            prototype_mass[:, class_idx, :] = mass_prototypes[:, start:end, class_idx]

        distances_by_class = self._reshape_prototypes(distances)
        similarity = self._reshape_prototypes(ed_ac)

        out = {
            "prob": prob,
            "uncertainty": uncertainty,
            "dst_mass": mass,
            "prototype_distances": distances_by_class,
            "prototype_similarity": similarity,
            "prototype_evidence": prototype_mass,
            "prototype_mass": prototype_mass,
        }

        if self.topk > 0:
            topk = min(self.topk, self.prototypes_per_class)
            top_mass, top_idx = torch.topk(prototype_mass, k=topk, dim=-1)
            top_similarity = torch.gather(similarity, dim=-1, index=top_idx)
            top_distances = torch.gather(distances_by_class, dim=-1, index=top_idx)
            out.update(
                {
                    "topk_proto_idx": top_idx,
                    "topk_proto_evidence": top_mass,
                    "topk_proto_mass": top_mass,
                    "topk_proto_similarity": top_similarity,
                    "topk_proto_distances": top_distances,
                }
            )

        return out

    def initialize_from_embeddings(self, embeddings, labels, random_state=0):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        labels = np.asarray(labels).astype(int)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.in_features:
            raise ValueError(
                f"Expected embeddings with shape (N, {self.in_features}), got {embeddings.shape}."
            )
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError("Embeddings and labels must contain the same number of rows.")

        working_embeddings = embeddings
        if self.normalize:
            norms = np.linalg.norm(working_embeddings, axis=1, keepdims=True)
            working_embeddings = working_embeddings / np.clip(norms, 1e-12, None)

        centers = []
        warnings = []
        global_center = working_embeddings.mean(axis=0, keepdims=True)
        rng = np.random.default_rng(random_state)
        for class_idx in range(self.num_classes):
            class_embeddings = working_embeddings[labels == class_idx]
            if len(class_embeddings) == 0:
                class_centers = np.repeat(global_center, self.prototypes_per_class, axis=0)
                warnings.append(f"class {class_idx} has no samples; using global mean prototypes")
            elif len(class_embeddings) >= self.prototypes_per_class:
                class_centers = _simple_kmeans(
                    class_embeddings,
                    self.prototypes_per_class,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
            else:
                repeat = int(np.ceil(self.prototypes_per_class / len(class_embeddings)))
                class_centers = np.tile(class_embeddings, (repeat, 1))[: self.prototypes_per_class]
                warnings.append(f"class {class_idx} has fewer samples than prototypes; repeated centers")
            centers.append(class_centers.astype(np.float32))

        self.initialize_prototypes(np.stack(centers, axis=0))
        return warnings


def _simple_kmeans(x, k, seed, max_iter=100):
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
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


PrototypeEDLHead = PrototypeDSTHead
