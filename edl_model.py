"""
EDL Model: Wraps the pretrained GLAM backbone with an Evidential Deep Learning head.

This module loads a pretrained GLAM checkpoint, replaces the final linear
classification head with an EDL output layer, and trains with Dirichlet-based
loss functions for uncertainty quantification.

Original project files are NOT modified.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import MulticlassConfusionMatrix
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch.distributed as dist

from pytorch_lightning import LightningModule, Trainer

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from model import GLAM
from edl_loss import (
    get_evidence,
    edl_digamma_loss,
    edl_log_loss,
    edl_mse_loss,
    compute_uncertainty,
    compute_dirichlet_prob,
)

from dataset.utils import get_specificity_with_sensitivity, pfbeta


class EDLModel(LightningModule):
    """
    Evidential Deep Learning model built on top of a pretrained GLAM backbone.

    The last linear classification head is replaced so that the model outputs
    raw logits which are then converted to evidence via softplus (or other
    activation). A Dirichlet distribution is constructed from the evidence,
    and the expected cross-entropy (digamma) loss is used for optimisation.
    """

    def __init__(
        self,
        pretrained_model: str = None,
        num_classes: int = 2,
        evidence_type: str = "softplus",
        edl_loss_type: str = "digamma",
        annealing_step: int = 10,
        lambda_kl: float = 0.1,
        pos_weight: float = None,
        learning_rate: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.05,
        batch_size: int = 32,
        max_epochs: int = 25,
        warm_up: int = 1,
        min_lr: float = 1e-8,
        adafactor: bool = False,
        sgd: bool = False,
        img_encoder: str = "dinov2_vitb14_reg",
        emb_dim: int = 512,
        linear_proj: bool = False,
        img_size: int = 336,
        crop_size: int = 336,
        freeze_backbone: bool = False,
        freeze_vit: bool = False,
        vit_grad_ckpt: bool = False,
        pretrained_encoder: str = None,
        devices: int = 1,
        rsna_mammo: bool = True,
        balance_training: bool = False,
        early_stop: bool = False,
        early_stop_patience: int = 5,
        early_stop_min_delta: float = 0.0,
        # GLAM kwargs (passed through to GLAM.load_from_checkpoint)
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ------------------------------------------------------------------
        # Build GLAM backbone
        # ------------------------------------------------------------------
        # Keep the same meaning as run.sh: the checkpoint under pretrain_model/
        # is an encoder checkpoint loaded through GLAM(pretrained_encoder=...),
        # not a full downstream LightningModule checkpoint.
        glam_defaults = {
            "embed": False,
            "vindr": False,
            "multi_label": False,
            "pred_density": False,
            "pred_mass": False,
            "pred_calc": False,
            "screen_only": False,
            "pool_feat": False,
            "pool_txt_feat": False,
            "slip": False,
            "symmetric_clip": False,
            "slip_loss_lambda": 1.0,
            "random_vit": False,
            "freeze_llm": False,
            "num_freeze_blocks": 0,
            "masked_lm_ratio": 0.0,
            "num_heads": 4,
            "accelerator": "gpu",
            "dev": False,
            "grad_ckpt": False,
            "balance_ratio": -1,
            "pos_weight": None,
            "weighted": False,
            "patch_contrast": False,
            "all_patch_contrast": False,
            "attn_pooler": False,
            "same_pos_contrast": False,
            "mega_patch_size": 8,
            "mega_patch_stride": 8,
            "softmax_temperature": 0.07,
            "ema": False,
            "ema_decay": 0.999,
            "ema_no_load": False,
            "agg_tokens": False,
            "train_sub_set": False,
            "data_pct": 1.0,
            "train_split": "train",
            "valid_split": "valid",
            "load_jpg": False,
            "aug_orig_img": False,
            "max_words": 144,
            "prob_diff_dcm": 0.5,
            "aligned_mlo": False,
            "fixed_view": False,
            "align_orientation": False,
            "remove_text": False,
            "balanced_test": False,
            "structural_cap": False,
            "simple_cap": False,
            "natural_cap": False,
            "raw_caption": False,
            "aug_text": False,
            "heavy_aug": False,
            "mask_ratio": 0.0,
            "mask_meta": -1.0,
            "extra_cap": None,
            "inter_view": False,
            "inter_side": False,
            "ext_img_prob": 0.5,
            "label_smoothing": 0.0,
            "less_train_neg": 0.0,
            "pos_smooth_only": False,
            "focal_loss": False,
            "focal_alpha": 1.0,
            "focal_gamma": 2.0,
            "asl": False,
            "asl_weighted": False,
            "gamma_neg": 4.0,
            "gamma_pos": 1.0,
            "asl_clip": 0.05,
            "save_last_k": 1,
            "instance_test_cap": False,
            "retrieval": False,
            "average_attn_weights": True,
            "pf1_threshold": False,
            "rsna_trans": False,
            "test_data_pct": 1.0,
            "bootstrap_test": False,
            "save_prediction": False,
            "extract_feat": False,
            "extract_train": False,
            "pred_only": False,
            "use_flash_attention": False,
            "train_by_epoch": True,
            "max_steps": -1,
        }
        glam_kwargs = {
            **glam_defaults,
            "rsna_mammo": rsna_mammo,
            "img_cls_ft": True,
            "num_classes": 1,
            "weighted_binary": True,
            "llm_type": kwargs.get("llm_type", "bert"),
            "img_encoder": img_encoder,
            "emb_dim": emb_dim,
            "linear_proj": linear_proj,
            "img_size": img_size,
            "crop_size": crop_size,
            "freeze_vit": freeze_vit,
            "vit_grad_ckpt": vit_grad_ckpt,
            "devices": devices,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "warm_up": warm_up,
            "min_lr": min_lr,
            "adafactor": adafactor,
            "sgd": sgd,
            "balance_training": balance_training,
            "early_stop": early_stop,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "pretrained_encoder": pretrained_model or pretrained_encoder,
            "num_workers": kwargs.get("num_workers", 0),
            "k_fold": kwargs.get("k_fold", 0),
            "fold": kwargs.get("fold", 0),
            "rsna_csv_path": kwargs.get("rsna_csv_path", None),
            "rsna_img_root": kwargs.get("rsna_img_root", None),
            "rsna_path_pattern": kwargs.get("rsna_path_pattern", None),
            "rsna_patient_col": kwargs.get("rsna_patient_col", "patient_id"),
            "rsna_image_col": kwargs.get("rsna_image_col", "image_id"),
            "rsna_label_col": kwargs.get("rsna_label_col", "cancer"),
            "rsna_split_col": kwargs.get("rsna_split_col", "split"),
            "rsna_train_split_value": kwargs.get("rsna_train_split_value", "training"),
            "rsna_test_split_value": kwargs.get("rsna_test_split_value", "test"),
        }

        if glam_kwargs["pretrained_encoder"] is not None:
            print(
                "\n### [EDL] Loading pretrained GLAM encoder from "
                f"{glam_kwargs['pretrained_encoder']}\n"
            )
        else:
            print("\n### [EDL] Initializing GLAM from scratch (no checkpoint)\n")
        self.glam = GLAM(**glam_kwargs)

        # Get the feature dimension from the existing global_embed head
        feature_dim = self.glam.img_encoder_q.feature_dim
        print(f"### [EDL] Backbone feature_dim = {feature_dim}")
        print(f"### [EDL] EDL num_classes = {num_classes}")
        if pos_weight is not None:
            print(f"### [EDL] Positive-class loss weight = {float(pos_weight):.6f}")

        # ------------------------------------------------------------------
        # Replace the classification head with an EDL head
        # ------------------------------------------------------------------
        # The original global_embed maps feature_dim -> emb_dim or num_classes.
        # We keep the original backbone (including global_embed for contrastive
        # features) but add a new EDL-specific head that maps feature_dim -> num_classes.
        # We reuse the img_encoder_q.forward() to get cls features, then apply edl_head.
        self.edl_head = nn.Linear(feature_dim, num_classes)

        if freeze_backbone:
            self._freeze_backbone_for_edl()

        # Print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"### [EDL] Total params: {total_params:,}")
        print(f"### [EDL] Trainable params: {trainable_params:,}")

        # ------------------------------------------------------------------
        # Metrics storage
        # ------------------------------------------------------------------
        self.num_classes = num_classes
        self.confmat = MulticlassConfusionMatrix(num_classes)
        self.all_scores_train = None
        self.all_labels_train = None
        self.all_scores_val = None
        self.all_labels_val = None
        # Test outputs (full detail for CSV export)
        self.all_evidence = None
        self.all_probs = None
        self.all_uncertainty = None
        self.all_scores = None
        self.all_labels = None
        self.img_path = []
        self.output_dir = kwargs.get("output_dir", os.getcwd())
        self.loss_plot_dir = os.path.join(self.output_dir, "plots")
        self.loss_history = {"epoch": [], "train_loss": [], "val_loss": []}

    def _freeze_backbone_for_edl(self):
        for param in self.glam.parameters():
            param.requires_grad = False
        for param in self.edl_head.parameters():
            param.requires_grad = True
        self.glam.eval()
        print("### [EDL] Freeze backbone enabled: training EDL head only")

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self.hparams, "freeze_backbone", False):
            self.glam.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward_edl(self, images: torch.Tensor):
        """
        Run images through the GLAM backbone and the EDL head.

        Returns:
            logits:     raw EDL logits  (B, C)
            evidence:   softplus(logits) (B, C)
            alpha:      evidence + 1    (B, C)
            prob:       alpha / S       (B, C)
            uncertainty: K / S          (B,)
        """
        # Extract visual features (same as visual_forward in GLAM)
        if self.hparams.freeze_backbone:
            with torch.no_grad():
                img_feat_q, patch_feat_q, img_full = self.glam.img_encoder_q(images)
                if self.glam.hparams.pool_feat:
                    img_feat_q = img_full.mean(dim=1)
        else:
            img_feat_q, patch_feat_q, img_full = self.glam.img_encoder_q(images)
            if self.glam.hparams.pool_feat:
                img_feat_q = img_full.mean(dim=1)

        # EDL head
        logits = self.edl_head(img_feat_q)
        evidence = get_evidence(logits, self.hparams.evidence_type)
        alpha = evidence + 1.0
        prob = compute_dirichlet_prob(evidence)
        uncertainty = compute_uncertainty(evidence)

        return logits, evidence, alpha, prob, uncertainty

    def _compute_loss(self, evidence, target):
        """Compute EDL loss based on the configured loss type."""
        epoch_num = self.current_epoch
        if self.hparams.edl_loss_type == "digamma":
            loss = edl_digamma_loss(
                evidence, target,
                epoch_num=epoch_num,
                annealing_step=self.hparams.annealing_step,
                lambda_kl=self.hparams.lambda_kl,
                pos_weight=self.hparams.pos_weight,
            )
        elif self.hparams.edl_loss_type == "log":
            loss = edl_log_loss(
                evidence, target,
                epoch_num=epoch_num,
                annealing_step=self.hparams.annealing_step,
                lambda_kl=self.hparams.lambda_kl,
                pos_weight=self.hparams.pos_weight,
            )
        elif self.hparams.edl_loss_type == "mse":
            loss = edl_mse_loss(
                evidence, target,
                epoch_num=epoch_num,
                annealing_step=self.hparams.annealing_step,
                lambda_kl=self.hparams.lambda_kl,
                pos_weight=self.hparams.pos_weight,
            )
        else:
            raise ValueError(f"Unknown EDL loss type: {self.hparams.edl_loss_type}")
        return loss

    def _shared_step(self, batch, batch_idx, split="train"):
        """Shared logic for training / validation / test steps."""
        img_key = "imgs"
        label_key = "multi_hot_label"

        images = batch[img_key]
        labels = batch[label_key]  # one-hot (B, 2) for binary classification

        # Convert one-hot to 2-class one-hot if needed
        # Original data uses n_classes=2 with one-hot: [1,0] for neg, [0,1] for pos
        if labels.shape[-1] == 1:
            # Convert binary single-value to 2-class one-hot
            labels = torch.cat([1 - labels, labels], dim=-1).float()

        logits, evidence, alpha, prob, uncertainty = self.forward_edl(images)

        loss = self._compute_loss(evidence, labels)

        # Accuracy and class-balance diagnostics
        preds = prob.argmax(dim=-1)
        targets = labels.argmax(dim=-1)
        acc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
        pos_rate = (targets == 1).float().mean()
        pred_pos_rate = (preds == 1).float().mean()

        # For val/test, accumulate predictions
        if split in ["val", "test"]:
            prob_np = prob.detach().float().cpu().numpy()
            labels_np = labels.detach().float().cpu().numpy()

            if split == "val":
                if self.all_scores_val is None:
                    self.all_scores_val = prob_np
                    self.all_labels_val = labels_np
                else:
                    self.all_scores_val = np.concatenate([self.all_scores_val, prob_np], axis=0)
                    self.all_labels_val = np.concatenate([self.all_labels_val, labels_np], axis=0)
            else:
                # test: also save evidence and uncertainty
                evidence_np = evidence.detach().float().cpu().numpy()
                uncertainty_np = uncertainty.detach().float().cpu().numpy()

                if self.all_scores is None:
                    self.all_scores = prob_np
                    self.all_labels = labels_np
                    self.all_evidence = evidence_np
                    self.all_probs = prob_np
                    self.all_uncertainty = uncertainty_np
                else:
                    self.all_scores = np.concatenate([self.all_scores, prob_np], axis=0)
                    self.all_labels = np.concatenate([self.all_labels, labels_np], axis=0)
                    self.all_evidence = np.concatenate([self.all_evidence, evidence_np], axis=0)
                    self.all_probs = np.concatenate([self.all_probs, prob_np], axis=0)
                    self.all_uncertainty = np.concatenate([self.all_uncertainty, uncertainty_np], axis=0)
                for i in range(len(batch["path"])):
                    self.img_path.append(batch["path"][i])

        return loss, acc, pos_rate, pred_pos_rate

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, acc, pos_rate, pred_pos_rate = self._shared_step(batch, batch_idx, "train")
        self.log_dict(
            {
                "train_loss_step": loss,
                "train_acc_step": acc,
                "train_pos_rate_step": pos_rate,
                "train_pred_pos_rate_step": pred_pos_rate,
            },
            on_step=True, on_epoch=False,
            batch_size=self.hparams.batch_size,
            sync_dist=True, prog_bar=True, logger=False, rank_zero_only=True,
        )
        self.log_dict(
            {"train_loss": loss, "train_acc": acc},
            on_step=False, on_epoch=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True, prog_bar=False, rank_zero_only=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, pos_rate, pred_pos_rate = self._shared_step(batch, batch_idx, "val")
        self.log_dict(
            {
                "val_loss_step": loss,
                "val_acc_step": acc,
                "val_pos_rate_step": pos_rate,
                "val_pred_pos_rate_step": pred_pos_rate,
            },
            on_step=True, on_epoch=False,
            batch_size=self.hparams.batch_size,
            sync_dist=True, prog_bar=True, logger=False, rank_zero_only=True,
        )
        self.log_dict(
            {"val_loss": loss, "val_acc": acc},
            on_step=False, on_epoch=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True, prog_bar=False, rank_zero_only=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc, pos_rate, pred_pos_rate = self._shared_step(batch, batch_idx, "test")
        self.log_dict(
            {
                "test_loss_step": loss,
                "test_acc_step": acc,
                "test_pos_rate_step": pos_rate,
                "test_pred_pos_rate_step": pred_pos_rate,
            },
            on_step=True, on_epoch=False,
            batch_size=self.hparams.batch_size,
            sync_dist=True, prog_bar=True, logger=False, rank_zero_only=True,
        )
        self.log_dict(
            {"test_loss": loss, "test_acc": acc},
            on_step=False, on_epoch=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True, prog_bar=False, rank_zero_only=True,
        )
        return loss

    # ------------------------------------------------------------------
    # Epoch-end hooks
    # ------------------------------------------------------------------
    def on_train_epoch_end(self):
        if self.global_rank == 0 and self.trainer is not None:
            metrics = self.trainer.callback_metrics
            train_loss = self._to_float(metrics.get("train_loss"))
            train_acc = self._to_float(metrics.get("train_acc"))
            fold = self.hparams.get("fold", "NA") if hasattr(self.hparams, "get") else getattr(self.hparams, "fold", "NA")
            msg = f"\n### [EDL] Fold {fold} | Epoch {self.current_epoch}"
            if train_loss is not None:
                msg += f" | train_loss: {train_loss:.4f}"
            if train_acc is not None:
                msg += f" | train_acc: {train_acc:.4f}"
            print(msg)
        self.all_scores_train = None
        self.all_labels_train = None

    def on_validation_epoch_end(self):
        if self.all_scores_val is not None and self.all_labels_val is not None:
            idx_label = np.argmax(self.all_labels_val, axis=-1)
            if self.num_classes == 2 and self.all_scores_val.ndim > 1:
                val_scores = self.all_scores_val[:, 1]
                idx_pred = np.argmax(self.all_scores_val, axis=-1)
            else:
                val_scores = np.squeeze(self.all_scores_val)
                idx_pred = (val_scores > 0.5).astype(int)

            try:
                if len(np.unique(idx_label)) < 2:
                    raise ValueError("Only one class in validation labels.")
                val_auc = 100 * roc_auc_score(idx_label, val_scores)
                val_bacc = 100 * balanced_accuracy_score(idx_label, idx_pred)
                val_f1 = 100 * f1_score(idx_label, idx_pred)
                best_threshold, best_bacc, best_f1 = self._find_best_binary_threshold(
                    idx_label, val_scores
                )
            except ValueError as e:
                if self.global_rank == 0:
                    fold = getattr(self.hparams, "fold", "NA")
                    print(f"### Fold {fold} | Val metric fallback: {e}")
                val_auc = 0.0
                val_bacc = 0.0
                val_f1 = 0.0
                best_threshold = 0.5
                best_bacc = 0.0
                best_f1 = 0.0

            self.log(
                "val_AUROC", float(val_auc),
                on_epoch=True, prog_bar=True, sync_dist=True, logger=True,
            )
            self.log_dict(
                {
                    "val_BACC_best": float(best_bacc),
                    "val_F1_best": float(best_f1),
                    "val_best_threshold": float(best_threshold),
                },
                on_epoch=True, prog_bar=False, sync_dist=True, logger=True,
            )
            if self.global_rank == 0:
                fold = getattr(self.hparams, "fold", "NA")
                print(f"\n### [EDL] Fold {fold} | Epoch {self.current_epoch} | val_AUROC: {val_auc:.4f}")
                print(f"### [EDL] Fold {fold} | Epoch {self.current_epoch} | val_BACC: {val_bacc:.4f}")
                print(f"### [EDL] Fold {fold} | Epoch {self.current_epoch} | val_F1: {val_f1:.4f}")
                print(
                    f"### [EDL] Fold {fold} | Epoch {self.current_epoch} | "
                    f"val_BACC_best: {best_bacc:.4f} @ threshold={best_threshold:.4f}"
                )
                print(f"### [EDL] Fold {fold} | Epoch {self.current_epoch} | val_F1_best: {best_f1:.4f}")
                self._update_loss_history_and_plot()

        self.all_scores_val = None
        self.all_labels_val = None

    def on_test_epoch_end(self):
        """At the end of testing, print metrics and save predictions."""
        if self.all_scores is None or self.all_labels is None:
            return

        idx_label = np.argmax(self.all_labels, axis=-1)
        idx_pred = np.argmax(self.all_scores, axis=-1)

        # Binary metrics
        if self.num_classes == 2 and self.all_scores.ndim > 1:
            pos_scores = self.all_scores[:, 1]
        else:
            pos_scores = self.all_scores.squeeze()

        acc = 100 * accuracy_score(idx_label, idx_pred)
        ba = 100 * balanced_accuracy_score(idx_label, idx_pred)
        try:
            auc = 100 * roc_auc_score(idx_label, pos_scores)
        except Exception as e:
            print(f"### Warning: AUC calculation failed: {e}")
            auc = 0.0

        print(f"### [EDL] Test Accuracy: {acc:.4f}")
        print(f"### [EDL] Test Balanced Accuracy: {ba:.4f}")
        print(f"### [EDL] Test AUC: {auc:.4f}")

        self.log("test_AUROC", float(auc), on_epoch=True, prog_bar=True, sync_dist=True, logger=True)

        # Reset
        self.confmat.reset()
        # Note: we do NOT reset all_scores / all_labels / all_evidence here
        # because the caller (edl_train.py) will read them for CSV export.

    @staticmethod
    def _to_float(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return None
            return float(x.detach().cpu().item())
        try:
            return float(x)
        except Exception:
            return None

    @staticmethod
    def _find_best_binary_threshold(labels, scores):
        thresholds = np.linspace(0.0, 1.0, 201)
        best_threshold = 0.5
        best_bacc = -1.0
        best_f1 = 0.0
        for threshold in thresholds:
            pred = (scores >= threshold).astype(int)
            bacc = 100 * balanced_accuracy_score(labels, pred)
            f1 = 100 * f1_score(labels, pred, zero_division=0)
            if bacc > best_bacc:
                best_threshold = float(threshold)
                best_bacc = float(bacc)
                best_f1 = float(f1)
        return best_threshold, best_bacc, best_f1

    def _update_loss_history_and_plot(self):
        if self.trainer is None or getattr(self.trainer, "sanity_checking", False):
            return

        metrics = self.trainer.callback_metrics
        train_loss = self._to_float(metrics.get("train_loss"))
        val_loss = self._to_float(metrics.get("val_loss"))
        if train_loss is None and val_loss is None:
            return

        epoch = int(self.current_epoch)
        if self.loss_history["epoch"] and self.loss_history["epoch"][-1] == epoch:
            self.loss_history["train_loss"][-1] = train_loss
            self.loss_history["val_loss"][-1] = val_loss
        else:
            self.loss_history["epoch"].append(epoch)
            self.loss_history["train_loss"].append(train_loss)
            self.loss_history["val_loss"].append(val_loss)

        os.makedirs(self.loss_plot_dir, exist_ok=True)
        fold = getattr(self.hparams, "fold", "NA")
        csv_path = os.path.join(self.loss_plot_dir, f"fold{fold}_loss_curve.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss\n")
            for ep, tr_loss, va_loss in zip(
                self.loss_history["epoch"],
                self.loss_history["train_loss"],
                self.loss_history["val_loss"],
            ):
                tr_text = "" if tr_loss is None else f"{tr_loss:.8f}"
                va_text = "" if va_loss is None else f"{va_loss:.8f}"
                f.write(f"{ep},{tr_text},{va_text}\n")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = self.loss_history["epoch"]
            plt.figure(figsize=(7, 4.5))
            if any(v is not None for v in self.loss_history["train_loss"]):
                plt.plot(
                    epochs,
                    self.loss_history["train_loss"],
                    marker="o",
                    linewidth=2,
                    label="train_loss",
                )
            if any(v is not None for v in self.loss_history["val_loss"]):
                plt.plot(
                    epochs,
                    self.loss_history["val_loss"],
                    marker="o",
                    linewidth=2,
                    label="val_loss",
                )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"EDL Loss Curve - Fold {fold}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(self.loss_plot_dir, f"fold{fold}_loss_curve.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"### [EDL] Loss curve saved: {plot_path}")
        except Exception as exc:
            print(f"### [EDL][Warning] Failed to save loss curve plot: {exc}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError("No trainable parameters found for EDL optimizer.")
        if self.hparams.adafactor:
            from transformers import Adafactor
            optimizer = Adafactor(
                parameters, self.hparams.learning_rate,
                beta1=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
                relative_step=False, scale_parameter=False,
            )
        elif self.hparams.sgd:
            optimizer = torch.optim.SGD(
                parameters, self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                parameters, self.hparams.learning_rate,
                betas=(self.hparams.momentum, 0.999),
                weight_decay=self.hparams.weight_decay,
            )

        first_cycle_steps = max(1, int(self.hparams.max_epochs))
        warmup_steps = max(0, int(self.hparams.warm_up))
        if first_cycle_steps == 1:
            warmup_steps = 0
        else:
            warmup_steps = min(warmup_steps, first_cycle_steps - 1)

        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=self.hparams.min_lr,
            warmup_steps=warmup_steps,
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
