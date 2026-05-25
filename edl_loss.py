"""
Evidential Deep Learning (EDL) loss functions.

References:
  - Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018.
  - EDL with Dirichlet distribution for uncertainty estimation.
"""

import torch
import torch.nn.functional as F
import math


def relu_evidence(logits: torch.Tensor) -> torch.Tensor:
    """Compute evidence using ReLU activation."""
    return F.relu(logits)


def exp_evidence(logits: torch.Tensor) -> torch.Tensor:
    """Compute evidence using exponential activation."""
    return torch.exp(torch.clamp(logits, min=-10, max=10))


def softplus_evidence(logits: torch.Tensor) -> torch.Tensor:
    """Compute evidence using softplus activation (smooth approximation of ReLU)."""
    return F.softplus(logits)


def get_evidence(logits: torch.Tensor, evidence_type: str = "softplus") -> torch.Tensor:
    """Get evidence from logits using the specified activation function."""
    if evidence_type == "relu":
        return relu_evidence(logits)
    elif evidence_type == "exp":
        return exp_evidence(logits)
    elif evidence_type == "softplus":
        return softplus_evidence(logits)
    else:
        raise ValueError(f"Unknown evidence type: {evidence_type}")


def kl_divergence_dirichlet(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute KL divergence between Dir(alpha) and Dir(1, ..., 1).

    KL[Dir(alpha) || Dir(1)] = log(Gamma(sum(alpha)) / Gamma(K))
                               - sum(log(Gamma(alpha_k)))
                               + sum((alpha_k - 1) * (psi(alpha_k) - psi(sum(alpha))))

    Args:
        alpha: Dirichlet parameters of shape (batch_size, num_classes).
        num_classes: Number of classes.

    Returns:
        KL divergence for each sample, shape (batch_size,).
    """
    beta = torch.ones_like(alpha)  # Dir(1, ..., 1)
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    sum_beta = torch.sum(beta, dim=-1, keepdim=True)

    # ln(Gamma(S_alpha) / Gamma(S_beta))
    ln_gamma_ratio_s = torch.lgamma(sum_alpha) - torch.lgamma(sum_beta)

    # sum(ln(Gamma(beta_k) / Gamma(alpha_k)))
    ln_gamma_ratio_k = torch.sum(torch.lgamma(beta) - torch.lgamma(alpha), dim=-1, keepdim=True)

    # sum((alpha_k - beta_k) * (digamma(alpha_k) - digamma(sum_alpha)))
    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(sum_alpha)
    weighted = torch.sum((alpha - beta) * (digamma_alpha - digamma_sum_alpha), dim=-1, keepdim=True)

    kl = (ln_gamma_ratio_s + ln_gamma_ratio_k + weighted).squeeze(-1)
    return kl


def get_kl_annealing_coef(epoch_num: int, annealing_step: int = 10, lambda_kl: float = 0.1) -> float:
    annealing_coef = min(1.0, float(epoch_num) / max(float(annealing_step), 1.0))
    return float(lambda_kl) * annealing_coef


def assert_kl_sanity(atol: float = 1e-6) -> None:
    alpha_uniform = torch.ones(2, 2)
    kl_uniform = kl_divergence_dirichlet(alpha_uniform, num_classes=2)
    if not torch.allclose(kl_uniform, torch.zeros_like(kl_uniform), atol=atol):
        raise AssertionError(f"KL sanity check failed for Dir(1)||Dir(1): {kl_uniform}")

    alpha_nonuniform = torch.tensor([[3.0, 1.0], [1.0, 4.0]])
    kl_nonuniform = kl_divergence_dirichlet(alpha_nonuniform, num_classes=2)
    if not torch.all(kl_nonuniform > 0):
        raise AssertionError(f"KL sanity check failed for non-uniform alpha: {kl_nonuniform}")


def _apply_binary_pos_weight(loss: torch.Tensor, target: torch.Tensor, pos_weight: float = None) -> torch.Tensor:
    """Apply BCE-style positive-class sample weighting for binary one-hot targets."""
    if pos_weight is None or target.shape[-1] < 2:
        return loss

    weight = torch.ones_like(loss)
    pos_weight_tensor = torch.as_tensor(pos_weight, dtype=loss.dtype, device=loss.device)
    weight = torch.where(target[..., 1] > 0.5, pos_weight_tensor, weight)
    return loss * weight


def edl_digamma_loss(
    evidence: torch.Tensor,
    target: torch.Tensor,
    epoch_num: int = 0,
    annealing_step: int = 10,
    lambda_kl: float = 0.1,
    pos_weight: float = None,
) -> torch.Tensor:
    """
    Expected Cross-Entropy (Type II Maximum Likelihood) loss for EDL.

    L_ce = sum_k y_k * (digamma(S) - digamma(alpha_k))

    Plus KL divergence regularization with annealing.

    Args:
        evidence: Evidence tensor of shape (batch_size, num_classes).
        target: One-hot target tensor of shape (batch_size, num_classes).
        epoch_num: Current epoch number for annealing.
        annealing_step: Number of epochs over which to anneal the KL term.
        lambda_kl: Maximum weight for the KL divergence term.

    Returns:
        Scalar loss value.
    """
    alpha = evidence + 1.0  # Dirichlet parameters
    S = torch.sum(alpha, dim=-1, keepdim=True)  # Dirichlet strength

    # Expected cross-entropy loss: sum_k y_k * (digamma(S) - digamma(alpha_k))
    loss_ce = torch.sum(target * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)
    loss_ce = _apply_binary_pos_weight(loss_ce, target, pos_weight)

    # KL divergence regularization (only for incorrect classes)
    # Remove the ground truth evidence for KL computation
    # Use alpha_tilde = alpha where non-target classes, and 1 for target class
    alpha_tilde = alpha.clone()
    # For one-hot target, set alpha_tilde at target position to 1
    alpha_tilde = alpha_tilde * (1 - target) + target  # target positions become 1

    kl = kl_divergence_dirichlet(alpha_tilde, alpha.shape[-1])

    # Annealing: gradually increase KL weight
    annealing_coef = get_kl_annealing_coef(epoch_num, annealing_step, lambda_kl)

    loss = loss_ce + annealing_coef * kl

    return loss.mean()


def edl_log_loss(
    evidence: torch.Tensor,
    target: torch.Tensor,
    epoch_num: int = 0,
    annealing_step: int = 10,
    lambda_kl: float = 0.1,
    pos_weight: float = None,
) -> torch.Tensor:
    """
    Negative log-likelihood of Dirichlet (MSE-type) loss for EDL.

    L_nll = sum_k y_k * (log(S) - log(alpha_k))

    Plus KL divergence regularization with annealing.

    Args:
        evidence: Evidence tensor of shape (batch_size, num_classes).
        target: One-hot target tensor of shape (batch_size, num_classes).
        epoch_num: Current epoch number for annealing.
        annealing_step: Number of epochs over which to anneal the KL term.
        lambda_kl: Maximum weight for the KL divergence term.

    Returns:
        Scalar loss value.
    """
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=-1, keepdim=True)

    # Negative log-likelihood
    loss_nll = torch.sum(target * (torch.log(S + 1e-10) - torch.log(alpha + 1e-10)), dim=-1)
    loss_nll = _apply_binary_pos_weight(loss_nll, target, pos_weight)

    # KL divergence regularization
    alpha_tilde = alpha.clone()
    alpha_tilde = alpha_tilde * (1 - target) + target

    kl = kl_divergence_dirichlet(alpha_tilde, alpha.shape[-1])

    annealing_coef = get_kl_annealing_coef(epoch_num, annealing_step, lambda_kl)

    loss = loss_nll + annealing_coef * kl

    return loss.mean()


def edl_mse_loss(
    evidence: torch.Tensor,
    target: torch.Tensor,
    epoch_num: int = 0,
    annealing_step: int = 10,
    lambda_kl: float = 0.1,
    pos_weight: float = None,
) -> torch.Tensor:
    """
    Sum-of-squares (MSE) loss for EDL.

    L_mse = sum_k (y_k - alpha_k / S)^2

    Plus KL divergence regularization with annealing.

    Args:
        evidence: Evidence tensor of shape (batch_size, num_classes).
        target: One-hot target tensor of shape (batch_size, num_classes).
        epoch_num: Current epoch number for annealing.
        annealing_step: Number of epochs over which to anneal the KL term.
        lambda_kl: Maximum weight for the KL divergence term.

    Returns:
        Scalar loss value.
    """
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=-1, keepdim=True)

    # MSE between target and Dirichlet mean
    pred = alpha / S
    loss_mse = torch.sum((target - pred) ** 2, dim=-1)
    loss_mse = _apply_binary_pos_weight(loss_mse, target, pos_weight)

    # KL divergence regularization
    alpha_tilde = alpha.clone()
    alpha_tilde = alpha_tilde * (1 - target) + target

    kl = kl_divergence_dirichlet(alpha_tilde, alpha.shape[-1])

    annealing_coef = get_kl_annealing_coef(epoch_num, annealing_step, lambda_kl)

    loss = loss_mse + annealing_coef * kl

    return loss.mean()


def compute_uncertainty(evidence: torch.Tensor) -> torch.Tensor:
    """
    Compute epistemic uncertainty from evidence.

    u = num_classes / sum(alpha) = num_classes / sum(evidence + 1)

    Args:
        evidence: Evidence tensor of shape (batch_size, num_classes).

    Returns:
        Uncertainty tensor of shape (batch_size,).
    """
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=-1)
    num_classes = evidence.shape[-1]
    return num_classes / S


def compute_dirichlet_prob(evidence: torch.Tensor) -> torch.Tensor:
    """
    Compute Dirichlet mean (predicted probability).

    p_k = alpha_k / S = (evidence_k + 1) / sum(evidence + 1)

    Args:
        evidence: Evidence tensor of shape (batch_size, num_classes).

    Returns:
        Probability tensor of shape (batch_size, num_classes).
    """
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=-1, keepdim=True)
    return alpha / S
