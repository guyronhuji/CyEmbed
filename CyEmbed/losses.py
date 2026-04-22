from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def reconstruction_loss(
    x_hat: Tensor,
    x: Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> Tensor:
    """Reconstruction loss for marker-level prediction."""
    if loss_type == "mse":
        return F.mse_loss(x_hat, x, reduction="mean")
    if loss_type == "huber":
        return F.huber_loss(x_hat, x, delta=huber_delta, reduction="mean")
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def entropy_penalty(w: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean per-cell categorical entropy of archetype weights."""
    entropy = -(w * (w.clamp_min(eps)).log()).sum(dim=1)
    return entropy.mean()


def separation_penalty(
    archetype_matrix: Tensor,
    mode: str = "cosine_sq",
    rbf_gamma: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Penalty to push archetypes apart."""
    if archetype_matrix.ndim != 2:
        raise ValueError("archetype_matrix must be 2D.")
    k = archetype_matrix.shape[0]
    if k < 2:
        return archetype_matrix.new_tensor(0.0)

    offdiag_mask = ~torch.eye(k, dtype=torch.bool, device=archetype_matrix.device)

    if mode in {"cosine_mean", "cosine_abs", "cosine_sq"}:
        normalized = archetype_matrix / (archetype_matrix.norm(dim=1, keepdim=True).clamp_min(eps))
        sim = normalized @ normalized.T
        offdiag = sim[offdiag_mask]
        if mode == "cosine_mean":
            return offdiag.mean()
        if mode == "cosine_abs":
            return offdiag.abs().mean()
        return (offdiag**2).mean()

    if mode == "rbf":
        dist_sq = torch.cdist(archetype_matrix, archetype_matrix, p=2) ** 2
        repulsion = torch.exp(-rbf_gamma * dist_sq)
        return repulsion[offdiag_mask].mean()

    raise ValueError(f"Unsupported separation mode: {mode}")


def balance_penalty(w: Tensor, mode: str = "l2_uniform", eps: float = 1e-8) -> Tensor:
    """Penalty against dead archetypes based on average usage."""
    if w.ndim != 2:
        raise ValueError("w must be 2D.")
    k = w.shape[1]
    target = 1.0 / max(k, 1)
    w_bar = w.mean(dim=0)

    if mode == "l2_uniform":
        return ((w_bar - target) ** 2).mean()
    if mode == "kl_uniform":
        target_vec = torch.full_like(w_bar, target)
        return (w_bar * ((w_bar.clamp_min(eps) / target_vec).log())).sum()
    if mode == "neg_entropy":
        entropy = -(w_bar * w_bar.clamp_min(eps).log()).sum()
        return math.log(max(k, 1)) - entropy
    raise ValueError(f"Unsupported balance mode: {mode}")


def gaussian_kl_standard_normal(mu: Tensor, logvar: Tensor, reduction: str = "mean") -> Tensor:
    """KL(q||p) for q=N(mu, diag(exp(logvar))) and p=N(0,I)."""
    per_sample = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unsupported KL reduction: {reduction}")


def total_loss(
    x_hat: Tensor,
    x: Tensor,
    w: Tensor,
    archetype_matrix: Tensor,
    *,
    recon_loss_type: str = "mse",
    huber_delta: float = 1.0,
    lambda_entropy: float = 0.0,
    lambda_sep: float = 0.0,
    lambda_balance: float = 0.0,
    separation_mode: str = "cosine_sq",
    balance_mode: str = "l2_uniform",
    rbf_gamma: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combined objective used during training."""
    recon = reconstruction_loss(x_hat, x, loss_type=recon_loss_type, huber_delta=huber_delta)
    entropy = entropy_penalty(w)
    separation = separation_penalty(archetype_matrix, mode=separation_mode, rbf_gamma=rbf_gamma)
    balance = balance_penalty(w, mode=balance_mode)
    total = recon + lambda_entropy * entropy + lambda_sep * separation + lambda_balance * balance
    return total, {
        "recon": recon,
        "entropy": entropy,
        "separation": separation,
        "balance": balance,
    }


def total_variational_loss(
    x_hat: Tensor,
    x: Tensor,
    w: Tensor,
    archetype_matrix: Tensor,
    *,
    kl_w: Tensor,
    kl_r: Tensor | None = None,
    beta_w: float = 1e-3,
    beta_r: float = 1e-3,
    recon_loss_type: str = "mse",
    huber_delta: float = 1.0,
    lambda_entropy: float = 0.0,
    lambda_sep: float = 0.0,
    lambda_balance: float = 0.0,
    separation_mode: str = "cosine_sq",
    balance_mode: str = "l2_uniform",
    rbf_gamma: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Objective for probabilistic archetype model."""
    recon = reconstruction_loss(x_hat, x, loss_type=recon_loss_type, huber_delta=huber_delta)
    entropy = entropy_penalty(w)
    separation = separation_penalty(archetype_matrix, mode=separation_mode, rbf_gamma=rbf_gamma)
    balance = balance_penalty(w, mode=balance_mode)
    kl_r_term = x_hat.new_tensor(0.0) if kl_r is None else kl_r
    total = (
        recon
        + float(beta_w) * kl_w
        + float(beta_r) * kl_r_term
        + lambda_entropy * entropy
        + lambda_sep * separation
        + lambda_balance * balance
    )
    return total, {
        "recon": recon,
        "kl_w": kl_w,
        "kl_r": kl_r_term,
        "entropy": entropy,
        "separation": separation,
        "balance": balance,
    }
