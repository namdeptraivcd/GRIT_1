"""Preservation loss for GRIT Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from grit.trust_region import TrustRegionProjection, project_to_kl_ball


@dataclass(frozen=True)
class PreservationLossResult:
    """Scalar loss plus projection diagnostics."""

    loss: torch.Tensor
    token_loss: torch.Tensor
    projection: TrustRegionProjection


def aggregate_preservation_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Aggregate preservation loss with TROLL/verl-compatible modes."""

    loss_mask = loss_mask.to(dtype=loss_mat.dtype)
    if reduction == "none":
        return loss_mat * loss_mask
    if reduction == "sum":
        return torch.sum(loss_mat * loss_mask)
    if reduction == "token-mean":
        return torch.sum(loss_mat * loss_mask) / loss_mask.sum().clamp_min(1.0)
    if reduction == "seq-mean-token-sum":
        if loss_mat.ndim < 2:
            raise ValueError("seq-mean-token-sum requires a batch x sequence loss matrix")
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).to(dtype=loss_mat.dtype)
        return torch.sum(seq_losses * seq_mask) / seq_mask.sum().clamp_min(1.0)
    if reduction == "seq-mean-token-mean":
        if loss_mat.ndim < 2:
            raise ValueError("seq-mean-token-mean requires a batch x sequence loss matrix")
        seq_lengths = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / seq_lengths.clamp_min(1.0)
        seq_mask = (seq_lengths > 0).to(dtype=loss_mat.dtype)
        return torch.sum(seq_losses * seq_mask) / seq_mask.sum().clamp_min(1.0)
    if reduction == "seq-mean-token-sum-norm":
        if loss_mat.ndim < 2:
            raise ValueError("seq-mean-token-sum-norm requires a batch x sequence loss matrix")
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        return torch.sum(seq_losses) / loss_mask.shape[-1]
    raise ValueError(f"unknown reduction: {reduction}")


def preservation_kl_loss(
    policy_logits: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    epsilon_pres: float,
    response_mask: torch.Tensor | None = None,
    selected_token_ids: torch.Tensor | None = None,
    top_k: int | None = None,
    default_probability: float = 1e-12,
    reduction: str = "seq-mean-token-mean",
) -> PreservationLossResult:
    """Compute the GRIT preservation regression loss.

    Gradients flow only through ``pi_tilde``. The TROLL-style geometric
    interpolation target is detached, matching the GRIT preservation objective:
    ``KL(pi_tilde || stopgrad(pi_proj))``.
    """

    projection = project_to_kl_ball(
        policy_logits,
        base_logits,
        epsilon=epsilon_pres,
        response_mask=response_mask,
        selected_token_ids=selected_token_ids,
        top_k=top_k,
        default_probability=default_probability,
    )
    policy_log_probs = projection.policy_log_probs
    policy_probs = policy_log_probs.exp()
    target_log_probs = projection.projected_log_probs.detach()
    token_loss = torch.sum(policy_probs * (policy_log_probs - target_log_probs), dim=-1)

    if response_mask is None:
        active = torch.ones_like(token_loss, dtype=token_loss.dtype)
    else:
        active = response_mask.to(dtype=token_loss.dtype)

    loss = aggregate_preservation_loss(token_loss, active, reduction)

    return PreservationLossResult(loss=loss, token_loss=token_loss, projection=projection)
