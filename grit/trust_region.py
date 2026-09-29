"""TROLL-style trust-region projection for GRIT Phase 3.

GRIT uses the TROLL token-level trust-region mechanism, but changes the anchor:

    TROLL: KL(pi_proj || pi_old) <= epsilon
    GRIT:  KL(pi_proj || pi_base) <= epsilon_pres

For scalability this module supports a sparse-default approximation: retain a
small token support, assign every dropped token a positive default probability
``p_d > 0``, and run the KL/projection on that sparse-default distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TrustRegionProjection:
    """Projected preservation targets and diagnostics."""

    projected_probs: torch.Tensor
    projected_log_probs: torch.Tensor
    token_kl: torch.Tensor
    projected_kl: torch.Tensor
    violation_mask: torch.Tensor
    active_mask: torch.Tensor
    support_mask: torch.Tensor | None
    eta: torch.Tensor
    policy_log_probs: torch.Tensor
    base_log_probs: torch.Tensor
    default_probability: float


def _check_matching_logits(policy_logits: torch.Tensor, base_logits: torch.Tensor) -> None:
    if policy_logits.shape != base_logits.shape:
        raise ValueError(
            f"policy_logits and base_logits must have the same shape, got "
            f"{tuple(policy_logits.shape)} and {tuple(base_logits.shape)}"
        )


def kl_from_log_probs(policy_log_probs: torch.Tensor, anchor_log_probs: torch.Tensor) -> torch.Tensor:
    """Compute token-level ``KL(policy || anchor)`` from log-probabilities."""

    if policy_log_probs.shape != anchor_log_probs.shape:
        raise ValueError(
            f"log-prob tensors must have the same shape, got "
            f"{tuple(policy_log_probs.shape)} and {tuple(anchor_log_probs.shape)}"
        )
    policy_probs = policy_log_probs.exp()
    return torch.sum(policy_probs * (policy_log_probs - anchor_log_probs), dim=-1)


def token_kl_from_logits(
    policy_logits: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    support_mask: torch.Tensor | None = None,
    default_probability: float = 1e-12,
) -> torch.Tensor:
    """Compute token-level ``KL(policy || base)``.

    When ``support_mask`` is provided, logits are first converted to a
    sparse-default distribution: retained tokens keep their normalized
    probability, dropped tokens receive default probability ``p_d > 0``, and the
    result is renormalized.
    """

    _check_matching_logits(policy_logits, base_logits)
    policy_log_probs = sparse_default_log_probs(
        policy_logits,
        support_mask=support_mask,
        default_probability=default_probability,
    )
    base_log_probs = sparse_default_log_probs(
        base_logits,
        support_mask=support_mask,
        default_probability=default_probability,
    )
    return kl_from_log_probs(policy_log_probs, base_log_probs)


def build_sparse_support_mask(
    policy_logits: torch.Tensor,
    *,
    base_logits: torch.Tensor | None = None,
    selected_token_ids: torch.Tensor | None = None,
    top_k: int | None = None,
) -> torch.Tensor | None:
    """Build the retained-token support for sparse KL/projection.

    ``top_k`` keeps policy top-k tokens and, when available, base-policy top-k
    tokens. ``selected_token_ids`` is always retained, matching the TROLL sparse
    path's selected-token safeguard.
    """

    if top_k is None and selected_token_ids is None:
        return None
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be positive when provided, got {top_k}")
    if base_logits is not None:
        _check_matching_logits(policy_logits, base_logits)

    vocab_size = policy_logits.shape[-1]
    support = torch.zeros_like(policy_logits, dtype=torch.bool)

    if top_k is not None:
        k = min(top_k, vocab_size)
        support.scatter_(-1, torch.topk(policy_logits, k=k, dim=-1).indices, True)
        if base_logits is not None:
            support.scatter_(-1, torch.topk(base_logits, k=k, dim=-1).indices, True)

    if selected_token_ids is not None:
        expected_shape = policy_logits.shape[:-1]
        if selected_token_ids.shape != expected_shape:
            raise ValueError(
                f"selected_token_ids must have shape {tuple(expected_shape)}, "
                f"got {tuple(selected_token_ids.shape)}"
            )
        if selected_token_ids.numel() == 0:
            return support
        if selected_token_ids.min().item() < 0 or selected_token_ids.max().item() >= vocab_size:
            raise ValueError("selected_token_ids contains ids outside the vocabulary")
        support.scatter_(-1, selected_token_ids.unsqueeze(-1), True)

    return support


def sparse_default_log_probs(
    logits: torch.Tensor,
    *,
    support_mask: torch.Tensor | None,
    default_probability: float = 1e-12,
) -> torch.Tensor:
    """Convert logits to dense log-probs with sparse-default dropped tokens."""

    if default_probability <= 0.0:
        raise ValueError(f"default_probability must be > 0, got {default_probability}")

    compute_logits = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
    log_probs = F.log_softmax(compute_logits, dim=-1)
    if support_mask is None:
        return log_probs
    if support_mask.shape != logits.shape:
        raise ValueError(
            f"support_mask must have shape {tuple(logits.shape)}, got {tuple(support_mask.shape)}"
        )

    probs = log_probs.exp()
    default = torch.as_tensor(default_probability, device=logits.device, dtype=torch.float32)
    sparse_probs = torch.where(support_mask, probs, default)
    sparse_probs = sparse_probs / sparse_probs.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(sparse_probs.dtype).tiny
    )
    return sparse_probs.log()


def geometric_interpolation_log_probs(
    policy_log_probs: torch.Tensor,
    anchor_log_probs: torch.Tensor,
    eta: torch.Tensor,
) -> torch.Tensor:
    """TROLL Eq. (4)-style geometric interpolation in log-prob space."""

    while eta.ndim < policy_log_probs.ndim:
        eta = eta.unsqueeze(-1)
    mixed_logits = (policy_log_probs + eta * anchor_log_probs) / (eta + 1.0)
    return mixed_logits - torch.logsumexp(mixed_logits, dim=-1, keepdim=True)


def _bracket_eta(
    policy_log_probs: torch.Tensor,
    anchor_log_probs: torch.Tensor,
    *,
    epsilon: float,
    violation_mask: torch.Tensor,
    max_eta: float,
    max_iterations: int,
) -> torch.Tensor:
    eta_high = torch.ones_like(violation_mask, dtype=policy_log_probs.dtype)
    for _ in range(max_iterations):
        projected = geometric_interpolation_log_probs(policy_log_probs, anchor_log_probs, eta_high)
        projected_kl = kl_from_log_probs(projected, anchor_log_probs)
        needs_more = (projected_kl > epsilon) & violation_mask & (eta_high < max_eta)
        eta_high = torch.where(needs_more, eta_high * 2.0, eta_high)
        if not bool(needs_more.any().item()):
            break
    return eta_high.clamp_max(max_eta)


def project_to_kl_ball(
    policy_logits: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    epsilon: float,
    response_mask: torch.Tensor | None = None,
    selected_token_ids: torch.Tensor | None = None,
    top_k: int | None = None,
    default_probability: float = 1e-12,
    tolerance: float = 1e-6,
    max_iterations: int = 40,
    max_eta: float = 1e12,
) -> TrustRegionProjection:
    """Project predictor distributions into a base-policy KL ball.

    Accepted tokens keep ``pi_tilde``. Violating tokens use TROLL's geometric
    interpolation with ``pi_base`` as anchor, and solve ``eta*`` by bracketing
    plus bisection.
    """

    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    _check_matching_logits(policy_logits, base_logits)
    if response_mask is not None and response_mask.shape != policy_logits.shape[:-1]:
        raise ValueError(
            f"response_mask must have shape {tuple(policy_logits.shape[:-1])}, "
            f"got {tuple(response_mask.shape)}"
        )

    support_mask = build_sparse_support_mask(
        policy_logits,
        base_logits=base_logits,
        selected_token_ids=selected_token_ids,
        top_k=top_k,
    )
    policy_log_probs = sparse_default_log_probs(
        policy_logits,
        support_mask=support_mask,
        default_probability=default_probability,
    )
    base_log_probs = sparse_default_log_probs(
        base_logits,
        support_mask=support_mask,
        default_probability=default_probability,
    )
    projection_policy_log_probs = policy_log_probs.detach()
    projection_base_log_probs = base_log_probs.detach()

    token_kl = kl_from_log_probs(projection_policy_log_probs, projection_base_log_probs)
    active_mask = torch.ones_like(token_kl, dtype=torch.bool)
    if response_mask is not None:
        active_mask = response_mask.bool()
    violation_mask = (token_kl > epsilon + tolerance) & active_mask

    eta_low = torch.zeros_like(token_kl)
    eta_high = _bracket_eta(
        projection_policy_log_probs,
        projection_base_log_probs,
        epsilon=epsilon,
        violation_mask=violation_mask,
        max_eta=max_eta,
        max_iterations=max_iterations,
    )
    for _ in range(max_iterations):
        eta_mid = (eta_low + eta_high) * 0.5
        projected = geometric_interpolation_log_probs(
            projection_policy_log_probs,
            projection_base_log_probs,
            eta_mid,
        )
        projected_kl = kl_from_log_probs(projected, projection_base_log_probs)
        too_far = projected_kl > epsilon
        eta_low = torch.where(too_far & violation_mask, eta_mid, eta_low)
        eta_high = torch.where((~too_far) & violation_mask, eta_mid, eta_high)

    eta = torch.where(violation_mask, eta_high, torch.zeros_like(token_kl))
    projected_log_probs = geometric_interpolation_log_probs(
        projection_policy_log_probs,
        projection_base_log_probs,
        eta,
    )
    projected_log_probs = torch.where(
        violation_mask.unsqueeze(-1),
        projected_log_probs,
        projection_policy_log_probs,
    )
    projected_probs = projected_log_probs.exp()
    projected_kl = kl_from_log_probs(projected_log_probs, projection_base_log_probs)

    return TrustRegionProjection(
        projected_probs=projected_probs,
        projected_log_probs=projected_log_probs,
        token_kl=token_kl,
        projected_kl=projected_kl,
        violation_mask=violation_mask,
        active_mask=active_mask,
        support_mask=support_mask,
        eta=eta,
        policy_log_probs=policy_log_probs,
        base_log_probs=base_log_probs,
        default_probability=default_probability,
    )
