from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from grit.preservation_loss import preservation_kl_loss as grit_preservation_kl_loss


@dataclass(frozen=True)
class PolicyBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor


def build_policy_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
) -> PolicyBatch:
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer requires a pad or EOS token")

    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    for prompt, response in zip(prompts, responses):
        try:
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            full_ids = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
            common = 0
            for prompt_id, full_id in zip(prompt_ids, full_ids):
                if prompt_id != full_id:
                    break
                common += 1
            if common == 0 or common == len(full_ids):
                raise ValueError("chat template did not expose an assistant response span")
            prompt_ids = full_ids[:common]
            response_ids = full_ids[common:]
        except (AttributeError, ValueError, TypeError):
            prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
            response_ids = tokenizer(response, add_special_tokens=False).input_ids
            if tokenizer.eos_token_id is not None:
                response_ids = response_ids + [tokenizer.eos_token_id]
        overflow = max(0, len(prompt_ids) + len(response_ids) - max_length)
        if overflow:
            prompt_ids = prompt_ids[min(overflow, max(0, len(prompt_ids) - 1)) :]
        remaining = max_length - len(prompt_ids)
        response_ids = response_ids[:remaining]
        if not response_ids:
            raise ValueError("response has no tokens after truncation")
        sequence = prompt_ids + response_ids
        mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        sequences.append(sequence)
        masks.append(mask)

    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    response_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    for index, (sequence, mask) in enumerate(zip(sequences, masks)):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence)
        attention[index, :length] = 1
        response_mask[index, :length] = torch.tensor(mask, dtype=torch.float32)
    return PolicyBatch(
        input_ids=input_ids.to(device),
        attention_mask=attention.to(device),
        response_mask=response_mask.to(device),
    )


def response_log_probs(model: nn.Module, batch: PolicyBatch) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    logits = outputs.logits[:, :-1].float()
    targets = batch.input_ids[:, 1:]
    token_log_probs = F.log_softmax(logits, dim=-1).gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    return token_log_probs, batch.response_mask[:, 1:]


def _resample(values: Sequence[float], length: int) -> list[float]:
    if length <= 0:
        return []
    if not values:
        return [0.5] * length
    if length == 1:
        return [float(values[-1])]
    return [
        float(values[round(index * (len(values) - 1) / (length - 1))])
        for index in range(length)
    ]


def token_advantages(
    response_mask: torch.Tensor,
    final_rewards: Sequence[float],
    prefix_scores: Sequence[Sequence[float]] | None,
    *,
    discount: float,
) -> torch.Tensor:
    advantages = torch.zeros_like(response_mask, dtype=torch.float32)
    for row in range(response_mask.shape[0]):
        length = int(response_mask[row].sum().item())
        if length == 0:
            continue
        if prefix_scores is None:
            values = [0.0] * length
        else:
            values = _resample(prefix_scores[row], length)
        next_values = values[1:] + [0.0]
        shaping = [discount * nxt - current for current, nxt in zip(values, next_values)]
        advantages[row, response_mask[row].bool()] = torch.tensor(
            [float(final_rewards[row]) + delta for delta in shaping],
            device=response_mask.device,
        )
    return advantages


def process_kappas(
    response_mask: torch.Tensor,
    prefix_scores: Sequence[Sequence[float]],
    *,
    discount: float,
    kappa_max: float,
    tau_kappa: float,
) -> torch.Tensor:
    if not 0.0 <= kappa_max <= 1.0:
        raise ValueError("kappa_max must be in [0, 1]")
    if tau_kappa <= 0.0:
        raise ValueError("tau_kappa must be positive")
    kappas = torch.zeros_like(response_mask, dtype=torch.float32)
    for row in range(response_mask.shape[0]):
        length = int(response_mask[row].sum().item())
        if length == 0:
            continue
        values = _resample(prefix_scores[row], length)
        next_values = values[1:] + [0.0]
        deltas = torch.tensor(
            [discount * nxt - current for current, nxt in zip(values, next_values)],
            device=response_mask.device,
        )
        values_kappa = kappa_max * torch.sigmoid(deltas / tau_kappa)
        kappas[row, response_mask[row].bool()] = values_kappa
    return kappas


def policy_gradient_loss(
    token_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    if advantages.shape != token_log_probs.shape:
        raise ValueError("advantages must match token log-probabilities")
    denominator = response_mask.sum().clamp_min(1.0)
    return -(token_log_probs * advantages.detach() * response_mask).sum() / denominator


def preservation_kl_loss(
    policy_model: nn.Module,
    base_model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
    epsilon: float,
    top_k: int | None = 64,
    default_probability: float = 1e-12,
    reduction: str = "seq-mean-token-mean",
) -> tuple[torch.Tensor, dict[str, float]]:
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    policy_logits = policy_model(**encoded).logits[:, :-1]
    with torch.no_grad():
        base_logits = base_model(**encoded).logits[:, :-1]
    token_mask = encoded["attention_mask"][:, 1:].bool()
    result = grit_preservation_kl_loss(
        policy_logits,
        base_logits,
        epsilon_pres=epsilon,
        response_mask=token_mask,
        selected_token_ids=encoded["input_ids"][:, 1:],
        top_k=top_k,
        default_probability=default_probability,
        reduction=reduction,
    )
    projection = result.projection
    active = projection.active_mask
    return result.loss, {
        "preservation_kl": float(projection.token_kl[active].mean().item()) if active.any() else 0.0,
        "projected_kl": float(projection.projected_kl[active].mean().item()) if active.any() else 0.0,
        "violation_fraction": float(
            (projection.violation_mask & active)
            .float()
            .sum()
            .div(active.sum().clamp_min(1))
            .item()
        ),
    }
