"""AdamW task-direction helpers for GRIT updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from grit.curvature import project_vector_with_module_projectors
from grit.projection import ProjectorBuildResult, default_module_filter


NamedParameter = tuple[str, nn.Parameter]
ModuleFilter = object


def _squared_norm(tensors) -> float:
    total = 0.0
    for tensor in tensors:
        norm = torch.linalg.vector_norm(tensor.detach())
        total += float(norm.item()) ** 2
    return total


@dataclass
class AdamWDirectionPreconditioner:
    """Compute a signed AdamW descent direction without a learning rate.

    This object intentionally does not mutate parameters. It only maintains
    AdamW moments. The caller applies the single global learning rate after
    combining the projected task direction and raw preservation correction.
    """

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        self.state: dict[str, dict[str, torch.Tensor | int]] = {}

    def state_dict(self) -> dict:
        packed: dict[str, dict[str, torch.Tensor | int]] = {}
        for name, state in self.state.items():
            packed[name] = {
                "step": int(state["step"]),
                "exp_avg": state["exp_avg"].detach().cpu(),
                "exp_avg_sq": state["exp_avg_sq"].detach().cpu(),
            }
        return {
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "state": packed,
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        self.betas = tuple(state_dict.get("betas", self.betas))  # type: ignore[assignment]
        self.eps = float(state_dict.get("eps", self.eps))
        self.weight_decay = float(state_dict.get("weight_decay", self.weight_decay))
        self.state = {}
        for name, state in state_dict.get("state", {}).items():
            self.state[name] = {
                "step": int(state["step"]),
                "exp_avg": state["exp_avg"].detach().clone(),
                "exp_avg_sq": state["exp_avg_sq"].detach().clone(),
            }

    @torch.no_grad()
    def directions(
        self,
        parameters: Sequence[NamedParameter],
        gradients: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        beta1, beta2 = self.betas
        updates: dict[str, torch.Tensor] = {}
        for name, parameter in parameters:
            grad = gradients[name].detach().to(device=parameter.device, dtype=torch.float32)
            state = self.state.get(name)
            if state is None:
                state = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(parameter.detach(), dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(parameter.detach(), dtype=torch.float32),
                }
                self.state[name] = state

            exp_avg = state["exp_avg"].to(device=parameter.device, dtype=torch.float32)
            exp_avg_sq = state["exp_avg_sq"].to(device=parameter.device, dtype=torch.float32)
            step = int(state["step"]) + 1

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(self.eps)
            adam_direction = exp_avg.div(bias_correction1).div(denom)
            direction = adam_direction.neg()
            if self.weight_decay != 0.0:
                direction = direction.add(
                    parameter.detach().to(torch.float32),
                    alpha=-self.weight_decay,
                )

            updates[name] = direction.to(device=parameter.device, dtype=parameter.dtype)
            state["step"] = step
            state["exp_avg"] = exp_avg.detach()
            state["exp_avg_sq"] = exp_avg_sq.detach()
        return updates


@torch.no_grad()
def adamw_task_directions(
    *,
    model: nn.Module,
    parameters: Sequence[NamedParameter],
    task_gradients: Mapping[str, torch.Tensor],
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
    task_preconditioner: AdamWDirectionPreconditioner,
    module_filter=None,
    missing_projector: str = "identity",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    """Return signed AdamW task directions and their projected counterpart."""

    if module_filter is None:
        module_filter = lambda name, module: default_module_filter(name, module)

    task_directions = task_preconditioner.directions(parameters, task_gradients)
    projected_task_directions = project_vector_with_module_projectors(
        model,
        task_directions,
        projectors,
        parameters=parameters,
        module_filter=module_filter,
        missing=missing_projector,
    )
    task_before_norm = _squared_norm(task_directions.values()) ** 0.5
    task_after_norm = _squared_norm(projected_task_directions.values()) ** 0.5
    task_removed = max(task_before_norm**2 - task_after_norm**2, 0.0) ** 0.5
    return task_directions, projected_task_directions, {
        "adamw_task_direction_norm": task_before_norm,
        "adamw_task_projected_direction_norm": task_after_norm,
        "adamw_task_removed_direction_norm": task_removed,
        "adamw_task_removed_fraction": task_removed / task_before_norm if task_before_norm > 0 else 0.0,
    }


@torch.no_grad()
def apply_grit_update(
    *,
    model: nn.Module,
    parameters: Sequence[NamedParameter],
    task_gradients: Mapping[str, torch.Tensor],
    preservation_gradients: Mapping[str, torch.Tensor],
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
    task_preconditioner: AdamWDirectionPreconditioner,
    learning_rate: float,
    lambda_pres: float,
    task_directions: Mapping[str, torch.Tensor] | None = None,
    projected_task_directions: Mapping[str, torch.Tensor] | None = None,
    module_filter=None,
    missing_projector: str = "identity",
) -> dict[str, float]:
    """Apply one-LR projected task and raw preservation update.

    ``theta += lr * (projected_task_direction - lambda * correction)``.
    """

    if learning_rate < 0:
        raise ValueError(f"learning_rate must be non-negative, got {learning_rate}")

    if task_directions is None or projected_task_directions is None:
        task_directions, projected_task_directions, task_metrics = adamw_task_directions(
            model=model,
            parameters=parameters,
            task_gradients=task_gradients,
            projectors=projectors,
            task_preconditioner=task_preconditioner,
            module_filter=module_filter,
            missing_projector=missing_projector,
        )
    else:
        task_before_norm = _squared_norm(task_directions.values()) ** 0.5
        task_after_norm = _squared_norm(projected_task_directions.values()) ** 0.5
        task_removed = max(task_before_norm**2 - task_after_norm**2, 0.0) ** 0.5
        task_metrics = {
            "adamw_task_direction_norm": task_before_norm,
            "adamw_task_projected_direction_norm": task_after_norm,
            "adamw_task_removed_direction_norm": task_removed,
            "adamw_task_removed_fraction": task_removed / task_before_norm if task_before_norm > 0 else 0.0,
        }
    final_deltas: dict[str, torch.Tensor] = {}
    for name, parameter in parameters:
        combined_direction = projected_task_directions[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        ).sub(
            preservation_gradients[name].detach().to(
                device=parameter.device,
                dtype=parameter.dtype,
            ),
            alpha=lambda_pres,
        )
        delta = combined_direction.mul(learning_rate)
        parameter.add_(delta)
        final_deltas[name] = delta.detach()

    final_norm = _squared_norm(final_deltas.values()) ** 0.5
    return {
        **task_metrics,
        "raw_preservation_direction_norm": _squared_norm(preservation_gradients.values()) ** 0.5,
        "grit_final_delta_norm": final_norm,
    }
