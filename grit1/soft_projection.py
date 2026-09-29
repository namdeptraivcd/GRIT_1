from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def soft_project_gradient_map(
    model: nn.Module,
    gradients: Mapping[str, torch.Tensor],
    projectors: Mapping[str, torch.Tensor],
    *,
    kappa: float,
    module_pattern: str = "mlp",
    missing: str = "identity",
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Apply ``P g + kappa (I-P) g`` to protected Linear weights.

    ``kappa=0`` is the original hard null-space projection and ``kappa=1`` is
    the identity map. Parameters outside the protected surface are unchanged.
    """

    if not 0.0 <= kappa <= 1.0:
        raise ValueError(f"kappa must be in [0, 1], got {kappa}")
    if missing not in {"identity", "zero"}:
        raise ValueError("missing must be identity or zero")

    module_by_name = dict(model.named_modules())
    output: dict[str, torch.Tensor] = {}
    before_sq = 0.0
    after_sq = 0.0
    projected_count = 0

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter_name not in gradients:
            continue
        gradient = gradients[parameter_name]
        module_name = parameter_name.removesuffix(".weight")
        module = module_by_name.get(module_name)
        is_protected = (
            parameter_name.endswith(".weight")
            and isinstance(module, nn.Linear)
            and module_pattern in module_name
        )
        if is_protected and module_name in projectors:
            projector = projectors[module_name].to(
                device=gradient.device, dtype=gradient.dtype
            )
            if projector.shape != (gradient.shape[-1], gradient.shape[-1]):
                raise ValueError(
                    f"projector {module_name!r} has shape {tuple(projector.shape)} "
                    f"for gradient {tuple(gradient.shape)}"
                )
            hard = gradient.matmul(projector)
            value = hard + kappa * (gradient - hard)
            projected_count += 1
        elif is_protected and missing == "zero":
            value = torch.zeros_like(gradient)
        else:
            value = gradient.clone()
        output[parameter_name] = value
        before_sq += float(gradient.detach().float().square().sum().item())
        after_sq += float(value.detach().float().square().sum().item())

    return output, {
        "projected_modules": float(projected_count),
        "gradient_norm_before": before_sq**0.5,
        "gradient_norm_after": after_sq**0.5,
        "projected_gradient_ratio": (after_sq / before_sq) if before_sq else 0.0,
        "kappa": float(kappa),
    }


def process_gated_project_gradient_map(
    model: nn.Module,
    advantage_gradients: Mapping[str, torch.Tensor],
    gated_gradients: Mapping[str, torch.Tensor],
    projectors: Mapping[str, torch.Tensor],
    *,
    module_pattern: str = "mlp",
    missing: str = "identity",
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Apply the exact aggregate ``g_A P + g_kappa (I-P)`` update."""

    if missing not in {"identity", "zero"}:
        raise ValueError("missing must be identity or zero")
    module_by_name = dict(model.named_modules())
    output: dict[str, torch.Tensor] = {}
    before_sq = 0.0
    after_sq = 0.0
    projected_count = 0

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter_name not in advantage_gradients:
            continue
        gradient_a = advantage_gradients[parameter_name]
        gradient_k = gated_gradients.get(parameter_name, torch.zeros_like(gradient_a))
        module_name = parameter_name.removesuffix(".weight")
        module = module_by_name.get(module_name)
        is_protected = (
            parameter_name.endswith(".weight")
            and isinstance(module, nn.Linear)
            and module_pattern in module_name
        )
        if is_protected and module_name in projectors:
            projector = projectors[module_name].to(
                device=gradient_a.device, dtype=gradient_a.dtype
            )
            expected = (gradient_a.shape[-1], gradient_a.shape[-1])
            if projector.shape != expected:
                raise ValueError(
                    f"projector {module_name!r} has shape {tuple(projector.shape)}, "
                    f"expected {expected}"
                )
            projected_a = gradient_a.matmul(projector)
            projected_k = gradient_k.matmul(projector)
            value = projected_a + (gradient_k - projected_k)
            projected_count += 1
        elif is_protected and missing == "zero":
            value = torch.zeros_like(gradient_a)
        else:
            value = gradient_a.clone()
        output[parameter_name] = value
        before_sq += float(gradient_a.detach().float().square().sum().item())
        after_sq += float(value.detach().float().square().sum().item())

    return output, {
        "projected_modules": float(projected_count),
        "gradient_norm_before": before_sq**0.5,
        "gradient_norm_after": after_sq**0.5,
        "projected_gradient_ratio": (after_sq / before_sq) if before_sq else 0.0,
    }
