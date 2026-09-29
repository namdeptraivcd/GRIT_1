"""Curvature HVP utilities for GRIT Phase 4.

GRIT's exact preservation gradient corrects the first-order preservation pull

    v = grad_{theta_tilde} L_pres(theta_tilde)

with the Hessian-vector product from differentiating through the task descent
predictor step:

    v - learning_rate * H_task(theta) P v

This module computes the product without materializing a Hessian.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from grit.projection import ProjectorBuildResult, default_module_filter


ModuleFilter = Callable[[str, nn.Module], bool]
ParameterFilter = Callable[[str, nn.Parameter], bool]
NamedParameter = tuple[str, nn.Parameter]
GradientMap = Mapping[str, torch.Tensor | None]


@dataclass(frozen=True)
class CurvatureCorrectionResult:
    """Gradient pieces and diagnostics for the Phase 4 correction."""

    gradients: dict[str, torch.Tensor]
    projected_vector: dict[str, torch.Tensor]
    hvp: dict[str, torch.Tensor]
    skipped_hvp: bool
    preservation_grad_norm: float
    projected_vector_norm: float
    hvp_norm: float


def trainable_named_parameters(
    model: nn.Module,
    parameter_filter: ParameterFilter | None = None,
) -> list[NamedParameter]:
    """Return trainable parameters, optionally filtered by name and tensor."""

    parameters: list[NamedParameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter_filter is not None and not parameter_filter(name, parameter):
            continue
        parameters.append((name, parameter))
    return parameters


def _zero_like_parameter(parameter: nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(parameter, memory_format=torch.preserve_format)


def _squared_norm(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        norm = torch.linalg.vector_norm(tensor.detach())
        total += float(norm.item()) ** 2
    return total


def _is_all_zero(tensors: Iterable[torch.Tensor]) -> bool:
    for tensor in tensors:
        detached = tensor.detach()
        if detached.numel() == 0:
            continue
        if bool((detached.max() != 0).item()) or bool((detached.min() != 0).item()):
            return False
    return True


def _linear_weight_to_module_names(model: nn.Module) -> dict[str, str]:
    names: dict[str, str] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameter_name = f"{module_name}.weight" if module_name else "weight"
            names[parameter_name] = module_name
    return names


def project_vector_with_module_projectors(
    model: nn.Module,
    vector: GradientMap,
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
    *,
    parameters: Sequence[NamedParameter] | None = None,
    module_filter: ModuleFilter | None = None,
    missing: str = "identity",
) -> dict[str, torch.Tensor]:
    """Apply the GRIT/NSPO block projector to a named parameter vector.

    Linear weights with a matching module projector use ``u_W = v_W @ P``.
    Parameters outside that surface default to identity in ``u = P v``:
    Phase 1 supplies nontrivial projectors only for protected Linear weights,
    while the full-theta block operator leaves unprojected parameters unchanged.
    Set ``missing="zero"`` only for a Linear-only ablation.
    """

    if missing not in {"zero", "identity"}:
        raise ValueError(f"missing must be 'zero' or 'identity', got {missing!r}")
    if module_filter is None:
        module_filter = lambda name, module: default_module_filter(name, module)
    if parameters is None:
        parameters = trainable_named_parameters(model)

    linear_weight_names = _linear_weight_to_module_names(model)
    module_by_name = dict(model.named_modules())
    projected: dict[str, torch.Tensor] = {}

    for parameter_name, parameter in parameters:
        value = vector.get(parameter_name)
        if value is None:
            projected[parameter_name] = _zero_like_parameter(parameter)
            continue
        if value.shape != parameter.shape:
            raise ValueError(
                f"vector for {parameter_name} has shape {tuple(value.shape)}, "
                f"expected {tuple(parameter.shape)}"
            )

        module_name = linear_weight_names.get(parameter_name)
        module = module_by_name.get(module_name, None) if module_name is not None else None
        if (
            module_name is not None
            and module is not None
            and module_filter(module_name, module)
            and module_name in projectors
        ):
            raw_projector = projectors[module_name]
            projector = raw_projector.projector if isinstance(raw_projector, ProjectorBuildResult) else raw_projector
            projector = projector.to(device=value.device, dtype=value.dtype)
            projected[parameter_name] = value.matmul(projector)
        elif missing == "identity":
            projected[parameter_name] = value.detach().clone()
        else:
            projected[parameter_name] = _zero_like_parameter(parameter)

    return projected


def hessian_vector_product(
    loss: torch.Tensor,
    parameters: Sequence[NamedParameter],
    vector: GradientMap,
    *,
    allow_unused: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute ``grad_params <grad_params loss, vector>``.

    This is the Pearlmutter HVP identity used by Phase 4. ``create_graph=True``
    is local to the first derivative so callers can skip this function entirely
    when ``vector`` is zero.
    """

    parameter_tensors = [parameter for _, parameter in parameters]
    vector_tensors = [
        torch.zeros_like(parameter) if vector.get(name) is None else vector[name].detach().to(parameter)
        for name, parameter in parameters
    ]
    if _is_all_zero(vector_tensors):
        return {name: _zero_like_parameter(parameter) for name, parameter in parameters}

    task_grads = torch.autograd.grad(
        loss,
        parameter_tensors,
        create_graph=True,
        retain_graph=True,
        allow_unused=allow_unused,
    )

    dot_terms = []
    for grad, direction in zip(task_grads, vector_tensors, strict=True):
        if grad is None:
            continue
        dot_terms.append(torch.sum(grad * direction))

    if not dot_terms:
        return {name: _zero_like_parameter(parameter) for name, parameter in parameters}

    dot = torch.stack(dot_terms).sum()
    if not dot.requires_grad:
        return {name: _zero_like_parameter(parameter) for name, parameter in parameters}

    hvp_tensors = torch.autograd.grad(
        dot,
        parameter_tensors,
        retain_graph=True,
        allow_unused=allow_unused,
    )
    return {
        name: _zero_like_parameter(parameter) if hvp is None else hvp.detach()
        for (name, parameter), hvp in zip(parameters, hvp_tensors, strict=True)
    }


def curvature_corrected_preservation_gradients(
    model: nn.Module,
    task_loss: torch.Tensor,
    preservation_gradients: GradientMap,
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
    *,
    learning_rate: float,
    parameters: Sequence[NamedParameter] | None = None,
    parameter_filter: ParameterFilter | None = None,
    module_filter: ModuleFilter | None = None,
    hvp_parameters: Sequence[NamedParameter] | None = None,
    missing_projector: str = "identity",
) -> CurvatureCorrectionResult:
    """Return ``v - learning_rate * H_task P v`` as named gradients.

    ``P`` is treated as a full-theta block operator: protected Linear weights
    use their Phase 1 projector, and parameters without a projector use the
    identity block. ``task_loss`` should have the sign used by the optimizer.
    If the training code minimizes ``task_loss = -J_task``, then the returned
    HVP is the Hessian of that minimization loss, not the Hessian of ``J_task``.
    """

    if learning_rate < 0:
        raise ValueError(f"learning_rate must be non-negative, got {learning_rate}")
    if parameters is None:
        parameters = trainable_named_parameters(model, parameter_filter)

    first_order: dict[str, torch.Tensor] = {}
    for name, parameter in parameters:
        grad = preservation_gradients.get(name)
        if grad is None:
            first_order[name] = _zero_like_parameter(parameter)
            continue
        if grad.shape != parameter.shape:
            raise ValueError(
                f"preservation gradient for {name} has shape {tuple(grad.shape)}, "
                f"expected {tuple(parameter.shape)}"
            )
        first_order[name] = grad.detach().clone()

    projected_vector = project_vector_with_module_projectors(
        model,
        first_order,
        projectors,
        parameters=parameters,
        module_filter=module_filter,
        missing=missing_projector,
    )

    if hvp_parameters is None:
        hvp_parameters = parameters
    hvp_parameter_names = {name for name, _parameter in hvp_parameters}
    hvp_projected_vector = {
        name: projected_vector[name]
        for name in hvp_parameter_names
        if name in projected_vector
    }

    skipped_hvp = learning_rate == 0.0 or not hvp_parameters or _is_all_zero(hvp_projected_vector.values())
    if skipped_hvp:
        hvp = {name: _zero_like_parameter(parameter) for name, parameter in parameters}
    else:
        partial_hvp = hessian_vector_product(task_loss, hvp_parameters, hvp_projected_vector)
        hvp = {name: _zero_like_parameter(parameter) for name, parameter in parameters}
        hvp.update(partial_hvp)

    gradients = {
        name: first_order[name] - hvp[name].to(first_order[name]).mul(learning_rate)
        for name, _parameter in parameters
    }
    preservation_grad_norm = _squared_norm(first_order.values()) ** 0.5
    projected_vector_norm = _squared_norm(projected_vector.values()) ** 0.5
    hvp_norm = _squared_norm(hvp.values()) ** 0.5

    return CurvatureCorrectionResult(
        gradients=gradients,
        projected_vector=projected_vector,
        hvp=hvp,
        skipped_hvp=skipped_hvp,
        preservation_grad_norm=preservation_grad_norm,
        projected_vector_norm=projected_vector_norm,
        hvp_norm=hvp_norm,
    )


def central_difference_curvature_corrected_preservation_gradients(
    model: nn.Module,
    task_loss_fn: Callable[[], torch.Tensor],
    preservation_gradients: GradientMap,
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
    *,
    learning_rate: float,
    radius: float,
    normalize_direction: bool = True,
    parameters: Sequence[NamedParameter] | None = None,
    parameter_filter: ParameterFilter | None = None,
    module_filter: ModuleFilter | None = None,
    hvp_parameters: Sequence[NamedParameter] | None = None,
    missing_projector: str = "identity",
) -> CurvatureCorrectionResult:
    """Approximate ``v - learning_rate * H_task P v`` with a central difference.

    The two task gradients are evaluated sequentially at ``theta + delta`` and
    ``theta - delta``. With ``normalize_direction=True``, ``delta`` is
    ``radius * P v / ||P v||``. The denominator is rescaled so that the result
    approximates ``H_task P v`` rather than ``H_task normalize(P v)``. The
    model parameters and PyTorch RNG state are restored after both evaluations.
    """

    if learning_rate < 0:
        raise ValueError(f"learning_rate must be non-negative, got {learning_rate}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if parameters is None:
        parameters = trainable_named_parameters(model, parameter_filter)

    first_order: dict[str, torch.Tensor] = {}
    for name, parameter in parameters:
        grad = preservation_gradients.get(name)
        if grad is None:
            first_order[name] = _zero_like_parameter(parameter)
            continue
        if grad.shape != parameter.shape:
            raise ValueError(
                f"preservation gradient for {name} has shape {tuple(grad.shape)}, "
                f"expected {tuple(parameter.shape)}"
            )
        first_order[name] = grad.detach().clone()

    projected_vector = project_vector_with_module_projectors(
        model,
        first_order,
        projectors,
        parameters=parameters,
        module_filter=module_filter,
        missing=missing_projector,
    )

    if hvp_parameters is None:
        hvp_parameters = parameters
    hvp_parameter_names = {name for name, _parameter in hvp_parameters}
    hvp_projected_vector = {
        name: projected_vector[name]
        for name in hvp_parameter_names
        if name in projected_vector
    }

    projected_vector_norm = _squared_norm(hvp_projected_vector.values()) ** 0.5
    skipped_hvp = learning_rate == 0.0 or not hvp_parameters or projected_vector_norm == 0.0
    if skipped_hvp:
        hvp = {name: _zero_like_parameter(parameter) for name, parameter in parameters}
    else:
        scale = (radius / projected_vector_norm) if normalize_direction else radius
        originals: list[tuple[nn.Parameter, torch.Tensor, torch.Tensor]] = []
        for name, parameter in hvp_parameters:
            direction = hvp_projected_vector[name].to(device=parameter.device, dtype=parameter.dtype)
            originals.append((parameter, parameter.detach().clone(), direction.mul(scale)))

        cpu_rng_before = torch.random.get_rng_state()
        cuda_rng_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        cpu_rng_after_plus: torch.Tensor | None = None
        cuda_rng_after_plus: list[torch.Tensor] | None = None
        try:
            with torch.no_grad():
                for parameter, original, perturbation in originals:
                    parameter.copy_(original + perturbation)
            plus_loss = task_loss_fn()
            plus_partial = torch.autograd.grad(
                plus_loss,
                [parameter for _, parameter in hvp_parameters],
                retain_graph=False,
                allow_unused=True,
            )
            cpu_rng_after_plus = torch.random.get_rng_state()
            if cuda_rng_before is not None:
                cuda_rng_after_plus = torch.cuda.get_rng_state_all()

            torch.random.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state_all(cuda_rng_before)
            with torch.no_grad():
                for parameter, original, perturbation in originals:
                    parameter.copy_(original - perturbation)
            minus_loss = task_loss_fn()
            minus_partial = torch.autograd.grad(
                minus_loss,
                [parameter for _, parameter in hvp_parameters],
                retain_graph=False,
                allow_unused=True,
            )
        finally:
            with torch.no_grad():
                for parameter, original, _perturbation in originals:
                    parameter.copy_(original)
            torch.random.set_rng_state(
                cpu_rng_before if cpu_rng_after_plus is None else cpu_rng_after_plus
            )
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state_all(
                    cuda_rng_before if cuda_rng_after_plus is None else cuda_rng_after_plus
                )

        difference_scale = 1.0 / (2.0 * scale)
        partial_hvp: dict[str, torch.Tensor] = {}
        for (name, parameter), plus_grad, minus_grad in zip(
            hvp_parameters, plus_partial, minus_partial, strict=True
        ):
            plus = _zero_like_parameter(parameter) if plus_grad is None else plus_grad.detach()
            minus = _zero_like_parameter(parameter) if minus_grad is None else minus_grad.detach()
            partial_hvp[name] = (plus - minus.to(plus)).mul(difference_scale)

        hvp = {name: _zero_like_parameter(parameter) for name, parameter in parameters}
        hvp.update(partial_hvp)

    gradients = {
        name: first_order[name] - hvp[name].to(first_order[name]).mul(learning_rate)
        for name, _parameter in parameters
    }
    preservation_grad_norm = _squared_norm(first_order.values()) ** 0.5
    full_projected_vector_norm = _squared_norm(projected_vector.values()) ** 0.5
    hvp_norm = _squared_norm(hvp.values()) ** 0.5

    return CurvatureCorrectionResult(
        gradients=gradients,
        projected_vector=projected_vector,
        hvp=hvp,
        skipped_hvp=skipped_hvp,
        preservation_grad_norm=preservation_grad_norm,
        projected_vector_norm=full_projected_vector_norm,
        hvp_norm=hvp_norm,
    )
