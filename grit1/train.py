from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import random
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import nn

from grit.curvature import central_difference_curvature_corrected_preservation_gradients
from grit.projection import load_projectors
from grit1.config import load_project_config
from grit1.data import distributed_batches, extract_prompt, load_rows, validate_dataset_role
from grit1.losses import (
    build_policy_batch,
    policy_gradient_loss,
    preservation_kl_loss,
    response_log_probs,
    token_advantages,
    process_kappas,
)
from grit1.prm import SafetyStreamPRM
from grit1.soft_projection import process_gated_project_gradient_map
from grit1.vllm_client import VLLMRolloutClient, VLLMSafetyClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRIT_1 first-order RL trainer with vLLM rollout and data parallelism"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", default="small_0_6b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--task-data", required=True)
    parser.add_argument("--preservation-data", required=True)
    parser.add_argument("--projectors", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rollout-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rollout-model", default="policy")
    parser.add_argument("--safety-url", default="http://127.0.0.1:8001")
    parser.add_argument("--safety-model", default=None)
    parser.add_argument("--prm-model", default=None)
    parser.add_argument("--use-prm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--soft-projection", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--trust-region", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--competence-gating", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-curvature", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--central-fd-radius", type=float, default=None)
    parser.add_argument(
        "--central-fd-normalize-direction",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--hvp-last-linear-layers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--preservation-batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lambda-pres", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--resume-optimizer", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def named_trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def last_projected_linear_parameters(
    model: nn.Module,
    parameters: Sequence[tuple[str, nn.Parameter]],
    projectors: Mapping[str, torch.Tensor],
    *,
    module_pattern: str,
    count: int,
) -> list[tuple[str, nn.Parameter]]:
    if count <= 0:
        return list(parameters)
    parameter_by_name = dict(parameters)
    selected: list[str] = []
    for module_name, module in model.named_modules():
        parameter_name = f"{module_name}.weight" if module_name else "weight"
        if (
            isinstance(module, nn.Linear)
            and module_pattern in module_name
            and module_name in projectors
            and parameter_name in parameter_by_name
        ):
            selected.append(parameter_name)
    selected_set = set(selected[-count:])
    return [(name, parameter) for name, parameter in parameters if name in selected_set]


def softened_projectors(
    projectors: Mapping[str, torch.Tensor], kappa: float
) -> dict[str, torch.Tensor]:
    if not 0.0 <= kappa <= 1.0:
        raise ValueError("kappa must be in [0, 1]")
    return {
        name: projector + kappa * (torch.eye(projector.shape[0], dtype=projector.dtype) - projector)
        for name, projector in projectors.items()
    }


def gradient_map(
    loss: torch.Tensor,
    parameters: Sequence[tuple[str, nn.Parameter]],
    *,
    retain_graph: bool = False,
) -> dict[str, torch.Tensor]:
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in parameters],
        allow_unused=True,
        retain_graph=retain_graph,
    )
    return {
        name: torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for (name, parameter), gradient in zip(parameters, gradients)
    }


@contextlib.contextmanager
def predictor_step(
    parameters: Sequence[tuple[str, nn.Parameter]],
    gradients: Mapping[str, torch.Tensor],
    learning_rate: float,
) -> Iterator[None]:
    with torch.no_grad():
        for name, parameter in parameters:
            parameter.add_(gradients[name].to(parameter), alpha=-learning_rate)
    try:
        yield
    finally:
        with torch.no_grad():
            for name, parameter in parameters:
                parameter.add_(gradients[name].to(parameter), alpha=learning_rate)


def all_reduce_gradients(gradients: dict[str, torch.Tensor], world_size: int) -> None:
    if world_size == 1:
        return
    for gradient in gradients.values():
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient.div_(world_size)


def global_normalize(values: torch.Tensor, mask: torch.Tensor, world_size: int) -> torch.Tensor:
    active = values[mask.bool()].float()
    statistics = torch.tensor(
        [active.sum(), active.square().sum(), active.numel()],
        dtype=torch.float64,
        device=values.device,
    )
    if world_size > 1:
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    count = statistics[2].clamp_min(1.0)
    mean = statistics[0] / count
    variance = (statistics[1] / count - mean.square()).clamp_min(1e-8)
    normalized = (values - mean.to(values.dtype)) / variance.sqrt().to(values.dtype)
    return normalized * mask


def scalar_mean(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(world_size)
    return float(tensor.item())


def competence_epsilon(
    cfg,
    prm: SafetyStreamPRM | None,
    prompts: list[str],
    current_responses: list[str],
    base_responses: list[str],
) -> float:
    if not cfg.competence_gating or prm is None or not base_responses:
        return cfg.epsilon_pres
    current = prm.score_batch(prompts, current_responses)
    base = prm.score_batch(prompts, base_responses)
    margins = [a.values[-1] - b.values[-1] for a, b in zip(current, base)]
    margin = sum(margins) / max(1, len(margins))
    gate = 1.0 / (1.0 + math.exp(-margin / max(cfg.tau_epsilon, 1e-6)))
    return cfg.epsilon_min + (cfg.epsilon_max - cfg.epsilon_min) * gate


def save_checkpoint(model, tokenizer, optimizer, output_dir: Path, rank: int) -> None:
    if rank != 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    (output_dir / "READY").write_text("ok\n")


def main() -> None:
    args = parse_args()
    rank, _local_rank, world_size, device = distributed_context()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    cfg = load_project_config(args.config, args.profile)
    model_path = args.model_path or cfg.profile.policy_model
    base_model_path = args.base_model_path or cfg.profile.base_model
    safety_model = args.safety_model or cfg.profile.safety_model
    prm_model = args.prm_model or cfg.profile.prm_model
    use_prm = cfg.process_shaping if args.use_prm is None else args.use_prm
    use_soft_projection = cfg.soft_projection if args.soft_projection is None else args.soft_projection
    use_competence = cfg.competence_gating if args.competence_gating is None else args.competence_gating
    cfg = cfg.__class__(**{**cfg.__dict__, "competence_gating": use_competence})
    learning_rate = args.learning_rate or cfg.learning_rate
    lambda_pres = cfg.lambda_pres if args.lambda_pres is None else args.lambda_pres
    update_config = cfg.raw.get("update", {})
    use_curvature = (
        bool(update_config.get("use_curvature", False))
        if args.use_curvature is None
        else args.use_curvature
    )
    curvature_mode = str(update_config.get("curvature_mode", "central_fd"))
    if use_curvature and curvature_mode != "central_fd":
        raise ValueError("only curvature_mode=central_fd is supported")
    central_fd_radius = (
        float(update_config.get("central_fd_radius", 0.05))
        if args.central_fd_radius is None
        else args.central_fd_radius
    )
    central_fd_normalize = (
        bool(update_config.get("central_fd_normalize_direction", True))
        if args.central_fd_normalize_direction is None
        else args.central_fd_normalize_direction
    )
    hvp_last_layers = (
        int(update_config.get("hvp_last_linear_layers", 1))
        if args.hvp_last_linear_layers is None
        else args.hvp_last_linear_layers
    )
    if use_curvature and central_fd_radius <= 0:
        raise ValueError("central_fd_radius must be positive")

    if use_prm and not prm_model:
        raise ValueError("--use-prm requires a prm_model in the selected profile or --prm-model")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        policy.config.use_cache = False
    policy.train()

    base = None
    if args.trust_region and lambda_pres > 0.0:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True, torch_dtype=dtype
        ).to(device)
        base.eval()
        for parameter in base.parameters():
            parameter.requires_grad_(False)

    projector_path = Path(args.projectors or cfg.projector_path)
    if not projector_path.exists():
        raise FileNotFoundError(f"missing projector artifact: {projector_path}")
    projectors = load_projectors(projector_path)

    task_rows = load_rows(args.task_data)
    preservation_rows = load_rows(args.preservation_data)
    validate_dataset_role(task_rows, expected="safety_task", source=args.task_data)
    validate_dataset_role(
        preservation_rows,
        expected="preservation",
        source=args.preservation_data,
    )
    task_batches = distributed_batches(
        task_rows,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        epoch=args.epoch,
        drop_last=True,
    )
    preservation_batches = list(
        distributed_batches(
            preservation_rows,
            batch_size=args.preservation_batch_size,
            rank=rank,
            world_size=world_size,
            seed=args.seed + 10_000,
            epoch=args.epoch,
            drop_last=True,
        )
    )
    if not preservation_batches:
        raise ValueError("preservation dataset is too small for the distributed batch size")
    preservation_iterator = itertools.cycle(preservation_batches)

    rollout = VLLMRolloutClient(args.rollout_url, args.rollout_model)
    safety = VLLMSafetyClient(args.safety_url, safety_model)
    prm = SafetyStreamPRM(prm_model, device) if use_prm and prm_model else None

    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.0)
    optimizer_path = Path(args.resume_optimizer) if args.resume_optimizer else None
    if optimizer_path and optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

    parameters = named_trainable_parameters(policy)
    completed = 0
    for step, task_rows_batch in enumerate(task_batches):
        if step >= args.max_steps:
            break
        prompts = [extract_prompt(row) for row in task_rows_batch]
        responses = rollout.generate(
            prompts,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        safety_scores = safety.score_batch(prompts, responses)
        final_rewards = [score.reward for score in safety_scores]
        prefix_scores = None
        if prm is not None:
            prefix_scores = [item.values for item in prm.score_batch(prompts, responses)]

        policy_batch = build_policy_batch(
            tokenizer,
            prompts,
            responses,
            device=device,
            max_length=args.max_length,
        )
        log_probs, response_mask = response_log_probs(policy, policy_batch)
        advantages = token_advantages(
            response_mask,
            final_rewards,
            prefix_scores,
            discount=cfg.process_discount,
        )
        advantages = global_normalize(advantages, response_mask, world_size)
        task_loss = policy_gradient_loss(log_probs, response_mask, advantages)
        task_gradients = gradient_map(
            task_loss, parameters, retain_graph=use_soft_projection and prefix_scores is not None
        )
        gated_gradients = {
            name: torch.zeros_like(value) for name, value in task_gradients.items()
        }
        mean_kappa = 0.0
        if use_soft_projection and prefix_scores is not None:
            kappas = process_kappas(
                response_mask,
                prefix_scores,
                discount=cfg.process_discount,
                kappa_max=cfg.kappa_max,
                tau_kappa=cfg.tau_kappa,
            )
            gated_loss = policy_gradient_loss(
                log_probs, response_mask, advantages * kappas
            )
            gated_gradients = gradient_map(gated_loss, parameters)
            mean_kappa = float(kappas[response_mask.bool()].mean().detach().item())
        projected_gradients, projection_metrics = process_gated_project_gradient_map(
            policy,
            task_gradients,
            gated_gradients,
            projectors,
            module_pattern=cfg.module_pattern,
            missing="identity",
        )
        del task_gradients, gated_gradients
        projection_metrics["kappa"] = mean_kappa

        preservation_metrics = {"preservation_kl": 0.0, "violation_fraction": 0.0}
        preservation_gradients = {name: torch.zeros_like(value) for name, value in projected_gradients.items()}
        preservation_correction = preservation_gradients
        curvature_metrics = {
            "central_fd_hvp_norm": 0.0,
            "central_fd_projected_vector_norm": 0.0,
            "central_fd_hvp_skipped": 1.0,
        }
        epsilon = cfg.epsilon_pres
        if base is not None and lambda_pres > 0.0:
            preserve_batch = next(preservation_iterator)
            preserve_prompts = [extract_prompt(row) for row in preserve_batch]
            base_responses = [
                str(row.get("base_response", "")).strip() for row in preserve_batch
            ]
            if use_competence and prm is not None and all(base_responses):
                current_responses = rollout.generate(
                    preserve_prompts,
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                )
                epsilon = competence_epsilon(
                    cfg, prm, preserve_prompts, current_responses, base_responses
                )
            preserve_texts = [
                prompt + ("\n" + response if response else "")
                for prompt, response in zip(preserve_prompts, base_responses)
            ]
            with predictor_step(parameters, projected_gradients, learning_rate):
                preservation_loss, preservation_metrics = preservation_kl_loss(
                    policy,
                    base,
                    tokenizer,
                    preserve_texts,
                    device=device,
                    max_length=args.max_length,
                    epsilon=epsilon,
                    top_k=int(cfg.raw.get("preservation", {}).get("top_k", 64)),
                    default_probability=float(
                        cfg.raw.get("preservation", {}).get("default_probability", 1e-12)
                    ),
                    reduction=str(
                        cfg.raw.get("preservation", {}).get(
                            "reduction", "seq-mean-token-mean"
                        )
                    ),
                )
                preservation_gradients = gradient_map(preservation_loss, parameters)

            preservation_correction = preservation_gradients
            if use_curvature:
                hvp_parameters = last_projected_linear_parameters(
                    policy,
                    parameters,
                    projectors,
                    module_pattern=cfg.module_pattern,
                    count=hvp_last_layers,
                )

                def task_loss_fn() -> torch.Tensor:
                    perturbed_log_probs, perturbed_mask = response_log_probs(
                        policy, policy_batch
                    )
                    return policy_gradient_loss(
                        perturbed_log_probs, perturbed_mask, advantages
                    )

                curvature_result = central_difference_curvature_corrected_preservation_gradients(
                    policy,
                    task_loss_fn,
                    preservation_gradients,
                    softened_projectors(projectors, mean_kappa),
                    learning_rate=learning_rate,
                    radius=central_fd_radius,
                    normalize_direction=central_fd_normalize,
                    parameters=parameters,
                    module_filter=lambda name, module: isinstance(module, nn.Linear)
                    and cfg.module_pattern in name,
                    hvp_parameters=hvp_parameters,
                    missing_projector="identity",
                )
                preservation_correction = curvature_result.gradients
                curvature_metrics = {
                    "central_fd_hvp_norm": curvature_result.hvp_norm,
                    "central_fd_projected_vector_norm": curvature_result.projected_vector_norm,
                    "central_fd_hvp_skipped": float(curvature_result.skipped_hvp),
                }
                del curvature_result

        combined = {
            name: projected_gradients[name] + lambda_pres * preservation_correction[name]
            for name, _parameter in parameters
        }
        all_reduce_gradients(combined, world_size)
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in parameters:
            parameter.grad = combined[name].to(parameter)
        del projected_gradients, preservation_gradients, preservation_correction, combined
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed += 1

        metrics = {
            "epoch": args.epoch,
            "step": step,
            "task_loss": scalar_mean(float(task_loss.detach().item()), device, world_size),
            "reward": scalar_mean(sum(final_rewards) / len(final_rewards), device, world_size),
            "unsafe_fraction": scalar_mean(
                sum(score.label == "Unsafe" for score in safety_scores) / len(safety_scores),
                device,
                world_size,
            ),
            "epsilon": scalar_mean(epsilon, device, world_size),
            "gradient_norm": scalar_mean(float(grad_norm), device, world_size),
            **{key: scalar_mean(value, device, world_size) for key, value in projection_metrics.items()},
            **{key: scalar_mean(value, device, world_size) for key, value in preservation_metrics.items()},
            **{key: scalar_mean(value, device, world_size) for key, value in curvature_metrics.items()},
        }
        if rank == 0:
            print(json.dumps(metrics, sort_keys=True), flush=True)

    if completed == 0:
        raise RuntimeError("no training steps ran; increase dataset size or reduce global batch size")
    if world_size > 1:
        dist.barrier()
    save_checkpoint(policy, tokenizer, optimizer, Path(args.output_dir), rank)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
