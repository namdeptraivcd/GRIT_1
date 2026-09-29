from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelProfile:
    name: str
    policy_model: str
    base_model: str
    safety_model: str
    prm_model: str | None


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]
    profile: ModelProfile
    learning_rate: float
    lambda_pres: float
    epsilon_pres: float
    projector_path: str
    module_pattern: str
    process_shaping: bool
    process_discount: float
    soft_projection: bool
    kappa_max: float
    tau_kappa: float
    competence_gating: bool
    epsilon_min: float
    epsilon_max: float
    tau_epsilon: float
    use_curvature: bool
    curvature_mode: str
    central_fd_radius: float
    central_fd_normalize_direction: bool
    hvp_last_linear_layers: int


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required configuration key: {key}")
    return mapping[key]


def load_project_config(path: str | Path, profile_name: str) -> ProjectConfig:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("config must contain a YAML mapping")

    profiles = _required(payload, "model_profiles")
    if profile_name not in profiles:
        raise ValueError(
            f"unknown model profile {profile_name!r}; available={sorted(profiles)}"
        )
    selected = profiles[profile_name]
    profile = ModelProfile(
        name=profile_name,
        policy_model=str(_required(selected, "policy_model")),
        base_model=str(_required(selected, "base_model")),
        safety_model=str(_required(selected, "safety_model")),
        prm_model=selected.get("prm_model"),
    )
    if profile.policy_model != profile.base_model:
        raise ValueError("policy_model and base_model must match at initialization")

    update = payload.get("update", {})
    preservation = payload.get("preservation", {})
    projectors = payload.get("projectors", {})
    techniques = payload.get("techniques", {})
    shaping = techniques.get("process_shaped_credit", {})
    soft = techniques.get("process_gated_soft_projection", {})
    competence = techniques.get("competence_gated_trust_region", {})

    return ProjectConfig(
        raw=payload,
        profile=profile,
        learning_rate=float(update.get("learning_rate", 1e-6)),
        lambda_pres=float(update.get("lambda_pres", 1.0)),
        epsilon_pres=float(preservation.get("epsilon_pres", 0.05)),
        projector_path=str(projectors.get("path", "artifacts/projectors.pt")),
        module_pattern=str(projectors.get("module_pattern", "mlp")),
        process_shaping=bool(shaping.get("enable", False)),
        process_discount=float(shaping.get("discount", 1.0)),
        soft_projection=bool(soft.get("enable", False)),
        kappa_max=float(soft.get("kappa_max", 0.0)),
        tau_kappa=float(soft.get("tau_kappa", 1.0)),
        competence_gating=bool(competence.get("enable", False)),
        epsilon_min=float(competence.get("epsilon_min", 0.05)),
        epsilon_max=float(competence.get("epsilon_max", 0.05)),
        tau_epsilon=float(competence.get("tau_epsilon", 1.0)),
        use_curvature=bool(update.get("use_curvature", False)),
        curvature_mode=str(update.get("curvature_mode", "central_fd")),
        central_fd_radius=float(update.get("central_fd_radius", 0.05)),
        central_fd_normalize_direction=bool(
            update.get("central_fd_normalize_direction", True)
        ),
        hvp_last_linear_layers=int(update.get("hvp_last_linear_layers", 1)),
    )
