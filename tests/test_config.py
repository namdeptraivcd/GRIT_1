from pathlib import Path

from grit1.config import load_project_config


def test_repository_config_loads_small_profile():
    config = load_project_config(Path(__file__).parents[1] / "config.yaml", "small_0_6b")
    assert config.profile.policy_model == "Qwen/Qwen3-0.6B"
    assert config.profile.base_model == config.profile.policy_model
    assert config.profile.prm_model == "Qwen/Qwen3Guard-Stream-0.6B"
    assert config.raw["update"]["use_curvature"] is True
    assert config.raw["update"]["curvature_mode"] == "central_fd"
    assert config.raw["update"]["central_fd_radius"] == 0.05
    assert config.use_curvature is True
    assert config.curvature_mode == "central_fd"
    assert config.central_fd_radius == 0.05
    assert config.central_fd_normalize_direction is True
    assert config.hvp_last_linear_layers == 1
