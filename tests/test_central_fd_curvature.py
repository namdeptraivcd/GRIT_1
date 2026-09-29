import torch
from torch import nn

from grit.curvature import central_difference_curvature_corrected_preservation_gradients


class QuadraticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(2, 1, bias=False)


def test_central_difference_recovers_quadratic_hvp_without_hessian():
    model = QuadraticModel()
    parameters = list(model.named_parameters())
    original = model.mlp.weight.detach().clone()

    def task_loss():
        return 0.5 * model.mlp.weight.square().sum()

    preservation = torch.tensor([[0.25, -0.5]])
    result = central_difference_curvature_corrected_preservation_gradients(
        model,
        task_loss,
        {"mlp.weight": preservation},
        {"mlp": torch.eye(2)},
        learning_rate=0.1,
        radius=0.01,
        normalize_direction=True,
        parameters=parameters,
        hvp_parameters=parameters,
    )

    torch.testing.assert_close(result.hvp["mlp.weight"], preservation, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        result.gradients["mlp.weight"], 0.9 * preservation, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(model.mlp.weight, original, atol=0.0, rtol=0.0)
    assert not result.skipped_hvp


def test_central_difference_reuses_rng_for_both_sides():
    model = QuadraticModel()
    parameters = list(model.named_parameters())
    preservation = torch.tensor([[0.25, -0.5]])

    def stochastic_task_loss():
        scale = torch.rand(())
        return 0.5 * scale * model.mlp.weight.square().sum()

    torch.manual_seed(7)
    expected_scale = torch.rand(())
    torch.manual_seed(7)
    result = central_difference_curvature_corrected_preservation_gradients(
        model,
        stochastic_task_loss,
        {"mlp.weight": preservation},
        {"mlp": torch.eye(2)},
        learning_rate=0.1,
        radius=0.01,
        normalize_direction=True,
        parameters=parameters,
        hvp_parameters=parameters,
    )

    torch.testing.assert_close(
        result.hvp["mlp.weight"], expected_scale * preservation, atol=1e-5, rtol=1e-5
    )
