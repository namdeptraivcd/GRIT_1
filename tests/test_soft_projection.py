import torch
from torch import nn

from grit1.soft_projection import (
    process_gated_project_gradient_map,
    soft_project_gradient_map,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(2, 2, bias=False)


def test_soft_projection_interpolates_hard_projection_and_identity():
    model = ToyModel()
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    projector = torch.diag(torch.tensor([1.0, 0.0]))

    projected, metrics = soft_project_gradient_map(
        model,
        {"mlp.weight": gradient},
        {"mlp": projector},
        kappa=0.5,
    )

    expected = torch.tensor([[1.0, 1.0], [3.0, 2.0]])
    torch.testing.assert_close(projected["mlp.weight"], expected)
    assert metrics["projected_modules"] == 1.0
    assert metrics["kappa"] == 0.5


def test_process_gated_projection_combines_two_gradient_components():
    model = ToyModel()
    gradient_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    gradient_k = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    projector = torch.diag(torch.tensor([1.0, 0.0]))

    projected, _metrics = process_gated_project_gradient_map(
        model,
        {"mlp.weight": gradient_a},
        {"mlp.weight": gradient_k},
        {"mlp": projector},
    )

    torch.testing.assert_close(
        projected["mlp.weight"], torch.tensor([[1.0, 6.0], [3.0, 8.0]])
    )
