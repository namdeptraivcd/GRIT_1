import torch
from torch import nn

from grit1.train import named_trainable_parameters, predictor_step


def test_predictor_step_restores_parameters():
    model = nn.Linear(2, 1, bias=False)
    parameters = named_trainable_parameters(model)
    before = model.weight.detach().clone()
    gradients = {"weight": torch.ones_like(model.weight)}

    with predictor_step(parameters, gradients, learning_rate=0.1):
        torch.testing.assert_close(model.weight, before - 0.1)

    torch.testing.assert_close(model.weight, before)
