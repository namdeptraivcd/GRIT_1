import torch

from grit1.losses import process_kappas, token_advantages


def test_process_shaping_uses_terminal_zero_potential():
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    advantages = token_advantages(
        mask,
        final_rewards=[1.0],
        prefix_scores=[[0.2, 0.6, 0.8]],
        discount=1.0,
    )
    torch.testing.assert_close(
        advantages,
        torch.tensor([[1.4, 1.2, 0.2]]),
    )


def test_without_prm_each_token_receives_final_reward():
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    advantages = token_advantages(
        mask,
        final_rewards=[0.25],
        prefix_scores=None,
        discount=1.0,
    )
    torch.testing.assert_close(advantages, torch.tensor([[0.25, 0.25, 0.0]]))


def test_process_kappa_is_bounded_by_kappa_max():
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    kappas = process_kappas(
        mask,
        [[0.2, 0.6, 0.8]],
        discount=1.0,
        kappa_max=0.2,
        tau_kappa=0.1,
    )
    assert torch.all(kappas >= 0)
    assert torch.all(kappas <= 0.2)
