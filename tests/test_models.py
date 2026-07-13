import pytest
import torch

from latent_mixup_bc.models import BCPolicy, local_latent_action_mixup, prepare_training_batch


def test_input_mixup_reuses_lambda_and_permutation_for_actions():
    states = torch.tensor([[0.0], [10.0]])
    actions = torch.tensor([[1.0], [3.0]])
    lam = torch.tensor([0.25, 0.75])
    permutation = torch.tensor([1, 0])
    batch = prepare_training_batch(
        "input_mixup_bc", states, actions, lam=lam, permutation=permutation
    )
    expected_states = lam[:, None] * states + (1 - lam[:, None]) * states[permutation]
    expected_actions = lam[:, None] * actions + (1 - lam[:, None]) * actions[permutation]
    assert torch.allclose(batch.representation, expected_states)
    assert torch.allclose(batch.target, expected_actions)
    assert torch.equal(batch.permutation, permutation)


def test_every_method_has_the_same_policy_shape():
    model = BCPolicy(11, 3, latent_dim=128)
    assert model(torch.zeros(7, 11)).shape == (7, 3)
    assert model.encode(torch.zeros(7, 11)).shape == (7, 128)


def test_latent_mixup_uses_caller_supplied_pairing():
    latent = torch.tensor([[0.0, 2.0], [10.0, 4.0]])
    actions = torch.tensor([[1.0], [5.0]])
    lam = torch.tensor([0.5, 0.25])
    permutation = torch.tensor([1, 0])
    batch = prepare_training_batch(
        "latent_mixup_bc", latent, actions, lam=lam, permutation=permutation,
        representation_is_latent=True,
    )
    assert torch.allclose(
        batch.representation,
        lam[:, None] * latent + (1 - lam[:, None]) * latent[permutation],
    )
    assert torch.allclose(
        batch.target,
        lam[:, None] * actions + (1 - lam[:, None]) * actions[permutation],
    )


def test_unknown_method_fails_loudly():
    with pytest.raises(ValueError, match="unknown method"):
        prepare_training_batch("mystery", torch.zeros(2, 1), torch.zeros(2, 1))


def test_local_mixup_selects_nearest_action_compatible_neighbor():
    latent = torch.tensor([[0.0], [0.1], [0.2]])
    actions = torch.tensor([[0.0], [2.0], [0.2]])
    result = local_latent_action_mixup(
        latent, actions, alpha=0.2, action_threshold=0.5,
        lam=torch.tensor([0.5, 0.5, 0.5]),
    )
    assert torch.equal(result.neighbor_index, torch.tensor([2, 1, 0]))
    assert torch.equal(result.valid_neighbor_mask, torch.tensor([True, False, True]))
    assert torch.allclose(result.latent[0], torch.tensor([0.1]))
    assert torch.allclose(result.action[0], torch.tensor([0.1]))
    assert torch.equal(result.latent[1], latent[1])
    assert torch.equal(result.action[1], actions[1])


def test_local_mixup_batch_size_one_is_identity():
    latent, actions = torch.randn(1, 4), torch.randn(1, 2)
    result = local_latent_action_mixup(latent, actions)
    assert torch.equal(result.latent, latent)
    assert torch.equal(result.action, actions)
    assert not result.valid_neighbor_mask.item()
