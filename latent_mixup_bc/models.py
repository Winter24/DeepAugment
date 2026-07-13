"""Controlled BC architecture and explicit augmentation transformations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


METHODS = ("vanilla_bc", "noise_bc", "input_mixup_bc", "latent_mixup_bc")


class BCPolicy(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, latent_dim: int = 128, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def act_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.policy_head(latent)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.act_from_latent(self.encode(states))


@dataclass(frozen=True)
class MixupBatch:
    representation: torch.Tensor
    target: torch.Tensor
    lambda_: torch.Tensor
    permutation: torch.Tensor


def sample_mixup(reference: torch.Tensor, alpha: float = 0.2):
    batch_size = reference.shape[0]
    if batch_size == 0:
        raise ValueError("cannot mix an empty batch")
    if alpha <= 0:
        lam = torch.ones(batch_size, device=reference.device, dtype=reference.dtype)
    else:
        concentration = torch.tensor(alpha, device=reference.device, dtype=reference.dtype)
        lam = torch.distributions.Beta(concentration, concentration).sample((batch_size,))
    permutation = torch.randperm(batch_size, device=reference.device)
    return lam, permutation


def _mix(values: torch.Tensor, lam: torch.Tensor, permutation: torch.Tensor):
    shape = (values.shape[0],) + (1,) * (values.ndim - 1)
    weight = lam.reshape(shape)
    return weight * values + (1.0 - weight) * values[permutation]


def prepare_training_batch(
    method: str,
    representation: torch.Tensor,
    actions: torch.Tensor,
    *,
    alpha: float = 0.2,
    noise_std: float = 0.05,
    lam: torch.Tensor | None = None,
    permutation: torch.Tensor | None = None,
    representation_is_latent: bool = False,
) -> MixupBatch:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method!r}")
    batch_size = representation.shape[0]
    if actions.shape[0] != batch_size:
        raise ValueError("representation and actions must have the same batch size")
    identity_lam = torch.ones(batch_size, device=representation.device, dtype=representation.dtype)
    identity_perm = torch.arange(batch_size, device=representation.device)

    if method == "vanilla_bc":
        return MixupBatch(representation, actions, identity_lam, identity_perm)
    if method == "noise_bc":
        if representation_is_latent:
            raise ValueError("noise_bc noise must be applied to normalized input states")
        noisy = representation + noise_std * torch.randn_like(representation)
        return MixupBatch(noisy, actions, identity_lam, identity_perm)

    if method == "latent_mixup_bc" and not representation_is_latent:
        raise ValueError("latent_mixup_bc requires encoded representations")
    if method == "input_mixup_bc" and representation_is_latent:
        raise ValueError("input_mixup_bc requires normalized input states")

    if (lam is None) != (permutation is None):
        raise ValueError("lam and permutation must be supplied together")
    if lam is None:
        lam, permutation = sample_mixup(representation, alpha)
    lam = lam.to(device=representation.device, dtype=representation.dtype)
    permutation = permutation.to(device=representation.device, dtype=torch.long)
    if lam.shape != (batch_size,) or permutation.shape != (batch_size,):
        raise ValueError("lambda and permutation must each have shape [batch_size]")
    return MixupBatch(
        representation=_mix(representation, lam, permutation),
        target=_mix(actions, lam.to(actions.dtype), permutation),
        lambda_=lam,
        permutation=permutation,
    )
