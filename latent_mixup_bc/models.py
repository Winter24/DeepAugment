"""Controlled BC architecture and explicit augmentation transformations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


METHODS = (
    "vanilla_bc", "noise_bc", "input_mixup_bc", "latent_mixup_bc",
    "local_latent_mixup_bc",
)


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


@dataclass(frozen=True)
class LocalMixupBatch:
    latent: torch.Tensor
    action: torch.Tensor
    lambda_: torch.Tensor
    neighbor_index: torch.Tensor
    valid_neighbor_mask: torch.Tensor


def local_latent_action_mixup(
    latent: torch.Tensor,
    actions: torch.Tensor,
    alpha: float = 0.2,
    action_threshold: float = 1.0,
    lam: torch.Tensor | None = None,
) -> LocalMixupBatch:
    if latent.ndim != 2 or actions.ndim != 2:
        raise ValueError("latent and actions must be rank-2 tensors")
    if latent.shape[0] != actions.shape[0]:
        raise ValueError("latent and actions must have the same batch size")
    if action_threshold < 0:
        raise ValueError("action_threshold must be non-negative")
    batch_size = latent.shape[0]
    if batch_size == 0:
        raise ValueError("cannot mix an empty batch")

    identity = torch.arange(batch_size, device=latent.device)
    if batch_size == 1:
        ones = torch.ones(1, device=latent.device, dtype=latent.dtype)
        return LocalMixupBatch(latent, actions, ones, identity, torch.zeros(1, device=latent.device, dtype=torch.bool))

    with torch.no_grad():
        latent_distance = torch.cdist(latent.detach(), latent.detach(), p=2.0)
        action_distance = torch.cdist(actions.detach(), actions.detach(), p=2.0)
        valid_pairs = action_distance <= action_threshold
        valid_pairs.fill_diagonal_(False)
        masked_distance = latent_distance.masked_fill(~valid_pairs, float("inf"))
        has_neighbor = valid_pairs.any(dim=1)
        nearest = masked_distance.argmin(dim=1)
        nearest = torch.where(has_neighbor, nearest, identity)

    if lam is None:
        if alpha <= 0:
            lam = torch.ones(batch_size, device=latent.device, dtype=latent.dtype)
        else:
            concentration = torch.tensor(alpha, device=latent.device, dtype=latent.dtype)
            lam = torch.distributions.Beta(concentration, concentration).sample((batch_size,))
    lam = lam.to(device=latent.device, dtype=latent.dtype).reshape(-1)
    if lam.shape != (batch_size,):
        raise ValueError("lambda must have shape [batch_size]")
    effective_lambda = torch.where(has_neighbor, lam, torch.ones_like(lam))
    latent_mix = _mix(latent, effective_lambda, nearest)
    action_mix = _mix(actions, effective_lambda.to(actions.dtype), nearest)
    return LocalMixupBatch(latent_mix, action_mix, effective_lambda, nearest, has_neighbor)


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
    action_threshold: float = 1.0,
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

    if method in ("latent_mixup_bc", "local_latent_mixup_bc") and not representation_is_latent:
        raise ValueError(f"{method} requires encoded representations")
    if method == "input_mixup_bc" and representation_is_latent:
        raise ValueError("input_mixup_bc requires normalized input states")

    if method == "local_latent_mixup_bc":
        if permutation is not None:
            raise ValueError("local_latent_mixup_bc selects its neighbor from distances")
        local = local_latent_action_mixup(
            representation, actions, alpha=alpha,
            action_threshold=action_threshold, lam=lam,
        )
        return MixupBatch(
            local.latent, local.action, local.lambda_, local.neighbor_index
        )

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
