"""Continuous flow matching and legal-state classification losses."""

import functools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import Example
from .model import coordinates, features
from .model.head import BlockStateHead
from .model.network import MATERIALS, STRUCTURE, Models

SIGMA = 1e-5


def interpolate(
    clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor | float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear path with data at t=0 and Gaussian noise at t=1."""
    t = torch.as_tensor(t, dtype=clean.dtype, device=clean.device).reshape(
        -1, *([1] * (clean.ndim - 1))
    )
    return (1 - t) * clean + (SIGMA + (1 - SIGMA) * t) * noise, (
        1 - SIGMA
    ) * noise - clean


def clean_estimate(
    x: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor | float
) -> torch.Tensor:
    # A shaped FP32 time keeps BF16 velocities from rounding the product first
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(
        -1, *([1] * (x.ndim - 1))
    )
    return (1 - SIGMA) * x - (SIGMA + (1 - SIGMA) * t) * velocity


def structure_loss(
    predicted: torch.Tensor, target: torch.Tensor, edit: torch.Tensor
) -> torch.Tensor:
    """Weights edit/context means by 0.7/0.3; an empty group gets zero weight."""
    error = (predicted.float() - target.float()).square().mean(1).flatten(1)
    mask = edit.flatten(1)
    n = mask.sum(1)
    other = mask.shape[1] - n
    edit_mean = (error * mask).sum(1) / n.clamp_min(1)
    context_mean = (error * ~mask).sum(1) / other.clamp_min(1)
    weight = torch.where(n == 0, 0.0, torch.where(other == 0, 1.0, 0.7))
    return (weight * edit_mean + (1 - weight) * context_mean).mean()


@functools.cache
def mean_data_weight() -> float:
    """E[1-t] for t=sigmoid(N(1,1)), integrated by Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite_e.hermegauss(129)
    return float((w / (1 + np.exp(1 + x))).sum() / np.sqrt(2 * np.pi))


def uniformity(
    table: torch.Tensor, ids: torch.Tensor, rng: torch.Generator
) -> torch.Tensor:
    """Log mean pairwise exp(-2*distance^2), over at most 512 distinct targets."""
    ids = ids.unique()
    if len(ids) < 2:
        return table.sum() * 0
    if len(ids) > 512:
        ids = ids[torch.randperm(len(ids), generator=rng, device=ids.device)[:512]]
    distance = torch.pdist(table[ids].float()).square()
    return torch.logsumexp(-2 * distance, dim=0) - np.log(len(distance))


def material_loss(
    head: BlockStateHead,
    predicted: torch.Tensor,
    velocity: torch.Tensor,
    estimate: torch.Tensor,
    io: torch.Tensor,
    targets: torch.Tensor,
    editable: torch.Tensor,
    palette: torch.Tensor,
    table: torch.Tensor,
    t: torch.Tensor,
    presentations: int,
    rng: torch.Generator,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One example's MSE + time-weighted full-vocabulary CE + uniformity.

    Invalid states have zero weight. Examples without editable coordinates
    contribute zero and remain in the outer batch denominator.
    """
    classes = head.target_classes(targets)
    weights = torch.where(editable, 1.0, 0.3) * (classes >= 0) * editable.any()
    denominator = weights.sum().clamp_min(1e-12)
    mse = (
        (predicted.float() - velocity.float()).square().mean(-1) * weights
    ).sum() / denominator
    ce = estimate.sum() * 0

    def cross_entropy(z, h, labels):
        return F.cross_entropy(head(z, h, palette).float(), labels, reduction="none")

    for start in range(0, len(estimate), chunk_size):
        stop = start + chunk_size
        values = checkpoint(
            cross_entropy,
            estimate[start:stop],
            io[start:stop],
            classes[start:stop].clamp_min(0),
            use_reentrant=False,
        )
        ce = ce + (values * weights[start:stop]).sum() / denominator
    ce = ce * (1 - t.squeeze()) / mean_data_weight()
    reg = uniformity(table, targets[classes >= 0], rng) * editable.any()
    coefficient = 0.5 * min(presentations / 40_000, 1.0)
    total = mse + coefficient * ce + 1e-3 * reg
    return total, dict(mse=mse.detach(), ce=ce.detach(), uniformity=reg.detach())


class TrainingLoss(nn.Module):
    """Computes one presentation; the trainer owns batching and optimizer state."""

    def __init__(self, models: Models, stage: str) -> None:
        super().__init__()
        if stage not in {"structure", "materials"}:
            raise ValueError("stage must be structure or materials")
        self.stage, self.config = stage, models.config
        if stage == "structure":
            self.context_encoder = models.context_encoder
            self.structure_flow = models.structure_flow
            self.structure_extra = models.structure_extra
            self.ss_encoder = models.ss_encoder
            self.latent_norm = models.latent_norm
        else:
            self.blockstate_flow = models.blockstate_flow
            self.blockstate_extra = models.blockstate_extra
            self.embedding = models.embedding
            self.categorical_head = models.categorical_head

    def forward(
        self, example: Example, presentations: int, seed: int
    ) -> tuple[torch.Tensor, dict]:
        m = self
        device = example.before.device
        rng = torch.Generator(device).manual_seed(seed)
        t = torch.sigmoid(torch.randn(1, device=device, generator=rng) + 1)
        text_keep, context_keep, style_keep, use_self = torch.rand(
            4, device=device, generator=rng
        ).unbind()
        cond = torch.where(text_keep >= 0.1, example.text, example.null_text)[None]
        style = torch.tensor([example.style], device=device)
        style = torch.where(style_keep >= 0.5, style, 5)
        module = m.structure_extra if self.stage == "structure" else m.blockstate_extra
        extra = module(
            style,
            torch.tensor([example.score], device=device),
            torch.tensor([example.fragment], device=device),
            torch.tensor([example.domain], device=device),
            torch.tensor([example.task], device=device),
        )
        keep = (context_keep >= 0.1).float().reshape(1)
        if self.stage == "structure":
            with torch.no_grad():
                clean = m.latent_norm(
                    m.ss_encoder((example.target != 0)[None, None].float())
                ).float()
            context = m.context_encoder(
                example.before[None], example.edit[None], example.observed[None], keep
            )
            noise = torch.randn(clean.shape, device=device, generator=rng)
            noisy, velocity = interpolate(clean, noise, t)
            previous = torch.zeros_like(clean)
            if use_self < 0.5:
                with torch.no_grad():
                    v = m.structure_flow(
                        torch.cat((noisy, context, previous), 1), t * 1000, cond, extra
                    )
                    previous = clean_estimate(noisy, v, t)
            predicted = m.structure_flow(
                torch.cat((noisy, context, previous), 1), t * 1000, cond, extra
            )
            edit = F.max_pool3d(example.edit[None, None].float(), 4, 4)
            edit = F.max_pool3d(edit, 5, 1, 2)[:, 0].bool()
            edit = coordinates.to_structure_grid(
                edit, m.config["structure_coordinate_system"]
            )
            total = structure_loss(predicted, velocity, edit)
            metrics = dict(mse=total.detach())
        else:
            active = (
                example.active if example.active is not None else example.target != 0
            )
            active = features.support(
                active.float() * 2 - 1,
                example.edit,
                example.observed,
                example.before,
                features.TASK_NAMES[example.task],
            )
            table = m.embedding.table()
            obs = features.material_observation(
                active,
                example.before,
                example.edit,
                example.observed,
                table,
                m.categorical_head,
            )
            _, x, y, z = obs.grid.coords.unbind(1)
            targets = torch.where(
                obs.known, example.before[x, y, z], example.target[x, y, z]
            )
            clean = table[targets]
            noise = torch.randn(clean.shape, device=device, generator=rng)
            noisy, velocity = interpolate(clean, noise, t)
            previous = torch.zeros_like(clean)
            if use_self < 0.5:
                with torch.no_grad():
                    v = m.blockstate_flow(
                        obs.inputs(noisy, previous, keep.expand(len(clean))),
                        t * 1000,
                        cond,
                        extra,
                    )
                    previous = clean_estimate(noisy, v.feats, t)
            predicted, io = m.blockstate_flow(
                obs.inputs(noisy, previous, keep.expand(len(clean))),
                t * 1000,
                cond,
                extra,
                return_io_feats=True,
            )
            total, metrics = material_loss(
                m.categorical_head,
                predicted.feats,
                velocity,
                clean_estimate(noisy, predicted.feats, t),
                io,
                targets,
                ~obs.known,
                obs.palette * keep,
                table,
                t,
                presentations,
                rng,
            )
        # Empty sparse examples still participate in every DDP reduction
        names = STRUCTURE if self.stage == "structure" else MATERIALS
        total = total + sum(
            p.reshape(-1)[0] * 0
            for name in names
            for p in getattr(m, name).parameters()
        )
        return total, metrics
