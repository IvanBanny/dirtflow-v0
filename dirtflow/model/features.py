"""Observed-world features in checkpoint channel order."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn
from torch.nn import functional as F

from . import coordinates, sparse
from .codec import DownsampleBlock3d, ResBlock3d
from .head import BlockStateHead
from .layers import TimestepEmbedder

TASK_NAMES = (
    "scratch_void",
    "scratch_ground",
    "completion",
    "replacement",
    "repair",
    "material_only",
)
NULL_STYLE_ID = 5
MC_DOMAIN_ID = 0
NUM_DOMAINS = 3


def dense_observation_features(
    before_ids: torch.Tensor,
    editable: torch.Tensor,
    observed: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    context_keep: torch.Tensor | None = None,
) -> torch.Tensor:
    if before_ids.shape != editable.shape or editable.shape != observed.shape:
        raise ValueError("before, editable and observed shapes must agree")
    if editable.dtype != torch.bool or observed.dtype != torch.bool:
        raise ValueError("editable and observed must be boolean")
    if embedding_table.shape[1] != 8:
        raise ValueError("v1 observation layout requires eight embedding channels")
    safe_ids = torch.where(observed, before_ids, 0)
    embeddings = F.embedding(safe_ids, embedding_table).movedim(-1, 1)
    known = observed & ~editable
    old = observed & editable
    occupancy = safe_ids != 0
    features = torch.cat(
        [
            (occupancy & known).unsqueeze(1),
            embeddings * known.unsqueeze(1),
            editable.unsqueeze(1),
            observed.unsqueeze(1),
            (occupancy & old).unsqueeze(1),
            embeddings * old.unsqueeze(1),
        ],
        dim=1,
    ).float()
    if context_keep is not None:
        features = features * context_keep.reshape(-1, 1, 1, 1, 1)
        features[:, 9] = editable
    return features


def neighbor_observations(coords: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 4 or observed.ndim != 4:
        raise ValueError("expected coordinates [L,4] and observed [B,X,Y,Z]")
    directions = coords.new_tensor(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    )
    neighbors = coords[:, None, 1:] + directions
    bounds = coords.new_tensor(observed.shape[1:])
    valid = ((neighbors >= 0) & (neighbors < bounds)).all(dim=-1)
    neighbors = torch.minimum(neighbors.clamp_min(0), bounds - 1).long()
    values = observed[
        coords[:, None, 0].long(),
        neighbors[..., 0],
        neighbors[..., 1],
        neighbors[..., 2],
    ]
    return (values & valid).float()


def sparse_observation_features(
    noisy: torch.Tensor,
    known_embedding: torch.Tensor,
    known: torch.Tensor,
    edit_logits: torch.Tensor,
    logit_valid: torch.Tensor,
    self_condition: torch.Tensor,
    old_edit_embedding: torch.Tensor,
    old_edit_valid: torch.Tensor,
    neighbor_observed: torch.Tensor,
    *,
    context_keep: torch.Tensor | None = None,
) -> torch.Tensor:
    length = noisy.shape[0]
    vectors = (noisy, known_embedding, self_condition, old_edit_embedding)
    if any((value.shape != (length, 8) for value in vectors)):
        raise ValueError("all v1 embedding features must have shape [L,8]")
    if neighbor_observed.shape != (length, 6):
        raise ValueError("neighbor observations must have shape [L,6]")
    known = known.reshape(length, 1).to(noisy.dtype)
    old_edit_valid = old_edit_valid.reshape(length, 1).to(noisy.dtype)
    if bool(((known != 0) & (old_edit_valid != 0)).any()):
        raise ValueError("known and old-edit observations must be disjoint")
    keep = 1 if context_keep is None else context_keep.reshape(length, 1)
    valid = logit_valid.reshape(length, 1).to(noisy.dtype) * (1 - known)
    return torch.cat(
        [
            noisy,
            known_embedding * known * keep,
            known * keep,
            edit_logits.reshape(length, 1) * valid,
            valid,
            self_condition,
            old_edit_embedding * old_edit_valid * keep,
            old_edit_valid * keep,
            neighbor_observed * keep,
        ],
        dim=1,
    )


class BlockStateEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 8, *, rms_norm: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.rms_norm = rms_norm
        weight = torch.randn(vocab_size, dim)
        self.weight = nn.Parameter(weight)

    def table(self) -> torch.Tensor:
        if not self.rms_norm:
            return self.weight
        values = self.weight.float()
        rms = values.pow(2).mean(dim=-1, keepdim=True).add(1e-08).rsqrt()
        return (values * rms).to(self.weight.dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.table())


class StructureContextEncoder(nn.Module):
    """Observed world [B,X,Y,Z] to native context [B,16,16,16,16]."""

    def __init__(
        self,
        *,
        in_channels: int = 20,
        out_channels: int = 16,
        widths: tuple[int, ...] = (32, 64, 128),
        vocab_size: int,
        embedding_dim: int = 8,
        embedding_rms_norm: bool = True,
        structure_coordinate_system: str = "minecraft_y_up",
    ) -> None:
        super().__init__()
        if in_channels != 20:
            raise ValueError("context encoder requires 20 observation channels")
        coordinates.validate_system(structure_coordinate_system)
        self.structure_coordinate_system = structure_coordinate_system
        self.embedding = BlockStateEmbedding(
            vocab_size, embedding_dim, rms_norm=embedding_rms_norm
        )
        self.stem = nn.Conv3d(in_channels, widths[0], 3, padding=1)
        stages: list[nn.Module] = []
        for i, width in enumerate(widths):
            stages.append(ResBlock3d(width))
            if i < len(widths) - 1:
                stages.append(DownsampleBlock3d(width, widths[i + 1]))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Conv3d(widths[-1], out_channels, 3, padding=1)

    def forward(
        self,
        before: torch.Tensor,
        edit: torch.Tensor,
        observed: torch.Tensor,
        context_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = dense_observation_features(
            before, edit, observed, self.embedding.table(), context_keep=context_keep
        )  # [B,20,X,Y,Z]
        context = self.head(self.stages(self.stem(features)))
        return coordinates.to_structure_grid(context, self.structure_coordinate_system)


class ExtraConditioning(nn.Module):
    """Caption metadata, domain and editing task added to the time embedding."""

    def __init__(self, model_channels: int) -> None:
        super().__init__()
        self.style_emb = nn.Embedding(NULL_STYLE_ID + 1, model_channels)
        self.score_emb = TimestepEmbedder(model_channels)
        self.fragment_emb = nn.Embedding(2, model_channels)
        self.domain_emb = nn.Embedding(NUM_DOMAINS, model_channels)
        self.task_emb = nn.Embedding(len(TASK_NAMES), model_channels)
        self.null_score = nn.Parameter(torch.zeros(model_channels))
        nn.init.zeros_(self.task_emb.weight)
        nn.init.zeros_(self.style_emb.weight)
        nn.init.zeros_(self.score_emb.mlp[2].get_parameter("weight"))
        nn.init.zeros_(self.score_emb.mlp[2].get_parameter("bias"))
        nn.init.zeros_(self.fragment_emb.weight)
        nn.init.zeros_(self.domain_emb.weight)

    def forward(
        self,
        style_ids: torch.Tensor,
        scores: torch.Tensor,
        fragment: torch.Tensor,
        domain_ids: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        style = self.style_emb(style_ids)
        known = (scores >= 0).float().unsqueeze(1)
        score = self.score_emb(scores.clamp(min=0.0) / 10.0 * 1000.0)
        return (
            style
            + score * known
            + (1 - known) * self.null_score
            + self.task_emb(task_ids)
            + self.fragment_emb(fragment.long())
            + self.domain_emb(domain_ids)
        )


def support(
    logits: torch.Tensor,
    edit: torch.Tensor,
    observed: torch.Tensor,
    before: torch.Tensor,
    task: str,
) -> torch.Tensor:
    """Keeps edit proposals and at most 8192 nearest observed context cells."""
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite occupancy logits")
    selected = (before != 0) & edit if task == "material_only" else (logits >= 0) & edit
    if task == "repair":
        selected |= (before != 0) & edit
    known = observed & ~edit
    retained = torch.zeros_like(known, memory_format=torch.contiguous_format)
    covered, remaining = edit.clone(), 8192
    for _ in range(max(logits.shape)):
        if not remaining or not (known & ~covered).any():
            break
        expanded = F.max_pool3d(covered[None, None].float(), 3, 1, 1)[0, 0].bool()
        layer = (expanded & ~covered & known).reshape(-1).nonzero().flatten()
        chosen = layer[:remaining]
        retained.view(-1)[chosen] = True
        remaining -= chosen.numel()
        covered = expanded
    return selected | retained


@dataclasses.dataclass
class MaterialObservation:
    """Observation features at one example's fixed sparse coordinates."""

    grid: sparse.SparseGrid
    known: torch.Tensor
    clean: torch.Tensor
    old: torch.Tensor
    old_valid: torch.Tensor
    neighbors: torch.Tensor
    palette: torch.Tensor
    confidence: torch.Tensor

    def inputs(
        self,
        noisy: torch.Tensor,
        previous: torch.Tensor,
        keep: torch.Tensor | None = None,
    ) -> sparse.SparseGrid:
        feats = sparse_observation_features(
            noisy,
            self.clean,
            self.known,
            self.confidence,
            ~self.known,
            previous,
            self.old,
            self.old_valid,
            self.neighbors,
            context_keep=keep,
        )
        return self.grid.replace_feats(feats)


def material_observation(
    active: torch.Tensor,
    before: torch.Tensor,
    edit: torch.Tensor,
    observed: torch.Tensor,
    table: torch.Tensor,
    head: BlockStateHead,
) -> MaterialObservation:
    """Builds observed features once; editable confidence is exactly zero."""
    grid = sparse.from_dense_mask(active[None])
    _, x, y, z = grid.coords.unbind(1)
    known = ~edit[x, y, z] & observed[x, y, z]
    clean = table[before[x, y, z] * known]
    old = table[before[x, y, z]]
    old_valid = edit[x, y, z] & observed[x, y, z]
    palette = table.new_zeros(1, head.output_vocab)
    classes = head.target_classes(
        torch.cat((before[observed], before.new_zeros(1))).unique()
    )
    palette[0, classes[classes >= 0]] = 1
    return MaterialObservation(
        grid,
        known,
        clean,
        old,
        old_valid,
        neighbor_observations(grid.coords, observed[None]),
        palette,
        table.new_zeros(len(grid)),
    )
