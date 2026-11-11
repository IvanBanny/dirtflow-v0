"""Categorical block states from sampled vectors and material-flow features."""

import torch
from torch import nn


class BlockStateHead(nn.Module):
    """Vectors [N,8] and features [N,C] to logits over K legal block states.

    Output classes index `output_to_global`, which contains input-registry IDs.
    The independent 128D keys retain the released checkpoint's parameter names.
    """

    output_to_global: torch.Tensor

    def __init__(
        self, output_to_global: torch.Tensor, feat_dim: int = 256, hidden: int = 256
    ) -> None:
        super().__init__()
        mapping = output_to_global.to(torch.long).clone()
        if mapping.ndim != 1 or not len(mapping) or mapping.min() < 0:
            raise ValueError("output map must be a nonempty vector of registry IDs")
        if len(mapping.unique()) != len(mapping):
            raise ValueError("output map contains duplicate states")
        self.register_buffer("output_to_global", mapping)
        self.output_vocab = len(mapping)
        self.mlp = nn.Sequential(
            nn.Linear(8 + feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 128),
        )
        self.keys = nn.Parameter(torch.randn(len(mapping), 128))
        self.class_bias = nn.Parameter(torch.zeros(len(mapping)))
        self.context_gamma = nn.Parameter(torch.zeros(()))
        self.scale = 128**-0.5

    def target_classes(self, ids: torch.Tensor) -> torch.Tensor:
        """Maps registry IDs to class IDs; unsupported states become -1."""
        ordered, classes = self.output_to_global.sort()
        pos = torch.searchsorted(ordered, ids).clamp_max(len(ordered) - 1)
        return torch.where(ordered[pos] == ids, classes[pos], -1)

    def forward(
        self, z: torch.Tensor, features: torch.Tensor, palette: torch.Tensor
    ) -> torch.Tensor:
        """Returns [N,K] logits; palette is a broadcastable [N,K] 0/1 indicator."""
        q = self.mlp(torch.cat((z, features), dim=-1))  # [N,8+C] -> [N,128]
        logits = q @ self.keys.T * self.scale  # [N,K]
        return logits + self.class_bias + self.context_gamma * palette

    @torch.no_grad()
    def decode(
        self,
        z: torch.Tensor,
        features: torch.Tensor,
        palette: torch.Tensor,
        *,
        allow_air: bool = True,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Returns registry IDs [N], bounding temporary logits to chunk_size*K."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        ids = torch.empty(len(z), dtype=torch.long, device=z.device)
        for start in range(0, len(z), chunk_size):
            stop = start + chunk_size
            context = palette if len(palette) == 1 else palette[start:stop]
            logits = self(z[start:stop], features[start:stop], context)
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite block-state logits")
            if not allow_air:
                logits[:, self.output_to_global == 0] = -torch.inf
                if not (self.output_to_global != 0).any():
                    raise ValueError("material-only decoding requires a non-air class")
            ids[start:stop] = self.output_to_global[logits.argmax(-1)]
        return ids
