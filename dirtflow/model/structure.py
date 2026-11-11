"""Dense flow over normalized occupancy latents."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .layers import (
    AbsolutePositionEmbedder,
    ModulatedTransformerCrossBlock,
    TimestepEmbedder,
)


class StructureFlowModel(nn.Module):
    """Latent, context and self-condition [B,32,R,R,R] to velocity [B,8,R,R,R]."""

    pos_emb: torch.Tensor

    def __init__(
        self,
        *,
        resolution: int = 16,
        in_channels: int = 32,
        out_channels: int = 8,
        model_channels: int = 1280,
        cond_channels: int = 768,
        num_blocks: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = False,
        self_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.use_checkpoint = True
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.self_conditioning = self_conditioning
        self.t_embedder = TimestepEmbedder(model_channels)
        pos_embedder = AbsolutePositionEmbedder(model_channels)
        coords = torch.stack(
            torch.meshgrid(*[torch.arange(resolution)] * 3, indexing="ij"), dim=-1
        ).reshape(-1, 3)
        self.register_buffer("pos_emb", pos_embedder(coords))
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_layer = nn.Linear(model_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        t_extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        # Voxel tokens
        h = x.reshape(B, self.in_channels, -1).permute(0, 2, 1)  # [B,R^3,32]
        h = self.input_layer(h)  # [B,R^3,D]
        h = h + self.pos_emb[None].to(h.dtype)
        t_emb = self.t_embedder(t)
        if t_extra is not None:
            t_emb = t_emb + t_extra

        # Attn
        for block in self.blocks:
            h = (
                checkpoint(block, h, t_emb, cond, use_reentrant=False)
                if self.training and self.use_checkpoint
                else block(h, t_emb, cond)
            )
        # Velocity
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)  # [B,R^3,8]
        r = self.resolution
        return h.permute(0, 2, 1).reshape(B, self.out_channels, r, r, r)
