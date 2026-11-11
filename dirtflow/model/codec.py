"""Occupancy codec: [B,1,64,64,64] and [B,8,16,16,16]."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from . import coordinates
from .layers import (
    ChannelLayerNorm32,
    zero_module,
)


def pixel_shuffle_3d(x: torch.Tensor, factor: int) -> torch.Tensor:
    B, C, H, W, D = x.shape
    C_ = C // factor**3
    x = x.reshape(B, C_, factor, factor, factor, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(B, C_, H * factor, W * factor, D * factor)


class ResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.norm1 = ChannelLayerNorm32(channels)
        self.norm2 = ChannelLayerNorm32(self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        )
        self.skip_connection = (
            nn.Conv3d(channels, self.out_channels, 1)
            if channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip_connection(x)


class DownsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_3d(self.conv(x), 2)


class SparseStructureDecoder(nn.Module):
    """Native latent [B,8,16,16,16] to world occupancy logits [B,1,64,64,64]."""

    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 8,
        num_res_blocks: int = 2,
        channels: tuple[int, ...] = (512, 128, 32),
        num_res_blocks_middle: int = 2,
        structure_coordinate_system: str = "minecraft_y_up",
    ) -> None:
        super().__init__()
        coordinates.validate_system(structure_coordinate_system)
        self.structure_coordinate_system = structure_coordinate_system
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(
            *[ResBlock3d(channels[0]) for _ in range(num_res_blocks_middle)]
        )
        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend((ResBlock3d(ch, ch) for _ in range(num_res_blocks)))
            if i < len(channels) - 1:
                self.blocks.append(UpsampleBlock3d(ch, channels[i + 1]))
        self.out_layer = nn.Sequential(
            ChannelLayerNorm32(channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        return coordinates.to_world_grid(
            self.out_layer(h), self.structure_coordinate_system
        )


class LatentNorm(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(channels))
        self.register_buffer("std", torch.ones(channels))

    def _view(self, ref: torch.Tensor, stat: torch.Tensor) -> torch.Tensor:
        return stat.view(1, -1, *[1] * (ref.dim() - 2)).to(ref.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self._view(z, self.mean)) / self._view(z, self.std)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * self._view(z, self.std) + self._view(z, self.mean)


class SparseStructureEncoder(nn.Module):
    """World occupancy [B,1,R,R,R] to the native posterior mean [B,8,R/4,R/4,R/4]."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 8,
        num_res_blocks: int = 2,
        channels: tuple[int, ...] = (32, 128, 512),
        num_res_blocks_middle: int = 2,
        structure_coordinate_system: str = "minecraft_y_up",
    ) -> None:
        super().__init__()
        coordinates.validate_system(structure_coordinate_system)
        self.structure_coordinate_system = structure_coordinate_system
        self.latent_channels = latent_channels
        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend(ResBlock3d(ch, ch) for _ in range(num_res_blocks))
            if i < len(channels) - 1:
                self.blocks.append(DownsampleBlock3d(ch, channels[i + 1]))
        self.middle_block = nn.Sequential(
            *[ResBlock3d(channels[-1]) for _ in range(num_res_blocks_middle)]
        )
        self.out_layer = nn.Sequential(
            ChannelLayerNorm32(channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels * 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = coordinates.to_structure_grid(x, self.structure_coordinate_system)
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h)
        h = self.out_layer(h)
        mean, _ = h.chunk(2, dim=1)
        return mean
