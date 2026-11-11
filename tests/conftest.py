import copy

import pytest
import torch

from dirtflow.data import Example
from dirtflow.model.network import Models


@pytest.fixture
def model_config():
    flow = dict(
        in_channels=32,
        out_channels=8,
        model_channels=24,
        cond_channels=12,
        num_blocks=2,
        num_heads=4,
        mlp_ratio=2,
        qk_rms_norm=True,
        self_conditioning=True,
    )
    return dict(
        feature_schema="v1",
        structure_coordinate_system="trellis_z_up",
        blockstate_coordinate_system="trellis_z_up",
        embedding=dict(dim=8, vocab_size=4, rms_norm=True),
        ss_vae=dict(
            latent_channels=8,
            channels=[4, 8, 12],
            num_res_blocks=1,
            num_res_blocks_middle=1,
        ),
        context_encoder=dict(in_channels=20, out_channels=16, widths=[4, 8, 12]),
        structure_flow=flow | dict(resolution=2),
        blockstate_flow=flow
        | dict(resolution=8, in_channels=42, io_block_channels=8, num_io_res_blocks=2),
        categorical_head=dict(kind="categorical_independent", hidden=16),
    )


@pytest.fixture
def models(model_config):
    torch.manual_seed(2)
    return Models(copy.deepcopy(model_config), torch.tensor([0, 2, 3]))


@pytest.fixture
def example():
    before = torch.zeros(8, 8, 8, dtype=torch.long)
    before[1:6, 1, 1:6] = 2
    target = before.clone()
    target[3:5, 2:4, 3:5] = 3
    edit = torch.zeros_like(before, dtype=torch.bool)
    edit[2:6, 2:6, 2:6] = True
    return Example(
        before,
        target,
        edit,
        torch.ones_like(edit),
        torch.randn(7, 12),
        torch.randn(7, 12),
        task=2,
    )
