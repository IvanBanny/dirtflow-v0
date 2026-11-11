import copy

import pytest
import torch
from torch.nn import functional as F

from dirtflow import checkpoint
from dirtflow.model import coordinates, features, sparse
from dirtflow.model.head import BlockStateHead


def test_frame_roundtrip_and_direction():
    world = torch.zeros(1, 1, 4, 4, 4)
    world[0, 0, 1, 2, 3] = 1
    native = coordinates.to_structure_grid(world, "trellis_z_up")
    assert native[0, 0, 1, 0, 2] == 1
    torch.testing.assert_close(coordinates.to_world_grid(native, "trellis_z_up"), world)


@pytest.mark.parametrize("lengths", [[3, 2], [0, 4], [0, 0], [2, 2]])
def test_attention_separates_examples(lengths):
    torch.manual_seed(1)
    total = sum(lengths)
    q, k, v = [torch.randn(total, 2, 4, requires_grad=True) for _ in range(3)]
    actual = sparse.varlen_self_attention(q, k, v, torch.tensor(lengths))
    expected = []
    start = 0
    for n in lengths:
        a, b, c = (x[start : start + n].transpose(0, 1) for x in (q, k, v))
        weights = (a @ b.transpose(-2, -1) / 2).softmax(-1)
        expected.append((weights @ c).transpose(0, 1))
        start += n
    expected = torch.cat(expected)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    if total:
        actual.square().sum().backward()
        assert all(
            x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v)
        )


def test_sparse_convolution_matches_dense_and_gradients():
    torch.manual_seed(3)
    mask = torch.rand(2, 4, 4, 4) > 0.55
    grid = sparse.from_dense_mask(mask)
    values = torch.randn(len(grid), 2, requires_grad=True)
    grid = grid.replace_feats(values)
    conv = sparse.SubMConv3d(2, 3).double()
    values = values.detach().double().requires_grad_()
    grid = grid.replace_feats(values)
    weight = conv.weight.detach().clone().requires_grad_()
    bias = conv.bias.detach().clone().requires_grad_()
    reference_values = values.detach().clone().requires_grad_()
    dense = sparse.to_dense(grid.replace_feats(reference_values)).movedim(-1, 1)
    kernel = weight.reshape(3, 3, 3, 2, 3).permute(4, 3, 0, 1, 2)
    expected = F.conv3d(dense, kernel, bias, padding=1).movedim(1, -1)[mask]
    actual = conv(grid).feats
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(
        values.grad, reference_values.grad, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(conv.weight.grad, weight.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(conv.bias.grad, bias.grad, atol=1e-10, rtol=1e-10)


def test_sparse_pool_uses_active_cell_mean():
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 3, 3, 3]])
    grid = sparse.SparseGrid(coords, torch.tensor([[2.0], [4.0], [9.0]]), 1, 4)
    coarse, parent = sparse.downsample(grid)
    torch.testing.assert_close(coarse.feats, torch.tensor([[3.0], [9.0]]))
    torch.testing.assert_close(
        sparse.upsample(coarse, parent, grid).feats, torch.tensor([[3.0], [3.0], [9.0]])
    )


def test_observation_channels_and_dropout():
    before = torch.tensor([[[[2, 3, 999]]]])
    edit = torch.tensor([[[[False, True, False]]]])
    seen = torch.tensor([[[[True, True, False]]]])
    table = torch.arange(32).reshape(4, 8).float()
    actual = features.dense_observation_features(before, edit, seen, table)
    assert actual.shape == (1, 20, 1, 1, 3)
    torch.testing.assert_close(actual[0, 1:9, 0, 0, 0], table[2])
    torch.testing.assert_close(actual[0, 12:20, 0, 0, 1], table[3])
    assert actual[..., 2].count_nonzero() == 0
    dropped = features.dense_observation_features(
        before, edit, seen, table, context_keep=torch.zeros(1)
    )
    assert dropped.sum() == 1 and dropped[0, 9, 0, 0, 1] == 1


def test_codec_flow_shapes_and_gradients(models, example):
    latent = models.ss_encoder((example.target != 0)[None, None].float())
    assert latent.shape == (1, 8, 2, 2, 2)
    assert models.ss_decoder(latent).shape == (1, 1, 8, 8, 8)
    context = models.context_encoder(
        example.before[None], example.edit[None], example.observed[None]
    )
    assert context.shape == (1, 16, 2, 2, 2)
    x = torch.cat((latent, context, torch.zeros_like(latent)), 1)
    y = models.structure_flow(x, torch.tensor([500.0]), example.text[None])
    assert y.shape == latent.shape
    y.square().mean().backward()
    assert models.structure_flow.input_layer.weight.grad.norm() > 0
    assert models.context_encoder.stem.weight.grad.norm() > 0


def test_material_coordinates_and_gradients(models):
    mask = torch.zeros(2, 8, 8, 8, dtype=torch.bool)
    mask[0, 1:3, 2:4, 4:6] = True
    mask[1, 4:6, 3:5, 1:3] = True
    grid = sparse.from_dense_mask(mask)
    grid = grid.replace_feats(torch.randn(len(grid), 42, requires_grad=True))
    prediction, io = models.blockstate_flow(
        grid, torch.tensor([200.0, 800.0]), torch.randn(2, 7, 12), return_io_feats=True
    )
    assert torch.equal(prediction.coords, grid.coords)
    assert prediction.feats.shape == (len(grid), 8)
    assert io.shape == (len(grid), 8)
    prediction.feats.square().sum().backward()
    assert grid.feats.grad.norm() > 0
    assert models.blockstate_flow.input_layer.weight.grad.norm() > 0


def test_recomputation_matches_direct_gradients(models):
    reference = copy.deepcopy(models.structure_flow)
    reference.use_checkpoint = False
    x = torch.randn(1, 32, 2, 2, 2)
    t, text = torch.tensor([300.0]), torch.randn(1, 7, 12)
    a = models.structure_flow(x, t, text)
    b = reference(x, t, text)
    torch.testing.assert_close(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    for p, q in zip(models.structure_flow.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_head_mapping_chunking_and_palette():
    head = BlockStateHead(torch.tensor([0, 5, 2]), feat_dim=4, hidden=8)
    torch.testing.assert_close(
        head.target_classes(torch.tensor([-1, 0, 1, 2, 5, 99])),
        torch.tensor([-1, 0, -1, 2, 1, -1]),
    )
    z, h = torch.randn(9, 8), torch.randn(9, 4)
    palette = torch.tensor([[0.0, 1.0, 0.0]])
    expected = head.output_to_global[head(z, h, palette).argmax(-1)]
    torch.testing.assert_close(head.decode(z, h, palette, chunk_size=2), expected)
    with torch.no_grad():
        head.context_gamma.fill_(1000)
    assert (head.decode(z, h, palette) == 5).all()
    varied = torch.eye(3).repeat(3, 1)
    expected = head.output_to_global[head(z, h, varied).argmax(-1)]
    torch.testing.assert_close(head.decode(z, h, varied, chunk_size=2), expected)
    with torch.no_grad():
        head.keys[0, 0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        head.decode(z, h, palette)


def test_trellis_input_expansion(models):
    flow = models.structure_flow
    source = {name: value.clone() for name, value in flow.state_dict().items()}
    source["input_layer.weight"] = source["input_layer.weight"][:, :8]
    checkpoint.transfer_trellis(flow, source)
    assert flow.input_layer.weight[:, 8:].count_nonzero() == 0
    x = torch.randn(4, 32)
    torch.testing.assert_close(
        flow.input_layer(x),
        F.linear(x[:, :8], source["input_layer.weight"], source["input_layer.bias"]),
    )
    del source["out_layer.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        checkpoint.transfer_trellis(flow, source)
