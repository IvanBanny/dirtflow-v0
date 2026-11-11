import copy
import dataclasses
import gc
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from dirtflow import checkpoint, training
from dirtflow.data import Example
from dirtflow.model import sparse
from dirtflow.objectives import TrainingLoss
from dirtflow.sampling import Sampler

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def cublas_workspace():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        yield


@pytest.fixture
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def cuda():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("requires CUDA with BF16 support")


@pytest.mark.parametrize("stage", ["structure", "materials"])
def test_cuda_training_step(models, example, stage):
    cuda()
    models.train_stage(stage, "cuda")
    import bitsandbytes

    config = training.Config(stage=stage, replay_initial=0, replay_final=0)
    optimizer = bitsandbytes.optim.AdamW8bit(
        training.parameter_groups(models, config), betas=(0.9, 0.95), is_paged=False
    )
    before = [p.detach().clone() for p in models.parameters() if p.requires_grad]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = TrainingLoss(models, stage)(example.to("cuda"), 40_000, 6)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(models.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, [p for p in models.parameters() if p.requires_grad])
    )


@pytest.mark.parametrize("stage", ["structure", "materials"])
def test_cuda_optimizer_resume(models, example, tmp_path, stage, deterministic):
    cuda()
    from test_training import cpu_config, write_dataset

    data = write_dataset(tmp_path, example)
    config = dataclasses.replace(
        cpu_config(stage), precision="bf16", optimizer="adamw8bit"
    )
    initial = copy.deepcopy(models)
    full = training.run(
        models, data, config, tmp_path / "full", torch.device("cuda"), "bundle"
    )
    split_models = copy.deepcopy(initial)
    first = training.run(
        split_models,
        data,
        config,
        tmp_path / "split",
        torch.device("cuda"),
        "bundle",
        max_updates=2,
    )
    del split_models, models
    gc.collect()
    final = training.run(
        initial,
        data,
        config,
        tmp_path / "split",
        torch.device("cuda"),
        "bundle",
        resume=first,
    )
    a, b = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in (full, final)
    ]
    for group in ("modules", "ema"):
        for name, state in a[group].items():
            for key, value in state.items():
                torch.testing.assert_close(value, b[group][name][key], atol=0, rtol=0)
    torch.testing.assert_close(
        a["optimizer"]["state"], b["optimizer"]["state"], atol=0, rtol=0
    )


def test_cuda_jagged_attention_backward():
    cuda()
    lengths = torch.tensor([2048, 2304], device="cuda")
    q, k, v = [
        torch.randn(
            4352, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    ]
    result = sparse.varlen_self_attention(q, k, v, lengths)
    result.float().square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v))


def test_structure_compile(models, example):
    flow = models.structure_flow.eval()
    x = torch.randn(1, 32, 2, 2, 2)
    t = torch.tensor([500.0])
    compiled = torch.compile(flow, fullgraph=True)
    torch.testing.assert_close(
        compiled(x, t, example.text[None]),
        flow(x, t, example.text[None]),
        atol=2e-5,
        rtol=1e-4,
    )


def test_released_bundle_generation(tmp_path, deterministic):
    cuda()
    path = os.environ.get("DIRTFLOW_TEST_BUNDLE")
    if not path:
        pytest.skip("set DIRTFLOW_TEST_BUNDLE to the released model directory")
    before = np.zeros((64, 64, 64), dtype=np.int64)
    edit = np.zeros_like(before, dtype=bool)
    edit[22:42, 22:42, 22:42] = True
    observed = np.zeros_like(edit)
    observed[14:50, 14:50, 14:50] = True
    sampler = Sampler(path)
    result = sampler.generate(
        before, edit, observed, "a small stone cottage with a wooden roof", seed=42
    )
    assert np.array_equal(result[~edit], before[~edit])
    assert np.isin(result, sampler.schema.output_to_global).all()
    assert (result[edit] != 0).any()
    np.save(tmp_path / "generation.npy", result)
    engine = os.environ.get("DIRTFLOW_ENGINE_SRC")
    if engine:
        del sampler
        gc.collect()
        torch.cuda.empty_cache()
        sys.path.insert(0, engine)
        from dirtflow_engine.sampling import Sampler as EngineSampler
        from dirtflow_engine.sampling import Settings as EngineSettings

        other = EngineSampler(path)
        expected = other.generate(
            before,
            edit,
            observed,
            "a small stone cottage with a wooden roof",
            [42],
            task="scratch_void",
            settings=EngineSettings(
                physics=False, batched_guidance=False, preview_every=0
            ),
        )[0]
        assert np.array_equal(result, expected), (
            f"{np.count_nonzero(result != expected)} state disagreements"
        )


@pytest.mark.parametrize("stage", ["structure", "materials"])
def test_released_weight_backward(stage):
    cuda()
    path = os.environ.get("DIRTFLOW_TEST_BUNDLE")
    if not path:
        pytest.skip("set DIRTFLOW_TEST_BUNDLE to the released model directory")
    models, schema, _ = checkpoint.load_bundle(pathlib.Path(path))
    models.train_stage(stage, "cuda")
    before = torch.zeros(64, 64, 64, dtype=torch.long, device="cuda")
    edit = torch.zeros_like(before, dtype=torch.bool)
    edit[28:36, 28:36, 28:36] = True
    target = before.clone()
    target[29:35, 29:32, 29:35] = schema.normalize_input("minecraft:stone")
    example = Example(
        before,
        target,
        edit,
        edit.clone(),
        torch.zeros(77, 768, device="cuda"),
        torch.zeros(77, 768, device="cuda"),
        task=0,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = TrainingLoss(models, stage)(example, 40_000, 4)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in models.parameters() if p.requires_grad], 1, error_if_nonfinite=True
    )
    assert torch.isfinite(loss) and norm > 0


def test_trellis_initialization_command(tmp_path):
    source = os.environ.get("DIRTFLOW_TRELLIS")
    bundle = os.environ.get("DIRTFLOW_TEST_BUNDLE")
    if not source or not bundle:
        pytest.skip("set DIRTFLOW_TRELLIS and DIRTFLOW_TEST_BUNDLE")
    import safetensors.torch

    source = pathlib.Path(source)
    dense = source / "ss_flow_txt_dit_XL_16l8_fp16.safetensors"
    sparse_flow = source / "slat_flow_txt_dit_XL_64l8p2_fp16.safetensors"
    output = tmp_path / "initialized"
    command = pathlib.Path(__file__).resolve().parents[1] / "initialize.py"
    subprocess.run(
        [
            sys.executable,
            str(command),
            "--bundle",
            bundle,
            "--structure-flow",
            str(dense),
            "--material-flow",
            str(sparse_flow),
            "--output",
            str(output),
        ],
        check=True,
    )
    models, _, _ = checkpoint.load_bundle(output)
    for flow, path in (
        (models.structure_flow, dense),
        (models.blockstate_flow, sparse_flow),
    ):
        with safetensors.safe_open(path, framework="pt") as tensors:
            expected = tensors.get_tensor("input_layer.weight").float()
        torch.testing.assert_close(
            flow.input_layer.weight[:, :8], expected, atol=0, rtol=0
        )
        assert not flow.input_layer.weight[:, 8:].any()
    del models
    gc.collect()
    engine = os.environ.get("DIRTFLOW_ENGINE_SRC")
    if engine:
        sys.path.insert(0, engine)
        from dirtflow_engine.model.load import load

        models, _, _, _ = load(str(output), device="cpu")
        assert models.structure_flow.input_layer.weight.shape[1] == 32
