import copy
import dataclasses
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from dirtflow import bundle, training
from dirtflow.data import Dataset
from dirtflow.objectives import (
    TrainingLoss,
    clean_estimate,
    interpolate,
    material_loss,
    mean_data_weight,
    structure_loss,
)


def write_dataset(path, example):
    archive = path / "example.npz"
    np.savez_compressed(
        archive,
        **{
            name: getattr(example, name).numpy()
            for name in ("before", "target", "edit", "observed", "text", "null_text")
        },
    )
    index = path / "index.jsonl"
    index.write_text(
        json.dumps(
            dict(
                path=archive.name,
                sha256=bundle.sha256(archive),
                task="completion",
                prompt="a small stone wall",
            )
        )
        + "\n"
    )
    return Dataset(index, 4)


def cpu_config(stage="structure"):
    return training.Config(
        stage=stage,
        presentations=12,
        batch_size=2,
        warmup=2,
        hold_fraction=0.5,
        optimizer="adamw",
        precision="fp32",
        replay_initial=0,
        replay_final=0,
        ema_stride=3,
        log_every=1,
    )


def test_linear_path_inverse():
    clean, noise = torch.randn(5, 8), torch.randn(5, 8)
    for t in (0.0, 0.2, 0.75, 1.0):
        x, velocity = interpolate(clean, noise, t)
        torch.testing.assert_close(
            clean_estimate(x, velocity, t), clean, atol=3e-7, rtol=1e-6
        )


def test_clean_estimate_keeps_velocity_product_in_fp32():
    x = torch.randn(3, 8, dtype=torch.float32)
    velocity = torch.randn(3, 8, dtype=torch.bfloat16)
    t = 0.3333333134651184
    time = torch.full((3, 1), t, dtype=torch.float32)
    expected = (1 - 1e-5) * x - (1e-5 + (1 - 1e-5) * time) * velocity
    torch.testing.assert_close(clean_estimate(x, velocity, t), expected, atol=0, rtol=0)


def test_structure_group_reduction():
    predicted = torch.tensor([[[[[1.0, 2.0, 3.0, 4.0]]]]])
    edit = torch.tensor([[[[True, False, False, False]]]])
    expected = 0.7 + 0.3 * (4 + 9 + 16) / 3
    assert float(
        structure_loss(predicted, torch.zeros_like(predicted), edit)
    ) == pytest.approx(expected)
    for empty in (torch.zeros_like(edit), torch.ones_like(edit)):
        assert float(
            structure_loss(predicted, torch.zeros_like(predicted), empty)
        ) == pytest.approx(7.5)


def test_material_loss_matches_full_vocabulary_reference(models):
    torch.manual_seed(7)
    n = 7
    v, target_v = torch.randn(n, 8, requires_grad=True), torch.randn(n, 8)
    z, io = torch.randn(n, 8, requires_grad=True), torch.randn(n, 8, requires_grad=True)
    targets = torch.tensor([0, 2, 3, 1, 2, 3, 0])
    editable = torch.tensor([True, True, True, True, False, False, False])
    palette = torch.tensor([[1.0, 1.0, 0.0]])
    table = models.embedding.table()
    time = torch.tensor([0.8])
    total, metrics = material_loss(
        models.categorical_head,
        v,
        target_v,
        z,
        io,
        targets,
        editable,
        palette,
        table,
        time,
        40_000,
        torch.Generator().manual_seed(1),
        chunk_size=2,
    )
    classes = models.categorical_head.target_classes(targets)
    weights = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.3, 0.3, 0.3])
    mse = ((v - target_v).square().mean(1) * weights).sum() / weights.sum()
    ce = (
        F.cross_entropy(
            models.categorical_head(z, io, palette),
            classes.clamp_min(0),
            reduction="none",
        )
        * weights
    ).sum() / weights.sum()
    ce *= 0.2 / mean_data_weight()
    assert float(metrics["mse"]) == pytest.approx(float(mse.detach()))
    assert float(metrics["ce"]) == pytest.approx(float(ce.detach()), rel=1e-6)
    total.backward()
    for value in (v, z, io, models.categorical_head.keys, models.embedding.weight):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.norm() > 0


@pytest.mark.parametrize("stage", ["structure", "materials"])
def test_training_loss_updates_only_selected_stage(models, example, stage):
    models.train_stage(stage, "cpu")
    loss, metrics = TrainingLoss(models, stage)(example, 40_000, 42)
    assert torch.isfinite(loss) and metrics
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in models.parameters()
        if p.requires_grad
    )
    assert all(p.grad is None for p in models.parameters() if not p.requires_grad)


def test_empty_material_example_has_zero_loss(models, example):
    empty = dataclasses.replace(
        example,
        before=torch.zeros_like(example.before),
        target=torch.zeros_like(example.target),
        observed=example.edit.clone(),
    )
    models.train_stage("materials", "cpu")
    loss, _ = TrainingLoss(models, "materials")(empty, 40_000, 2)
    assert loss == 0
    loss.backward()
    assert all(
        p.grad is not None and p.grad.count_nonzero() == 0
        for p in models.parameters()
        if p.requires_grad
    )


@pytest.mark.parametrize("stage", ["structure", "materials"])
def test_resume_matches_uninterrupted(models, example, tmp_path, stage):
    data = write_dataset(tmp_path, example)
    config = cpu_config(stage)
    uninterrupted, interrupted, resumed = [copy.deepcopy(models) for _ in range(3)]
    full = training.run(
        uninterrupted, data, config, tmp_path / "full", torch.device("cpu"), "source"
    )
    first = training.run(
        interrupted,
        data,
        config,
        tmp_path / "resume",
        torch.device("cpu"),
        "source",
        max_updates=2,
    )
    final = training.run(
        resumed,
        data,
        config,
        tmp_path / "resume",
        torch.device("cpu"),
        "source",
        resume=first,
    )
    a, b = [torch.load(p, weights_only=True) for p in (full, final)]
    assert a["presentations"] == b["presentations"] == 12
    assert a["ema_last"] == b["ema_last"]
    for group in ("modules", "ema"):
        for name, state in a[group].items():
            for key, value in state.items():
                torch.testing.assert_close(value, b[group][name][key], atol=0, rtol=0)
    for key, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, b["optimizer"]["state"][key][name], atol=0, rtol=0
            )
    changed = dataclasses.replace(config, seed=5)
    with pytest.raises(ValueError, match="config"):
        training.run(
            resumed,
            data,
            changed,
            tmp_path / "resume",
            torch.device("cpu"),
            "source",
            resume=final,
        )


def test_lr_ema_and_decay_policy(models):
    config = cpu_config()
    assert training.lr_factor(0, config) == 0
    assert training.lr_factor(2, config) == 1
    assert training.lr_factor(6, config) == 1
    assert training.lr_factor(12, config) == pytest.approx(1 / 30)
    models.train_stage("structure", "cpu")
    groups = training.parameter_groups(models, config)
    entries = {name: group for group in groups for name in group["param_names"]}
    assert entries["context_encoder.embedding.weight"]["weight_decay"] == 0
    assert entries["structure_extra.task_emb.weight"]["weight_decay"] == 0
    assert entries["structure_flow.input_layer.weight"]["base_lr"] == config.backbone_lr


def test_dataset_rejects_changed_payload(tmp_path, example):
    data = write_dataset(tmp_path, example)
    (tmp_path / "example.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        data.sample(1)
