import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from dirtflow import bundle, checkpoint
from dirtflow.schema import Schema


@pytest.fixture
def model_dir(tmp_path, models):
    root = tmp_path / "bundle"
    root.mkdir()
    states = [
        "minecraft:air",
        "dirtflow:unknown",
        "minecraft:oak_planks",
        "minecraft:stone",
    ]
    (root / "registry.json").write_text(
        json.dumps(dict(format="dirtflow-blockstate-vocab", version=1, states=states))
    )
    entries = [
        dict(state=name, is_air=name.endswith(":air"), default=True)
        for name in (states[0], states[2], states[3])
    ]
    (root / "schema.json").write_text(
        json.dumps(
            dict(
                format="dirtflow-game-state-schema",
                version=1,
                minecraft_version="1.21.11",
                states=entries,
            )
        )
    )
    np.savez(root / "mapping.npz", output_to_global=np.array([0, 2, 3], dtype=np.int64))
    (root / "clip").mkdir()
    for name in ("config.json", "vocab.json", "merges.txt", "model.safetensors"):
        (root / "clip" / name).write_text("{}")
    (root / "clip-manifest.json").write_text(
        json.dumps(
            dict(files={p.name: bundle.sha256(p) for p in (root / "clip").iterdir()})
        )
    )
    torch.save(dict(modules=models.module_states()), root / "weights.pt")
    manifest = dict(
        format=bundle.FORMAT,
        version=1,
        weight_set="stage1-ema+stage2-live",
        module_weight_sets=dict.fromkeys(bundle.STAGE1, "ema")
        | dict.fromkeys(bundle.STAGE2, "live"),
        model=models.config,
        modules=dict.fromkeys(bundle.STAGE1 + bundle.STAGE2, "weights.pt"),
        assets=dict(
            registry="registry.json",
            schema="schema.json",
            mapping="mapping.npz",
            text_encoder="clip",
            clip_manifest="clip-manifest.json",
        ),
        inference_policy=dict(
            editable_confidence="zero_normalized_editable",
            parameter_storage="float32",
            position_frequencies="native_float32",
        ),
    )
    manifest["files"] = {
        p.relative_to(root).as_posix(): bundle.sha256(p)
        for p in root.rglob("*")
        if p.is_file()
    }
    (root / "bundle.json").write_text(json.dumps(manifest))
    return root


def test_bundle_load_includes_encoder_and_frequencies(model_dir, models):
    loaded, schema, manifest = checkpoint.load_bundle(model_dir)
    assert len(schema.states) == 4
    for name, value in models.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)
    assert loaded.blockstate_flow.pos_embedder.freqs.device.type == "cpu"
    assert not any(p.requires_grad for p in loaded.parameters())
    assert (
        manifest["inference_policy"]["editable_confidence"]
        == "zero_normalized_editable"
    )


def test_schema_defaults_air_and_unknown(model_dir):
    manifest = bundle.validate_bundle(model_dir)
    schema = Schema.load(model_dir, manifest["assets"])
    assert schema.normalize_input("stone") == 3
    assert schema.normalize_input("minecraft:air") == 0
    assert schema.normalize_input("minecraft:stone[bad=property]") == 1
    assert schema.normalize_input("unregistered") == 1


def test_export_roundtrip_and_stage_ownership(model_dir, models, tmp_path):
    state = {
        name: {
            key: value.clone()
            for key, value in getattr(models, name).state_dict().items()
        }
        for name in ("structure_flow", "structure_extra", "context_encoder")
    }
    state["structure_flow"]["out_layer.bias"].add_(2)
    path = tmp_path / "trained.pt"
    torch.save(
        dict(
            format="dirtflow-training",
            stage="structure",
            model_config=models.config,
            bundle_sha256=bundle.sha256(model_dir / "bundle.json"),
            ema=state,
        ),
        path,
    )
    output = tmp_path / "export"
    checkpoint.export_bundle(model_dir, output, structure=path)
    loaded, _, _ = checkpoint.load_bundle(output)
    torch.testing.assert_close(
        loaded.structure_flow.out_layer.bias, models.structure_flow.out_layer.bias + 2
    )
    torch.testing.assert_close(
        loaded.ss_encoder.input_layer.weight, models.ss_encoder.input_layer.weight
    )
    torch.testing.assert_close(loaded.embedding.weight, models.embedding.weight)
    with pytest.raises(FileExistsError):
        checkpoint.export_bundle(model_dir, output)


def test_bundle_rejects_changed_bytes(model_dir):
    (model_dir / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        checkpoint.load_bundle(model_dir)


@pytest.mark.parametrize(
    "relative", ["../secret", "/tmp/secret", "C:/secret", "..\\secret", ""]
)
def test_bundle_rejects_escaping_paths(tmp_path, relative):
    with pytest.raises(ValueError):
        bundle.local_path(tmp_path, relative)


def test_atomic_checkpoint_failure_preserves_previous(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    checkpoint.save_atomic(path, dict(value=torch.tensor([1])))

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        checkpoint.save_atomic(path, dict(value=2))
    assert torch.load(path, weights_only=True)["value"].item() == 1
    assert not path.with_suffix(".pt.tmp").exists()


@pytest.mark.parametrize("problem", ["missing", "shape", "nonfinite", "mapping"])
def test_export_rejects_incompatible_updates(model_dir, models, tmp_path, problem):
    state = {k: v.clone() for k, v in models.categorical_head.state_dict().items()}
    if problem == "missing":
        state.pop("class_bias")
    elif problem == "shape":
        state["class_bias"] = torch.zeros(100)
    elif problem == "nonfinite":
        state["class_bias"][0] = torch.nan
    else:
        state["output_to_global"][1] = 1
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        checkpoint.write_bundle(model_dir, output, {"categorical_head": state}, "test")
    assert not output.exists()


def test_train_resume_and_export_commands(model_dir, example, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    archive = data / "example.npz"
    np.savez(
        archive,
        **{
            name: getattr(example, name).numpy()
            for name in ("before", "target", "edit", "observed", "text", "null_text")
        },
    )
    index = data / "index.jsonl"
    index.write_text(
        json.dumps(
            dict(
                path=archive.name,
                sha256=bundle.sha256(archive),
                task="completion",
                prompt="a wall",
            )
        )
    )
    manifest = bundle.validate_bundle(model_dir)
    assets = {
        key: manifest["files"][manifest["assets"][key]]
        for key in ("registry", "schema", "mapping", "clip_manifest")
    }
    (data / "metadata.json").write_text(json.dumps(dict(assets=assets)))
    config = tmp_path / "config.py"
    config.write_text(
        "from dirtflow.training import Config\n"
        "def get_config():\n"
        "    return Config(presentations=4, batch_size=2, warmup=0, "
        'optimizer="adamw", precision="fp32", replay_initial=0, replay_final=0)\n'
    )
    root = pathlib.Path(__file__).resolve().parents[1]
    output = tmp_path / "run"
    command = [
        sys.executable,
        str(root / "train.py"),
        "--config",
        str(config),
        "--bundle",
        str(model_dir),
        "--data",
        str(index),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]
    subprocess.run(command + ["--max-updates", "1"], check=True, cwd=tmp_path)
    saved = output / "checkpoint.pt"
    assert torch.load(saved, weights_only=True)["presentations"] == 2
    subprocess.run(command + ["--resume", str(saved)], check=True, cwd=tmp_path)
    exported = tmp_path / "exported"
    subprocess.run(
        [
            sys.executable,
            str(root / "export.py"),
            "--bundle",
            str(model_dir),
            "--structure",
            str(saved),
            "--output",
            str(exported),
        ],
        check=True,
        cwd=tmp_path,
    )
    _, _, result = checkpoint.load_bundle(exported)
    assert result["source_snapshot"] == "research-export"
