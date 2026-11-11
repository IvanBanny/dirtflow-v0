"""Portable model bundles and atomic training checkpoints."""

import copy
import json
import os
import pathlib
import shutil

import torch

from . import bundle
from .model.network import MATERIALS, STRUCTURE, Models
from .schema import Schema


def load_bundle(path: str | pathlib.Path) -> tuple[Models, Schema, dict]:
    """Verifies a bundle and loads every model tensor on CPU in FP32."""
    root = pathlib.Path(path).resolve()
    manifest = bundle.validate_bundle(root)
    schema = Schema.load(root, manifest["assets"])
    if manifest["model"]["embedding"]["vocab_size"] != len(schema.states):
        raise ValueError("embedding size disagrees with input registry")
    with torch.device("meta"):
        models = Models(manifest["model"], torch.from_numpy(schema.output_to_global))
    modules = dict(models.named_children())
    for relative in sorted(set(manifest["modules"].values())):
        payload = torch.load(
            root / relative, map_location="cpu", weights_only=True, mmap=True
        )
        expected = {
            name for name, file in manifest["modules"].items() if file == relative
        }
        if set(payload["modules"]) != expected:
            raise ValueError("payload module ownership differs from bundle")
        for name, state in payload["modules"].items():
            modules[name].load_state_dict(state, strict=True, assign=True)
    # These analytic frequencies are nonpersistent and must survive meta loading.
    from .model.layers import AbsolutePositionEmbedder

    for module in models.modules():
        if isinstance(module, AbsolutePositionEmbedder):
            count = module.channels // module.in_channels // 2
            module.freqs = 1.0 / 10000 ** (torch.arange(count).float() / count)
    models.float().eval().requires_grad_(False)
    if not torch.equal(
        models.categorical_head.output_to_global,
        torch.from_numpy(schema.output_to_global),
    ):
        raise ValueError("checkpoint output classes differ from schema")
    return models, schema, manifest


def save_atomic(path: pathlib.Path, payload: dict) -> None:
    """Replaces one checkpoint after its complete payload has reached disk."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_bundle(
    source: pathlib.Path,
    destination: pathlib.Path,
    structure: pathlib.Path | None = None,
    materials: pathlib.Path | None = None,
) -> None:
    """Exports structure EMA and material LIVE weights for the public engine.

    Omitted stages retain the source bundle's selected weights. Training
    checkpoints must belong to this bundle's model and asset configuration.
    The destination must not exist.
    """
    manifest = bundle.validate_bundle(source)
    if destination.exists():
        raise FileExistsError(destination)
    updates = {}
    for path, stage in ((structure, "structure"), (materials, "materials")):
        if path is None:
            continue
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if saved.get("format") != "dirtflow-training" or saved["stage"] != stage:
            raise ValueError(f"expected a {stage} training checkpoint: {path}")
        if saved["model_config"] != manifest["model"] or saved[
            "bundle_sha256"
        ] != bundle.sha256(source / "bundle.json"):
            raise ValueError("checkpoint model or source bundle differs")
        selected = saved["ema"] if stage == "structure" else saved["modules"]
        expected = STRUCTURE if stage == "structure" else MATERIALS
        if set(selected) != set(expected):
            raise ValueError(f"incomplete or unexpected {stage} weights")
        updates.update(selected)
    write_bundle(source, destination, updates, "research-export")


def write_bundle(
    source: pathlib.Path,
    destination: pathlib.Path,
    updates: dict,
    snapshot: str,
    lineage: dict | None = None,
) -> None:
    """Writes verified module replacements alongside unchanged codec and assets."""
    manifest = bundle.validate_bundle(source)
    if manifest["weight_set"] != "stage1-ema+stage2-live":
        raise ValueError(
            "export requires a source bundle with structure EMA and material LIVE"
        )
    if destination.exists():
        raise FileExistsError(destination)
    if not set(updates) <= set(STRUCTURE + MATERIALS):
        raise ValueError("only flow stages can replace source bundle modules")
    payloads = {
        relative: torch.load(
            source / relative, map_location="cpu", weights_only=True, mmap=True
        )
        for relative in sorted(set(manifest["modules"].values()))
    }
    for name, state in updates.items():
        reference = payloads[manifest["modules"][name]]["modules"][name]
        if state.keys() != reference.keys():
            raise ValueError(f"replacement keys differ for {name}")
        for key, value in state.items():
            original = reference[key]
            if value.shape != original.shape or value.dtype != original.dtype:
                raise ValueError(f"replacement shape or dtype differs for {name}.{key}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"nonfinite replacement tensor: {name}.{key}")
            if not value.is_floating_point() and not torch.equal(value, original):
                raise ValueError(f"replacement changed a fixed mapping: {name}.{key}")
    destination.mkdir(parents=True)
    result = copy.deepcopy(manifest)
    for relative in manifest["files"]:
        if relative in manifest["modules"].values():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    for relative, payload in payloads.items():
        for name in payload["modules"]:
            if name in updates:
                payload["modules"][name] = updates[name]
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
        result["files"][relative] = bundle.sha256(target)
    result["weight_set"] = "stage1-ema+stage2-live"
    result["module_weight_sets"] = dict.fromkeys(bundle.STAGE1, "ema") | dict.fromkeys(
        bundle.STAGE2, "live"
    )
    if updates:
        result["source_snapshot"] = snapshot
        result["lineage"] = {
            "parent_bundle_sha256": bundle.sha256(source / "bundle.json"),
            **(lineage or {}),
        }
    (destination / "bundle.json").write_text(json.dumps(result, indent=2) + "\n")
    bundle.validate_bundle(destination)


def transfer_trellis(model: torch.nn.Module, source: dict[str, torch.Tensor]) -> None:
    """Loads TRELLIS weights, expanding only the eight-column input projection.

    Sparse kernels use spconv's [out,kx,ky,kz,in] layout in the source and
    [kx*ky*kz,in,out] here. Every parameter must be present exactly once.
    """
    target = model.state_dict()
    converted = {}
    for name, tensor in source.items():
        name = name.replace(".conv1.conv.", ".conv1.").replace(
            ".conv2.conv.", ".conv2."
        )
        if name not in target or name in converted:
            raise ValueError(f"unexpected or duplicate pretrained tensor: {name}")
        expected = target[name]
        if tensor.ndim == 5 and expected.ndim == 3:
            tensor = tensor.permute(1, 2, 3, 4, 0).reshape(
                -1, tensor.shape[-1], tensor.shape[0]
            )
        if name == "input_layer.weight" and tensor.shape != expected.shape:
            if tensor.ndim != 2 or tensor.shape != (expected.shape[0], 8):
                raise ValueError("pretrained input projection must have eight columns")
            expanded = torch.zeros_like(expected)
            expanded[:, :8] = tensor
            tensor = expanded
        if tensor.shape != expected.shape or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid pretrained tensor: {name}")
        converted[name] = tensor
    model.load_state_dict(converted, strict=True)
