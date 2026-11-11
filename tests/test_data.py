import json

import numpy as np
import pytest
import torch

from dirtflow import bundle, tasks
from dirtflow.data import Dataset


@pytest.fixture
def sources(tmp_path):
    null = np.zeros((77, 768), dtype=np.float32)
    np.save(tmp_path / "null_text.npy", null)
    metadata = dict(
        assets={"registry": "registry-identity"},
        null_text_sha256=bundle.sha256(tmp_path / "null_text.npy"),
    )
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    records = []
    for i in range(4):
        source = np.full((5 + i, 6, 4), 2 + i % 2, dtype=np.int64)
        path = tmp_path / f"{i}.npz"
        np.savez_compressed(
            path, source=source, text=np.full((1, 77, 768), i, np.float32)
        )
        records.append(
            dict(
                kind="source",
                path=path.name,
                sha256=bundle.sha256(path),
                captions=["a stone wall"],
                domain=i // 2,
                material=3,
            )
        )
    index = tmp_path / "index.jsonl"
    index.write_text("\n".join(json.dumps(row) for row in records))
    return index, metadata


@pytest.mark.parametrize("stage", ["structure", "materials"])
def test_online_tasks_are_repeatable_and_vary(sources, stage):
    index, metadata = sources
    data = Dataset(index, 4, metadata["assets"])
    tasks_seen, masks = set(), set()
    for seed in range(12):
        a, b = [data.sample(seed, stage=stage) for _ in range(2)]
        for name in ("before", "target", "edit", "observed", "text", "null_text"):
            assert torch.equal(getattr(a, name), getattr(b, name))
        a.validate(4)
        tasks_seen.add(a.task)
        masks.add(a.edit.numpy().tobytes())
    assert len(tasks_seen) >= 3 and len(masks) >= 3
    replay = data.sample(0, replay_share=1, stage="structure")
    assert replay.domain == 1 and replay.task == 0
    assert replay.edit.sum() == 58**3


def test_prepared_identity_binds_assets_null_and_metadata(sources):
    index, metadata = sources
    original = Dataset(index, 4, metadata["assets"]).sha256
    with pytest.raises(ValueError, match="registry or text encoder"):
        Dataset(index, 4, {"registry": "another-registry"})
    metadata["revision"] = 2
    (index.parent / "metadata.json").write_text(json.dumps(metadata))
    assert Dataset(index, 4).sha256 != original
    np.save(index.parent / "null_text.npy", np.ones((77, 768), np.float32))
    with pytest.raises(ValueError, match="null-text"):
        Dataset(index, 4)


def test_large_thin_object_survives_fitting():
    source = np.zeros((200, 5, 100), dtype=np.int64)
    source[:, 2, 49] = 3
    source[100, 2, :] = 2
    result, box = tasks.fit(source, np.random.default_rng(0))
    assert box.max() <= 58 and np.prod(box) <= 10_000
    assert (result != 0).any(axis=(1, 2)).all()
    assert (result != 0).any(axis=(0, 1)).all()
