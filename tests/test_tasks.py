import dataclasses

import numpy as np
import pytest
import torch
from scipy import ndimage

from dirtflow import tasks
from dirtflow.model.features import TASK_NAMES, support


@pytest.mark.parametrize("task", TASK_NAMES)
def test_tasks_preserve_protected_world(task):
    source = np.zeros((12, 10, 8), dtype=np.int64)
    source[2:10, 1:8, 2:6] = 3
    source[4:8, 3:8, 3:5] = 0
    result = tasks.construct(
        source,
        task,
        np.random.default_rng(5),
        old_source=np.full((8, 8, 8), 2, dtype=np.int64),
    )
    before, target, edit, seen = [
        result[k] for k in ("before", "target", "edit", "observed")
    ]
    assert 0 < edit.sum() <= 10_000
    assert not (edit & ~seen).any()
    assert np.array_equal(before[seen & ~edit], target[seen & ~edit])
    assert not before[~seen].any()
    if task == "material_only":
        assert np.array_equal(before[edit] != 0, target[edit] != 0)
    elif task.startswith("scratch") or task == "completion":
        assert not before[edit].any()


def test_boundary_additions_are_connected_and_bounded():
    occupied = np.zeros((24, 24, 24), dtype=bool)
    occupied[7:17, 7:17, 7:17] = True
    valid = np.ones_like(occupied)
    result = tasks.boundary_additions(occupied, valid, 19)
    assert 50 <= result.sum() <= 250
    assert not (result & occupied).any()
    cross = ndimage.generate_binary_structure(3, 1)
    assert not (result & ~ndimage.binary_dilation(occupied, cross, iterations=2)).any()
    assert ndimage.label(occupied | result, cross)[1] == 1
    assert np.array_equal(result, tasks.boundary_additions(occupied, valid, 19))


def test_support_repair_and_material_only():
    edit = torch.zeros(8, 8, 8, dtype=torch.bool)
    edit[2:6, 2:6, 2:6] = True
    before = torch.zeros_like(edit, dtype=torch.long)
    before[3, 3, 3] = 2
    logits = torch.full(edit.shape, -1.0)
    logits[4, 4, 4] = 0
    seen = edit.clone()
    repair = support(logits, edit, seen, before, "repair")
    assert repair.sum() == 2
    material = support(logits, edit, seen, before, "material_only")
    assert material.sum() == 1 and material[3, 3, 3]


def test_context_limit_distance_and_ties():
    edit = torch.zeros(24, 24, 24, dtype=torch.bool)
    edit[10:14, 10:14, 10:14] = True
    seen = torch.ones_like(edit)
    before = torch.zeros_like(edit, dtype=torch.long)
    actual = support(torch.full(edit.shape, -1.0), edit, seen, before, "completion")
    coords = torch.nonzero(~edit)
    distance = torch.maximum(
        (10 - coords).clamp_min(0), (coords - 13).clamp_min(0)
    ).amax(1)
    expected = coords[torch.argsort(distance, stable=True)[:8192]]
    mask = torch.zeros_like(edit)
    mask[tuple(expected.T)] = True
    assert torch.equal(actual, mask)


@pytest.mark.parametrize("field", ["target", "observed", "text"])
def test_example_boundary_validation(example, field):
    value = getattr(example, field).clone()
    if field == "target":
        value[0, 0, 0] = 99
    elif field == "observed":
        value[:] = False
    else:
        value[0, 0] = torch.nan
    with pytest.raises(ValueError):
        dataclasses.replace(example, **{field: value}).validate(4)
