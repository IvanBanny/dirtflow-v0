"""Editing pairs and connected air candidates from complete block grids."""

import heapq

import numpy as np
import scipy.ndimage

from .model.features import TASK_NAMES


def construct(
    source: np.ndarray,
    task: str,
    rng: np.random.Generator,
    *,
    old_source: np.ndarray | None = None,
    material: int = 2,
) -> dict[str, np.ndarray]:
    """Places a native-scale source crop in a 64-cube and constructs an edit.

    Sources use registry IDs and XYZ axes. Replacement requires a separately
    chosen source. Scratch fits the whole object to the edit; completion and
    repair use a crop at native block scale.
    """
    if task not in TASK_NAMES or source.ndim != 3 or source.dtype != np.int64:
        raise ValueError("expected a known task and an int64 XYZ source grid")
    if not all(source.shape) or np.any(source < 0):
        raise ValueError("source must be nonempty with nonnegative state IDs")
    source, _ = trim(source)
    scratch = task.startswith("scratch_")
    if scratch:
        source, size = fit(source, rng)
    else:
        for _ in range(1024):
            size = rng.integers(4, 59, size=3)
            if np.prod(size) <= 10_000:
                break
        else:
            raise RuntimeError("failed to draw an edit box")
    radius = int(rng.choice([r for r in (3, 5, 8) if size.max() + 2 * r <= 64]))
    start = (64 - size) // 2
    extent = size if scratch else size + 2 * radius
    crop_size = np.minimum(source.shape, extent)
    counts = np.count_nonzero(source, axis=(1, 2)).cumsum()
    chosen = int(rng.integers(counts[-1]))
    x = int(np.searchsorted(counts, chosen, side="right"))
    offset = chosen - (int(counts[x - 1]) if x else 0)
    yz = np.flatnonzero(source[x])[offset]
    center = np.array((x, yz // source.shape[2], yz % source.shape[2]))
    crop_start = np.clip(center - crop_size // 2, 0, np.array(source.shape) - crop_size)
    crop = source[tuple(slice(a, a + n) for a, n in zip(crop_start, crop_size))]
    target = np.zeros((64, 64, 64), dtype=np.int64)
    placed = start - (0 if scratch else radius) + (extent - crop_size) // 2
    if task == "scratch_ground":
        placed[1] = start[1]
    target[tuple(slice(a, a + n) for a, n in zip(placed, crop_size))] = crop
    edit = np.zeros_like(target, dtype=bool)
    edit[tuple(slice(a, a + n) for a, n in zip(start, size))] = True
    observed = np.zeros_like(edit)
    observed[tuple(slice(a - radius, a + n + radius) for a, n in zip(start, size))] = (
        True
    )
    if task in {"scratch_ground", "material_only"} and material <= 1:
        raise ValueError("ground and material-only tasks require a legal non-air state")
    if task == "scratch_ground":
        target[:, start[1] - 1, :] = material
    before = np.where(observed, target, 0)
    if scratch or task == "completion":
        before[edit] = 0
    elif task == "repair":
        before[edit & (rng.random(target.shape) < 0.5)] = 0
    elif task == "material_only":
        before[edit & (target != 0)] = material
    else:
        if old_source is None or old_source.ndim != 3:
            raise ValueError("replacement requires another source grid")
        before[edit] = 0
        old_size = np.minimum(old_source.shape, size)
        old_start = np.array(
            [rng.integers(n - s + 1) for n, s in zip(old_source.shape, old_size)]
        )
        old = old_source[tuple(slice(a, a + n) for a, n in zip(old_start, old_size))]
        before[tuple(slice(a, a + n) for a, n in zip(start, old_size))] = old
    return dict(before=before, target=target, edit=edit, observed=observed)


def fit(
    source: np.ndarray, rng: np.random.Generator, max_volume: int = 10_000
) -> tuple[np.ndarray, np.ndarray]:
    """Fits complete occupancy in a box; occupied source bins remain occupied."""
    source, _ = trim(source)
    shape = np.array(source.shape)
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        candidate = np.maximum(1, np.floor(shape * middle).astype(int))
        box = np.maximum(4, candidate)
        if box.max() <= 58 and np.prod(box) <= max_volume:
            low = middle
        else:
            high = middle
    box = np.maximum(4, shape)
    if box.max() > 58 or np.prod(box) > max_volume:
        shape = np.maximum(1, np.floor(shape * low).astype(int))
        source = _resize(source, shape)
        box = np.maximum(4, shape)
    for axis in rng.permutation(3):
        candidate = box.copy()
        candidate[axis] += int(rng.integers(5))
        if candidate.max() <= 58 and np.prod(candidate) <= max_volume:
            box = candidate
    return source, box


def global_structure(
    source: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Whole-object structure training on a 58-cube interior."""
    source, _ = fit(source, rng, max_volume=58**3)
    target = np.zeros((64, 64, 64), dtype=np.int64)
    start = (64 - np.array(source.shape)) // 2
    target[tuple(slice(a, a + n) for a, n in zip(start, source.shape))] = source
    edit = np.zeros_like(target, dtype=bool)
    edit[3:61, 3:61, 3:61] = True
    return dict(
        before=np.zeros_like(target),
        target=target,
        edit=edit,
        observed=np.ones_like(edit),
    )


def boundary_additions(
    occupied: np.ndarray, valid: np.ndarray, seed: int
) -> np.ndarray:
    """Adds 5-25% face-connected air within distance two, capped at 1,024 cells."""
    if occupied.shape != valid.shape or occupied.ndim != 3:
        raise ValueError("boundary masks must have matching XYZ shapes")
    seeds = np.random.SeedSequence(seed).spawn(2)
    fraction = np.random.default_rng(seeds[0]).uniform(0.05, 0.25)
    budget = min(1024, int(occupied.sum() * fraction))
    added = np.zeros_like(occupied, dtype=bool)
    if not budget:
        return added
    cross = scipy.ndimage.generate_binary_structure(3, 1)
    distance = np.zeros(occupied.shape, dtype=np.int8)
    reached = occupied.copy()
    for layer in range(1, 3):
        frontier = (
            scipy.ndimage.binary_dilation(reached, structure=cross) & valid & ~reached
        )
        distance[frontier] = layer
        reached |= frontier
    field = scipy.ndimage.gaussian_filter(
        np.random.default_rng(seeds[1]).standard_normal(occupied.shape), sigma=1.5
    )
    priority = field / max(float(field.std()), 1e-8) - distance
    queued = distance == 1
    heap = [(-float(priority[tuple(p)]), tuple(p)) for p in np.argwhere(queued)]
    heapq.heapify(heap)
    for _ in range(budget):
        if not heap:
            break
        _, position = heapq.heappop(heap)
        added[position] = True
        for axis in range(3):
            for direction in (-1, 1):
                neighbor = list(position)
                neighbor[axis] += direction
                if not 0 <= neighbor[axis] < occupied.shape[axis]:
                    continue
                index = tuple(neighbor)
                if distance[index] and not queued[index]:
                    queued[index] = True
                    heapq.heappush(heap, (-float(priority[index]), index))
    return added


def trim(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Removes only exterior all-air planes, with bounded temporary storage."""
    bounds = []
    for axis in range(3):
        present = np.any(source != 0, axis=tuple(i for i in range(3) if i != axis))
        indices = np.flatnonzero(present)
        if not len(indices):
            raise ValueError("whole-object scratch requires nonempty source occupancy")
        bounds.append((int(indices[0]), int(indices[-1]) + 1))
    return source[tuple(slice(a, b) for a, b in bounds)], np.array(
        [a for a, _ in bounds]
    )


def _resize(source: np.ndarray, shape: np.ndarray) -> np.ndarray:
    source_shape = np.array(source.shape)
    centers = [
        np.minimum(((np.arange(n) + 0.5) * old / n).astype(int), old - 1)
        for old, n in zip(source_shape, shape, strict=True)
    ]
    result = source[np.ix_(*centers)].copy()
    flat = result.reshape(-1)
    for x in range(source.shape[0]):
        y, z = np.nonzero(source[x])
        mapped_x = min(x * int(shape[0]) // source.shape[0], int(shape[0]) - 1)
        mapped_y = y * shape[1] // source.shape[1]
        mapped_z = z * shape[2] // source.shape[2]
        destinations = (mapped_x * shape[1] + mapped_y) * shape[2] + mapped_z
        unique, first = np.unique(destinations, return_index=True)
        missing = flat[unique] == 0
        flat[unique[missing]] = source[x, y[first[missing]], z[first[missing]]]
    return result
