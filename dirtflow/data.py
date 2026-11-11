"""Prepared editing examples and deterministic weighted sampling."""

import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import torch

from . import bundle, tasks
from .model.features import TASK_NAMES


@dataclasses.dataclass
class Example:
    """One native XYZ canvas, its target, and conditioning metadata.

    State arrays are int64 [R,R,R]; masks are boolean with the same shape.
    `text` and `null_text` contain frozen CLIP features [77,768]. Smaller
    canvases and text widths can be used with correspondingly smaller models.
    """

    before: torch.Tensor
    target: torch.Tensor
    edit: torch.Tensor
    observed: torch.Tensor
    text: torch.Tensor
    null_text: torch.Tensor
    task: int
    style: int = 5
    score: float = -1.0
    fragment: bool = False
    domain: int = 0
    active: torch.Tensor | None = None

    def validate(self, vocab_size: int) -> None:
        shape = self.before.shape
        if len(shape) != 3 or len(set(shape)) != 1 or shape[0] % 4:
            raise ValueError("example must have a cubic canvas divisible by four")
        for value in (self.before, self.target):
            if value.shape != shape or value.dtype != torch.long:
                raise ValueError("state grids must be int64 with matching shapes")
            if (value < 0).any() or (value >= vocab_size).any():
                raise ValueError("example state lies outside the input registry")
        for mask in (self.edit, self.observed, self.active):
            if mask is not None and (mask.shape != shape or mask.dtype != torch.bool):
                raise ValueError("example masks must be boolean with matching shapes")
        if not self.edit.any() or (self.edit & ~self.observed).any():
            raise ValueError("edit must be nonempty and fully observed")
        known = self.observed & ~self.edit
        if (self.target[known] != self.before[known]).any():
            raise ValueError("target changes protected observations")
        if self.before[~self.observed].any():
            raise ValueError("unobserved storage must be zero")
        if (
            not 0 <= self.task < len(TASK_NAMES)
            or not 0 <= self.style <= 5
            or not 0 <= self.domain < 3
        ):
            raise ValueError("invalid task, caption style or data domain")
        if not np.isfinite(self.score) or not (
            self.score == -1 or 0 <= self.score <= 10
        ):
            raise ValueError("caption score must be -1 or in 0..10")
        if self.text.ndim != 2 or self.text.shape != self.null_text.shape:
            raise ValueError("text and null text must be matching token matrices")
        if (
            not self.text.is_floating_point()
            or not self.null_text.is_floating_point()
            or not torch.isfinite(self.text).all()
            or not torch.isfinite(self.null_text).all()
        ):
            raise ValueError("text features must be finite floating-point values")
        if (
            self.task == 5
            and ((self.before != 0) != (self.target != 0))[self.edit].any()
        ):
            raise ValueError("material-only examples must preserve occupancy")

    def to(self, device: torch.device | str) -> "Example":
        return dataclasses.replace(
            self,
            **{
                field.name: getattr(self, field.name).to(device)
                for field in dataclasses.fields(self)
                if isinstance(getattr(self, field.name), torch.Tensor)
            },
        )


def presentation_seed(seed: int, index: int) -> int:
    """Gives each presentation its own stream, independent of rank and restart."""
    raw = hashlib.blake2b(f"{seed}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(raw, "little") % (2**63 - 1)


class Dataset:
    """Weighted JSONL index of NPZ examples; paths are relative to the index.

    Source rows construct a fresh task on every presentation. Explicit example
    rows retain supplied before/target pairs. Asset identity is checked on first
    access, and the index digest is recorded on resume.
    """

    def __init__(
        self, path: pathlib.Path, vocab_size: int, asset_hashes: dict | None = None
    ) -> None:
        self.root = path.resolve().parent
        raw = path.read_bytes()
        identity = hashlib.sha256(raw)
        self.records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not self.records:
            raise ValueError("dataset index is empty")
        self.vocab_size = vocab_size
        self.verified: set[tuple[str, str]] = set()
        self.null_text = None
        metadata_path = self.root / "metadata.json"
        if metadata_path.is_file():
            metadata_bytes = metadata_path.read_bytes()
            identity.update(metadata_bytes)
            metadata = json.loads(metadata_bytes)
            if asset_hashes is not None and metadata["assets"] != asset_hashes:
                raise ValueError(
                    "prepared data uses a different registry or text encoder"
                )
            if "null_text_sha256" in metadata:
                if (
                    bundle.sha256(self.root / "null_text.npy")
                    != metadata["null_text_sha256"]
                ):
                    raise ValueError("prepared null-text features changed")
                self.null_text = np.load(
                    self.root / "null_text.npy", allow_pickle=False
                )
        elif asset_hashes is not None:
            raise ValueError(
                "prepared data needs metadata.json with bundle asset hashes"
            )
        self.sha256 = identity.hexdigest()
        weights = np.array([r.get("weight", 1.0) for r in self.records], dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("sampling weights must be finite and positive")
        weights /= weights.max()
        self.probabilities = weights / weights.sum()
        self.domains = np.array([r.get("domain", 0) for r in self.records])
        self.pools = [
            np.flatnonzero(self.domains == 0),
            np.flatnonzero(self.domains != 0),
        ]
        self.cumulative = [
            np.cumsum(self.probabilities[p] / self.probabilities[p].sum())
            if len(p)
            else np.array([])
            for p in self.pools
        ]
        self.sources = [
            np.array(
                [i for i in p if self.records[i].get("kind") == "source"], dtype=int
            )
            for p in self.pools
        ]
        for row in self.records:
            if not 0 <= row.get("domain", 0) < 3:
                raise ValueError("dataset domain must be 0, 1 or 2")
            if row.get("kind") == "source":
                if not row.get("captions"):
                    raise ValueError("source row needs captions")
            elif row["task"] not in TASK_NAMES or not isinstance(row["prompt"], str):
                raise ValueError("dataset row needs a known task and text prompt")

    def _read(self, row: dict) -> dict[str, np.ndarray]:
        path = bundle.local_path(self.root, row["path"])
        identity = row["path"], row["sha256"]
        if identity not in self.verified:
            if bundle.sha256(path) != row["sha256"]:
                raise ValueError(f"example digest changed: {row['path']}")
            self.verified.add(identity)
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in archive.files}
        if "null_text" not in values and self.null_text is not None:
            values["null_text"] = self.null_text
        return values

    def sample(
        self, seed: int, replay_share: float = 0.0, stage: str = "structure"
    ) -> Example:
        rng = np.random.default_rng(seed)
        replay = rng.random() < replay_share
        pool = self.pools[int(replay)]
        if not len(pool):
            raise ValueError("dataset lacks the requested Minecraft/replay population")
        position = min(
            int(np.searchsorted(self.cumulative[int(replay)], rng.random())),
            len(pool) - 1,
        )
        selected = int(pool[position])
        row = self.records[selected].copy()
        arrays = self._read(row)
        if row.get("kind") == "source":
            shares = (
                (0.25, 0.25, 0.25, 0.15, 0.10, 0)
                if stage == "structure"
                else (0.25, 0.25, 0.20, 0.15, 0.05, 0.10)
            )
            task = str(rng.choice(TASK_NAMES, p=shares))
            global_shape = stage == "structure" and (replay or rng.random() < 0.5)
            if global_shape:
                task = "scratch_void"
                values = tasks.global_structure(arrays["source"], rng)
            else:
                companion = None
                if task == "replacement":
                    candidates = self.sources[int(replay)]
                    if len(candidates) < 2:
                        raise ValueError(
                            "replacement requires another source in the same domain"
                        )
                    position = int(np.searchsorted(candidates, selected))
                    other = int(rng.integers(len(candidates) - 1))
                    other += other >= position
                    companion = self._read(self.records[int(candidates[other])])[
                        "source"
                    ]
                values = tasks.construct(
                    arrays["source"],
                    task,
                    rng,
                    old_source=companion,
                    material=row["material"],
                )
            caption = int(rng.integers(len(arrays["text"])))
            values.update(text=arrays["text"][caption], null_text=arrays["null_text"])
            if stage == "materials":
                occupied = (values["target"] != 0) & values["edit"]
                added = (
                    tasks.boundary_additions(
                        occupied,
                        values["edit"] & (values["target"] == 0),
                        int(rng.integers(2**63)),
                    )
                    if task != "material_only" and rng.random() < 0.5
                    else np.zeros_like(occupied)
                )
                values["active"] = occupied | added
            arrays = values
            row.update(
                task=task,
                style=caption if len(row["captions"]) == 5 else 5,
                fragment=not task.startswith("scratch"),
            )
        tensors = {name: torch.from_numpy(value) for name, value in arrays.items()}
        example = Example(
            before=tensors["before"],
            target=tensors["target"],
            edit=tensors["edit"],
            observed=tensors["observed"],
            text=tensors["text"],
            null_text=tensors["null_text"],
            active=tensors.get("active"),
            task=TASK_NAMES.index(row["task"]),
            style=row.get("style", 5),
            score=row.get("score", -1.0),
            fragment=row.get("fragment", False),
            domain=row.get("domain", 0),
        )
        example.validate(self.vocab_size)
        return example
