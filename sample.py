"""Sample a native block grid, with optional observed context."""

import argparse
import pathlib

import numpy as np

from dirtflow import voxel
from dirtflow.download import fetch_model
from dirtflow.model.features import TASK_NAMES
from dirtflow.sampling import Sampler, Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--bundle", type=pathlib.Path)
    parser.add_argument(
        "--input", type=pathlib.Path, help="NPZ with before, edit and observed arrays"
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--size", type=int, nargs=3, default=(20, 20, 20), metavar=("X", "Y", "Z")
    )
    parser.add_argument("--task", choices=TASK_NAMES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--structure-steps", type=int, default=30)
    parser.add_argument("--material-steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.input:
        with np.load(args.input, allow_pickle=False) as saved:
            before, edit, observed = (
                saved[name] for name in ("before", "edit", "observed")
            )
        task = args.task or "replacement"
    else:
        if any(not 1 <= n <= 58 for n in args.size) or np.prod(args.size) > 10_000:
            parser.error("size must fit 58 cells per axis and 10,000 editable cells")
        before = np.zeros((64, 64, 64), dtype=np.int64)
        edit = np.zeros_like(before, dtype=bool)
        observed = np.zeros_like(edit)
        start = (64 - np.array(args.size)) // 2
        radius = max(r for r in (3, 5, 8) if max(args.size) + 2 * r <= 64)
        edit[tuple(slice(a, a + n) for a, n in zip(start, args.size))] = True
        observed[
            tuple(slice(a - radius, a + n + radius) for a, n in zip(start, args.size))
        ] = True
        task = args.task or "scratch_void"
    sampler = Sampler(args.bundle or fetch_model(), args.device)
    result = sampler.generate(
        before,
        edit,
        observed,
        args.prompt,
        seed=args.seed,
        task=task,
        settings=Settings(args.structure_steps, args.material_steps, args.guidance),
    )
    states = np.unique(result)
    states = np.concatenate(([0], states[states != 0]))
    palette = [sampler.schema.states[i] for i in states]
    mapping = np.zeros(len(sampler.schema.states), dtype=np.int64)
    mapping[states] = np.arange(len(states))
    flat = mapping[result].transpose(2, 1, 0).reshape(-1)
    positions = np.flatnonzero(flat).tolist()
    tensor = voxel.Tensor(
        (0, 0, 0), (64, 64, 64), palette, positions, flat[positions].tolist()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(voxel.encode(tensor))
    print(args.output)


if __name__ == "__main__":
    main()
