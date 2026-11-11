"""Initialize new world interfaces around the pretrained TRELLIS XL flows."""

import argparse
import pathlib

import safetensors.torch
import torch

from dirtflow import bundle, checkpoint
from dirtflow.model.network import MATERIALS, STRUCTURE, Models
from dirtflow.schema import Schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=pathlib.Path,
        required=True,
        help="Provides adapted codec, normalization, registry and CLIP",
    )
    parser.add_argument("--structure-flow", type=pathlib.Path, required=True)
    parser.add_argument("--material-flow", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = bundle.validate_bundle(args.bundle)
    schema = Schema.load(args.bundle, manifest["assets"])
    torch.manual_seed(args.seed)
    fresh = Models(manifest["model"], torch.from_numpy(schema.output_to_global))
    checkpoint.transfer_trellis(
        fresh.structure_flow, safetensors.torch.load_file(str(args.structure_flow))
    )
    checkpoint.transfer_trellis(
        fresh.blockstate_flow, safetensors.torch.load_file(str(args.material_flow))
    )
    # Initial LIVE and EMA values coincide before the first optimizer update
    checkpoint.write_bundle(
        args.bundle,
        args.output,
        {name: getattr(fresh, name).state_dict() for name in STRUCTURE + MATERIALS},
        "trellis-initialization",
        dict(
            structure_flow_sha256=bundle.sha256(args.structure_flow),
            material_flow_sha256=bundle.sha256(args.material_flow),
            seed=args.seed,
        ),
    )


if __name__ == "__main__":
    main()
