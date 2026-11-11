"""Train one flow stage from a model bundle or resume a saved optimizer."""

import argparse
import os
import pathlib
import runpy

import torch
import torch.distributed as dist

from dirtflow import bundle, checkpoint, training
from dirtflow.data import Dataset
from dirtflow.download import fetch_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path)
    parser.add_argument("--resume", type=pathlib.Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    config = runpy.run_path(str(args.config))["get_config"]()
    if not isinstance(config, training.Config):
        raise TypeError("get_config() must return training.Config")
    device = torch.device(args.device)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        path = args.bundle or fetch_model()
        models, schema, manifest = checkpoint.load_bundle(path)
        assets = {
            key: manifest["files"][manifest["assets"][key]]
            for key in ("registry", "schema", "mapping", "clip_manifest")
        }
        dataset = Dataset(args.data, len(schema.states), assets)
        training.run(
            models,
            dataset,
            config,
            args.output,
            device,
            bundle.sha256(path / "bundle.json"),
            resume=args.resume,
            max_updates=args.max_updates,
        )
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
