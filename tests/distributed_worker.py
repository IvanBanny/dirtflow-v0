"""Two-process accumulation check, launched by test_distributed.py."""

import pathlib
import sys

import torch
import torch.distributed as dist

from dirtflow.data import Dataset
from dirtflow.model.network import Models
from dirtflow.training import Config, run

root = pathlib.Path(sys.argv[1])
saved = torch.load(root / "initial.pt", weights_only=True)
models = Models(saved["model"], torch.tensor([0, 2, 3]))
models.load_state_dict(saved["state"])
dist.init_process_group("gloo")
try:
    run(
        models,
        Dataset(root / "index.jsonl", 4),
        Config(**saved["config"]),
        root / "distributed",
        torch.device("cpu"),
        "source",
    )
finally:
    dist.destroy_process_group()
