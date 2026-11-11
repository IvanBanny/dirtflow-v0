import copy
import dataclasses
import os
import pathlib
import subprocess
import sys

import pytest
import torch
from test_training import cpu_config, write_dataset

from dirtflow import training


@pytest.mark.parametrize("stage", ["structure", "materials"])
def test_ddp_matches_accumulated_single_process(models, example, tmp_path, stage):
    config = dataclasses.replace(
        cpu_config(stage), presentations=4, warmup=0, batch_size=4
    )
    dataset = write_dataset(tmp_path, example)
    torch.save(
        dict(
            model=models.config,
            state=models.state_dict(),
            config=dataclasses.asdict(config),
        ),
        tmp_path / "initial.pt",
    )
    expected = training.run(
        copy.deepcopy(models),
        dataset,
        config,
        tmp_path / "single",
        torch.device("cpu"),
        "source",
    )
    root = pathlib.Path(__file__).resolve().parents[1]
    env = os.environ | {
        "PYTHONPATH": str(root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(root / "tests/distributed_worker.py"),
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    a, b = [
        torch.load(path, weights_only=True)
        for path in (expected, tmp_path / "distributed/checkpoint.pt")
    ]
    for name, state in a["modules"].items():
        for key, value in state.items():
            # Summation order differs between rank reduction and accumulation
            torch.testing.assert_close(
                value, b["modules"][name][key], atol=2e-6, rtol=1e-5
            )
