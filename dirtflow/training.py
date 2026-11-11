"""Sample-clock training with accumulation, DDP and exact checkpoint state."""

import contextlib
import dataclasses
import json
import math
import pathlib
from typing import cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import checkpoint
from .data import Dataset, presentation_seed
from .model.network import MATERIALS, STRUCTURE, Models
from .objectives import TrainingLoss


@dataclasses.dataclass(frozen=True)
class Config:
    """Horizon, LR and EMA use presentations; log/save intervals use updates."""

    stage: str = "structure"
    seed: int = 0
    presentations: int = 2_040_000
    batch_size: int = 6
    warmup: int = 8_000
    hold_fraction: float = 0.2
    backbone_lr: float = 3e-5
    new_lr: float = 3e-4
    final_lr_ratio: float = 1 / 30
    weight_decay: float = 0.05
    optimizer: str = "adamw8bit"
    precision: str = "bf16"
    ema_horizon: float = 80_000
    ema_stride: int = 64
    replay_initial: float = 0.3
    replay_final: float = 0.05
    save_every: int = 10_000
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.stage not in {"structure", "materials"}:
            raise ValueError("stage must be structure or materials")
        for name in (
            "presentations",
            "batch_size",
            "ema_stride",
            "save_every",
            "log_every",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.presentations % self.batch_size
            or not 0 <= self.warmup <= self.presentations
        ):
            raise ValueError("horizon must contain whole batches and include warmup")
        if not 0 <= self.hold_fraction < 1 or not 0 < self.final_lr_ratio <= 1:
            raise ValueError("invalid learning-rate schedule")
        if self.warmup > self.hold_fraction * self.presentations:
            raise ValueError("warmup exceeds the learning-rate hold phase")
        if self.optimizer not in {"adamw", "adamw8bit"} or self.precision not in {
            "fp32",
            "bf16",
        }:
            raise ValueError("unsupported optimizer or precision")
        if not 0 <= self.replay_final <= self.replay_initial <= 1:
            raise ValueError("invalid replay schedule")
        if self.stage == "materials" and self.replay_initial:
            raise ValueError("material training requires Minecraft targets")
        if (
            not all(
                math.isfinite(x) and x > 0
                for x in (self.backbone_lr, self.new_lr, self.ema_horizon)
            )
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError(
                "learning rates and EMA horizon must be positive and finite"
            )


def lr_factor(n: int, config: Config) -> float:
    if n < config.warmup:
        return n / max(config.warmup, 1)
    hold = config.presentations * config.hold_fraction
    progress = min(1.0, max(0.0, (n - hold) / (config.presentations - hold)))
    return (
        config.final_lr_ratio
        + (1 - config.final_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    )


def replay_share(n: int, config: Config) -> float:
    progress = min(1.0, max(0.0, (n / config.presentations - 0.15) / 0.85))
    return (
        config.replay_final
        + (config.replay_initial - config.replay_final)
        * (1 + math.cos(math.pi * progress))
        / 2
    )


def parameter_groups(models: Models, config: Config) -> list[dict]:
    trunk = (
        models.structure_flow if config.stage == "structure" else models.blockstate_flow
    )
    backbone = {id(p) for p in trunk.parameters()}
    embeddings = {
        id(p)
        for module in models.modules()
        if isinstance(module, torch.nn.Embedding)
        for p in module.parameters()
    }
    groups: dict[tuple[bool, bool], dict] = {}
    for name, parameter in models.named_parameters():
        if not parameter.requires_grad:
            continue
        pretrained = id(parameter) in backbone
        decay = (
            parameter.ndim >= 2
            and id(parameter) not in embeddings
            and "embedding" not in name.split(".")
        )
        key = (pretrained, decay)
        if key not in groups:
            lr = config.backbone_lr if pretrained else config.new_lr
            groups[key] = dict(
                params=[],
                param_names=[],
                lr=lr,
                base_lr=lr,
                weight_decay=config.weight_decay if decay else 0.0,
            )
        groups[key]["params"].append(parameter)
        groups[key]["param_names"].append(name)
    return list(groups.values())


class EMA:
    def __init__(self, models: Models, names: tuple[str, ...], config: Config) -> None:
        self.config = config
        self.last = 0
        self.states = {
            name: {
                k: v.detach().cpu().clone()
                for k, v in getattr(models, name).state_dict().items()
            }
            for name in names
        }

    @torch.no_grad()
    def update(self, models: Models, n: int, force: bool = False) -> None:
        delta = n - self.last
        if not delta or delta < self.config.ema_stride and not force:
            return
        # Preserve the original EMA's eight-sample reference clock
        beta = min(
            math.exp(-8 / self.config.ema_horizon), (1 + n / 8) / (10 + n / 8)
        ) ** (delta / 8)
        for name, state in self.states.items():
            for key, value in getattr(models, name).state_dict().items():
                if value.is_floating_point():
                    state[key].lerp_(
                        value.detach().to("cpu", dtype=state[key].dtype), 1 - beta
                    )
                else:
                    state[key].copy_(value)
        self.last = n


def run(
    models: Models,
    dataset: Dataset,
    config: Config,
    output: pathlib.Path,
    device: torch.device,
    bundle_sha256: str,
    *,
    resume: pathlib.Path | None = None,
    max_updates: int | None = None,
) -> pathlib.Path:
    """Runs on one process or an initialized torchrun process group.

    Global batches contain consecutive presentation IDs. Each presentation's
    data and noise depend only on the run seed and its ID. Resume requires the
    same recipe, index and world size; optimizer and EMA state are restored.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if config.batch_size % world_size:
        raise ValueError("global batch_size must be divisible by world size")
    if config.precision == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("BF16 training requires a CUDA device with BF16 support")
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive")
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and (output / "checkpoint.pt").exists():
        raise FileExistsError(
            "output has a checkpoint; use --resume or a new directory"
        )
    models.train_stage(config.stage, device)
    groups = parameter_groups(models, config)
    optimizer: torch.optim.Optimizer
    if config.optimizer == "adamw8bit":
        if device.type != "cuda":
            raise ValueError("AdamW8bit requires CUDA")
        import bitsandbytes

        optimizer = bitsandbytes.optim.AdamW8bit(
            groups, betas=(0.9, 0.95), eps=1e-8, is_paged=False
        )
    else:
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    names = STRUCTURE if config.stage == "structure" else MATERIALS
    ema = (
        EMA(models, names, config)
        if rank == 0 and config.stage == "structure"
        else None
    )
    n = 0
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        expected = dict(
            format="dirtflow-training",
            stage=config.stage,
            config=dataclasses.asdict(config),
            model_config=models.config,
            dataset_sha256=dataset.sha256,
            bundle_sha256=bundle_sha256,
            world_size=world_size,
        )
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"resume differs in {key}")
        if set(saved["modules"]) != set(names):
            raise ValueError("resume has incomplete stage weights")
        for name, state in saved["modules"].items():
            getattr(models, name).load_state_dict(state, strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        n = saved["presentations"]
        if not 0 <= n <= config.presentations or n % config.batch_size:
            raise ValueError("invalid checkpoint presentation count")
        if ema is not None:
            ema.states, ema.last = saved["ema"], saved["ema_last"]
        del saved
    loss_module: torch.nn.Module = TrainingLoss(models, config.stage)
    if world_size > 1:
        loss_module = DistributedDataParallel(
            loss_module,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(output / "events"), purge_step=n if resume else None)
        (output / "config.json").write_text(
            json.dumps(dataclasses.asdict(config), indent=2) + "\n"
        )
    local_batch = config.batch_size // world_size
    stop = (
        config.presentations
        if max_updates is None
        else min(config.presentations, n + max_updates * config.batch_size)
    )
    path = output / "checkpoint.pt"
    try:
        while n < stop:
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_factor(
                    n + config.batch_size, config
                )
            metrics: dict[str, torch.Tensor] = {}
            for micro in range(local_batch):
                index = n + micro * world_size + rank
                seed = presentation_seed(config.seed, index)
                example = dataset.sample(
                    seed, replay_share(n, config), config.stage
                ).to(device)
                sync = (
                    cast(DistributedDataParallel, loss_module).no_sync()
                    if world_size > 1 and micro + 1 < local_batch
                    else contextlib.nullcontext()
                )
                with (
                    sync,
                    torch.autocast(
                        device.type,
                        dtype=torch.bfloat16,
                        enabled=config.precision == "bf16",
                    ),
                ):
                    loss, parts = loss_module(example, n + config.batch_size, seed)
                    loss = loss / local_batch
                loss.backward()
                for key, value in dict(
                    loss=loss.detach() * local_batch, **parts
                ).items():
                    metrics[key] = metrics.get(key, 0) + value / local_batch
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in models.parameters() if p.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            n += config.batch_size
            if ema is not None:
                ema.update(models, n, force=n == config.presentations)
            update = n // config.batch_size
            if update % config.log_every == 0 or n == stop:
                if world_size > 1:
                    for value in metrics.values():
                        dist.all_reduce(value)
                        value /= world_size
                if writer is not None:
                    for key, value in metrics.items():
                        writer.add_scalar(key, float(value), n)
                    writer.add_scalar("grad_norm", float(grad_norm), n)
                    writer.add_scalar(
                        "lr/backbone", config.backbone_lr * lr_factor(n, config), n
                    )
                    print(
                        f"presentations {n}: loss {float(metrics['loss']):.6f}",
                        flush=True,
                    )
            if update % config.save_every == 0 or n == stop:
                if rank == 0:
                    checkpoint.save_atomic(
                        path,
                        dict(
                            format="dirtflow-training",
                            version=1,
                            stage=config.stage,
                            config=dataclasses.asdict(config),
                            model_config=models.config,
                            dataset_sha256=dataset.sha256,
                            bundle_sha256=bundle_sha256,
                            world_size=world_size,
                            presentations=n,
                            modules={
                                name: getattr(models, name).state_dict()
                                for name in names
                            },
                            optimizer=optimizer.state_dict(),
                            ema=ema.states if ema else {},
                            ema_last=ema.last if ema else 0,
                        ),
                    )
                if world_size > 1:
                    dist.barrier()
    finally:
        if writer is not None:
            writer.close()
    return path
