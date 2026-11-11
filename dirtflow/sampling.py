"""Two-stage research sampling with exact preservation outside the edit."""

import dataclasses
import math
import pathlib

import numpy as np
import torch

from . import checkpoint
from .model import features
from .model.network import MATERIALS, STRUCTURE
from .objectives import clean_estimate, interpolate
from .text import TextEncoder


@dataclasses.dataclass(frozen=True)
class Settings:
    structure_steps: int = 30
    material_steps: int = 30
    guidance: float = 5.0

    def __post_init__(self) -> None:
        if any(
            type(n) is not int or n < 1
            for n in (self.structure_steps, self.material_steps)
        ):
            raise ValueError("Euler step counts must be positive integers")
        if not math.isfinite(self.guidance) or self.guidance < 0:
            raise ValueError("guidance must be finite and nonnegative")


class Sampler:
    """Loads one bundle; each generation owns its noise and coordinate cache."""

    def __init__(self, path: str | pathlib.Path, device: str = "cuda") -> None:
        self.root = pathlib.Path(path)
        self.device = torch.device(device)
        if self.device.type == "cuda" and (
            not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("CUDA sampling requires BF16 support")
        self.models, self.schema, self.manifest = checkpoint.load_bundle(self.root)
        if (
            self.manifest["inference_policy"]["editable_confidence"]
            != "zero_normalized_editable"
        ):
            raise ValueError("research sampling requires corrected editable confidence")
        self.text = TextEncoder(
            self.root / self.manifest["assets"]["text_encoder"],
            str(self.device),
            self.manifest["model"]["text_encoder"]["use_attention_mask"],
        )

    def _place(self, stage: str) -> None:
        names = (
            STRUCTURE + ("ss_decoder", "latent_norm")
            if stage == "structure"
            else MATERIALS
        )
        for name, module in self.models.named_children():
            module.to(self.device if name in names else "cpu")

    @torch.no_grad()
    def generate(
        self,
        before: np.ndarray,
        edit: np.ndarray,
        observed: np.ndarray,
        prompt: str,
        *,
        seed: int = 0,
        task: str = "scratch_void",
        settings: Settings = Settings(),
    ) -> np.ndarray:
        """Returns legal registry IDs [64,64,64] without placement postprocessing.

        Arrays use XYZ axes, int64 states and boolean masks. Both guidance
        branches retain observations. The final head uses the last conditioned
        trunk features, with no extra network call at time zero.
        """
        if (
            before.shape != (64, 64, 64)
            or edit.shape != before.shape
            or observed.shape != before.shape
        ):
            raise ValueError("generation requires matching 64-cube arrays")
        if before.dtype != np.int64 or edit.dtype != bool or observed.dtype != bool:
            raise ValueError("expected int64 states and boolean masks")
        if not edit.any() or (edit & ~observed).any():
            raise ValueError("edit must be nonempty and fully observed")
        coordinates = np.argwhere(edit)
        if edit.sum() > 10_000 or (np.ptp(coordinates, axis=0) + 1 > 58).any():
            raise ValueError("edit must fit 58 cells per axis and 10,000 cells")
        if (before < 0).any() or (before >= len(self.schema.states)).any():
            raise ValueError("state ID outside input registry")
        if before[~observed].any():
            raise ValueError("unobserved storage must be zero")
        if (
            task not in features.TASK_NAMES
            or type(seed) is not int
            or not 0 <= seed < 2**63
        ):
            raise ValueError("unknown task or invalid seed")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be text")
        canvas, mask, seen = [
            torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
            for x in (before, edit, observed)
        ]
        m = self.models
        rng = torch.Generator(self.device).manual_seed(seed)
        null, text = [self.text.encode([p]) for p in ("", prompt)]

        def extra(module):
            with torch.autocast(self.device.type, enabled=False):
                return module(
                    torch.tensor([5], device=self.device),
                    torch.tensor([9.0], device=self.device),
                    torch.tensor([False], device=self.device),
                    torch.tensor([0], device=self.device),
                    torch.tensor([features.TASK_NAMES.index(task)], device=self.device),
                )

        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            if task == "material_only":
                logits = (canvas != 0).float() * 2 - 1
            else:
                self._place("structure")
                context = m.context_encoder(canvas[None], mask[None], seen[None])
                conditioning = extra(m.structure_extra)
                z = torch.randn(1, 8, 16, 16, 16, device=self.device, generator=rng)
                previous = torch.zeros_like(z)
                times = torch.linspace(1, 0, settings.structure_steps + 1)
                for t, next_t in zip(times[:-1].tolist(), times[1:].tolist()):
                    x = torch.cat((z, context, previous), dim=1)
                    time = torch.tensor([t * 1000], device=self.device)
                    v_null, v_text = [
                        m.structure_flow(x, time, c, conditioning) for c in (null, text)
                    ]
                    velocity = v_null + settings.guidance * (v_text - v_null)
                    previous = clean_estimate(z, velocity, t)
                    z = z - (t - next_t) * velocity
                logits = m.ss_decoder(m.latent_norm.inverse(z))[0, 0].float()
            active = features.support(logits, mask, seen, canvas, task)
            output = canvas.clone().masked_fill(mask, 0)
            if active.any():
                self._place("materials")
                table = m.embedding.table()
                obs = features.material_observation(
                    active, canvas, mask, seen, table, m.categorical_head
                )
                conditioning = extra(m.blockstate_extra)
                latent = torch.randn(obs.clean.shape, device=self.device, generator=rng)
                previous = torch.zeros_like(latent)
                has_known = bool(obs.known.any())
                times = torch.linspace(1, 0, settings.material_steps + 1)
                for t, next_t in zip(times[:-1].tolist(), times[1:].tolist()):
                    grid = obs.inputs(latent, previous)
                    time = torch.tensor([t * 1000], device=self.device)
                    v_null = m.blockstate_flow(grid, time, null, conditioning)
                    v_text, io = m.blockstate_flow(
                        grid, time, text, conditioning, return_io_feats=True
                    )
                    velocity = v_null.feats + settings.guidance * (
                        v_text.feats - v_null.feats
                    )
                    previous = clean_estimate(latent, velocity, t)
                    latent = latent - (t - next_t) * velocity
                    if has_known:
                        noise = torch.randn(
                            obs.clean.shape, device=self.device, generator=rng
                        )
                        known, _ = interpolate(obs.clean, noise, next_t)
                        latent[obs.known] = known[obs.known].to(latent.dtype)
                prediction = m.categorical_head.decode(
                    latent, io, obs.palette, allow_air=task != "material_only"
                )
                _, x, y, z = obs.grid.coords.unbind(1)
                output[x, y, z] = torch.where(obs.known, canvas[x, y, z], prediction)
        result = output.cpu().numpy()
        result[~edit] = before[~edit]
        if task == "material_only" and (((result != 0) != (before != 0)) & edit).any():
            raise RuntimeError("material-only output changed occupied coordinates")
        return result
