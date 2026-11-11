"""The modules shared by training, research sampling and engine bundles."""

import copy

import torch
from torch import nn

from .codec import LatentNorm, SparseStructureDecoder, SparseStructureEncoder
from .features import BlockStateEmbedding, ExtraConditioning, StructureContextEncoder
from .head import BlockStateHead
from .sparse_flow import BlockStateFlowModel
from .structure import StructureFlowModel

STRUCTURE = ("context_encoder", "structure_flow", "structure_extra")
MATERIALS = ("blockstate_flow", "blockstate_extra", "embedding", "categorical_head")


class Models(nn.Module):
    """Owns the numerical modules; checkpoint and data I/O live outside the model."""

    def __init__(self, config: dict, output_to_global: torch.Tensor) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        if config["feature_schema"] != "v1":
            raise ValueError("expected observation feature schema v1")
        emb, vae = config["embedding"], config["ss_vae"]
        structure, materials = config["structure_flow"], config["blockstate_flow"]
        if (
            emb["dim"] != 8
            or structure["in_channels"] != 32
            or materials["in_channels"] != 42
            or not structure.get("self_conditioning")
            or not materials.get("self_conditioning")
        ):
            raise ValueError("expected eight latent channels and 32/42 flow inputs")
        if config["categorical_head"]["kind"] != "categorical_independent":
            raise ValueError(
                "expected the continuous model with independent class keys"
            )
        codec_args = dict(
            latent_channels=vae["latent_channels"],
            num_res_blocks=vae["num_res_blocks"],
            num_res_blocks_middle=vae["num_res_blocks_middle"],
            structure_coordinate_system=config["structure_coordinate_system"],
        )
        self.ss_encoder = SparseStructureEncoder(
            channels=tuple(vae["channels"]), **codec_args
        )
        self.ss_decoder = SparseStructureDecoder(
            channels=tuple(reversed(vae["channels"])), **codec_args
        )
        self.latent_norm = LatentNorm(vae["latent_channels"])
        self.context_encoder = StructureContextEncoder(
            **config["context_encoder"],
            vocab_size=emb["vocab_size"],
            embedding_dim=emb["dim"],
            embedding_rms_norm=emb["rms_norm"],
            structure_coordinate_system=config["structure_coordinate_system"],
        )
        shared = (
            "resolution",
            "in_channels",
            "out_channels",
            "model_channels",
            "cond_channels",
            "num_blocks",
            "num_heads",
            "mlp_ratio",
            "qk_rms_norm",
            "qk_rms_norm_cross",
            "self_conditioning",
        )
        self.structure_flow = StructureFlowModel(
            **{k: structure[k] for k in shared if k in structure}
        )
        widths = materials["io_block_channels"]
        self.blockstate_flow = BlockStateFlowModel(
            **{k: materials[k] for k in shared if k in materials},
            io_block_channels=(widths,) if isinstance(widths, int) else tuple(widths),
            num_io_res_blocks=materials["num_io_res_blocks"],
            coordinate_system=config["blockstate_coordinate_system"],
        )
        self.embedding = BlockStateEmbedding(
            emb["vocab_size"], emb["dim"], rms_norm=emb["rms_norm"]
        )
        self.categorical_head = BlockStateHead(
            output_to_global,
            feat_dim=self.blockstate_flow.io_channels,
            hidden=config["categorical_head"]["hidden"],
        )
        self.structure_extra = ExtraConditioning(structure["model_channels"])
        self.blockstate_extra = ExtraConditioning(materials["model_channels"])

    def train_stage(self, stage: str, device: torch.device | str) -> None:
        """Freezes the other stage and places only the required modules on device."""
        names = {"structure": STRUCTURE, "materials": MATERIALS}[stage]
        self.eval().requires_grad_(False)
        for name, module in self.named_children():
            module.to(
                device
                if name in names
                or stage == "structure"
                and name in ("ss_encoder", "latent_norm")
                else "cpu"
            )
            if name in names:
                module.train().requires_grad_(True)

    def module_states(self) -> dict[str, dict[str, torch.Tensor]]:
        """Returns the checkpoint's named module state dictionaries."""
        return {name: module.state_dict() for name, module in self.named_children()}
