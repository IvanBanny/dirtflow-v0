"""Frozen CLIP token features with the trained padding convention."""

import pathlib
from typing import Any, cast

import torch
import transformers


class TextEncoder:
    def __init__(self, path: pathlib.Path, device: str, attention_mask: bool) -> None:
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            path, local_files_only=True
        )
        self.model = (
            cast(Any, transformers.CLIPTextModel)
            .from_pretrained(path, local_files_only=True)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.device = device
        self.attention_mask = attention_mask

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> torch.Tensor:
        """Returns CLIP token features [B,77,768], with checkpoint padding semantics."""
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        args = {"input_ids": tokens["input_ids"]}
        if self.attention_mask:
            args["attention_mask"] = tokens["attention_mask"]
        return self.model(**args).last_hidden_state
