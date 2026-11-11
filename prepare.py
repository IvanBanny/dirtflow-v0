"""Prepare complete block grids and frozen CLIP features for online task sampling."""

import argparse
import json
import pathlib

import numpy as np

from dirtflow import bundle, voxel
from dirtflow.schema import Schema
from dirtflow.text import TextEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=pathlib.Path,
        required=True,
        help="JSONL: path, captions, optional score, weight and domain",
    )
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = bundle.validate_bundle(args.bundle)
    schema = Schema.load(args.bundle, manifest["assets"])
    text = TextEncoder(
        args.bundle / manifest["assets"]["text_encoder"],
        args.device,
        manifest["model"]["text_encoder"]["use_attention_mask"],
    )
    null = text.encode([""])[0].cpu().float().numpy()
    rows = [
        json.loads(line)
        for line in args.sources.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) < 2:
        raise ValueError("replacement tasks need at least two source builds")
    args.output.mkdir(parents=True)
    np.save(args.output / "null_text.npy", null)
    assets = {
        key: manifest["files"][manifest["assets"][key]]
        for key in ("registry", "schema", "mapping", "clip_manifest")
    }
    metadata = dict(
        assets=assets, null_text_sha256=bundle.sha256(args.output / "null_text.npy")
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output / "index.jsonl").open("w") as index:
        for i, row in enumerate(rows):
            source_path = bundle.local_path(args.sources.resolve().parent, row["path"])
            tensor = voxel.decode(source_path.read_bytes())
            mapping = np.array(
                [schema.normalize_input(state) for state in tensor.palette],
                dtype=np.int64,
            )
            source = mapping[tensor.array()]
            if not source.any():
                raise ValueError(f"source has no occupied cells: {source_path}")
            captions = row["captions"]
            if not 1 <= len(captions) <= 5 or any(
                not isinstance(c, str) or not c.strip() for c in captions
            ):
                raise ValueError("captions must contain one to five nonempty strings")
            encoded = text.encode(captions).cpu().float().numpy()
            path = args.output / f"{i:07d}.npz"
            np.savez_compressed(path, source=source, text=encoded)
            record = dict(
                kind="source",
                path=path.name,
                sha256=bundle.sha256(path),
                captions=captions,
                score=row.get("score", -1.0),
                domain=row.get("domain", 0),
                weight=row.get("weight", 1.0),
                material=schema.normalize_input("minecraft:stone"),
                source_sha256=bundle.sha256(source_path),
            )
            index.write(json.dumps(record) + "\n")
    print(args.output / "index.jsonl")


if __name__ == "__main__":
    main()
