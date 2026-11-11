"""Export trained stages into a bundle accepted by the Dirtflow engine."""

import argparse
import pathlib

from dirtflow.checkpoint import export_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--structure", type=pathlib.Path)
    parser.add_argument("--materials", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    export_bundle(args.bundle, args.output, args.structure, args.materials)


if __name__ == "__main__":
    main()
