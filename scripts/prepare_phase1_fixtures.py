"""Extract ten deterministic VisDrone images from an official DET archive."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from wam_drones.detection.fixtures import extract_visdrone_fixtures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/samples/phase1_visdrone")
    )
    args = parser.parse_args(argv)
    for path in extract_visdrone_fixtures(args.archive, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
