"""CLI for creating an experiment metadata record."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from wam_drones.experiments.record import initialise_experiment


def repository_root() -> Path:
    """Resolve the repository root from the installed source tree."""
    return Path(__file__).resolve().parents[3]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", help="exp_YYYYMMDD_short_name")
    args = parser.parse_args(argv)
    meta_path = initialise_experiment(args.experiment_id, repository_root())
    print(meta_path.relative_to(repository_root()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
