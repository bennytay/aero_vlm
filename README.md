# µAeroVLM

µAeroVLM is a research project for a low-cost, fully local,
language-conditioned quadcopter. The first implementation is deliberately L1:
an onboard parser accepts a bounded vocabulary and command grammar. It is not a
full generative VLM running onboard.

The canonical requirements, architecture, safety rules, and implementation
sequence are in [PROJECT_SPEC.md](PROJECT_SPEC.md). The frozen Phase 0 target
vocabulary is in [configs/vocabulary_v0.yaml](configs/vocabulary_v0.yaml), and
the milestone checklist is in [docs/milestones.md](docs/milestones.md).

## Setup and checks

Python 3.11 and [`uv`](https://docs.astral.sh/uv/) are required. From the
repository root:

```shell
uv sync
uv run pytest
uv run ruff check .
uv run mypy src tests
```

No GPU, NVIDIA software, dataset, or model weights are required for Phase 0.

## Commands and contracts

The frozen L1 command grammar accepts uppercase `FIND`, `APPROACH`, or `HOLD`,
one exact lowercase vocabulary label, and an optional positive stand-off
distance in metres. Examples:

```text
FIND red backpack
APPROACH water bottle 2.5
HOLD black vehicle
```

`CommandState` and `TargetObservation` are implemented in
`src/wam_drones/contracts.py`; their generated JSON Schemas are under
`interfaces/schemas/`. Image coordinates and boxes are normalised to `[0, 1]`,
with `(0, 0)` at the top-left, +x right, and +y down. The named stale-observation
threshold is 200 ms.

## Experiments

Experiment IDs use `exp_YYYYMMDD_short_name`. Create a record with:

```shell
uv run python -m wam_drones.experiments.init exp_YYYYMMDD_short_name
```

The helper writes `evaluation/experiments/<experiment_id>/meta.json` and a
report stub. The current experiment is
`exp_20260828_phase0_controls`.

Phase 0 stops at reproducible engineering controls. The next phase is Phase 1,
the desktop teacher baseline—not hardware bring-up.
