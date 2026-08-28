# Experiment exp_20260828_phase0_controls: Phase 0 controls

## Question

Does the repo tooling work?

## Hypothesis

A new contributor can sync the Python 3.11 environment and run all Phase 0
checks without model or hardware dependencies.

## Configuration

- Git commit: `5ec033c2c8aebb14dc2c835a669bf3b837a53799`
- Config hash: `sha256:157ffceb4eccb4519e57b6e906f76f6fb6e11b1f8a147f03e84b9f4301998ac2`
- Dataset manifest hash: N/A (no dataset used)
- Hardware revision: desktop environment; exact hardware TBD
- Firmware versions: N/A (no firmware used)
- Model hash: N/A (no model trained)
- Random seed: 0

## Protocol

Run `uv sync`, `uv run pytest`, `uv run ruff check .`, and
`uv run mypy src tests` from a clean checkout.

## Acceptance criteria

Dependency sync succeeds on Python 3.11 without GPU requirements, and every
test, lint, and type-check command exits successfully.

## Results

- `uv sync`: passed with CPython 3.11 and 19 resolved packages
- `uv run pytest`: 19 passed
- `uv run ruff check .`: passed
- `uv run mypy src tests`: passed with 12 source files checked

## Failures and anomalies

None recorded.

## Conclusion

The Phase 0 repository tooling works on the tested desktop environment without
GPU, dataset, model, firmware, or flight dependencies. The M0 engineering
controls exit gate is satisfied; Phase 1 may begin with the desktop teacher
baseline.
