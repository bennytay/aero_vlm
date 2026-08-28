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
The default `uv sync` remains light. Teacher, distillation, and ONNX tools are
installed only with `uv sync --extra training`.

## Phase 1 frozen teacher baseline

Phase 1 assembles official validation subsets into a hashed manifest and runs
frozen MobileCLIP2-S0 zero-shot inference. Raw images, checkpoint weights, and
large generated artefacts remain ignored.

```shell
uv run python -m wam_drones.assemble_v0
uv sync --extra training
uv run python -m wam_drones.teacher.evaluate
uv run python -m wam_drones.teacher.run_target path/to/images \
  --vocabulary configs/vocabulary_v0.yaml
```

The source/licence map is at
[`data/manifests/sources_v0.yaml`](data/manifests/sources_v0.yaml), the validated
scene manifest is at
[`data/manifests/dataset_v0.json`](data/manifests/dataset_v0.json), and the full
result is in
[`evaluation/experiments/exp_20260828_teacher_v0_prompts/report.md`](evaluation/experiments/exp_20260828_teacher_v0_prompts/report.md).

On 1,141 held-out validation scenes, the photo prompt reached 75.99% top-1 and
95.97% top-3 against a 9.09% chance baseline. The aerial prompt reached 73.88%
top-1 and 92.81% top-3. The original 11-way rejection was weak. Phase 1b
freezes a ten-prompt cosine rule at **T=0.170**, reducing validation
`no_target` false positives from 48% to 18% while retaining 80.25% macro recall
on the supported scope. See
[`exp_20260828_teacher_threshold`](evaluation/experiments/exp_20260828_teacher_threshold/report.md)
for the full curve and limitations.

## Phase 2 compact student

The student is MobileNetV2-0.35 at 160x160 with a 64-D L2-normalised embedding.
It learns a fixed train-only projection of frozen teacher image embeddings with
cosine and contrastive losses; it is not a class-only model and the teacher
remains frozen.

```shell
uv sync --extra training
uv run python -m wam_drones.student.train --device cpu --epochs 30
```

Generated checkpoints, ONNX, predictions, and large embeddings are gitignored.
The supported v0 scope is water bottle, black vehicle, bicycle, cardboard box,
and sports ball. The other names remain accepted by the exact parser but are
parked pending licensed evidence. `no_target` is a reject result, not a sixth
object. Open Images does not verify colour-qualified names, and these results
do not establish aerial-domain performance.

## Commands and contracts

The frozen L1 command grammar accepts uppercase `FIND`, `APPROACH`, or `HOLD`,
one exact lowercase vocabulary label, and an optional positive stand-off
distance in metres. Examples:

```text
FIND bicycle
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
report stub. The current experiment is `exp_20260828_student_v0_distill`.

Phase 2 remains a desktop student experiment, not an onboard VLM. Hardware
bring-up, firmware, quantisation, SITL, and flight control remain out of scope.
