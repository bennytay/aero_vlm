# Aerial VLM step-by-step implementation plan

**Status:** implementation plan. Steps 0–3 are done; the immediate next work
is Step 4, a trimmed demo — not the training pipeline below it.

**Companion research:** [`vlm_research_brief.md`](vlm_research_brief.md)
**Runtime boundary:** one RGB frame plus text in; validated structured response out

The runtime can serve a live camera loop: submit the most recently decoded RGB
frame and a question, then consume its validated response before submitting a
later frame. It has no temporal state or future-frame access. The sampled-video
CLI is an offline measurement harness for this same single-frame interface; it
does not make the VLM retrospective or video-native.

## MVP roadmap (revised 2026-09-08)

Priority shifted to shipping a working demo before investing further in
training infrastructure. Fine-tuning and polish come after the product
exists, not before.

**Done:**

- **Step 0** — tracking boundary hardened (Phase 4: BoT-SORT selected over
  ByteTrack, HOTA 0.4136 vs 0.3566).
- **Step 1** — VLM contracts frozen (`contracts.py`), schemas checked into
  `interfaces/schemas/`, round-trip tests passing.
- **Step 2** — zero-shot spike run. Qwen3-VL-2B is the selected local
  candidate: 100% parse success, 76.5% (13/17) semantic match on the
  corrected smoke suite, mean latency 1.36s after a bf16 config fix.
  SmolVLM2 dropped (fails the output-contract gate outright). Full record in
  `evaluation/experiments/exp_20260908_vlm_zero_shot/report.md`.
- **Step 3 (MVP scope)** — Phase 4 sequence split locked
  (`configs/vlm/data/locked_split_v1.json`) with a leakage guard
  (`assert_no_held_out_leakage` in `src/wam_drones/vlm/splits.py`). The full
  300–500-prompt evaluation suite is deferred, not built.

**Next: Step 4 below — ship the demo.** Runs the existing zero-shot
Qwen3-VL-2B directly through the already-working `wam-vlm infer` CLI. No
training, adapter, or data factory required.

**Deferred until there's a concrete reason to train** (see "Deferred:
training and evaluation pipeline" below — former Steps 4–10, kept for
reference): the teacher/data factory, external dataset ingestion, detector
baselines, a formal zero-shot model-selection report, the first aerial SFT,
ablations, and final evaluation. None of it blocks the demo. Do not start it
without revisiting this decision first.

## Repository decision

Keep detection, tracking, and the VLM in this repository for the research phase.
This should be a modular monorepo, not one interwoven pipeline.

The components share:

- VisDrone manifests and source-frame identity;
- image/video decoding rules;
- detection and track contracts used by the supervision factory;
- experiment metadata and leakage checks; and
- an evaluation story in which YOLO/BoT-SORT is a baseline and teacher.

A new repository now would require versioning and synchronizing all of those
interfaces before there is evidence that the VLM is independently useful. It
would also make it easier to accidentally evaluate on a frame that entered VLM
training through another checkout.

The dependency direction must still be strict:

```text
dataset/manifests ───────────────┐
                                ├──> detection ──> tracking ──> saved tracks
common media/provenance ─────────┤                           │
                                │                           v
                                └────────────────────> VLM data factory
                                                            │
RGB frame + text ────────────────────────────────────> VLM runtime
                                                            │
detector baseline + GT + human labels ───────────────> VLM evaluation
```

The VLM runtime does not import or call detection/tracking. Only the offline
data factory and evaluator may read their saved artifacts.

Split the VLM into a separate repository later only if at least one is true:
it becomes a reusable package for projects that do not use this detector; it
needs an independent release cadence or public model/data release; its
licence obligations conflict with the detector repository; CI and dependency
weight make normal detector development impractical; or a stable versioned
interface package already exists between the repos. If that happens, first
extract a tiny `wam-perception-contracts` package (or published JSON
Schemas), then split the VLM. Do not copy contract definitions between
repositories.

## Target repository structure

Evolve toward this layout incrementally. `(deferred)` marks files that belong
to the training/evaluation pipeline below and are not needed for the Step 4
demo.

```text
configs/
  datasets/                    # existing dataset provenance
  detection/                   # migrate only when touching old config paths
  tracking/                    # actual BoT-SORT/ByteTrack configs
  vlm/
    models/                    # done
    data/                      # done (locked_split_v1.json)
    training/                  # (deferred)
    eval/                      # (deferred)

data/
  manifests/                   # existing versioned source manifests
  processed/vlm/               # (deferred) ignored generated supervision
  samples/vlm_demo/            # Step 4: licensed demo clip + manifest

interfaces/schemas/
  detection_v1.schema.json     # existing
  frame_tracks_v1.schema.json  # existing
  vlm_response_v1.schema.json           # done
  vlm_supervision_record_v1.schema.json # done
  vlm_inference_record_v1.schema.json   # done

src/wam_drones/
  dataset/                     # existing converters/manifests/splits
  detection/                   # existing YOLO code, unchanged
  tracking/                    # existing offline BoT-SORT/ByteTrack implementation
  vlm/
    contracts.py                # done
    prompts.py                  # done
    inference.py                # done
    teacher.py                  # done
    splits.py                   # done (MVP scope)
    cli.py                      # done
    frame_source.py             # (deferred; folded into inference.py so far)
    structured_decode.py        # (deferred; folded into contracts.py so far)
    data_factory.py             # (deferred)
    baselines.py                 # (deferred)
    evaluation.py                # (deferred)

scripts/
  generate_schemas.py          # existing generator
  render_vlm_showcase.py       # Step 4: thin script, not a config-driven renderer

tests/
  vlm/
    test_contracts.py           # done
    test_teacher.py             # done
    test_splits.py              # done
    test_inference.py           # done
    test_data_factory.py        # (deferred)
    test_evaluation.py          # (deferred)

evaluation/experiments/
  exp_20260908_vlm_zero_shot/  # done (Step 2)
  exp_YYYYMMDD_vlm_sft/        # (deferred)
  exp_YYYYMMDD_vlm_final_eval/ # (deferred)

runs/vlm_demo/                 # ignored rendered output and audit artifacts
```

Do not perform a large config migration at the same time as the VLM work. New
VLM files can use `configs/vlm/`; existing detector paths should remain stable.

In `pyproject.toml`, keep the default install lightweight and add optional
groups rather than placing Torch/Transformers in core dependencies:

- `vlm-inference`: Transformers, Accelerate, image processor, constrained
  decoding backend — everything the demo needs;
- `vlm-train`: inference group plus TRL, PEFT, bitsandbytes and datasets —
  not needed until the deferred training pipeline starts;
- `vlm-eval`: metric and report dependencies — deferred; and
- `vlm-demo`: OpenCV/Pillow plus whatever the Step 4 script needs to render.

Pin complete working versions in `uv.lock`. CUDA-only packages must not break a
normal macOS `uv sync` or detector-only test run.

## Step-by-step implementation

Each phase ends in a gate. Do not proceed because code exists; proceed because
the artifact at the gate is correct.

### Step 0 — harden the existing tracking boundary (done)

1. Treat `src/wam_drones/tracking/offline.py` and the selected
   `configs/tracking/botsort_phase4.yaml` as the existing tracker; do not replace
   it with the separate Ultralytics tracking call in general inference.
2. Run `wam-detect phase4-evaluate` on one complete development sequence and
   validate every `tracks.jsonl` row as `FrameTracks v1`.
3. Add or derive a provenance index mapping global frame ID to sequence ID,
   original one-based frame number/path, decoded dimensions, and source hash.
4. Record detector checkpoint, tracker config, protocol, thresholds, and code
   revision with the teacher artifact. The current frame contract alone does
   not carry all of that provenance.
5. Implement a read-only `vlm/teacher.py` reader for `FrameTracks`; keep
   observed, propagated, stale, age, hit, motion, and confidence information
   available to the acceptance filters.
6. Validate the `.tracks.jsonl` sidecar from `wam-detect demo-video` separately
   because its presentation visibility filter may hide tracks that should not
   define training truth.

Gate (met): one command round-trips a complete sequence, preserves track IDs
and normalized boxes, and proves every record maps one-to-one back to its
source RGB frame. The teacher must consume evaluation-grade tracks, not
filtered demo overlays.

### Step 1 — freeze the VLM contracts (done)

1. Implement Pydantic discriminated unions for `caption`, `answer`, and `point`.
2. Add explicit abstention statuses and require no point for every non-found
   point response.
3. Use `relative_0_1000_xy`; centralize pixel/grid conversion in one tested
   function.
4. Add an outer inference audit record containing image hash, source frame,
   model revision, prompt revision, preprocessing dimensions, raw generation,
   parse errors, and validated response.
5. Generate and check in the three JSON Schemas.
6. Write negative tests for extra keys, mixed coordinate order, invalid status,
   point-on-abstain, NaN, and out-of-range coordinates.

Gate (met): contract round trips are deterministic, generated schemas have no
diff, and malformed generations remain failures rather than being silently
repaired.

### Step 2 — add an image-only pretrained VLM spike (done)

1. Define a small `VLMBackend` protocol with `generate(rgb, question)`.
2. Implement the Transformers backend without importing detector code.
3. Make Qwen3-VL-2B and SmolVLM2-2.2B the primary local candidates. Keep
   Qwen3-VL-4B and Qwen3.5-4B as optional quality ceilings. Do not run
   Miril-DroneVLM-2B-2 locally on the 8 GB development GPU; reserve it for a
   separately resourced reference comparison.
4. Fix decoding settings and record exact processor/model revisions.
5. Support schema-constrained decoding when compatible, plus an unconstrained
   deterministic research mode for comparison.
6. Implement `wam-vlm infer IMAGE --question ...`.
7. Implement sampled-video inference as independent frames. Do not add temporal
   model inputs.
8. Create a smoke suite spanning captions, counts, presence, pointing, absent
   targets, ambiguous targets, and low-visibility images.

Gate (met, with one known gap): each primary local model ran the same input
contract and produced an audit JSONL with peak memory, preprocessing
resolution, latency, parse success, and semantic failures recorded. Qwen3-VL-2B
selected on measured evidence; SmolVLM2 dropped. Known gap: the smoke suite
ended at 17 cases after deduplication, short of the original 30–50-case
target, and has no genuine low-visibility example. Not worth closing before
the demo — revisit only if it starts limiting real decisions.

### Step 3 — lock splits and the evaluation set (done at MVP scope)

**MVP scope decision (2026-09-08):** implemented only the sequence-boundary
lock and a leakage guard. Deferred: the MOT training/checkpoint split, DET
perceptual-hash dedup, cross-dataset dedup, temporal sampling, and — the
largest cost — authoring 300–500 human-audited prompts. None of it is needed
until real VLM training data is actually being assembled.

What exists: `configs/vlm/data/locked_split_v1.json` preserves the exact
Phase 4 tracking split (three development sequences, four held-out
sequences) verbatim from
`evaluation/experiments/exp_20260831_phase4_tracking/meta.json`.
`src/wam_drones/vlm/splits.py` loads it and exposes
`assert_no_held_out_leakage(candidate_sequence_ids)`, which any future
sampling code (deferred data factory, smoke-suite expansion, etc.) must call
before drawing frames from a sequence. `tests/vlm/test_splits.py` covers the
loader and the guard.

Gate (MVP scope, met): `configs/vlm/data/locked_split_v1.json` is checked in,
loads deterministically, and `assert_no_held_out_leakage` rejects any of the
four held-out sequence IDs.

Gate (full scope, deferred): an automated leakage report proves that no test
sequence, source group, image hash, or near-duplicate occurs in
train/development. Do not attempt this until Step 5+ below actually starts.

### Step 4 — ship the MVP demo

Runs the already-working zero-shot Qwen3-VL-2B model (Step 2) directly
through `wam-vlm infer`. No training, adapter, or data factory needed — this
proves the capability exists, not final quality.

**Cut from a fuller showcase spec to move faster**, because none of these
affect whether the capability is real:

- Cached-mode rendering — live-only for v1.
- A scripted, narrated, multi-segment storyboard — a handful of short,
  labeled clips is enough to show the capability.
- Configurable fonts/title-cards/hold-times — hardcode something readable
  now; make it configurable only if it's reused enough to be worth it.
- The detector-baseline-comparison case — nice supporting evidence, not
  required to prove the VLM itself works. Add later if there's time.
- A separate `src/wam_drones/vlm/demo.py` module and
  `configs/vlm/demo/showcase_v1.yaml` config layer — one script calling the
  existing `wam-vlm infer` path is enough until the same rendering logic is
  needed a second time.

**Kept, because they're what makes the demo honest, not decoration:**

- Never hand-edit a JSON response to make a case look better. Replace a poor
  case with a different frame, or show the failure honestly.
- The model call must receive only RGB + text, nothing else — this is the
  entire point of the runtime boundary.
- At least one abstention case (absent or ambiguous target). "The model
  refuses to guess" is the most convincing evidence the contract has
  operational discipline, not a nice-to-have to cut.
- "Offline research — not flight control" visible somewhere in the output.

Files:

```text
scripts/render_vlm_showcase.py     # loads frames, calls wam-vlm infer, renders output
data/samples/vlm_demo/README.md    # source + licence note for the demo clip(s)
data/samples/vlm_demo/source_manifest.json
runs/vlm_demo/<run_id>/
  predictions.jsonl                # raw audit record per case — the proof it's not staged
  showcase.mp4 (or stills/gif if a rendered video isn't worth building yet)
```

1. **Only item that needs you, not me:** get 4–6 short clips or stills of
   redistributable aerial footage with a recorded source, licence, download
   date, and permitted use. VisDrone is fine for a private/internal demo,
   not for anything posted publicly — its terms are research/non-commercial.
2. For each demo case, run the real model through the existing CLI and keep
   the raw audit JSONL.
3. Minimum case set (trimmed from 7 to 4): one caption, one count or
   presence answer, one unambiguous pointing case, one abstention case.
   Add more only if it's cheap once this works.
4. Render simply: source frame beside the question and the validated JSON
   (raw JSON, then a one-line plain-language gloss). Draw a point only when
   `status == "found"`; never draw one for any abstention status.

Gate (trimmed): every displayed result traces to a real `wam-vlm infer` call
against real footage with its audit JSONL kept; nothing fabricated or
hand-edited; at least one abstention case shown, not only successes.

## Deferred: training and evaluation pipeline

Everything below is real planning, kept for when there's a concrete reason to
train — do not start it opportunistically. It assumes the demo (Step 4)
already exists and something specific about its quality needs to improve.
Resume numbering as Step 5 onward when this phase actually starts.

### Teacher / data factory

Read official DET/MOT ground truth as preferred evidence, and saved BoT-SORT
outputs for unlabelled footage; convert into a canonical supervision record
before any model-specific chat format. Compute post-VLM-resize visibility per
target. Stabilize predicted tracks using temporal hits, class consistency,
geometry, observed/stale state, and teacher agreement; reject identity-switch,
duplicate-track, ignored-region, heavy-truncation, and uncertain-count
windows. Generate task families: class presence/exact counts, coarse layout,
target-centre pointing with ambiguous/absent counterparts, short captions,
`review_region` cues from conservative occupancy, and unknown/not-visible/
ambiguous/no-candidate examples. Store teacher evidence and rejection reasons
for audit but strip them from the model-visible export. Cap prompts per image
and source-balance the dataset. Produce a review pack (images, teacher
overlays, prompts, expected JSON, rejection rationale).

Gate: manually audit a stratified sample before training; report negative-label
precision and count correctness separately for GT-derived and tracker-derived
examples.

### External aerial/remote-sensing data

Start with VRSBench for caption/VQA/grounding diversity; add AeroCaps if its
licence is acceptable; optionally a bounded RSVLM-QA slice. Keep HRVQA as a
separately reported, heavily subsampled ablation (huge, templated,
non-commercial/share-alike). Keep FloodNet-VQA evaluation-only unless its
no-derivatives terms are cleared. Keep ReVA out of first-stage training
(contains VisDrone/UAVDT and temporal questions) until sequence-level
deduplication exists. Maintain a dataset bill of materials: source URL,
revision, licence, upstream dataset, local hash, permitted use, assigned
split.

Gate: every training row maps to a licensed source image and annotation, and
no external source overwhelms VisDrone through repeated templates.

### Detector and shortcut baselines

Deterministic detector answers for supported counts/presence; a selected box
centre for supported detector-only pointing; a deliberately simple
object-list caption from detections; the same 3×3 occupancy review-region
heuristic. Add question-only/blank-image and shuffled-image VLM evaluations.
Score YOLO/BoT-SORT itself against official GT to expose the teacher ceiling.

Gate: every VLM metric table has detector-only, question-only,
shuffled-image, and pretrained-model rows, with detector-native questions and
richer semantic questions reported separately.

### Formal zero-shot model selection

Run the locked development suite on both primary local candidates with
identical image preprocessing budgets where architectures permit. Report
valid/schema JSON separately from answer correctness. Measure caption
hallucination, count accuracy/MAE, point-in-box and normalized error,
abstention precision/recall, and false-point rate. Inspect failures by
processed target size and density. Select the smallest base that meets the
locked-suite quality and abstention thresholds; run a larger candidate only
if the smaller one shows a measured capacity ceiling.

Gate: commit an experiment report and immutable config. The demo must not
decide the winner. (Note: the Step 2 zero-shot spike already selected
Qwen3-VL-2B on the 17-case smoke suite as an interim decision — this step is
the fuller, locked-suite version of that same decision.)

### First aerial SFT

A thin TRL/PEFT training entry point driven entirely by YAML. Start 4-bit
QLoRA, frozen vision tower, trainable merger/projector and LM attention/MLP
adapters. Completion-only loss; verify image tokens are never truncated.
Bounded 640-class image budget, micro-batch 1–2, gradient checkpointing,
measured accumulation. Mix roughly 55–65% audited VisDrone synthetic, 20–25%
VRSBench, 5–10% AeroCaps, 10–15% diverse/retention data, treated as an
ablation. Evaluate by sequence through 1–3 epochs; save base revision,
adapter, processor, config, dataset hashes, environment, peak VRAM. Do not
merge or quantize the only research checkpoint; retain the adapter.

Gate: the adapter improves semantic metrics over zero-shot without increasing
false presence, false point, or overconfident non-abstention. JSON validity
alone cannot pass.

### Targeted ablations

One change at a time: frozen vision vs. LoRA on the last vision blocks;
640-class vs. higher-resolution image budget; GT-only vs. GT plus stabilized
tracker supervision; Qwen3-VL-2B vs. SmolVLM2-2.2B after aerial SFT; the
selected efficient base vs. an optional 4B quality ceiling; the selected base
vs. Miril only on separately provisioned hardware. Hard-example SFT next; DPO
only from reviewed on-policy rejected/chosen pairs (hallucination and
abstention); GRPO/GSPO optional.

Gate: select the smallest configuration that materially improves the locked
suite. Do not pick a larger model because its prose looks better in the demo.

### Final evaluation

Freeze the selected checkpoint and all inference settings. Run the four
Phase 4 held-out MOT-val sequences once for the final result; report the
three tracker-development sequences separately. Report aggregate and
per-sequence metrics with sequence-level bootstrap intervals. Retain every
prompt, raw generation, validated output, parse failure, and overlay. Publish
the detector-only comparison and image-ablation results beside the VLM
result. State where the VLM adds value and where YOLO/BoT-SORT remains
better.

Gate: a reproducible report can be regenerated from pinned artifacts without
network calls other than fetching pinned model/data revisions.

## Recommended next work session

1. Source or confirm 4–6 licence-clear demo clips/stills (the one item that
   needs you, not me).
2. Write `scripts/render_vlm_showcase.py`: call `wam-vlm infer` on each demo
   case, keep the raw audit JSONL, render source frame + question + JSON +
   plain-language gloss, point overlay only on `status == "found"`.
3. Cover the 4 minimum cases (caption, count/presence, unambiguous point,
   abstention) and ship.
4. Only after that exists: decide whether anything in "Deferred: training and
   evaluation pipeline" is actually worth starting, based on what the demo
   reveals is missing.

## Still out of scope

No step above adds flight hardware, PX4/ArduPilot, MAVLink, radios, navigation,
onboard deployment, motor commands, closed-loop control, or physical landing
claims. Video remains a source of sampled independent RGB frames. YOLO and
BoT-SORT remain separate and are not rewritten or jointly trained with the VLM.
