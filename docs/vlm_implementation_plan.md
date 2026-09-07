# Aerial VLM step-by-step implementation plan

**Status:** implementation plan, not an implemented training stack

**Companion research:** [`vlm_research_brief.md`](vlm_research_brief.md)
**Runtime boundary:** one RGB frame plus text in; validated structured response out

The runtime can serve a live camera loop: submit the most recently decoded RGB
frame and a question, then consume its validated response before submitting a
later frame. It has no temporal state or future-frame access. The sampled-video
CLI is an offline measurement harness for this same single-frame interface; it
does not make the VLM retrospective or video-native.

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

1. it becomes a reusable package for projects that do not use this detector;
2. it needs an independent release cadence or public model/data release;
3. its licence obligations conflict with the detector repository;
4. CI and dependency weight make normal detector development impractical; or
5. a stable versioned interface package already exists between the repos.

If that happens, first extract a tiny `wam-perception-contracts` package (or
published JSON Schemas), then split the VLM. Do not copy contract definitions
between repositories.

## Target repository structure

Evolve toward this layout incrementally:

```text
configs/
  datasets/                    # existing dataset provenance
  detection/                   # migrate only when touching old config paths
  tracking/                    # actual BoT-SORT/ByteTrack configs
  vlm/
    models/
    data/
    training/
    eval/
    demo/

data/
  manifests/                   # existing versioned source manifests
  processed/vlm/               # ignored generated supervision
  samples/                     # tiny checked-in test fixtures only

interfaces/schemas/
  detection_v1.schema.json     # existing
  frame_tracks_v1.schema.json  # existing
  vlm_response_v1.schema.json
  vlm_supervision_record_v1.schema.json
  vlm_inference_record_v1.schema.json

src/wam_drones/
  dataset/                     # existing converters/manifests/splits
  detection/                   # existing YOLO code, unchanged
  tracking/                    # existing offline BoT-SORT/ByteTrack implementation
  media/                       # shared decoded-frame metadata, added only if useful
  vlm/
    contracts.py
    prompts.py
    frame_source.py
    inference.py
    structured_decode.py
    teacher.py
    data_factory.py
    splits.py
    baselines.py
    evaluation.py
    demo.py
    cli.py

scripts/
  generate_schemas.py          # extend existing generator
  render_vlm_showcase.py       # thin executable wrapper

tests/
  vlm/
    test_contracts.py
    test_teacher.py
    test_data_factory.py
    test_splits.py
    test_inference.py
    test_evaluation.py
    test_demo.py

evaluation/experiments/
  exp_YYYYMMDD_vlm_zero_shot/
  exp_YYYYMMDD_vlm_sft/
  exp_YYYYMMDD_vlm_final_eval/

runs/vlm_demo/                 # ignored rendered videos and audit artifacts
```

Do not perform a large config migration at the same time as the VLM work. New
VLM files can use `configs/vlm/`; existing detector paths should remain stable.

In `pyproject.toml`, keep the default install lightweight and add optional
groups rather than placing Torch/Transformers in core dependencies:

- `vlm-inference`: Transformers, Accelerate, image processor, constrained
  decoding backend;
- `vlm-train`: inference group plus TRL, PEFT, bitsandbytes and datasets;
- `vlm-eval`: metric and report dependencies; and
- `vlm-demo`: OpenCV/Pillow plus the chosen video/text renderer.

Pin complete working versions in `uv.lock`. CUDA-only packages must not break a
normal macOS `uv sync` or detector-only test run.

## Step-by-step implementation

Each phase ends in a gate. Do not proceed because code exists; proceed because
the artifact at the gate is correct.

### Step 0 — harden the existing tracking boundary

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

Gate: one command round-trips a complete sequence, preserves track IDs and
normalized boxes, and proves every record maps one-to-one back to its source
RGB frame. The teacher must consume evaluation-grade tracks, not filtered demo
overlays.

### Step 1 — freeze the VLM contracts

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

Gate: contract round trips are deterministic, generated schemas have no diff,
and malformed generations remain failures rather than being silently repaired.

### Step 2 — add an image-only pretrained VLM spike

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
8. Create a 30–50-case smoke suite spanning captions, counts, presence,
   pointing, absent targets, ambiguous targets, and low-visibility images.

Gate: each primary local model can run the same input contract and produces an
audit JSONL. Peak memory, preprocessing resolution, latency, parse success,
and semantic failures are recorded. Select no training base yet from showcase
examples.

### Step 3 — lock splits and the evaluation set

**MVP scope decision (2026-09-08):** priority shifted to getting the product
to exist before investing in a research-grade evaluation suite. Items 1 and
8 (the sequence-boundary lock and a leakage guard) were implemented now,
scoped down from a full automated report to a single checked-in split file
plus a guard function future data-factory code must call. Items 2–7 (MOT
training/checkpoint split, DET perceptual-hash dedup, cross-dataset dedup,
temporal sampling, and — the largest cost — authoring 300–500 human-audited
prompts) are deferred until real VLM training data is actually being
assembled (Step 4+), not attempted speculatively ahead of that need. The
Step 2 smoke suite (17 cases, `configs/vlm/smoke_suite_v1.json`) remains the
working sanity check until then.

What exists now: `configs/vlm/data/locked_split_v1.json` preserves the exact
Phase 4 tracking split (three development sequences, four held-out
sequences) verbatim from
`evaluation/experiments/exp_20260831_phase4_tracking/meta.json`.
`src/wam_drones/vlm/splits.py` loads it and exposes
`assert_no_held_out_leakage(candidate_sequence_ids)`, which any future
sampling code (Step 4 data factory, smoke-suite expansion, etc.) must call
before drawing frames from a sequence, so the four held-out sequences can
never silently enter training or prompt-authoring data. `tests/vlm/test_splits.py`
covers the loader and the guard.

The full original scope below is the target once Step 4+ actually needs it —
do not skip straight to authoring the 300–500-prompt suite without expanding
this scope decision first.

1. Preserve the Phase 4 split: reserve its four held-out MOT-val sequences for
   final VLM testing and keep its three development sequences as diagnostics.
   — **done at MVP scope**, see above.
2. Split complete MOT-train sequences into VLM training and checkpoint
   development groups. — deferred.
3. Group DET stills by the existing scene proxy, then join perceptual-hash
   duplicates into connected components. — deferred.
4. Deduplicate across DET, MOT, and any external dataset before generating
   questions. — deferred.
5. Sample temporally separated frames; cap frames per sequence. — deferred.
6. Author 300–500 human-audited prompts independently from training templates.
   — deferred.
7. Balance task, class, count, density, processed object size, occlusion,
   positive, negative, ambiguous, and unknown cases. — deferred.
8. Hash the suite and prompt file. Never hard-mine or tune against final test.
   — **partially done at MVP scope**: the leakage guard exists
   (`assert_no_held_out_leakage`); the full automated report and suite/prompt
   hashing are deferred to when item 6 actually produces a suite to hash.

Gate (full scope, deferred): an automated leakage report proves that no test
sequence, source group, image hash, or near-duplicate occurs in
train/development.

Gate (MVP scope, met): `configs/vlm/data/locked_split_v1.json` is checked in,
loads deterministically, and `assert_no_held_out_leakage` rejects any of the
four held-out sequence IDs — verified in `tests/vlm/test_splits.py`.

### Step 4 — implement the teacher/data factory

1. Read official DET/MOT ground truth as the preferred evidence.
2. Read saved BoT-SORT outputs for unlabelled recorded footage.
3. Convert evidence into a canonical supervision record before producing any
   model-specific chat format.
4. Compute post-VLM-resize visibility for each target.
5. Stabilize predicted tracks using temporal hits, class consistency, geometry,
   observed/stale state, and teacher agreement.
6. Reject identity-switch, duplicate-track, ignored-region, heavy truncation,
   and uncertain-count windows.
7. Generate task families:
   - class presence and exact counts;
   - coarse layout and relative-position answers;
   - target-centre pointing with ambiguous/absent counterparts;
   - short factual captions;
   - `review_region` cues based on conservative occupancy; and
   - unknown/not-visible/ambiguous/no-candidate examples.
8. Store teacher evidence and rejection reasons for audit but strip them from
   the model-visible export.
9. Cap prompts per image and source-balance the resulting dataset.
10. Produce a review pack with images, teacher overlays, prompts, expected JSON,
    and rejection rationale.

Gate: manually audit a stratified sample before training. Negative-label
precision and count correctness must be reported separately for GT-derived and
tracker-derived examples.

### Step 5 — add external aerial/remote-sensing data

1. Start with VRSBench for caption/VQA/grounding diversity.
2. Add AeroCaps if its exact repository licence is acceptable.
3. Optionally add a bounded RSVLM-QA slice for richer VQA.
4. Keep HRVQA as a separately reported, heavily subsampled ablation because it
   is huge, templated, and non-commercial/share-alike.
5. Keep FloodNet-VQA evaluation-only unless its no-derivatives terms are cleared
   for the intended training use.
6. Keep ReVA out of first-stage training because it contains VisDrone/UAVDT and
   temporal questions; use it later only after sequence-level deduplication.
7. Maintain a dataset bill of materials with source URL, revision, licence,
   upstream dataset, local hash, permitted use, and assigned split.

Gate: every training row maps to a licensed source image and annotation, and no
external source overwhelms VisDrone through repeated templates.

### Step 6 — build the detector and shortcut baselines

1. Implement deterministic detector answers for supported counts/presence.
2. Use a selected box centre for supported detector-only pointing questions.
3. Generate a deliberately simple object-list caption from detections.
4. Implement the same 3×3 occupancy review-region heuristic.
5. Add question-only/blank-image and shuffled-image VLM evaluations.
6. Score YOLO/BoT-SORT itself against official GT to expose the teacher ceiling.

Gate: every VLM metric table has detector-only, question-only, shuffled-image,
and pretrained-model rows. Detector-native questions and richer semantic
questions are reported separately.

### Step 7 — run the zero-shot model selection

1. Run the locked development suite on both primary local candidates.
2. Use identical image preprocessing budgets where architectures permit.
3. Report valid/schema JSON separately from answer correctness.
4. Measure caption hallucination, count accuracy/MAE, point-in-box and normalized
   error, abstention precision/recall, and false-point rate.
5. Inspect failures by processed target size and density.
6. Select the smallest base that meets the locked-suite quality and abstention
   thresholds. Run a larger optional candidate only when the smaller models
   show a measured capacity ceiling.

Gate: commit an experiment report and immutable config. The demo must not decide
the winner.

### Step 8 — train the first aerial SFT

1. Add a thin TRL/PEFT training entry point driven entirely by YAML.
2. Begin with 4-bit QLoRA, frozen vision tower, trainable merger/projector, and
   LM attention/MLP adapters.
3. Use completion-only loss and verify image tokens are never truncated.
4. Start at a bounded 640-class image budget, micro-batch 1–2, gradient
   checkpointing, and measured accumulation.
5. Mix roughly 55–65% audited VisDrone synthetic, 20–25% VRSBench, 5–10%
   AeroCaps, and 10–15% diverse/retention data, then treat the mix as an
   ablation.
6. Evaluate by sequence throughout 1–3 epochs and save the exact base revision,
   adapter, processor, config, dataset hashes, environment, and peak VRAM.
7. Do not merge or quantize the only research checkpoint; retain the adapter.

Gate: the adapter improves semantic metrics over zero-shot without increasing
false presence, false point, or overconfident non-abstention. JSON validity
alone cannot pass.

### Step 9 — run targeted ablations

Run one change at a time:

1. frozen vision versus LoRA on the last vision blocks;
2. 640-class versus higher-resolution image budget;
3. GT-only versus GT plus stabilized tracker supervision;
4. Qwen3-VL-2B versus SmolVLM2-2.2B after aerial SFT;
5. the selected efficient base versus an optional 4B quality ceiling; and
6. the selected base versus Miril only on separately provisioned hardware.

Use hard-example SFT next. Add DPO only from reviewed on-policy rejected/chosen
pairs, primarily for hallucination and abstention. GRPO/GSPO remains optional.

Gate: select the smallest configuration that materially improves the locked
suite. Do not pick a larger model because its prose looks better in the demo.

### Step 10 — run final evaluation

1. Freeze the selected checkpoint and all inference settings.
2. Run the four Phase 4 held-out MOT-val sequences once for the final VLM
   result; report the three tracker-development sequences separately.
3. Report aggregate and per-sequence metrics with sequence-level bootstrap
   intervals.
4. Retain every prompt, raw generation, validated output, parse failure, and
   overlay.
5. Publish the detector-only comparison and image-ablation results beside the
   VLM result.
6. State where the VLM adds value and where YOLO/BoT-SORT remains better.

Gate: a reproducible report can be regenerated from pinned artifacts without
network calls other than fetching pinned model/data revisions.

## Showcase demo implementation

The showcase is a renderer over the same inference and contracts, not a second
demo-only inference path.

### Files

```text
scripts/render_vlm_showcase.py
src/wam_drones/vlm/demo.py
configs/vlm/demo/showcase_v1.yaml
data/samples/vlm_demo/README.md
data/samples/vlm_demo/source_manifest.json
runs/vlm_demo/<run_id>/
  showcase.mp4
  predictions.jsonl
  render_manifest.json
  frames/
  overlays/
```

The source clip should be redistributable drone footage with a recorded URL,
author, licence, download date, SHA-256, and permitted showcase use. VisDrone
may be used for a private research demo, but its research/non-commercial terms
make it a poor default for a broadly posted promotional video. Until original
footage exists, use explicitly licensed/public-domain aerial footage.

### Runnable interface

```shell
uv run python scripts/render_vlm_showcase.py \
  --config configs/vlm/demo/showcase_v1.yaml \
  --input path/to/licensed_aerial_clip.mp4 \
  --model path/to/selected_adapter \
  --output runs/vlm_demo/showcase_v1/showcase.mp4
```

The config specifies:

- source crop/resize policy and frame sample timestamps;
- a fixed prompt deck and expected response type per timestamp;
- model, processor, adapter, schema, and prompt revisions;
- deterministic generation settings;
- font, colors, title cards, hold times, and output FPS/resolution;
- whether to use live generation or previously audited cached generations; and
- optional detector/teacher explainer segments, clearly separated from VLM
  runtime segments.

Cached mode is appropriate for rendering and editing a reproducible showcase,
but the manifest must say it is cached. A separate `--live` run demonstrates
that the same results come from the model. Never hand-edit JSON to make a demo
pass; replace a poor case or show the failure honestly.

### Visual layout

Use a 16:9 1080p canvas:

```text
┌────────────────────────────────┬──────────────────────────┐
│                                │  INPUT                   │
│        original RGB frame      │  “Where is the bus?”     │
│                                │                          │
│     optional point overlay     │  OUTPUT                  │
│                                │  validated JSON          │
│                                │  plain-language summary  │
└────────────────────────────────┴──────────────────────────┘
       VLM input: RGB frame + text only • offline research
```

Display raw JSON briefly, then highlight the parsed fields used to draw the
point or abstention card. Use a compact crosshair only for `status=found`.
Never draw a point for `not_visible`, `ambiguous`, `unknown`, or `no_candidate`.
Label a review-region cue as “operator review cue—not a safe landing claim.”

### Demonstration cases

Choose different held-out or licence-cleared frames for each capability:

1. **Scene description:** ask for a concise overview that includes useful
   context beyond the ten detector classes.
2. **Count:** ask for one VisDrone class. Show the detector baseline beside the
   VLM so this is not presented as unique VLM value.
3. **Spatial VQA:** ask where objects are concentrated or how two visible
   regions relate; answer in language.
4. **Pointing:** request one unambiguous, model-visible target and draw its
   returned 0–1000 point after conversion.
5. **Absent target:** ask for an object not present and show a null point.
6. **Ambiguous target:** ask for “the car” when several similar cars are visible;
   show that the model refuses to choose arbitrarily.
7. **Review cue:** request a visually unobstructed area for human inspection and
   display the safety disclaimer.

Do not use only successful positives. The absent and ambiguous cases are the
most persuasive evidence that the contract has operational discipline.

### Suggested 90-second showcase storyboard and narration

**0–8 seconds — title**

Visual: clean aerial footage and title “Aerial perception, now queryable.”

Narration: “This project already detects objects and maintains identities in
recorded aerial video. I added a research vision-language layer for asking
questions about individual frames.”

**8–18 seconds — existing geometry system**

Visual: YOLO boxes and track IDs for a short sequence.

Narration: “YOLO and BoT-SORT remain the source of geometry and identity. They
are not replaced.”

**18–28 seconds — teacher/runtime boundary**

Visual: boxes become example question/JSON pairs, then the boxes disappear.
Large label: “Training supervision only.”

Narration: “During dataset creation, stable tracks teach counts, presence, and
coarse locations. At inference, the model receives no boxes, counts, or track
IDs—only one RGB frame and text.”

**28–42 seconds — caption and VQA**

Visual: full frame, question, validated caption/answer JSON.

Narration: “It can summarize the visible scene and answer structured questions.
The response is validated against a versioned JSON schema.”

**42–55 seconds — pointing**

Visual: unambiguous target request, JSON point, converted crosshair.

Narration: “For review tasks, it can return a coarse image point on a visible
target. Coordinates are resolution-independent and remain separate from the
tracker.”

**55–68 seconds — abstention**

Visual: absent then ambiguous prompts; clear `not_visible` and `ambiguous`
cards with no reticle.

Narration: “Just as importantly, it can say that a target is absent, ambiguous,
or not reliably visible instead of inventing a point.”

**68–79 seconds — baseline comparison**

Visual: count result beside deterministic detector-only result; then a semantic
question the detector cannot express.

Narration: “The detector remains the stronger baseline for raw counts. The VLM
is useful where language and scene context add something the class list cannot.”

**79–90 seconds — scope and close**

Visual: architecture boundary and final project name.

Narration: “This is offline research on recorded frames. It is not a flight
controller, a landing system, or a safety sensor. Every result remains available
for human review and reproducible evaluation.”

### Demo quality gates

Before publishing:

- regenerate the video from a clean environment and pinned artifacts;
- verify the model call receives only RGB plus text;
- validate every displayed JSON response;
- visually confirm all coordinate transforms after crop/letterbox reversal;
- include at least one abstention and one honest limitation;
- show the detector baseline for detector-native claims;
- retain the source licence manifest and output hashes;
- add subtitles and ensure JSON is readable at normal playback speed; and
- put “offline research—not flight control” in the video and description.

## Recommended first work session

The next implementation session should stop after Steps 0–2:

1. validate the existing BoT-SORT output and add the frame-provenance join;
2. implement and test `vlm_response_v1` plus the audit envelope;
3. add the Transformers backend and `wam-vlm infer`;
4. run the two primary local models on 30–50 fixed frames; and
5. write the zero-shot spike report with actual memory and latency.

That provides enough evidence to choose the training base and enough reusable
infrastructure for the showcase. Building the data factory or trainer before
this spike would prematurely lock the project to one model family.

## Still out of scope

No step above adds flight hardware, PX4/ArduPilot, MAVLink, radios, navigation,
onboard deployment, motor commands, closed-loop control, or physical landing
claims. Video remains a source of sampled independent RGB frames. YOLO and
BoT-SORT remain separate and are not rewritten or jointly trained with the VLM.
