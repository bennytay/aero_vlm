# Aerial VLM research brief and implementation outline

**Status:** research and design only

**Date:** 2026-09-08

**Scope:** offline inference on one recorded RGB frame plus one text request

## Decision in one page

Build a separate `wam_drones.vlm` research layer. It may consume detections and
tracks while creating training data and evaluation references, but the VLM
runtime must accept only the RGB frame and text. It must not be inserted into,
or become a dependency of, the YOLO/BoT-SORT path.

Start with a locked zero-shot comparison of:

1. `Qwen/Qwen3-VL-4B-Instruct`;
2. `Qwen/Qwen3.5-4B`; and
3. `MirilAI/Miril-DroneVLM-2B-2`.

The first model I would fine-tune is **Qwen3-VL-4B-Instruct**. It is Apache-2.0,
has a current Transformers implementation, was designed for visual grounding,
and has a mature official multimodal fine-tuning path with independent support
in TRL, Unsloth, Axolotl, vLLM, MLX, and llama.cpp. This is a tooling and
experimental-control decision, not a claim that it has the highest 2026 vendor
benchmark score.

The mandatory same-size ablation is **Qwen3.5-4B**. Its native early-fusion
vision-language architecture and reported spatial scores are compelling, and
it may simply be the better base. It is newer, so promote it to the main model
only after (a) a repeatable LoRA/QLoRA smoke test, and (b) a win on the locked
VisDrone suite. The stronger-size check is Qwen3.5-9B or Qwen3-VL-8B. The
reported 4B-to-9B improvement is real but modest on generic visual benchmarks,
so it does not justify doubling the first experiment's cost before aerial
testing.

Use **QLoRA SFT with the vision tower frozen**, training language adapters plus
the multimodal merger/projector first. Then ablate LoRA on the last vision
blocks at a much lower learning rate. Do not begin with full fine-tuning. Do
hard-example SFT before DPO; reserve DPO for on-policy, reviewed preference
pairs. Do not begin with GRPO/GSPO.

Use a **versioned discriminated-union JSON Schema**, enforced at decode time
when the runtime supports it and always validated with Pydantic. Do not use
tool-calling: no tool is being invoked, and model-specific tool tokens would
make the research contract less portable. Use integer `x,y` coordinates on a
resolution-independent 0–1000 grid, in ordinary `xy` order. A point is a review
cue, never a control target.

Most importantly, do not call empty-looking image regions “safe.” Detection
boxes can supervise **visually unobstructed review regions**, not ground-plane
safety, clearance, slope, wires, surface support, or landing suitability.

## What was inspected in this checkout

The repository was inspected at `/Users/tayy/dev/wam_drones` on `main`, first
at Phase 3 and then reconciled with the Phase 4/5 work through base commit
`2e3c075` before this brief was committed.

- The frozen ten-class vocabulary is in
  `src/wam_drones/detection/vocabulary.py` and
  `configs/vocabulary_detection_v1.yaml`: `pedestrian`, `person`, `bicycle`,
  `car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, `motor`.
- `Detection`/`FrameDetections` and `TrackObservation`/`FrameTracks` are strict,
  immutable Pydantic contracts in `src/wam_drones/detection/contracts.py`.
  Boxes are normalized `xyxy`; track rows include ID, velocity, age, hits,
  time-since-update, observed/stale flags, and capture time.
- The matching JSON Schemas already exist under `interfaces/schemas/`.
- Folder inference uses stable lexical image order. Video inference uses
  OpenCV `VideoCapture`, increments a zero-based `frame_id`, and yields one
  frame at a time; FPS falls back to 30 when metadata is invalid.
- `wam-detect run` writes `predictions.jsonl`, `metrics.json`, annotations, and
  an optional MP4. That legacy/general inference path serializes
  `FrameDetections`; its internal Ultralytics `model.track(...)` call does not
  define the project's selected offline tracker boundary.
- Phase 4 adds the native offline tracker in
  `src/wam_drones/tracking/offline.py`. It supports ByteTrack and BoT-SORT;
  `configs/tracking/botsort_phase4.yaml` fixes BoT-SORT with camera-motion
  compensation and ReID disabled.
- `wam-detect phase4-evaluate` and the Phase 5 path serialize validated
  `FrameTracks` records to `tracks.jsonl`. `wam-detect demo-video` also writes
  an adjacent `.tracks.jsonl` and `.metrics.json` for a source video. This is
  the concrete teacher interface; the VLM work should read it rather than
  changing tracker behavior.
- The Phase 4 development comparison selected BoT-SORT over ByteTrack by HOTA
  (0.4136 versus 0.3566). Its four-sequence held-out report records HOTA 0.4076,
  IDF1 0.8591, and track recall 0.3719. The modest recall and remaining identity
  switches make temporal stability filters and official GT essential for
  synthetic-label quality.
- VisDrone-DET train/val/test and VisDrone-MOT train/val are on disk. MOT full
  annotations contain target IDs, pixel boxes, score, class, truncation,
  occlusion, and ignore regions. The manifests contain 56 MOT-train sequences
  (24,201 frames) and 7 MOT-val sequences (2,846 frames).
- The completed 100-epoch Phase 3 run reports mAP50-95 0.1676 and recall
  0.4063, up from 0.0449 and 0.1088 for the COCO baseline. It also reports
  12.329 unmatched predictions per frame. Detector predictions should still be
  treated as a noisy teacher; official VisDrone annotations are the stronger
  reference where available.
- The repository already found DET/MOT cross-split near-duplicates. VLM split
  construction must extend the existing hash-based checks rather than assume
  the official task splits are independent across tracks.
- The repo host is macOS and the detector run used MPS. For planning, I assume
  development/quantized inference on Apple Silicon and training on one Linux
  NVIDIA GPU with about 24 GB VRAM. A CUDA training environment is not present
  in this checkout.

## Model survey

Vendor benchmark numbers below are useful screening signals, not evidence of
VisDrone performance. Resolution, total multimodal parameter count, image-token
budget, prompt, and quantization can dominate the apparent size comparison.

| Candidate | Class | Why it is relevant | Main reservations | Practical memory envelope |
|---|---:|---|---|---|
| Qwen3-VL-4B-Instruct | small | Apache-2.0; dynamic-resolution image path; strong native grounding; official controls for vision/merger/LLM tuning; broad Transformers/vLLM/MLX/GGUF support | Superseded on some generic scores by Qwen3.5; aerial tiny objects are unproven | BF16 inference roughly 10–14 GB; 4-bit roughly 5–8 GB; 24 GB is a sensible QLoRA target at bounded image resolution |
| Qwen3.5-4B | small | Apache-2.0; current native multimodal model; vendor reports strong counting, RefCOCO, and spatial results; Transformers/vLLM/SGLang support | Newer training/runtime path; less public task-specific fine-tuning history; vendor scores are not aerial scores | Similar 4B planning envelope, but measure activations before committing a full run |
| SmolVLM2-2.2B-Instruct | tiny/edge | Apache-2.0; low-cost Transformers baseline; still-image and video-aware family | Expected quality and precise grounding ceiling is lower; not aerial-specialized | About 5 GB-class BF16 weights before activations; comfortable quantized inference; easiest low-VRAM SFT control |
| Gemma 4 E2B-it / Miril-DroneVLM-2B-2 | edge-oriented | Apache-2.0 in the current release; Gemma 4 is multimodal; Miril implements the exact detector-labels-to-image/text/JSON pattern | “E2B” is not total model size—the HF artifact reports about 5B parameters; Miril uses WALDO ontology, has limited adoption, and its technical report is not peer reviewed | Gemma/Miril card reports about 5 GB for E2B runtime in one MLX implementation; allow materially more for BF16 training activations |
| PaliGemma 2 3B | task-transfer baseline | Designed to be fine-tuned for downstream vision-language tasks; fixed 224/448/896 variants make resolution ablations clean | Older, gated, Gemma terms rather than Apache-2.0, and less natural as a conversational structured-output model | 3B BF16 weights are about 6 GB; 448/896 image activations determine the real training requirement |
| InternVL3.5-4B/8B | runner-up family | Apache-2.0, strong general multimodal family, open staged checkpoints | 4B-named model is about 5B total; historically more custom-code friction; adds another architecture before the core question is answered | 4B/5B QLoRA should fit 24 GB; 8B is more comfortable at 32–48 GB for high-resolution work |
| Qwen3.5-9B or Qwen3-VL-8B | stronger | Better language/spatial capacity and a clean scale ablation | Generic 4B-to-9B gains are modest on several relevant metrics; image-token activations make 24 GB tight | 4-bit inference roughly 8–12 GB; QLoRA may fit 24 GB only with batch 1, checkpointing, and capped pixels; 32–48 GB is the comfortable research target |

The envelopes are planning estimates, not promises. A VLM's image-token count
and trainable vision path make text-only “minimum VRAM” tables optimistic. The
spike must record peak allocated/reserved VRAM for each resolution and backend.

Primary sources:

- [Qwen3-VL-4B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
  and [official fine-tuning code](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and
  [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) model cards
- [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- [PaliGemma 2 3B](https://huggingface.co/google/paligemma2-3b-pt-448)
- [InternVL3.5-4B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-4B-Instruct)
- [Gemma 4 E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [Miril-DroneVLM-2B-2](https://huggingface.co/MirilAI/Miril-DroneVLM-2B-2)

### General base versus remote-sensing model

Start from a current general model and add aerial SFT. Remote-sensing models
such as GeoChat, LHRS-Bot, EarthGPT, and SkyEyeGPT contributed valuable data and
task formulations, but many are older 7B LLaVA/Vicuna-era stacks aimed at
satellite/orthophoto imagery. Domain naming alone does not guarantee stronger
counting or grounding. The 2025
[CHOICE benchmark](https://proceedings.neurips.cc/paper_files/paper/2025/file/befe25a01cf4dbe9635e85f835d31250-Paper-Datasets_and_Benchmarks_Track.pdf)
shows a mixed result: current general VLMs are often stronger overall, while
some remote-specific models retain advantages on particular tasks. Likewise,
[GEOBench-VLM](https://openaccess.thecvf.com/content/ICCV2025/papers/Danish_GEOBench-VLM_Benchmarking_Vision-Language_Models_for_Geospatial_Tasks_ICCV_2025_paper.pdf)
finds that density, tiny objects, localization, and temporal reasoning remain
hard across both groups.

Miril is the most relevant domain checkpoint because its runtime really is
image plus text and its WALDO labels are training-time evidence. It should be
included in the zero-shot suite and optionally used as an initialization
ablation. It should not be the sole starting point: its ontology differs from
VisDrone, its public evidence is one organization’s 2026 technical report, and
its own BF16 results show that valid JSON can coexist with mediocre precise
point recovery. That is exactly why this project needs its own locked suite.

## Training recipe

### Stage 1: supervised adaptation

Use Transformers + TRL + PEFT as the reference implementation. TRL directly
supports single-image conversational/prompt-completion VLM data; importantly,
its documentation warns against truncating away image tokens. Keep the
reference path simple enough to inspect, and use Unsloth only as a measured
memory/speed backend. Axolotl is reasonable once a stable config is desired.
llama.cpp and MLX are deployment/evaluation paths, not the canonical trainer.

Reference: [TRL VLM SFT documentation](https://huggingface.co/docs/trl/en/sft_trainer),
[Unsloth Qwen3-VL guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune),
and [Axolotl multimodal support](https://github.com/axolotl-ai-cloud/axolotl).

Initial recipe:

- 4-bit NF4 QLoRA for the language model, rank 16 or 32, adapters on attention
  and MLP projections;
- train the small multimodal merger/projector; freeze the vision tower;
- completion-only loss—never learn to reproduce the system or user prompt;
- BF16 compute on supported CUDA hardware, gradient checkpointing, micro-batch
  1–2, and gradient accumulation to the measured stable effective batch;
- 1–3 epochs with frequent evaluation by sequence, not by random row;
- cap image pixels explicitly and start near the detector's 640-pixel scale,
  then test a higher-resolution condition. Never silently resize differently
  between training and evaluation;
- source-balanced batches rather than concatenating million-row templated
  datasets and letting them dominate; and
- retain a small general-image instruction slice if general VQA regression is
  observed.

After the first result, compare:

1. frozen vision + trainable merger + LM LoRA;
2. vision LoRA on the last blocks at roughly one-tenth the LM adapter learning
   rate; and
3. higher input resolution with the vision tower still frozen.

This separates “the visual features are inadequate” from “the model was shown
too few pixels.” Full vision-tower training combines both changes and is a poor
first diagnostic. Full-parameter fine-tuning is not justified by a modest,
partly synthetic corpus on one GPU; revisit it only after LoRA saturates, the
labels pass audit, and at least 48–80 GB or multi-GPU capacity is available.

### Preference/RL stages

Do **hard-example mining followed by another short SFT pass** first. Generate
on the train split, select false-presence, wrong-count, wrong-route,
ambiguous-point, too-small-target, and malformed/null-coordinate failures, have
the target corrected from teacher evidence plus human review, and mix them back
without oversampling one sequence.

DPO is worthwhile only after there are on-policy pairs from the actual SFT
checkpoint: same image/prompt, rejected model answer, and reviewed correction.
Current VLM work shows that DPO can reduce hallucination, but the result depends
strongly on how close the preference data is to the policy being optimized;
see [OPA-DPO (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Mitigating_Hallucinations_in_Large_Vision-Language_Models_via_DPO_On-Policy_Data_CVPR_2025_paper.pdf).
Use DPO mainly for abstention, false-premise rejection, and concise contract
semantics—not as a substitute for missing visual resolution.

GRPO/GSPO is technically available for VLMs, but it is not the second milestone.
It multiplies generation cost and invites reward hacking. The only clean rewards
here are exact count/presence, point distance, status correctness, and schema
validity; rewarding schema heavily would teach formatting rather than vision.
Consider a narrow later experiment only if SFT plateaus and these rewards are
evaluated on independent sequences. TRL exposes custom multimodal reward
functions, but support is not evidence that the method is needed:
[TRL GRPO documentation](https://huggingface.co/docs/trl/en/grpo_trainer).

## Output contract

Use JSON because it is portable, auditable, easy to version beside the existing
Pydantic contracts, and supported by constrained decoders. Use JSON **Schema**,
not “please return JSON” prompt wording. vLLM supports schema-guided output, and
llguidance is also integrated into llama.cpp/SGLang/vLLM. Constrained decoding
guarantees syntax, not visual truth, so report semantic metrics separately.

References: [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)
and [llguidance](https://github.com/guidance-ai/llguidance).

Recommended discriminated union, abbreviated:

```json
{
  "schema_version": "vlm_response_v1",
  "type": "point",
  "status": "found",
  "description": "The requested car is left of the road centre.",
  "point": {
    "x": 238,
    "y": 604,
    "coordinate_system": "relative_0_1000_xy",
    "semantics": "target_center"
  }
}
```

The three variants should be:

- `caption`: `status = ok|unknown`, `text` required only for `ok`;
- `answer`: `status = ok|not_visible|ambiguous|unknown`, plus a typed answer
  `{kind: boolean|count|label|text, value: ...}` only for `ok`; and
- `point`: `status = found|not_visible|ambiguous|unknown|no_candidate`, plus a
  point only for `found`.

Point semantics are `target_center` or `review_region`. `review_region` means a
coarse visual inspection cue, not “free space” or a landing location. Do not
emit model self-confidence; it is not calibrated evidence. Keep the raw
generation, validated result, model revision, prompt revision, image hash, and
preprocessing metadata in an outer audit record, but do not put source paths,
frame IDs, detector counts, or track IDs in the model-visible prompt.

Coordinate rules:

- integer `x` is left-to-right and `y` is top-to-bottom, both inclusive 0–1000;
- use `xy`, not the Gemma-specific `[y,x]` convention;
- the application owns letterbox/crop inversion and pixel conversion;
- training targets normally use box centres, with an optional visible-interior
  point for heavily clipped boxes;
- coordinates must be null/absent for every non-`found` status; and
- a point is drawable only after schema and semantic-status validation.

Tool-calling is the wrong abstraction for the first version. It wraps the same
JSON in model/runtime-specific control tokens, suggests an action side effect,
and complicates cross-model evaluation. Plain text plus a repair parser is also
inferior: silent repairs destroy the distinction between model failure and
application behavior. A failed parse should remain a failed parse in the audit.

## Data factory: tracks to language supervision

### Canonical record

Create a versioned, provenance-rich record before exporting to any model's chat
format:

```text
sample_id, image_path, image_sha256, dataset, source_split,
sequence_id, frame_number, split, task_family, question,
expected_response, prompt_template_id, teacher_source,
teacher_evidence, label_confidence, rejection_reasons
```

`teacher_evidence` may contain boxes, track IDs, detector revision, temporal
window, and filtering decisions. It is never serialized into the model input.
The final training view contains only image, system contract, question, and
expected response.

### Evidence priority

1. Official VisDrone annotations, including MOT target IDs and
   occlusion/truncation.
2. Stable BoT-SORT tracks over fine-tuned YOLO detections for recorded data
   without ground truth.
3. Single-frame detector output only for positive, high-confidence facts—not
   for definitive absence.

Keep provenance so results can be stratified by teacher source. Synthetic data
is not ground truth merely because it was produced deterministically.

### Stabilizing tracks

For every candidate anchor frame, inspect a symmetric temporal window only in
the data factory. Accept a track-derived object when it has sufficient hits,
is observed at the anchor, is not stale, has stable majority class, has
consistent box motion/scale, and exceeds a model-visible size threshold after
the exact VLM resize. Merge likely duplicate tracks and reject identity-switch
windows. Use robust median confidence and geometry rather than one-frame peaks.

Do not copy tracker flicker into labels:

- debounce entry/exit with multiple observations;
- derive anchor-frame count from stable tracks present at that anchor, not the
  union over a window;
- reject count questions when teacher versions disagree, tracks cross/merge,
  or ignored/occluded regions make completeness doubtful;
- never derive a negative-presence label from one detector miss; and
- sample at most a bounded number of correlated frames per track/sequence.

### Task generators

**Presence and counts.** Generate exact questions only from complete GT or
stable, high-recall temporal evidence. Balance zero, one, few, and dense counts;
balance classes rather than reflecting VisDrone's car frequency. Add
counterfactual absent-class prompts, but only where absence is trustworthy.

**Spatial VQA and pointing.** Geometry can support `left/right/top/bottom`,
nearest-to-edge, largest visible instance, and box-centre questions when the
relation has a clear margin. An underspecified request with several matching
objects must produce `ambiguous` and no point. Color, make/model, intent, and
activity cannot be inferred from a track class and should not be synthesized
without separate reviewed image evidence.

**Captions.** Template captions are useful for coverage but dangerous as the
whole corpus: they teach a detector verbalizer. Produce short factual scene
summaries from audited object facts and coarse layout, then mix human-authored
aerial captions. Any richer LLM-generated caption must be checked against the
known object inventory and sampled for human review. Preserve the broad
VisDrone ontology (`motor`, `person`, etc.) rather than inventing narrower nouns.

**Hazards and review regions.** Boxes can generate statements such as “people
and vehicles occupy the centre-right” or “the frame is densely occupied.” They
cannot prove collision or landing safety. A coarse candidate review region may
be the least-occupied 3×3 cell after dilating reliable boxes and excluding
image borders/ignored regions. Label it `review_region`, and prefer `unknown` or
`no_candidate` when the field of view, occlusion, or scene type is unsuitable.
A later segmentation dataset plus human review can improve this, but it still
does not create physical safety evidence.

**Abstention.** Include absent targets, several indistinguishable targets,
attributes unavailable from the image, objects below the post-resize visibility
threshold, severe blur/occlusion/truncation, questions outside the ontology,
and frames dominated by ignored regions. Pair positive and negative versions
of near-identical question templates so language priors cannot solve the task.

### Public-data mix

Do not indiscriminately concatenate all remote-sensing corpora. Satellite,
orthophoto, and low-altitude drone views are related but not identical domains.

| Dataset | Use | Value | Caution |
|---|---|---|---|
| [VRSBench](https://vrsbench.github.io/) | Main external SFT/eval mix | 29,614 human-verified detailed captions, 52,472 references, 123,221 QA pairs; directly covers caption/VQA/grounding | Built on DOTA-v2 and DIOR; DOTA imagery is academic-only. Preserve source splits and licenses |
| [AeroCaps / Aerial Mirage](https://openaccess.thecvf.com/content/WACV2025/html/Basak_Aerial_Mirage_Unmasking_Hallucinations_in_Large_Vision_Language_Models_WACV_2025_paper.html) | Small high-value caption/hallucination set | 1,256 UAV images and four human captions each; LID explicitly studies aerial hallucination | Small; verify repository license before training; use LID chiefly for evaluation/hard negatives |
| [RSVLM-QA](https://github.com/StarZi0213/RSVLM-QA) | Modern diverse VQA supplement | 13,820 images, 162,373 QA pairs, CC BY 4.0; counts come from segmentation | Many richer annotations are GPT-4.1-generated; satellite/orthophoto distribution; audit and source-balance |
| [HRVQA](https://hrvqa.nl/) | Optional high-resolution count/attribute ablation | 53,512 1024×1024 images and 1,070,240 QA pairs | Templated and CC BY-NC-SA; subsample heavily or it will dominate; not default if artifact reuse needs permissive terms |
| [FloodNet-VQA](https://github.com/BinaLab/FloodNet-VQA) | Evaluation-only disaster-domain check | Low-altitude post-flood imagery and VQA | CC BY-NC-ND is restrictive; do not mix into training without legal review; “damage” labels are domain-specific |
| [ReVA](https://huggingface.co/datasets/ReVA-Benchmark/ReVA) | Later external video QA evaluation | Drone video QA across Hawk_UAV, VisDrone, UAVDT, and ERA | Contains VisDrone itself, so deduplicate at source/sequence level; temporal questions are out of the first model's contract |
| [SkyEye-968k](https://huggingface.co/datasets/ZhanYang-nwpu/SkyEye-968k) | Source to mine selectively, not default bulk mix | Large unified RS instruction collection with captions, VQA, and grounding | Aggregates many upstream datasets and generated examples; provenance/licence and leakage audits are mandatory |

A reasonable first source-balanced target is 55–65% audited VisDrone synthetic,
20–25% VRSBench, 5–10% AeroCaps, and 10–15% diverse VQA/general-retention data.
Treat that as an ablation, not a sacred ratio. Cap questions per image and use
task-balanced sampling so one image with dozens of templates is not dozens of
independent visual examples.

### Split protocol

Split before generating questions. Keep every video sequence, track, burst,
and near-duplicate connected component in exactly one split.

- Preserve the repository's Phase 4 protocol. Its four `held_out_sequences`
  form the primary final VLM video test and must never enter synthesis, prompt
  development, hard mining, or checkpoint selection. The three
  `development_sequences` may be used for diagnostics, but not for VLM
  training; create the actual VLM checkpoint-development split from MOT-train.
- Split complete MOT-train sequences into train and development groups.
- For DET stills, use the repository's leading source token as a scene proxy,
  then improve it with perceptual-hash connected components. A random image
  split is unacceptable.
- Extend deduplication across DET, MOT, and every external corpus. If any frame
  in a held-out sequence matches a training image, quarantine the entire
  connected group from the primary test.
- Sample held-out frames with a minimum temporal gap and report confidence
  intervals by sequence, not by treating adjacent frames as independent.
- Freeze question templates and test cases with hashes before SFT.

## Evaluation that demonstrates visual work

Use the same locked cases for zero-shot and every checkpoint. The primary suite
should have a few hundred human-audited questions, stratified by task, class,
density, post-resize object size, occlusion, negative/ambiguous status, and
sequence. Test questions should be independently paraphrased, not copies of
training templates.

### Baselines

1. **Question-only VLM:** blank/mean image or text-only ablation. This measures
   template and class-prior shortcuts.
2. **Shuffled-image VLM:** preserve prompts but permute images within a batch.
   Visually grounded performance must collapse appropriately.
3. **Detector-only:** deterministic YOLO answers for supported counts/presence,
   box-centre point for supported target requests, templated object caption, and
   the same coarse occupancy-grid review cue. It is a separate system, never a
   hidden VLM input.
4. **Teacher ceiling:** score YOLO/BoT-SORT against official GT so synthetic
   label noise is visible.
5. **Pretrained and aerial checkpoint:** compare base Qwen, Miril, and SFT using
   identical preprocessing and decoding.

The VLM need not beat the detector on detector-native counts. If it merely
reproduces them less accurately, it has not added value. It should add useful
caption/VQA semantics on questions the detector cannot represent while
retaining disciplined abstention.

### Metrics

**Contract:** raw valid JSON, schema validity, correct union route, no extra
keys, and null-coordinate discipline. Report them, but never aggregate them
into “VLM accuracy.”

**Answer:** exact and normalized accuracy for boolean/label; exact-count
accuracy, MAE, and ±1 accuracy for counts; macro scores by class and density;
and human-audited semantic correctness for open text. Report performance on
detector-answerable and detector-unanswerable questions separately.

**Caption:** object hallucination/omission against GT where possible, semantic
coverage, and a blinded human rubric for factuality, usefulness, and unsupported
specificity. CIDEr/BERTScore/LLM judging may be secondary regression signals,
not the sole truth. Aerial Mirage specifically warns that using another LLM as
the hallucination judge is not automatically reliable.

**Pointing:** point-inside-target-box, normalized centre error, error divided by
box diagonal, accuracy at multiple thresholds, and breakdown by target size.
For negative/ambiguous cases measure false-point rate. For `review_region`,
measure coarse-cell agreement, clearance from dilated annotated boxes, and
human-reviewed visual plausibility—never “safe landing accuracy.”

**Abstention:** precision/recall/F1 for each non-answer status, false-positive
answer rate on absent targets, false-point rate, and coverage versus error. If
status token probabilities are available consistently, add calibration/ECE;
do not invent a self-reported confidence field.

**Grounding controls:** paired images with different counts, crops that remove
the requested object, image shuffles, and plausible false-premise questions.
These are essential evidence that the image changes the answer.

Checkpoint selection uses sequence groups drawn from MOT-train. The four
Phase 4 held-out MOT-val sequences are run once for the final VLM result, with
per-sequence bootstrap intervals and all raw predictions retained. Report the
three Phase 4 development sequences separately as a diagnostic slice.

## Module and artifact layout

This layout adds one package and leaves detection/tracking imports pointed in
their current direction:

```text
src/wam_drones/vlm/
  __init__.py
  contracts.py          # response + audit Pydantic models
  prompts.py            # versioned system contract and task wording
  frame_source.py       # RGB still/folder and sampled-video frames only
  inference.py          # VLMBackend protocol + Transformers adapter
  structured_decode.py # schema-guided decode, validate, never silent repair
  teacher.py            # read-only adapters for FrameTracks/GT/detections
  data_factory.py       # evidence filters and supervision generators
  splits.py             # sequence/scene/hash grouping
  baselines.py          # detector-only and question/image ablations
  evaluation.py         # type-specific metrics
  cli.py

interfaces/schemas/
  vlm_response_v1.schema.json
  vlm_supervision_record_v1.schema.json
  vlm_inference_record_v1.schema.json

configs/vlm/
  models/{qwen3_vl_4b,qwen35_4b,miril_2b2}.yaml
  data/visdrone_sft_v1.yaml
  training/qwen3_vl_4b_qlora_sft_v1.yaml
  eval/heldout_sequences_v1.yaml

evaluation/experiments/
  exp_YYYYMMDD_vlm_zero_shot/
  exp_YYYYMMDD_vlm_sft/
```

`teacher.py` depends on existing detection contracts; detection never imports
VLM code. The runtime inference adapter never accepts teacher fields. The video
source samples decoded frames at an explicit interval and records source frame
index/time. It does not call the detector. A later shared media iterator is a
possible refactor, but is not required to begin and should not disturb the
working detector path.

### CLI outline

```shell
# Locked zero-shot or fine-tuned inference on one image
uv run wam-vlm infer image.jpg \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --question "How many cars are visible?" \
  --output runs/vlm/inference.jsonl

# Offline video means sampled independent frames
uv run wam-vlm infer-video clip.mp4 \
  --sample-every-seconds 2 \
  --questions configs/vlm/eval/questions_v1.json \
  --output runs/vlm/video_answers.jsonl

# Build but do not yet train a provenance-rich corpus
uv run wam-vlm synthesize \
  --manifest data/manifests/visdrone_mot_train_v1.json \
  --tracks path/to/frame_tracks_v1.jsonl \
  --config configs/vlm/data/visdrone_sft_v1.yaml \
  --output data/processed/vlm/visdrone_sft_v1.jsonl

uv run wam-vlm audit-data \
  data/processed/vlm/visdrone_sft_v1.jsonl \
  --report evaluation/experiments/exp_.../data_audit.json

# Thin config-driven TRL/PEFT entry point
uv run wam-vlm train \
  --config configs/vlm/training/qwen3_vl_4b_qlora_sft_v1.yaml

uv run wam-vlm evaluate \
  --suite configs/vlm/eval/heldout_sequences_v1.yaml \
  --model path/to/adapter \
  --with-detector-baseline \
  --output evaluation/experiments/exp_.../eval
```

The inference JSONL is an audit envelope with frame metadata plus the model's
bare validated response. At model-call time the only content is RGB pixels,
system contract, and user text.

## Milestones and gates

### M0 — interface and held-out protocol

- Validate the existing BoT-SORT `tracks.jsonl` against `FrameTracks v1` and
  add a provenance index that maps each global frame ID to sequence, original
  frame number/path, dimensions, source hash, checkpoint, and tracker config;
  do not change tracker behavior.
- Freeze response schema, preprocessing, sequence groups, dedup exclusions,
  and 300–500 manually audited eval prompts.
- Gate: an automated check proves no held-out sequence/hash enters synthesis.

### M1 — pretrained zero-shot spike

- Run Qwen3-VL-4B, Qwen3.5-4B, Miril, and optionally SmolVLM2 on the same suite.
- Measure quality, peak VRAM, latency, image resolution, structured decoding
  compatibility, and quantized/BF16 deltas.
- Gate: select the base from repo-specific evidence. Default to Qwen3-VL-4B if
  results are tied or Qwen3.5 training support is unstable.

### M2 — data factory and audit

- Generate balanced presence/count/point/abstain examples from GT and stable
  tracks; add bounded factual captions and external aerial data.
- Render a stratified human review pack including accepted and rejected labels.
- Gate: teacher disagreement, sequence leakage, task balance, visibility
  thresholds, and negative-label precision meet documented thresholds.

### M3 — synthetic SFT

- Train the frozen-vision QLoRA baseline, then run one vision-LoRA or
  higher-resolution ablation.
- Gate: semantic metrics improve over zero-shot without increasing false-point
  or false-presence rates; JSON validity alone cannot pass the gate.

### M4 — compare with detector baseline

- Report detector-only, question-only, shuffled-image, base VLM, aerial SFT,
  and (if useful) Miril on locked MOT-val sequences.
- Gate: document exactly where language adds value and where the detector
  remains superior.

### M5 — optional hardening

- Mine on-policy failures and do hard-example SFT.
- Only then test DPO for abstention/hallucination, or the 8B/9B scale step.
- GRPO/GSPO remains optional and requires a pre-registered verifiable reward.

## Open questions and ablations

- Should Phase 4 emit a small frame-provenance sidecar directly, or should the
  VLM teacher reconstruct it from the frozen protocol and per-sequence files?
  Prefer an explicit sidecar if reconstruction is not one-to-one.
- Does Qwen3.5-4B beat Qwen3-VL-4B on tiny post-resize targets enough to offset
  its newer training path?
- 2B/4B/9B scale versus 640/high-resolution pixels: which buys more grounding
  per GB?
- Frozen vision, last-block vision LoRA, or resolution increase?
- GT-only supervision versus GT plus stable predicted tracks; report both so a
  noisy teacher cannot hide behind more data.
- General base plus aerial SFT versus Miril warm start.
- Template-only language versus templates plus human aerial captions.
- 0–1000 point tokens versus nine-cell coarse review regions; keep the API
  coordinate convention fixed while ablating the training target.
- How small is too small after the exact model processor? Define visibility in
  processed pixels, not original pixels.
- Constrained decode versus unconstrained deterministic decode: compare
  semantic quality because forcing a schema can change token probabilities.
- Is external RS data helping drone frames, or merely generic scene vocabulary?
  Report by source and by low-altitude versus orthophoto domain.

No blocking hardware question is required for this design phase: the actual
host establishes macOS/MPS, VisDrone GT is present, and 24 GB CUDA is an
explicit planning assumption. Before M3, record the exact NVIDIA GPU and VRAM;
that determines QLoRA image budget and whether the 9B run is reasonable.

## Explicitly out of scope

- drone purchase, airframe, propulsion, battery, payload, or camera selection;
- flight controller, PX4, ArduPilot, MAVLink, radios, telemetry, navigation,
  path planning, mission planning, landing logic, or closed-loop control;
- onboard/edge deployment, real-time scheduling, thermal/power optimization,
  Hailo/ONNX export, or flight certification;
- rewriting or jointly training YOLO or BoT-SORT;
- detector boxes, counts, identities, timestamps, or tracks as VLM runtime
  inputs;
- native temporal/video VLM training in the first study—video is sampled into
  independent still frames;
- claims of physical free-space, collision clearance, or landing safety from
  2-D appearance;
- person identification, face recognition, or surveillance identity inference;
- a full production training platform, model registry, serving stack, or UI;
  and
- automated action from a VLM point. Points are optional review cues only.
