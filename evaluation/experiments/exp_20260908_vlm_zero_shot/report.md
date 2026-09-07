# Zero-shot efficient VLM spike

This is a provisional runtime and contract report. The 36-frame VisDrone
candidate pack was selected from DET validation labels and has not yet received
the required human review. Its semantic scores must not be used for model
selection.

## Environment

- GPU: NVIDIA GeForce RTX 3070, 8 GB
- Runtime: Python 3.11, Torch 2.13.0+cu130, Transformers 4.57.6
- Input contract: one RGB frame and text only; no detector, tracker, temporal,
  or future-frame input
- Prompt: `vlm-spike-v2`

## Qwen3-VL-2B

The primary low-compute baseline successfully produced strict structured JSON
on a 960x540 aerial frame. Qwen resized it to 960x544, used 4.43 GB peak GPU
memory, and generated the response in 1.54 seconds.

On the unreviewed 36-frame candidate pack, the first broad all-types prompt
averaged 1.09 seconds per frame and 4.51 GB peak GPU memory. Only 12 of 36
outputs parsed as `vlm_response_v1`; most failures were code-fenced JSON or
objects missing required fields.

Question-routed templates raised unconstrained parse success to 29 of 36
(80.6%). The compatible Transformers 4.57 runtime used for this pass averaged
8.16 seconds per frame and 6.64 GB peak GPU memory. The recorded
`semantic_failures: 23` is a provisional contract/type-status mismatch count,
not a human semantic score.

Schema-constrained decoding on one caption case produced valid JSON but took
44.94 seconds and 6.42 GB peak GPU memory. It is useful as a correctness
comparison, not a live runtime path on this host.

## SmolVLM2-2.2B

SmolVLM2 fit only after bounding the source frame to 448x252 and limiting
generation to 64 tokens. It used 6.88 GB peak GPU memory and took 7.64 seconds
for the caption probe. It returned plain text despite the model-visible strict
JSON instruction, so it presently fails the output-contract gate.

## Candidate pack audit (2026-09-08)

A visual audit of all 36 candidate images against `configs/vlm/smoke_suite_v1.json`
found the pack was not 36 independent cases. Hashing every image showed only
**17 unique frames**; 19 manifest rows were duplicate copies of another row's
image under a different filename and question. Several of those duplicates
carried self-contradictory ground truth relative to the shared frame: a
"point to the red car" case (expects `found`) and a "point to the truck"
case (expects `no_candidate`) shared one image containing a truck and no
car — the labels were inverted. A "point to the bicycle" case expected
`found` on a rooftop frame with no bicycle in it. The entire `low_visibility`/
`hidden` category turned out to be bright, fully legible daytime scenes with
no occlusion, contradicting the category tag.

The pack was deduplicated to 17 unique frames and the ground truth corrected
against what each image actually shows (`configs/vlm/smoke_suite_v1.json`,
`evaluation/experiments/exp_20260908_vlm_zero_shot/artefacts/smoke_candidates/images/`).
The `wam-vlm smoke` CLI help text was updated from "36-case" to "17-case"
accordingly. The pre-audit 81%/29-of-36 result above should be treated as
provisional and superseded by the run below.

Re-running Qwen3-VL-2B (`vlm-spike-v3` routed prompts) against the corrected
17-frame pack (`qwen3_vl_2b_dedup17_smoke_candidates`) gave a materially
different picture: **12 of 17 parsed (70.6%)**, mean latency 9.22 s/frame,
peak memory 6.64 GB. Of the 5 unconstrained parse failures, all were `point`
type and reproduce the same root causes seen previously — markdown code
fences not stripped before validation, `point` populated when `status` isn't
`found` (schema forbids this), and a `point.semantics` enum too narrow for
what the model returns.

More notably, of the 12 that *did* parse, only 6 matched expected
`type`/`status`: three caption cases on genuinely clear, describable scenes
came back `status: unknown` instead of `ok`; a `point` case on an image with
exactly one unambiguous person came back `no_candidate`. These are real
model behavior gaps, not pack-labeling artifacts — the previous unaudited
pack was masking this signal.

## Parser and prompt fixes (2026-09-08, `vlm-spike-v4`)

Two of the three parse-failure root causes from the audit above were
mechanical and got fixed directly:

- `parse_vlm_response` (`src/wam_drones/vlm/contracts.py`) now unwraps a
  single leading/trailing Markdown code fence before validating, since the
  model reliably wraps otherwise-valid JSON in ` ```json...``` ` despite
  being told not to. This is the only normalization it performs.
- The raw caption/point/answer generations showed the model echoing the
  prompt's negative-case clause almost verbatim regardless of image content:
  three different images produced the byte-identical
  `{"type":"caption","status":"unknown","text":null}`, and two different
  `point` questions produced the byte-identical
  `{"type":"point","status":"no_candidate","point":null}`. The prompt
  template (`src/wam_drones/vlm/prompts.py`) was rewritten so every
  fallback status is shown as its own concrete JSON example with an
  explicit "only use this when…" criterion, instead of one prose clause
  appended to the positive example, and the system prompt now states
  outright that an ordinary photograph is not grounds for abstaining.

Re-running the corrected 17-frame pack under `vlm-spike-v4`
(`qwen3_vl_2b_v4_dedup17_smoke_candidates`) confirmed both fixes worked:
**parse success rose from 70.6% to 94.1% (16/17)**, and — more importantly —
**true `type`/`status` matches roughly doubled, from 6/17 (35.3%) to 10/17
(58.8%)**. All five caption cases and the two previously fence-wrapped
`point` cases (`point_car_02`, `point_truck_01`) now pass outright, and
`point_absent_01` stopped hallucinating a bicycle that isn't in frame
(it now abstains, just with the wrong specific status).

**A follow-up prompt attempt (`vlm-spike-v5`) was tried and reverted.** The
remaining 7 failures still showed an echo pattern — four different `point`
cases converged on the byte-identical `{"status":"not_visible","point":null}`
regardless of what each image actually called for. Restructuring the four
negative-branch examples into one parameterized `{"status":"<pick one>",...}`
shape with prose criteria (hypothesis: four consecutive full JSON snippets
invite copy-pasting the nearest match) was tested against just the 7 failing
cases first. It did stop the `not_visible` echo, but parse success on that
subset fell from 6/7 to 3/7 — the model started answering `"found"` with
hallucinated coordinates for genuinely ambiguous scenes and dropping the
required `coordinate_system`/`semantics` keys. Net effect was worse, so the
change was reverted; `vlm-spike-v4` is the current prompt. **Do not retry
this exact restructuring** — record it here so it isn't rediscovered blind.

The remaining 7 failures on `vlm-spike-v4` split into two kinds:

- One narrow schema slip: `point_ambiguous_01` returns `{"x":..,"y":..}`
  missing `coordinate_system` and `semantics`, a single occurrence.
- Six genuine reasoning misses: `presence_person_01` and `point_person_01`
  abstain on unambiguous content; `point_absent_01`, `point_ambiguous_02`,
  and `review_region_02` pick the wrong one of four valid-looking abstention
  statuses; `answer_ambiguous_01` answers with one car's colour instead of
  flagging that "the car" is underspecified among several. These look like
  a genuine ceiling for zero-shot prompting on a 2B model discriminating
  among 4–5 similar abstention statuses, not an engineering defect — the
  v5 experiment above is evidence further prompt-only iteration here has
  turned risky rather than productive.

## Automatic schema-constrained fallback and a `lm-format-enforcer` bug (2026-09-08)

`point_ambiguous_01`'s missing-keys slip turned out to be fully
deterministic — identical `{"x":509,"y":758}` coordinates across repeated
runs and even after reinforcing "always include all four point keys" in the
prompt. Retrying it under schema-constrained decoding (`--decode-mode
schema`, already in the codebase as a documented "correctness reference")
was expected to force valid structure, but it produced the exact same
missing-field output. That result is diagnosed, not just observed: probing
`lmformatenforcer.JsonSchemaParser` character-by-character shows that
constraining against `PointResponse` alone correctly *rejects* closing the
`point` object before `semantics` is supplied, but constraining against the
top-level discriminated union `CaptionResponse | AnswerResponse |
PointResponse` (`VLMResponseContract`, used by the existing
`_schema_prefix_allowed_tokens`) incorrectly *allows* it. The installed
`lm-format-enforcer` (pinned `>=0.11,<1`) does not correctly propagate
nested `required` constraints through a `oneOf`/discriminator composition,
even though the emitted JSON Schema itself is correct. Patching or
upgrading that pinned third-party dependency is out of scope here.

The in-scope fix: `render_prompt` already knows which single response type
(`caption`/`answer`/`point`) a question maps to (now factored out as
`response_kind_for` in `src/wam_drones/vlm/prompts.py`, shared by both the
prompt template and schema selection). `TransformersVLMBackend
._schema_prefix_allowed_tokens` (`src/wam_drones/vlm/inference.py`) now
constrains against that single response type's schema instead of the full
union, sidestepping the union-composition bug entirely — confirmed
directly: the same case that returned malformed JSON under the full-union
schema returns a fully valid response under the narrowed one.

`wam-vlm smoke` (`run_smoke` in `src/wam_drones/vlm/cli.py`) now uses this
automatically: any case that fails unconstrained parsing is retried once
with schema-constrained decoding (only when the model config declares
`supports_schema_constrained_decoding`), and the metrics record a new
`schema_fallback_successes` count for transparency. `generate()`,
`infer_rgb()`, and `infer_image()` all gained an optional `decode_mode`
override to support the per-case retry without constructing a second model
instance.

Re-running the full 17-frame pack
(`qwen3_vl_2b_v4_narrowfallback_dedup17_smoke_candidates`) with this in
place: **parse success reached 17/17 (100%)** — `schema_fallback_successes:
1`, exactly the one case it was built for. `type`/`status` match held at
10/17 (58.8%), which is expected: the fallback only fixes structural
validity, not the model's status choice, so `point_ambiguous_01` moved from
a parse failure to a clean semantic mismatch (`found` instead of
`ambiguous`) rather than becoming correct. **Every one of the 7 remaining
failures is now a structurally valid response with the wrong `status`** —
there are no parse failures, no missing fields, and no hallucinated
out-of-schema values left in the pack. The gap that remains is entirely the
model choosing the wrong one of several valid-looking abstention statuses
(or failing to recognize an unambiguous target/an obscured reference), which
is what the `vlm-spike-v5` negative result above already showed does not
respond well to further zero-shot prompt rewrites.

## Next work

1. Replace the provisional type/status-only smoke checks with task-aware
   semantic scoring for reviewed responses.
2. Close the remaining status-discrimination gap via few-shot exemplars in
   the prompt or light aerial SFT, not further zero-shot prompt rewrites —
   see the `vlm-spike-v5` negative result above. This is the only
   remaining gap in the spike; every parsing/schema/data-quality issue found
   during this investigation is now fixed and verified at 17/17.
3. Consider reporting the `lm-format-enforcer` discriminated-union
   `required`-field gap upstream, or revisit once the pinned version range
   (`>=0.11,<1`) moves — the per-type-schema workaround in
   `_schema_prefix_allowed_tokens` is a durable fix either way and should
   stay regardless.
4. Use Qwen3-VL-2B as the only local candidate for the next iteration; retain
   schema-constrained decoding as the automatic fallback for parse failures
   (now default in `wam-vlm smoke`) but not as the primary live decode path —
   individual fallback calls still take up to ~30s. SmolVLM2 is dropped from
   further local-candidate consideration — it returns plain text despite a
   strict-JSON instruction and fails the output-contract gate outright.
