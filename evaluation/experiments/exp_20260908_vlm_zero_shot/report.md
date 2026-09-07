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

On the unreviewed 36-frame candidate pack in deterministic unconstrained mode,
it averaged 1.09 seconds per frame and 4.51 GB peak GPU memory. Only 12 of 36
outputs parsed as `vlm_response_v1`; most failures were code-fenced JSON or
objects missing required fields. The recorded `semantic_failures: 31` is a
provisional contract/type-status mismatch count, not a human semantic score.

Schema-constrained decoding on one caption case produced valid JSON but took
44.94 seconds and 6.42 GB peak GPU memory. It is useful as a correctness
comparison, not a live runtime path on this host.

## SmolVLM2-2.2B

SmolVLM2 fit only after bounding the source frame to 448x252 and limiting
generation to 64 tokens. It used 6.88 GB peak GPU memory and took 7.64 seconds
for the caption probe. It returned plain text despite the model-visible strict
JSON instruction, so it presently fails the output-contract gate.

## Next work

1. Visually audit every candidate image, question, expected status, and
   expected response before treating the smoke pack as an evaluation suite.
2. Replace the provisional type/status-only smoke checks with task-aware
   semantic scoring for reviewed responses.
3. Use Qwen3-VL-2B as the only local candidate for the next iteration. Improve
   its unconstrained JSON reliability through prompt routing or aerial SFT;
   retain constrained decoding only as a measured reference.
