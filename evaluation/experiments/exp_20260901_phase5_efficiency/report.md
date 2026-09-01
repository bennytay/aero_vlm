# Experiment exp_20260901_phase5_efficiency: efficient operating point

## Question

Which measured pipeline profile preserves small-object tracking quality with the
lowest compute and energy use?

## Status

**Complete (desktop gate).** Development selection and one sealed held-out
confirmation are complete. No target-hardware latency, power, energy, thermal,
or deployment claim is made.

## Frozen baseline

Use the Phase 3 native-class checkpoint identified in `meta.json`, the seven
official VisDrone-MOT validation sequences, and both Phase 4 tracker configs.
The three development sequences select profiles; the four Phase 4 held-out
sequences remain sealed until selection is recorded.

## Matrix and selection rule

The complete pre-registered matrix and quality floor are in
`configs/experiments/phase5_efficiency.yaml`. Measure the end-to-end pipeline
from decoded frame availability through tracking, optional tile merge, and
optional preview encoding; do not substitute neural-network-only timing.

On detector-skipped frames the runner updates the tracker with no detections.
Its output therefore retains motion-propagated tracks with
`observed_this_frame: false` and a positive `time_since_update_frames`; it never
labels them as fresh detections.

## Development results

The frozen baseline was measured on all three development sequences (1,046
frames): 640px FP32, detector cadence one, tiling off, selected BoT-SORT with
camera-motion compensation and ReID disabled, and preview off. It achieved HOTA
**0.4110**, IDF1 **0.7832**, and track recall **0.4388**, satisfying the
pre-registered development floors of 0.38 HOTA and 0.35 track recall.

The measured decoded-frame-to-track mean pipeline time was **13.10 ms/frame**;
detector time was 5.34 ms/frame and tracker time was 1.75 ms/frame. Raw
per-frame timings, validated track JSONL, MOT text, and sequence metrics are in
`artefacts/dev_640_fp32_cadence1_tiles_off_botsort_preview_off_clean/`. This
desktop run does not make a power, energy, or MaixCAM2 deployment claim.

| Profile | HOTA | Track recall | Pipeline ms/frame | Development decision |
| --- | ---: | ---: | ---: | --- |
| 640 FP32, cadence 1, BoT-SORT | 0.4110 | 0.4388 | 13.10 | retain accuracy reference |
| 512 FP32, cadence 1, BoT-SORT | 0.3963 | 0.3806 | 11.97 | qualified, slower than cadence-2 candidate |
| 416 FP32, cadence 1, BoT-SORT | 0.3891 | 0.3024 | 11.22 | reject: recall floor missed |
| 640 FP32, cadence 2, BoT-SORT | 0.4076 | 0.3972 | 9.93 | selected for one held-out confirmation |
| 640 FP32, cadence 3, BoT-SORT | 0.3616 | 0.3599 | 8.75 | reject: HOTA floor missed |
| 640 FP16, cadence 2, BoT-SORT | 0.4066 | 0.3965 | 11.01 | qualified, slower on this host |
| 512 FP32, cadence 2, scheduled tiles, BoT-SORT | 0.3915 | 0.3873 | 11.46 | qualified, slower than 640 cadence 2 |
| 640 FP32, cadence 2, ByteTrack | 0.3421 | 0.3833 | 10.28 | reject: HOTA floor missed |
| 640 FP32, cadence 2, BoT-SORT, preview on | 0.4076 | 0.3972 | 21.14 | preview is not a low-latency mode |

The explicit selection record is `selection.json`. It retains 640px FP32
cadence-one BoT-SORT as the accuracy reference and selects the 640px
cadence-two profile for a single sealed held-out confirmation.

## Held-out confirmation

The pre-registered 640px FP32 cadence-two BoT-SORT profile was then run once on
the four held-out sequences (1,800 frames). It produced HOTA **0.3749**, IDF1
**0.8200**, and track recall **0.3288**, with mean pipeline time **13.52
ms/frame**. It is rejected as a Phase 6 candidate: the development speed gain
does not transfer with sufficient held-out small-object tracking quality.

The output remains useful as a documented negative result. Per-frame timings,
tracks, MOT files, and sequence metrics are in
`artefacts/held_out_640_fp32_cadence2_tiles_off_botsort_preview_off/`. The
held-out data was not used for further tuning.

## Conclusion and next phase

Phase 5 retains the 640px FP32 every-frame BoT-SORT reference for Phase 6 ONNX
and target-INT8 parity work. It does **not** retain a desktop low-latency
cadence-two deployment candidate, nor does it identify a low-power setting:
NVML failed to initialise on the desktop measurement host, so valid host power
samples were unavailable. Phase 6 must measure complete pipelines and power on
MaixCAM2 before selecting either an onboard quality or low-power profile.
