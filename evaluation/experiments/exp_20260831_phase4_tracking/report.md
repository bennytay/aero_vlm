# Experiment exp_20260831_phase4_tracking: offline MOT evaluation

## Question

Can the Phase 3 detector be associated into stable, native-class trajectories
on public aerial video without selecting tracker thresholds on the final score?

## Protocol

The fine-tuned native YOLO26n checkpoint was run at 640px FP32 on all seven
official VisDrone-MOT validation sequences. Three whole sequences (1,046
frames) form the tracking-development set; four disjoint sequences (1,800
frames) form the held-out set. Frames retain source time from their original
frame number at 30 FPS even when inference finishes later.

All thresholds are versioned in `configs/tracking/*.yaml`. ByteTrack and
BoT-SORT were compared on development only. BoT-SORT uses camera-motion
compensation and has ReID explicitly disabled. The pre-recorded selection rule
was to keep simpler ByteTrack unless BoT-SORT gained at least 0.03 HOTA.

## Development comparison

| Tracker | HOTA | IDF1 | ID switches | Fragmentation | Track recall | Tracker ms/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ByteTrack | 0.3566 | 0.7119 | 1,271 | 1,106 | 0.4195 | 1.730 |
| BoT-SORT, no ReID, CMC | 0.4136 | 0.7870 | 856 | 1,055 | 0.4389 | 1.899 |

BoT-SORT gains 0.0570 HOTA, exceeding the material-improvement threshold by
0.0270, while adding 0.169 ms/frame tracker time. It was therefore fixed as
the selected configuration before held-out scoring.

## Held-out result

The selected BoT-SORT configuration produced stable IDs on all four held-out
sequences: HOTA **0.4076**, IDF1 **0.8591**, 754 identity switches, 443
fragmentations, and 0.3719 track recall. Mean detector time was 18.217
ms/frame and tracker time was 0.680 ms/frame; those boundaries are reported
separately.

| Camera-motion magnitude | Track recall |
| --- | ---: |
| Low | 0.3884 |
| Medium | 0.3347 |
| High | 0.3832 |

Object-size and occlusion results, per-sequence scores, MOTChallenge text
outputs, timing, and validated frame-track JSONL are in the ignored local
artefact directories. They are reproducible with `wam-detect phase4-evaluate`.

## Conclusion

The Phase 4 gate is met for offline public-video tracking. Keep the BoT-SORT
without-ReID configuration as the Phase 5 moving-camera tracker, while
treating small, partly occluded objects as the dominant remaining recall risk.
