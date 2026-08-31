# Experiment exp_20260831_phase3_visdrone_det: sequence-safe VisDrone detector fine-tuning

## Question

Does deterministic fine-tuning of COCO-pretrained YOLO26n at 640px improve
native-class VisDrone DET performance on the primary sequence-safe validation
set without collapsing a class?

## Configuration

- Implementation commit at start: `3d55f38c55bc0bac59763ae4a464032700239bde`
- Device: NVIDIA GeForce RTX 3070, CUDA device 0
- Public starting checkpoint: `models/yolo26n.pt`, SHA-256
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- Input: 640px aspect-ratio-preserving letterbox; no crop or stretch
- Classes: native 10 VisDrone classes; `pedestrian` and `person` remain separate
- Seed / epochs / batch / precision: 20260831 / 100 / 8 / FP32 (`amp: false`)
- Training images: 6471 official DET-train images
- Primary validation images: 547 official DET-val images, excluding only
  `0000323_02601_d_0000645`; raw data and official labels were not edited

Ultralytics AutoBatch profiling terminated before selecting a batch. A
controlled one-epoch probe found batch 16 invoked TaskAlignedAssigner CUDA OOM
retries, while batch 8 completed without OOM retries. Batch 8 is therefore the
documented hardware-specific fixed setting.

## Results

| Metric | COCO baseline | Fine-tuned best.pt |
| --- | ---: | ---: |
| mAP50-95 | 0.0449 | 0.1676 |
| AP50 | 0.0755 | 0.3061 |
| Precision | 0.6683 | 0.6996 |
| Recall | 0.1088 | 0.4063 |
| False positives/frame | 3.819 | 12.329 |

Fine-tuning improves mAP50-95 by 0.1228 and raises small-box recall from
0.0727 to 0.3739. All ten native classes improve in mAP50-95; the gate is met,
although the higher recall comes with substantially more unmatched predictions.

| Class | mAP50-95 baseline -> fine | AP-small baseline -> fine | Recall baseline -> fine | Precision baseline -> fine | FP/frame baseline -> fine |
| --- | ---: | ---: | ---: | ---: | ---: |
| pedestrian | 0.000 -> 0.146 | 0.000 -> 0.139 | 0.000 -> 0.299 | 0.000 -> 0.637 | 0.000 -> 2.748 |
| person | 0.019 -> 0.099 | 0.018 -> 0.099 | 0.050 -> 0.203 | 0.244 -> 0.641 | 1.463 -> 1.062 |
| bicycle | 0.004 -> 0.027 | 0.002 -> 0.027 | 0.001 -> 0.061 | 0.167 -> 0.350 | 0.009 -> 0.265 |
| car | 0.244 -> 0.471 | 0.153 -> 0.407 | 0.272 -> 0.677 | 0.783 -> 0.772 | 1.934 -> 5.124 |
| van | 0.000 -> 0.211 | 0.000 -> 0.141 | 0.000 -> 0.308 | 0.000 -> 0.529 | 0.000 -> 0.984 |
| truck | 0.049 -> 0.163 | 0.005 -> 0.060 | 0.068 -> 0.209 | 0.255 -> 0.600 | 0.272 -> 0.190 |
| tricycle | 0.000 -> 0.091 | 0.000 -> 0.072 | 0.000 -> 0.156 | 0.000 -> 0.466 | 0.000 -> 0.342 |
| awning-tricycle | 0.000 -> 0.055 | 0.000 -> 0.047 | 0.000 -> 0.092 | 0.000 -> 0.353 | 0.000 -> 0.165 |
| bus | 0.107 -> 0.263 | 0.009 -> 0.141 | 0.144 -> 0.392 | 0.429 -> 0.676 | 0.088 -> 0.086 |
| motor | 0.025 -> 0.151 | 0.022 -> 0.150 | 0.010 -> 0.286 | 0.628 -> 0.652 | 0.053 -> 1.364 |

### Breakdown highlights

| Slice | Recall baseline -> fine | Precision baseline -> fine | FP/frame baseline -> fine |
| --- | ---: | ---: | ---: |
| Small (<32px) | 0.073 -> 0.374 | 0.562 -> 0.672 | 3.709 -> 11.945 |
| Medium (32-96px) | 0.549 -> 0.803 | 0.446 -> 0.287 | 3.612 -> 10.561 |
| Large (>=96px) | 0.629 -> 0.657 | 0.011 -> 0.004 | 3.503 -> 10.186 |
| Dense (16+ objects) | 0.107 -> 0.405 | 0.662 -> 0.697 | 3.814 -> 12.303 |
| No occlusion | 0.184 -> 0.529 | 0.601 -> 0.594 | 3.737 -> 11.086 |
| Partial occlusion | 0.059 -> 0.337 | 0.362 -> 0.506 | 3.585 -> 11.333 |
| Heavy occlusion | 0.009 -> 0.196 | 0.015 -> 0.097 | 3.503 -> 10.276 |

The complete box-size, density, occlusion, and per-scene statistics are saved
in the versioned evaluation metrics artefacts.

## Manual review

All 25 selected false negatives and all 25 selected false positives were
visually inspected before any architecture change. The FNs are predominantly
4-10px distant targets, including several occluded targets. The sampled
high-confidence FPs are visually real vehicles; their unmatched status is best
explained by official-label omission/ignore treatment or fine-grained vehicle
class boundaries, rather than by obvious hallucination.

## Checkpoints

- best.pt: `sha256:7468c1bcdc4b8e22e57f3d6a356c79fa786bacdc14f9e19d8fe7ac4993a62871`
- last.pt: `sha256:5a064f94fd09000e48c4442999d424968e9cda5c2d5c7ec20dc3aa7745e95a3e`

Checkpoint bytes are deliberately not versioned. The versioned artefact
metadata preserves their identities, metrics, review record, and training CSV.

## Conclusion

The Phase 3 gate is met. Keep the native vocabulary and use the fine-tuned
best checkpoint for the next evaluation stage; do not interpret the increased
unmatched-prediction count without accounting for the reviewed label coverage
and vehicle-category ambiguity.
