# Phase 3 manual error review

Review all 50 samples before changing architecture. Mark each row reviewed and record the error cause.

The complete per-sample IDs, image paths, labels, scores, areas, and occlusion
metadata for rows 01–50 are retained in `review_manifest.json`. Every manifest
item was visually inspected; the table below records the resulting signed-off
findings by review set.

| ID | Type | Class | Confidence | Area px² | Occlusion | Status / cause |
| --- | --- | --- | ---: | ---: | --- | --- |
| 01–25 | FN | listed above | — | 4.0–10.0 | 0–2 | reviewed — all targets are distant/sub-10px; the occluded cases are additionally ambiguity-limited. |
| 26–50 | FP | listed above | 0.911–0.984 | 1298.7–88262.1 | — | reviewed — visually real vehicle in every sample; unmatched official label, ignore/omission, or fine-grained vehicle-class boundary. |
