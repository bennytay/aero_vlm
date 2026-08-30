# Experiment exp_20260831_phase1_public_detector: public detector smoke test

## Question

Can the untouched public YOLO26n model run through the v1 detection contract
and export to portable ONNX before any training investment?

## Configuration

- Ultralytics: `8.4.135`
- Checkpoint: `yolo26n.pt`
- Checkpoint SHA-256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- Input: 640 x 640 letterboxed by Ultralytics
- Precision: FP32
- Device: CPU for the portable baseline
- Tracking preview: Ultralytics ByteTrack with persistent video state

## Protocol

1. Install the optional group with `uv sync --group detection`.
2. Run ten small, locally prepared VisDrone images as one folder.
3. Run one short public VisDrone video clip with tracking enabled.
4. Validate every line in `predictions.jsonl` as `FrameDetections`.
5. Export ONNX, parse it with `onnx.checker`, then run the same image through
   PyTorch and ONNX and match same-class boxes greedily by IoU.

## Acceptance criteria

- The checkpoint digest matches before model deserialization.
- All output frames validate against the v1 project contract.
- A non-empty annotated MP4 is produced from the clip.
- The ONNX model passes `onnx.checker`.
- Parity matches same-class boxes at IoU >= 0.5 and records unmatched boxes.

## Results

The lightweight suite passes with 25 tests, clean Ruff checks, and clean strict
Mypy checks. On an Apple M2 CPU, the public image produced five mapped detections
at 109.40 ms decoded-frame-to-contract latency. The 11-frame public clip produced
an annotated tracked MP4 and averaged 34.33 ms per frame (28.18 ms minimum,
77.69 ms maximum). Three lowest-confidence frames were retained for review.

The ONNX file parsed successfully. Against the same public image, all five boxes
matched: mean IoU 0.9658, minimum IoU 0.9028, and maximum confidence difference
0.1059. Raw runtime artefacts remain ignored by Git.

The deterministic ten-image VisDrone extractor is implemented and tested, but
the official VisDrone archive was not downloaded during this run. Its measured
fixture result remains explicitly pending rather than being inferred from the
non-aerial public smoke image.

## Failures and anomalies

The public checkpoint uses COCO classes. Only the six unambiguous overlapping
classes are projected into the ten-class VisDrone contract. Unsupported COCO
classes are dropped, and the four VisDrone-only distinctions cannot be emitted
before fine-tuning.

Tracking initially failed because Ultralytics lazily requires `lap` and tried to
self-install it with pip, which uv environments do not include. The optional
dependency group now pins `lap==0.5.12`; the rerun passed.

## Conclusion

The portable public-model path, tracked preview, ONNX export, parity comparison,
latency capture, and error-frame retention all pass. The only remaining evidence
item is running the prepared ten-image smoke command after an official VisDrone
DET archive is placed locally.
