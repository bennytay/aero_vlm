# MaixCAM2 model export and contract adapter

This is the planned MaixCAM2 deployment path. It does not claim that a model
has been converted or run on the device.

## Export policy

1. Pin the Ultralytics package version, checkpoint SHA-256, class order, input
   size, and preprocessing alongside the experiment record.
2. Start with the fine-tuned detect-only YOLO26n checkpoint. Export static
   batch-one ONNX with `opset=17`, `dynamic=False`, and one fixed `imgsz`.
3. Convert ONNX through Pulsar2 or MaixHub into a `.mud` project and `.axmodel`.
   Preserve the converter version and `.mud` model type, input shape, class
   order, preprocessing, quantisation/calibration settings, and output decode
   settings in the experiment record.
4. Use 20–100 representative aerial image crops for INT8 calibration. Include
   small, occluded, sparse, and dense targets; do not calibrate only on clean
   close-up objects.
5. Compare desktop FP32, ONNX, and `.axmodel` boxes on the same fixtures before
   starting live-camera work.

If YOLO26n fails to compile after three documented Pulsar2/MaixHub attempts,
switch to YOLO11n at the same input sizes. YOLOv8n is also an allowed
compatibility fallback. Do not redesign the runtime around a broken compiler
path.

## Runtime boundary

Load the compiled detect-only model with the matching MaixPy API
(`nn.YOLO26`, `nn.YOLO11`, or `nn.YOLOv8`). The device adapter must convert each
MaixPy object to the existing `Detection` schema with the frozen class ID/name,
confidence, normalised `xyxy` box, model name, frame ID, and original capture
timestamp. Assemble these into `FrameDetections`; then run the repository's
BoT-SORT + camera-motion compensation tracker with ReID disabled and emit
`FrameTracks`/JSONL.

No schema is changed for this device. Device identity (`maixcam2`),
`model_format` (`axmodel`), MaixPy/Pulsar2 versions, calibration-set hash, and
pipeline boundary belong in the versioned experiment record rather than in the
Phase 0–4 runtime contracts.
