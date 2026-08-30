# Experiment exp_20260831_phase0_foundation: detection contract controls

## Question

Are the detection vocabulary and runtime contracts precise, versioned, and
reproducible?

## Hypothesis

Frozen schemas and validation tests reject inconsistent labels, boxes, timing,
lifecycle state, and duplicate track IDs.

## Configuration

- Git commit: `fce56104fd146b5d4152dc1dfaa57d7889d0f921`
- Config hash:
  `sha256:42b75c41ad31b0c65e71d33b16f498bd03c82f0bfb844e60f200b07a79d61bab`
- Dataset manifest hash: not applicable
- Hardware: development computer; contract-only checks
- Model format: Pydantic JSON Schema
- Input size: not applicable
- Numeric precision: not applicable
- Pipeline boundary: configuration load through contract serialisation
- Dataset split: deterministic unit fixtures
- Random seed: 0

## Protocol

1. Load the checked-in detection vocabulary.
2. Construct valid detection, frame, track, and frame-track records.
3. Round-trip public records through JSON.
4. Generate schemas and compare them with the checked-in files.
5. Exercise invalid labels, mismatched IDs, inverted and zero-area boxes,
   reversed timestamps, mixed frames, inconsistent track lifecycle state,
   duplicate track IDs, and non-increasing frame sequences.
6. Run the complete test, lint, and static-type suite.

## Acceptance criteria

- The vocabulary contains exactly the ten ordered native VisDrone classes.
- Every public contract round-trips through JSON.
- Checked-in schemas match the models exactly.
- Invalid geometry, labels, timing, lifecycle state, and duplicate IDs fail
  validation.
- Tests, Ruff, MyPy, and `git diff --check` pass.

## Results

- Pytest: 17 passed, 0 skipped
- Ruff: 0 errors
- MyPy: 0 errors
- Schema parity: passed for all four public contracts
- Vocabulary coverage: 10 of 10 ordered classes

## Failures and anomalies

The first sequence test reversed both frame ID and capture time, so the frame-ID
guard correctly failed before the timestamp guard. The test was separated into
two cases so each invariant is verified independently. No product defect was
found.

## Conclusion

Phase 0 passes. The vocabulary, runtime records, and experiment context are
versioned and mechanically checked. Phase 1 can consume these contracts without
inventing class order, box conventions, timing semantics, or track staleness.
