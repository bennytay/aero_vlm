# Experiment exp_20260828_teacher_threshold: ten-prompt reject threshold

## Question

Can a cosine threshold over the ten target prompts reject empty scenes more
reliably than the Phase 1 11-way argmax, without collapsing supported targets?

## Hypothesis

A Pareto threshold will reduce `no_target` false positives to at most 20% while
retaining at least 75% macro recall on the five supported classes.

## Protocol

The experiment reused all 1,141 cached, unit-normalised MobileCLIP2-S0 image
embeddings from Phase 1. It encoded the default photo prompts, computed raw
cosines for the ten frozen target names, and swept `T=0.10..0.60` in 0.005
steps. The rule is argmax over ten, followed by rejection when the best cosine
is below T. No teacher parameters or images were changed.

The comparison is the previous 11-way argmax including a `no_target` text.
Metrics use all 841 target scenes and 300 `no_target` scenes; the supported
headline comprises water bottle, black vehicle, bicycle, cardboard box, and
sports ball.

## Results

The selected threshold is **T=0.170**, the lowest grid point meeting the 20%
false-positive gate. It reduced `no_target` FP from 48.0% to **18.0%** (TNR
82.0%) while retaining **80.25% supported macro recall** and 81.45% top-1 over
all target-bearing images. It rejected 126 of 841 target scenes.

Supported recalls at T=0.170:

- water bottle: 59.28%
- black vehicle: 90.50%
- bicycle: 92.31%
- cardboard box: 80.70%
- sports ball: 78.48%

The immediate Pareto neighborhood shows the trade-off: T=0.165 yields 22.33%
FP and 82.13% supported macro recall; T=0.175 yields 14.0% FP and 77.31% macro
recall. The complete deterministic curve is in `metrics.json`.

## Failures and limitations

Water-bottle recall falls to 59.28%, the largest supported-class cost. The
threshold was fitted and evaluated on the same Phase 1 validation collection,
so a separately sourced test set is still required. Thin parked classes are
not headline evidence. Open Images does not verify the colour adjectives, and
the image domain is not aerial.

## Conclusion

The hypothesis passes on this collection. T=0.170 is frozen as the v0 flight
decision threshold and production `TargetObservation` now reports invisible
with null label/id below it. This is a similarity confidence, not a softmax or
invented 1.0. Phase 2 may compare a compact student under this same rule, but
the threshold remains subject to external-domain validation.
