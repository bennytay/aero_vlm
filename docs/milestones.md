# Project milestones

These GitHub-style milestones mirror the implementation plan in
`PROJECT_SPEC.md` and do not require network access.

- **M0 Reproducible repository** — desktop environment, tests, configuration,
  experiment records, and documentation.
- **M1 Desktop semantic prototype** — image/video target selection and
  localisation through the shared observation contract.
- **M2 MCU semantic prototype** — constrained-device target selection with
  measured latency, memory, power, and accuracy.
- **M3 Simulated autonomous behaviour** — target centring and approach in
  simulation with failure handling.
- **M4 Stable manual aircraft** — safe, stable manual flight before AI authority.
- **M5 First language-conditioned flight** — supervised bounded-vocabulary
  target acquisition and approach.
- **M6 Efficiency result** — mission performance and resource measurements for
  the constrained implementation.
- **M7 Research comparison** — fair comparison with the upper-bound system and
  documented ablations.

## Phase 0 / M0 checklist

- [x] Repository structure, `uv`, linting, and tests
- [x] Frozen version 0 vocabulary
- [x] Frozen command grammar and parser tests
- [x] `TargetObservation` and `CommandState` contracts
- [x] Experiment metadata record and report template
- [x] Bench inventory template
- [x] Canonical specification linked from README
