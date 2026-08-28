# µAeroVLM Project Specification

**Status:** Canonical engineering specification, version 0.2  
**Last reviewed:** 28 August 2026  
**Working name:** µAeroVLM, pronounced “micro Aero VLM”  
**Repository:** `wam_drones`  
**Primary objective:** Build and evaluate a very low-cost, fully local, language-conditioned autonomous quadcopter by compiling useful vision-language capabilities into a microcontroller-class perception and control system.

This document is the project’s source of truth. Future design discussions, implementation work, purchases, experiments, and reviews should begin here. When a decision changes, update this document and its decision log instead of allowing a chat transcript to become the only record.

---

## 1. Executive summary

The project will build a low-cost quadcopter that can execute bounded language-conditioned tasks such as:

- “Find the red backpack.”
- “Approach the traffic cone.”
- “Follow the person in the yellow jacket.”
- “Circle the blue vehicle.”
- “Inspect the solar panel.”

The central research question is:

> How much useful vision-language-guided aerial autonomy can be retained when onboard AI is constrained by extreme cost, memory, power, and weight limits?

The current compute study is leaning toward an ESP32-P4-class device rather than an NVIDIA Jetson. This is a design option, not a requirement. A conventional generative VLM will not fit on that class of hardware. The current model architecture is therefore leaning toward distilling a larger vision-language teacher into a small visual student, caching or precomputing text embeddings, running semantic perception at a low rate, tracking targets at a higher rate, and leaving flight stabilisation to an independent flight controller.

The result should be described accurately as a **distilled language-conditioned aerial policy** or **microcontroller-scale open-vocabulary aerial perception system**. It should only be called a fully general onboard VLM if arbitrary text is encoded and interpreted onboard.

The current comparison plan has two implementation tiers:

1. **Constrained implementation:** A microcontroller-class computer running compressed semantic perception, tracking, short-horizon latent prediction, and deterministic control. The current hardware lean is ESP32-P4.
2. **Upper-bound implementation:** A low-cost Linux/NPU board running a materially larger vision-language model. Current candidates include MaixCAM2 and RK3588-class boards.

The flight-control and semantic-compute responsibilities remain separate. The flight-control subsystem controls attitude, rates, arming, motor output, and vehicle failsafes. The semantic subsystem sends bounded motion setpoints through a documented interface and never commands individual motors. The current implementation lean is ArduPilot with MAVLink.

---

## 2. How to use this specification

Every technical statement falls into one of seven categories:

- **Requirement:** A part-independent statement of what every valid version 1 design must satisfy.
- **Design option:** A candidate implementation. Options may be compared or replaced without changing the underlying requirement.
- **Current lean:** The option presently favoured for prototyping. A lean is not a requirement or final selection.
- **Decision:** A process, scope, safety, or architecture decision that is intentionally fixed. Change it only through an explicit decision-log entry.
- **Verified:** Confirmed through an official source, code, or a reproduced experiment.
- **Target:** A measurable engineering goal that has not yet been achieved.
- **Hypothesis:** A research idea that must be tested.

Numbers from papers are not treated as verified on this project’s hardware until reproduced. Vendor performance claims are reference points, not acceptance results.

### 2.1 Governing authoring rules

This is an engineering specification, not a commercial PRD. It must not contain personas, market segmentation, commercial pricing, SKUs, go-to-market plans, sales language, or product-positioning claims.

All future edits must follow these rules:

1. Requirements state **what must be true**, not which part must be used.
2. A requirement must not name a specific MCU, board, motor, sensor, model family, firmware project, or supplier unless compatibility with that named external system is itself the required interface.
3. If a proposed requirement implies a part choice, rewrite it as a performance, physical, safety, or interface constraint.
4. Every requirement must include one primary verification method: **measure**, **inspect**, **demo**, or **analysis**.
5. Unknown values are written as **TBD**, followed by the measurement, experiment, or calculation required to resolve them.
6. Candidate parts and architectures belong in the separate Design Options section. List two or three credible alternatives and compare cost, mass, power, complexity, and relevant technical limits.
7. Mark a preferred candidate as **leaning toward**. Do not use “shall use” for a design option.
8. Costs in this document are engineering BOM constraints or comparison inputs. They are not retail pricing, packaging, or commercial plans.

Example:

- Bad requirement: “The aircraft shall use an ESP32-P4.”
- Good requirement: “The semantic-compute subsystem shall execute the deployed perception model within the latency, memory, mass, power, and cost limits in PERF-001 through PHY-006.”

When asking another person or LLM to review the project, request that they separate:

1. Factual errors
2. Architecture objections
3. Missing safety cases
4. Unmeasured assumptions
5. Better experiments
6. Scope changes

Reviews should propose changes to this document, not merely produce a second competing plan.

---

## 3. Project and research definition

### 3.1 Intended result

A low-cost quadcopter should accept a supported command, recognise or search for the requested target, generate safe motion setpoints, maintain target awareness over time, and complete a bounded aerial behaviour without cloud inference.

The first complete demonstration is:

> From a stationary armed position, accept `find and approach <target>`, take off under pilot supervision, visually locate one target from a known candidate vocabulary, approach to a configured stand-off distance, hold the target near the image centre, and stop or return safely when confidence is lost.

### 3.2 Research contribution

The intended contribution is the complete efficiency architecture, not one unusually small neural network:

- Distil a large image-text teacher into a micro visual encoder.
- Encode language once per command instead of once per frame.
- Use nested or Matryoshka embeddings so deployment can trade accuracy for memory.
- Run semantic inference only when required.
- Track the selected target between semantic updates.
- Predict short-horizon latent state rather than future video.
- Couple learned perception to deterministic geometric control and safety checks.
- Evaluate cost and energy per successful mission, not model accuracy alone.

### 3.3 Non-goals for version 1

Version 1 will not attempt:

- Arbitrary conversation with the drone
- Unbounded natural-language planning
- End-to-end motor control from a VLM
- Video generation
- Fully learned attitude stabilisation
- Swarm coordination
- GPS-denied SLAM in large unknown buildings
- Human-free safety certification
- Delivery or payload release
- Operations beyond visual line of sight
- Custom carbon-frame design

These may become later projects. They are not prerequisites for validating the core efficiency claim.

### 3.4 Language-capability levels

The project must declare which level each demonstration uses:

| Level | Language processing | Honest description |
|---|---|---|
| L0 | Fixed numeric target ID | Semantic target-conditioned policy |
| L1 | Onboard parser over a fixed command grammar and stored object prototypes | Onboard language-conditioned policy with bounded vocabulary |
| L2 | Arbitrary phrase converted once to an embedding by a phone or ground station | Edge-assisted open-vocabulary policy; flight inference is local |
| L3 | Arbitrary phrase encoded onboard by a local text encoder | Fully onboard open-vocabulary policy |
| L4 | Local generative VLM performs multi-step reasoning | Fully onboard VLM/VLA system |

The constrained implementation initially targets L1. L2 is a useful experiment but must not be reported as fully onboard. The upper-bound implementation targets L3 and may explore L4.

---

## 4. Engineering requirements

The requirements below govern version 1. Candidate parts in section 6 are valid only if they satisfy these requirements. Target values may be tightened after measurements; any change must preserve the requirement ID and record the reason.

Verification methods mean:

- **Measure:** Instrument or benchmark the built system and retain raw data.
- **Inspect:** Check a physical artefact, configuration, drawing, log, or source file against an objective criterion.
- **Demo:** Execute the defined behaviour under a repeatable test protocol and retain logs/video.
- **Analysis:** Use a documented calculation, simulation, or traceable evidence set.

### 4.1 Operational requirements

| ID | Requirement | Verification |
|---|---|---|
| OPS-001 | The aircraft shall accept a bounded onboard command containing one behaviour and one target concept. Version 1 shall support at least 10 target concepts and the behaviours `FIND`, `APPROACH`, and `HOLD`. | **Demo:** Execute every supported command through the production command interface and confirm the parsed state and resulting state-machine path. |
| OPS-002 | The constrained implementation shall perform perception and action selection without cloud inference after a command is accepted. | **Inspect:** Disable all external network paths, inspect runtime configuration, and review network captures and logs for a complete mission. |
| OPS-003 | Given a supported visible target in the declared operating envelope, the system shall output target visibility, confidence, and normalised localisation. Required acquisition time is **TBD**; resolve it using closed-loop simulation and handheld trials to determine the maximum delay that still meets controller convergence. | **Demo:** Run the scene-separated acquisition protocol and report acquisition-time distribution, misses, and false acquisitions. |
| OPS-004 | When the target is temporarily occluded, the system shall either reacquire the same target or explicitly enter `TARGET_LOST`; it shall not reuse stale localisation beyond the validity limits. Maximum reacquisition time is **TBD**; resolve it from mission-level success versus delay. | **Demo:** Run controlled occlusion, exit/re-entry, and crossing-object trials with identity-switch counts. |
| OPS-005 | After persistent target loss, commanded forward velocity shall reach zero within 250 ms before search yaw is permitted. | **Measure:** Inject target loss with timestamped logs and calculate time from invalid observation to zero-forward command. |
| OPS-006 | The version 1 mission shall complete `find and approach <target>` from a stationary start, stop at a configured stand-off distance, and hold the target near image centre. Required success rate, centre error, distance error, and completion time are **TBD**; establish them from a manually controlled baseline and at least 30 simulated trials before freezing acceptance thresholds. | **Demo:** Execute the frozen mission protocol over repeated trials in simulation, then in a controlled flight area. |
| OPS-007 | A pilot shall be able to override autonomous motion immediately through an independent control path. Maximum end-to-end override latency is **TBD**; resolve it from transmitter input to flight-control mode/output response measurement. | **Measure:** Instrument and trigger override in bench, restrained, and flight tests. |

### 4.2 Performance requirements

| ID | Requirement | Verification |
|---|---|---|
| PERF-001 | The constrained semantic pipeline shall sustain at least 10 completed semantic observations per second at the deployed input resolution, including preprocessing and postprocessing. | **Measure:** Run a 30-minute live-camera benchmark and report median, p95, worst-case latency, dropped frames, and resets. |
| PERF-002 | The tracking pipeline shall update localisation at at least 20 Hz while a target is valid. | **Measure:** Replay timestamped motion sequences and report update-rate distribution and missed deadlines. |
| PERF-003 | The safety/setpoint loop shall run at at least 20 Hz, and no transmitted control intent shall be older than 250 ms. | **Measure:** Inspect timestamped intent and transport logs under normal load and induced compute overload. |
| PERF-004 | The deployed semantic model artefact for the constrained configuration shall be no larger than 2 MiB. Working memory is **TBD**; resolve it by measuring peak internal and external memory with camera, tracking, communications, and logging active concurrently. | **Measure:** Record model binary size and runtime high-water marks on target hardware. |
| PERF-005 | Quantisation shall not reduce the primary scene-separated target-selection metric by more than **TBD percentage points**. Resolve the limit by relating quantised accuracy loss to mission success during Phase 4. | **Analysis:** Compare floating-point, exported, quantised-desktop, and device outputs on the frozen validation set. |
| PERF-006 | The complete AI loop shall run continuously for at least 30 minutes without reset, watchdog trip, unbounded allocation growth, or missed-deadline rate above **TBD**. Resolve the deadline allowance from controller stability tests. | **Measure:** Conduct a thermal and memory soak with live camera, inference, tracking, communications, and logging. |

### 4.3 Physical and environmental requirements

| ID | Requirement | Verification |
|---|---|---|
| PHY-001 | Total take-off mass is **TBD**. Resolve it from component measurements, required thrust-to-weight margin, legal mass category, desired flight time, and motor/propulsion test data before the propulsion system is frozen. | **Analysis:** Maintain a mass budget and propulsion calculation; confirm final value on a calibrated scale. |
| PHY-002 | The AI payload, including compute, camera, regulator, storage, mount, wiring, and cooling, shall have mass no greater than 60 g. | **Measure:** Weigh the complete flight-ready AI payload as installed. |
| PHY-003 | The installed component envelope and camera field of view are **TBD**. Resolve them after bench hardware inventory by measuring actual boards, connectors, bend radii, cooling clearance, propeller clearance, and required camera angle. | **Inspect:** Compare measured installation and CAD drawings against the frozen envelope and clearance drawing. |
| PHY-004 | The semantic-compute subsystem shall operate from one documented regulated power interface. Nominal voltage, tolerance, peak current, ripple limit, connector, and transient requirement are **TBD**; resolve them from candidate-board datasheets and measured worst-case load before regulator selection. | **Measure:** Sweep input tolerance and record rail voltage, ripple, peak current, resets, and flight-controller sensor interference. |
| PHY-005 | Average AI payload power during the frozen mission shall not exceed 2 W. Peak power and allowed duration are **TBD**; resolve them with a logging power analyser during worst-case concurrent operation. | **Measure:** Integrate power over repeated mission traces and report average, peak, energy, and thermal state. |
| PHY-006 | The production-oriented AI module and camera engineering BOM shall not exceed US$30 equivalent, excluding development-only tools and shipping. The complete prototype-aircraft BOM shall not exceed US$300 equivalent, excluding transmitter, charger, and development tools. | **Analysis:** Audit the dated BOM, quantities, currency conversion, and development-only exclusions. |
| PHY-007 | The version 1 operating environment is **TBD** for temperature, wind, lighting, precipitation, dust, altitude, and indoor/outdoor use. Resolve each limit through sensor datasheets, legal constraints, and progressive environmental tests before autonomous outdoor acceptance. | **Analysis:** Publish the operating-envelope table with evidence; **demo** boundary cases that can be tested safely. |
| PHY-008 | The airframe shall preserve required sensor function and mechanical clearance under expected vibration and acceleration. Quantitative vibration limits are **TBD**; resolve them from flight-controller vibration logs, camera image quality, fastener inspection, and mount tests. | **Measure:** Review vibration spectra, image blur, mount displacement, and post-flight mechanical inspection. |

### 4.4 Safety requirements

| ID | Requirement | Verification |
|---|---|---|
| SAFE-001 | Semantic or learned-control software shall not command individual motors or bypass the stabilisation subsystem. | **Inspect:** Review interfaces, message permissions, source paths, and hardware wiring. |
| SAFE-002 | All autonomous velocity, acceleration, altitude, range, and yaw-rate commands shall pass through fixed configurable bounds outside the learned model. Exact bounds are **TBD** and shall be frozen per flight-test level. | **Inspect:** Review configuration and tests; **demo** saturation with out-of-range synthetic model outputs. |
| SAFE-003 | Loss of semantic-compute heartbeat or valid setpoints shall trigger a deterministic fallback sequence. Timeout values are **TBD** except the 250 ms intent-age limit; resolve them in simulation before flight. | **Demo:** Inject heartbeat loss, stale observations, communications loss, and process reset in simulation and hardware-in-the-loop tests. |
| SAFE-004 | RC override, disarm, and flight-controller failsafes shall remain available when the semantic computer is unpowered, failed, or disconnected. | **Demo:** Repeat the override and failsafe protocol with the semantic subsystem disconnected. |
| SAFE-005 | Autonomous flight testing shall progress through simulation, propeller-free bench test, handheld/gimbal test, pilot-observe flight, limited-authority flight, and only then full version 1 authority. | **Inspect:** Verify signed test records and gate results before each authority increase. |
| SAFE-006 | Low confidence, unsupported commands, unsupported targets, and stale tracks shall not produce forward approach motion. | **Demo:** Inject each invalid condition and verify neutral or fallback output. |

### 4.5 Interface and data requirements

| ID | Requirement | Verification |
|---|---|---|
| IF-001 | The semantic subsystem shall receive timestamped vehicle attitude, velocity, flight state, battery state, and available range/position estimates through a documented full-duplex interface. | **Inspect:** Review the interface schema and capture a valid end-to-end trace. |
| IF-002 | The semantic subsystem shall transmit bounded local-motion intent containing forward, lateral, vertical, yaw/yaw-rate, timestamp, confidence, and reason code at at least 20 Hz during autonomous control. | **Demo:** Exercise the interface in simulation and inspect all required fields and rates. |
| IF-003 | Image, vehicle, and command coordinate frames, units, sign conventions, and timestamps shall follow section 5.5 and shall never be inferred from unlabeled vectors. | **Inspect:** Review schemas and coordinate-conversion tests. |
| IF-004 | Runtime logs shall preserve command, observation, vehicle state, proposed action, safety-modified action, state transition, model/config identity, and timing counters. | **Inspect:** Decode a complete mission log and verify required fields and monotonic ordering. |
| IF-005 | Desktop video, live camera, constrained compute, and upper-bound compute implementations shall produce the same versioned `TargetObservation` contract. | **Demo:** Run contract tests against each available producer. |

### 4.6 Bench inventory constraints

The current bench inventory is unknown and must not be inferred from conversation. Complete this table before finalising purchases or physical interfaces:

| Item category | Owned exact item | Quantity | Condition | Measured mass | Interfaces/accessories | Evidence/action |
|---|---|---:|---|---:|---|---|
| Development computer and OS | TBD | TBD | TBD | N/A | TBD | Record model, OS, RAM, GPU/NPU, free storage, ports |
| Camera/compute boards | TBD | TBD | TBD | TBD | TBD | Photograph labels; record part/revision and run power-on test |
| Flight controllers | TBD | TBD | TBD | TBD | TBD | Record firmware support, sensor set, UARTs, and connector kit |
| Frame/propulsion components | TBD | TBD | TBD | TBD | TBD | Record frame size, motors, ESC, props, and ratings |
| Batteries and charger | TBD | TBD | TBD | TBD | TBD | Record chemistry, cell count, capacity, C rating, connector, health |
| RC transmitter/receiver | TBD | TBD | TBD | TBD | TBD | Record protocol, firmware, and failsafe capability |
| Sensors | TBD | TBD | TBD | TBD | TBD | Record range, update rate, voltage, bus, field of view |
| Power supplies/BECs/meters | TBD | TBD | TBD | TBD | TBD | Record voltage/current/ripple capability and calibration status |
| Fabrication and soldering tools | TBD | TBD | TBD | N/A | TBD | Identify missing crimp, solder, print, and measurement capabilities |

Verification: **Inspect** every item physically, photograph the bench, copy exact part numbers and revisions, weigh flight-bound items, and attach the completed inventory to the BOM before ordering duplicates.

---

## 5. Reference architecture, current lean

### 5.1 Responsibility boundary

```text
Operator / command source
        |
        v
Language parser or text embedding
        |
        v
Semantic perception ---- RGB camera
        |
        v
Target tracker <-------- optical flow / frame history
        |
        v
Latent state estimator <- range, FC attitude, velocity
        |
        v
Behaviour controller ---> candidate velocity/yaw commands
        |
        v
Safety shield ----------> confidence, range, geofence, timeouts
        |
        v
Documented motion-setpoint interface
        |
        v
Independent flight-control subsystem
        |
        v
ESC -> motors
```

The semantic computer owns semantic interpretation and high-level motion intent. The independent flight-control subsystem owns stabilisation and vehicle safety functions. The current prototype is leaning toward MAVLink Guided setpoints and ArduPilot, but the boundary applies to any conforming design.

### 5.2 Runtime rates

Initial target rates are:

| Loop | Target rate | Purpose |
|---|---:|---|
| Flight-controller attitude/rate loop | FC-managed, typically hundreds of Hz | Stable flight |
| AI safety and setpoint loop | 20–50 Hz | Bound and refresh setpoints |
| Image capture | 20–30 Hz | Current visual observation |
| Lightweight tracker | 20–30 Hz | Maintain target between semantic updates |
| Semantic encoder | 5–15 Hz | Recognise or reacquire requested target |
| Language processing | Once per new command | Produce command state and target prototype |
| Latent dynamics rollout | 10–30 Hz | Score short-horizon candidate actions |
| Persistent logging | Buffered, asynchronous | Preserve evidence without blocking control |

These are targets. Measured rates replace them after hardware benchmarking.

### 5.3 Core runtime state

The AI application maintains:

```text
CommandState
  command_id
  behaviour: FIND | APPROACH | FOLLOW | CIRCLE | INSPECT | HOLD
  target_label
  target_embedding[d]
  desired_distance_m
  command_timeout_ms

PerceptionState
  timestamp_us
  visible
  confidence
  bbox or heatmap
  center_x_normalized
  center_y_normalized
  embedding[d]

VehicleState
  armed
  flight_mode
  attitude
  angular_rates
  local_position if available
  velocity
  range_down
  range_forward if installed
  battery
  RC override state

ControlIntent
  timestamp_us
  forward_velocity_mps
  lateral_velocity_mps
  vertical_velocity_mps
  yaw_rate_rps
  confidence
  reason_code
```

Use fixed-size binary structures onboard. JSON may be used at desktop boundaries and during early prototypes, but not in the timing-sensitive MCU loop.

### 5.4 State machine

```text
DISARMED
  -> READY
  -> TAKEOFF
  -> SEARCH
  -> ACQUIRE
  -> TRACK
  -> EXECUTE_BEHAVIOUR
  -> HOLD_SUCCESS
  -> LAND or RETURN

Any active state
  -> DEGRADED when semantic confidence or sensors fall below threshold
  -> HOLD when the command stream expires briefly
  -> LAND/RETURN on persistent failure
  -> PILOT_OVERRIDE immediately on RC override
```

Every state transition must be logged with its trigger.

### 5.5 Coordinate, timing, and validity conventions

Use one convention at every interface and convert only at named boundaries:

- Image coordinates are normalised to `[0, 1]` with `(0, 0)` at the top-left, positive x to the right, and positive y downward.
- Bounding boxes use `[x_min, y_min, x_max, y_max]` in normalised image coordinates unless a field name explicitly says pixels.
- Camera bearing error is positive when the target lies to the right of image centre.
- Vehicle-local commands use the coordinate convention required by the selected MAVLink message. For `LOCAL_NED`, x is north/forward relative to the chosen frame, y east/right, and z down. Never infer the frame from an unlabeled vector.
- Angular quantities are radians in software interfaces. Human-facing logs may add degrees as a derived field.
- Distances are metres, velocities metres per second, accelerations metres per second squared, and time intervals seconds or explicitly suffixed milliseconds/microseconds.
- Runtime timestamps use a monotonic clock. UTC wall time is metadata, not a control-loop clock.
- Every observation and intent carries a timestamp and maximum age.
- Confidence values lie in `[0, 1]`, but must be calibrated before they are treated as probabilities.

Initial validity targets:

- A `TargetObservation` older than 200 ms is stale.
- A control intent older than 250 ms is not transmitted.
- MAVLink setpoints are refreshed at 20 Hz or faster during AI control.
- After two stale semantic observations, the tracker may continue only under a separately defined tracking timeout.
- After persistent target loss, forward motion becomes zero before any search yaw begins.

These are conservative starting targets. Replace them with measured values and a decision-log entry.

---

## 6. Design options

Design options are candidates, not requirements. The trade-off tables use approximate engineering comparisons until measured values and dated supplier quotes are added. “Leaning toward” identifies the current prototype direction only.

### 6.1 Airframe options

| Option | Cost | Mass/payload | Power/endurance | Complexity | Current assessment |
|---|---|---|---|---|---|
| Commodity 5-inch true-X | Low; exact landed cost TBD | Moderate frame mass with useful payload margin | Commodity propulsion; flight time TBD from propulsion calculation | Low; wide parts availability | **Leaning toward** for the first prototype |
| Commodity 3.5–4-inch true-X | Low to moderate | Lower vehicle mass but tighter AI payload and cooling margin | May improve total energy for a light payload; endurance TBD | Medium; integration is denser | Retain if the AI payload falls well below PHY-002 |
| Commodity 6–7-inch frame | Moderate | Highest payload and physical clearance | Potentially longer endurance but larger batteries and higher kinetic energy | Medium; less suitable for early controlled tests | Upper-payload fallback, not current lean |

The current 5-inch lean is based on replaceable commodity parts and expected payload margin. It is not frozen. Resolve the choice using PHY-001 through PHY-003, propulsion calculations, measured component mass, and controlled thrust tests.

CAD remains limited to mounts, guards, cooling, and cable management until physical dimensions and measured temperatures are known. A custom carbon frame is outside version 1.

### 6.2 Flight-control options

| Option | Cost | Mass/power | Software/interface | Complexity | Current assessment |
|---|---|---|---|---|---|
| H7 FPV-format controller running ArduPilot; MAVLink setpoints | Low to moderate; exact quote TBD | Low flight-controller mass and power | Mature SITL and Guided control; broad board support | Medium | **Leaning toward** |
| H7 controller running PX4; MAVLink Offboard setpoints | Low to moderate | Similar class of mass and power | Strong Offboard interface and simulation ecosystem | Medium | Credible alternative if board/tooling fit is better |
| Linux-integrated autopilot/companion platform | High relative to project target | Higher mass and power | Simplifies high-level integration | Lower software split but higher hardware cost | Comparator only unless MCU split proves infeasible |

Candidate H7 boards must be screened against SAFE-001 through SAFE-004 and IF-001/IF-002, including independent RC override, logging, sensor availability, full-duplex communication, and setpoint timeout behaviour. Matek H743-family boards are one documented example, not a requirement. [ArduPilot Matek H743 documentation](https://ardupilot.org/copter/docs/common-matekh743-wing.html)

### 6.3 Semantic-compute options

#### Option A: ESP32-P4-class microcontroller

Verified vendor specifications include a dual-core RISC-V CPU up to 400 MHz, 768 KB on-chip SRAM, AI instruction extensions, MIPI-CSI camera input, image-processing hardware, and parts with 16 or 32 MB PSRAM. ESP32-P4 does not itself provide Wi-Fi, so wireless operation requires a companion ESP32-C/S device or a board that integrates one. [ESP32-P4 product page](https://www.espressif.com/en/products/socs/esp32-p4)

Intended workload:

- RGB capture and preprocessing
- INT8 distilled visual encoder
- Target heatmap or region scoring
- Cached class embeddings
- Optical-flow or lightweight image tracking
- Small recurrent latent model
- MAVLink over UART
- Binary logging

It is not expected to run a normal 256M-parameter generative VLM.

Trade-off: lowest prospective cost and power with the highest model-conversion and memory risk. **Leaning toward** for the constrained implementation, subject to PERF-001 through PERF-006.

#### Option B: 256 MB Linux/NPU camera module

MaixCAM Pro integrates a 1 TOPS INT8 NPU, 256 MB DDR3, camera, Wi-Fi, Linux, and an RTOS core. It is useful for testing a larger vision encoder before full ESP32 compression. [MaixCAM Pro specifications](https://wiki.sipeed.com/hardware/en/maixcam/maixcam_pro.html)

Trade-off: more memory and accelerator throughput with higher power, cost, and vendor-toolchain dependence. Candidate intermediate platform.

#### Option C: 1–8 GB Linux/NPU board

Sipeed lists MaixCAM2 with 1 or 4 GB LPDDR4, 3.2 TOPS INT8, onboard camera, storage, Wi-Fi, and Transformer/LLM/VLM support. As of this document’s date it is newly introduced and its availability and production software maturity must be treated as risks. [MaixCAM2 specifications](https://wiki.sipeed.com/hardware/en/maixcam/maixcam2.html)

An RK3588-class 4–8 GB board is another option in this tier.

Trade-off: supports onboard text encoding and small generative VLM experiments, but likely fails the constrained implementation’s mass, power, and cost requirements. Use as an upper-bound comparator unless measurements show otherwise.

### 6.4 Perception-architecture options

| Option | Accuracy/flexibility | Compute/memory | Complexity | Current assessment |
|---|---|---|---|---|
| Cached text prototypes plus distilled visual encoder | Bounded vocabulary onboard; arbitrary phrases possible only at L2 | Lowest recurrent language cost | Medium training and deployment work | **Leaning toward** for constrained L1 |
| Tiny onboard text encoder plus distilled visual encoder | True L3 phrase encoding if it fits | Higher flash/RAM and command latency | High compression and operator risk | Research extension after L1 |
| Small generative VLM | Richest prompt and reasoning capability | Highest memory, latency, and energy | High | Upper-bound comparator |

### 6.5 Localisation options

| Option | Cost/power | Expected quality | Complexity | Current assessment |
|---|---|---|---|---|
| Class-agnostic proposals then semantic crop scoring | Repeated crop inference can be expensive | Easy to inspect and debug | Medium | **Leaning toward** for first desktop/device baseline |
| Dense semantic heatmap from shared backbone | Efficient reuse of features | May localise small targets poorly without multi-scale features | Higher model-design complexity | Second implementation if proposal cost is excessive |
| Fixed-vocabulary tiny detector | Lowest deployment risk | Does not preserve useful language/open-vocabulary behaviour | Low | Control baseline, not final semantic claim |

### 6.6 Navigation-sensor options

| Option | Cost/mass/power | Capability | Complexity | Current assessment |
|---|---|---|---|---|
| GNSS outdoors plus downward flow/range at low altitude | Low to moderate | Independent navigation and height cues | Medium integration | **Leaning toward** for first outdoor/low-altitude tests |
| Monocular RGB plus flight-controller state | Lowest added sensor cost | Weak independent collision/range evidence | High algorithmic and safety risk | Perception ablation only |
| Stereo/depth or scanning lidar | Highest cost, mass, and power | Better geometric safety and mapping | High integration | Upper-bound safety/perception comparator |

Exact sensor models must remain options until range, update rate, field of view, bus, voltage, environmental rating, mass, and landed cost are checked against the requirements.

### 6.7 Power-system options

| Option | Cost/mass | Electrical/flight effect | Complexity | Current assessment |
|---|---|---|---|---|
| 4S propulsion battery plus dedicated regulated AI rail | Commodity components and moderate voltage | Simpler initial setup; current draw and endurance TBD | Low to medium | **Leaning toward** for prototype |
| 6S propulsion battery plus dedicated regulated AI rail | Potentially lower propulsion current for equivalent power | Requires compatible propulsion and regulator selection | Medium | Alternative after propulsion analysis |
| Separate small AI battery | Added mass and charging burden | Electrical isolation and independent runtime | Medium | Diagnostic option if shared-rail noise cannot be controlled |

### 6.8 Bench-order options and timing

After completing the bench inventory, source one candidate from each required bench function:

- One constrained-compute camera development board; **leaning toward** an ESP32-P4 camera board
- Compatible RGB camera and lens for that board
- MicroSD card
- USB power meter or logging power analyser
- Candidate regulated supply/BEC selected only after PHY-004 input measurements are available
- Candidate flight controller satisfying SAFE-001 through SAFE-004 and IF-001/IF-002; **leaning toward** an ArduPilot-supported H7 board
- USB-UART tools, connectors, wire, soldering supplies
- Optional intermediate or upper-bound compute board

Delay the complete propulsion purchase until the known-device detector or first student model runs successfully and PHY-001 propulsion analysis is complete, unless shipping lead times justify ordering reversible standard components earlier.

### 6.9 Sensor functions required by the current architecture

The current low-cost architecture is leaning toward the following functions. Exact parts remain open:

- **RGB camera:** semantic perception input. Use short exposure and adequate lighting to reduce motion blur.
- **Downward optical flow:** velocity assistance and target tracking experiments.
- **Downward ToF/range sensor:** low-altitude height reference.
- **Optional forward range sensor:** inexpensive independent collision cue.
- **GPS/compass:** outdoor baseline and return capability.
- **FC IMU/barometer:** primary attitude and altitude-estimation inputs.

No learned perception output is considered sufficient by itself to guarantee collision avoidance.

### 6.10 Cost, mass, and power accounting

Keep three budgets separate:

1. **Development cost:** evaluation boards, tools, spare parts, power analyser, charger, and one-time equipment.
2. **Prototype unit cost:** the exact parts installed on the first working aircraft.
3. **Deployable unit cost:** estimated small-volume cost after removing screens, debug hardware, oversized connectors, and development-only accessories.

Initial project targets, not achieved facts:

| Metric | Target |
|---|---:|
| Production-oriented AI module plus camera | Below US$30 |
| Development AI board plus camera | Below US$75 |
| Average AI electrical power during a mission | Below 2 W |
| AI board, camera, regulator, mount, and cooling | Below 60 g |
| Complete prototype aircraft excluding transmitter, charger, and development tools | Below US$300 equivalent |
| Later small-volume deployable aircraft | Below US$200 equivalent |

Maintain a BOM spreadsheet with supplier, order date, currency, tax, shipping, quantity, measured mass, expected replacement rate, and whether the item is development-only. Report both raw component cost and landed Australian cost. Do not make a “cheap drone” claim by excluding required regulators, storage, sensors, or mounting hardware.

### 6.11 Teacher, student, and temporal-model options

| Function | Option | Compute/deployment trade-off | Semantic/control trade-off | Current assessment |
|---|---|---|---|---|
| Teacher | MobileCLIP2-S0 | Efficient desktop teacher with released code; not MCU-sized | Useful image-text embedding baseline | **Leaning toward** for the first baseline |
| Teacher | Larger SigLIP/CLIP encoder | Higher training and labelling cost | Potentially stronger teacher embeddings | Later teacher ablation |
| Teacher | Generative VLM plus detector | Highest offline cost and pipeline complexity | Can create captions, boxes, masks, and hard negatives | Data-generation option with human audit |
| Student | MCU-friendly convolutional encoder | Best current embedded operator support | May lose fine-grained and small-target accuracy | **Leaning toward** |
| Student | Tiny vision transformer | Attention/operator and memory risk | May preserve global semantics better | Export feasibility experiment |
| Student | Fixed-vocabulary detector | Lowest deployment risk | Does not preserve the target language capability | Required control baseline |
| Temporal model | Calibrated kinematic/filter model | Minimal learned compute | Limited semantic dynamics | Baseline required before learned prediction |
| Temporal model | Small GRU | Small recurrent state and common operators | Can learn action-conditioned target evolution | **Leaning toward** after tracker baseline |
| Temporal model | Small state-space or temporal-convolution model | Operator and tooling risk varies | May improve longer temporal context | Later ablation |

### 6.12 Software and deployment-toolchain options

| Layer | Option | Cost/mass/power effect | Complexity and compatibility | Current assessment |
|---|---|---|---|---|
| Training | Python, PyTorch, OpenCLIP/MobileCLIP, ONNX | Development-computer cost only; no flight mass or power | Broad model ecosystem and a direct export path; dependency pinning required | **Leaning toward** |
| Training | Python, JAX/Flax, ONNX | Development-computer cost only | Strong accelerator tooling; more conversion work for the current embedded candidates | Alternative if training throughput becomes limiting |
| Training | TensorFlow/Keras, TFLite export | Development-computer cost only | Mature quantisation path, but weaker fit with the current teacher code | Alternative if the selected runtime favours TFLite |
| Constrained runtime | Vendor MCU SDK plus vendor inference runtime | Lowest prospective flight mass, power, and unit component cost | Tight operator and memory constraints; board-specific integration | **Leaning toward** ESP-IDF plus ESP-DL if the ESP32-P4 option passes section 4 |
| Constrained runtime | TFLite Micro or CMSIS-NN-class runtime | Similar MCU resource class | More portable across supported MCUs; camera and operator integration may require more work | Portability alternative |
| NPU runtime | Vendor Linux/RTOS NPU SDK | Higher mass and power than MCU options | Easier deployment of larger networks; vendor conversion tools and kernels vary | Upper-bound alternative |
| Flight integration | SITL plus a documented setpoint protocol | No flight mass or power during simulation | Enables failure injection and automated tests before hardware | **Leaning toward** ArduPilot SITL plus MAVLink |
| Flight integration | Alternative autopilot simulator plus the same versioned intent contract | No flight mass or power during simulation | Preserves architecture portability; requires a second adapter and test matrix | Alternative if the selected controller is not ArduPilot |

---

## 7. Machine-learning reference architecture, current lean

### 7.1 Teacher-student design

The desktop teacher creates a semantically useful embedding space. The MCU student learns to map each image or image region into a much smaller representation that remains comparable to stored text prototypes.

The current teacher lean is:

- MobileCLIP2-S0 for reproducible image-text embeddings
- Optional larger SigLIP, CLIP, or VLM teachers for later ablations
- Optional teacher detector to create boxes, masks, captions, and hard negatives

Apple’s published MobileCLIP2-S0 architecture contains an 11.4M-parameter image encoder and 63.4M-parameter text encoder. This is a teacher or intermediate baseline, not the ESP32 deployment model. [MobileCLIP2 repository](https://github.com/apple/ml-mobileclip)

The current student lean is:

- MobileNetV2 width 0.35 or similarly MCU-friendly convolutional backbone
- 160×160 first input resolution, then 224×224 if affordable
- 64-dimensional L2-normalised output
- INT8 weights and activations
- Optional multi-scale features for target localisation
- Later Matryoshka dimensions: 16, 32, 64, 128

### 7.2 Why language is not decoded every frame

Autoregressive decoding repeatedly loads a large language model and creates a growing token cache. Drone control normally needs a small structured intent, not prose. Therefore:

1. Parse the command once.
2. Convert its target phrase to one embedding.
3. Cache that embedding.
4. Compare it with current visual features.
5. Convert geometric target error into bounded motion.

This preserves the semantic matching capability while removing most language computation from the control loop.

### 7.3 TinyVLM research lead and caveat

The February 2026 TinyVLM preprint proposes a related decoupled architecture using stored text embeddings, a MobileNetV2 0.35 visual student, Matryoshka embeddings, and INT8 prototypes. It reports a visual encoder requiring 892 KB flash and 285 KB RAM. It also reports ESP32-S3 performance, but the paper states that only its MAX78000 result was measured and other MCU results were projected. Its code was promised but not linked at the time of review. Treat the paper as a hypothesis source and comparison baseline, not reproduced evidence. [TinyVLM](https://arxiv.org/abs/2603.00136)

### 7.4 Localisation

A single whole-image embedding cannot reliably tell the controller where a target is. Version 1 will compare two approaches:

1. **Proposal then classify:** A tiny class-agnostic proposal head produces candidate boxes; the semantic encoder scores each crop against the command embedding.
2. **Dense semantic heatmap:** The backbone produces a spatial feature grid; each grid vector is compared with the target embedding to produce a heatmap.

The first baseline is leaning toward proposal then classify because it is easier to debug. Move to a dense head if crop inference is too slow or fails PERF-001.

### 7.5 Tracking

After a semantic acquisition, use a lightweight tracker to maintain the box between semantic updates. Candidate implementations include:

- Sparse optical flow around target features
- Correlation-filter tracking
- Template matching at reduced resolution
- A tiny learned tracking head

The semantic encoder must periodically verify the tracked target. Tracker confidence alone cannot preserve identity after occlusion or crossing objects.

### 7.6 Latent world model

The current world-model lean is a small task-relevant future-state predictor rather than a pixel generator:

```text
z_t = [target bearing, box scale, target confidence, optical flow,
       vehicle velocity, yaw rate, range cues, hidden state]

z_(t+1) = f(z_t, candidate_action)
```

Candidate implementations:

- GRU with 64–128 hidden dimensions
- Small temporal convolution
- Small state-space model if supported efficiently

Outputs:

- Predicted target visibility
- Predicted bearing and apparent scale
- Collision-risk proxy
- Progress toward desired stand-off distance
- Uncertainty

The current controller architecture evaluates a small discrete set of candidate actions and selects the best admissible action. A deterministic safety layer rejects unsafe candidates as required by SAFE-002.

### 7.7 Training objectives

The first training baseline is leaning toward:

```text
L_total = λ_embed L_cosine
        + λ_contrast L_contrastive
        + λ_loc L_localisation
        + λ_temp L_temporal
        + λ_action L_behaviour_cloning
```

Introduce losses in that order. Do not train the entire system jointly before each component has an independent baseline.

### 7.8 Quantisation and export

The constrained deployment path is currently leaning toward:

```text
PyTorch checkpoint
    -> ONNX with static input shapes
    -> ESP-PPQ calibration and INT8 quantisation
    -> .espdl model
    -> ESP-IDF component
    -> measured device inference
```

ESP-DL supports model loading and inference, common quantised operators, static memory planning, and conversion through ESP-PPQ. Current official guidance requires ESP-IDF 5.3 or newer and recommends checking operator support before designing/exporting the model. [ESP-DL](https://github.com/espressif/esp-dl)

The model must be designed around supported deployment operators. A desktop model that cannot be exported is not a successful model.

---

## 8. Data strategy

### 8.1 Data sources

Use a hybrid data pipeline:

- Public image-text data for initial teacher/student distillation
- Public object detection datasets for localisation
- Aerial datasets such as VisDrone/UAVDT for viewpoint adaptation, subject to licence review
- Simulation for controlled navigation trajectories and rare failure cases
- Handheld collection using the intended camera and lens
- Real drone video after safe manual flight is available

The handheld rig matters because it captures real exposure, lens distortion, compression, target size, and background statistics before the drone is ready.

### 8.2 Version 0 dataset

Start with 10 target concepts and approximately 200–1,000 useful images per concept, including negatives. Exact quantity matters less than diversity and clean splits.

Include:

- Multiple instances of each target
- Large and small image scale
- Partial occlusion
- Blur
- Lighting variation
- Confusing backgrounds
- Similar non-target objects
- Aerial and eye-level viewpoints
- Frames with no target

Split by physical scene or recording session, not randomly by frame. Adjacent video frames in train and validation would inflate results.

### 8.3 Annotation schema

Each frame may include:

```json
{
  "image_id": "session03_frame001242",
  "session_id": "session03",
  "timestamp_us": 124200000,
  "objects": [
    {
      "label": "red backpack",
      "bbox_xyxy": [112, 76, 183, 191],
      "occluded": false,
      "track_id": 4
    }
  ],
  "no_target": false,
  "camera": "prototype_camera_v1",
  "environment": "indoor_lab_a"
}
```

Maintain dataset licences, collection consent, and privacy notes beside each dataset manifest.

### 8.4 Teacher-generated data

Teacher labels are not ground truth. Sample and manually audit them. Track which labels came from humans and which came from a model. Do not evaluate the student only against teacher-generated labels.

---

## 9. Software reference stack and repository layout, current lean

This section describes the implementation currently favoured for the first prototype. It is not a requirements list. The alternatives are compared in section 6.12; changing a toolchain shall not change the contracts or acceptance criteria in section 4.

Create this structure as implementation begins:

```text
wam_drones/
├── README.md
├── PROJECT_SPEC.md
├── pyproject.toml
├── configs/
│   ├── models/
│   ├── training/
│   └── experiments/
├── data/
│   ├── README.md
│   ├── manifests/
│   └── samples/
├── training/
│   ├── teachers/
│   ├── students/
│   ├── distillation/
│   ├── localisation/
│   ├── temporal/
│   └── export/
├── evaluation/
│   ├── perception/
│   ├── control/
│   ├── energy/
│   └── missions/
├── firmware/
│   └── esp32_p4/
│       ├── main/
│       ├── components/
│       └── sdkconfig.defaults
├── flight/
│   ├── sitl/
│   ├── mavlink/
│   ├── controller/
│   └── safety/
├── interfaces/
│   ├── schemas/
│   └── generated/
├── hardware/
│   ├── bom/
│   ├── wiring/
│   ├── power/
│   └── mounts/
├── logs/
│   └── README.md
├── scripts/
└── tests/
```

Large raw datasets, model weights, firmware build outputs, and flight logs should not be committed directly to Git. Store manifests, hashes, download instructions, and small fixtures in the repository.

### 9.1 Desktop technology stack, current lean

- Python 3.11 unless a deployment dependency requires another pinned version
- `uv` for environments and dependency locking
- PyTorch for training
- OpenCLIP/MobileCLIP2 for teacher inference
- `timm` for vision backbones
- ONNX for model interchange
- Albumentations or torchvision transforms
- pytest for tests
- Ruff for linting/formatting
- mypy or Pyright for important typed interfaces
- MLflow or Weights & Biases for experiment records
- OpenCV for video, proposals, and tracking baselines

### 9.2 MCU technology stack, current lean

- ESP-IDF 5.3 or newer, pinned after first successful build
- ESP-DL
- ESP-PPQ
- C/C++ and FreeRTOS
- MAVLink C library
- Fixed allocation after startup wherever practical
- Watchdog and task-health monitoring
- Binary or CBOR logs with desktop conversion tools

### 9.3 Flight technology stack, current lean

- ArduPilot Copter
- ArduPilot SITL for simulation
- MAVLink 2 over UART on hardware and UDP in simulation
- Pymavlink for desktop prototyping
- MAVLink C on the MCU
- Mission Planner or QGroundControl for setup and review

ArduPilot SITL runs the autopilot without hardware and can simulate multicopters, optical flow, lidar, and other sensors. It is the current-lean simulation environment before real autonomous flight; SAFE-005 requires simulation but does not prescribe this simulator. [ArduPilot SITL](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html)

MAVLink’s offboard-control interface supports position, velocity, acceleration, and attitude setpoints. ArduPilot accepts relevant setpoints in Guided mode and can exit external-control modes when updates stop. This project will initially use local-frame velocity and yaw/yaw-rate commands, subject to ArduPilot’s exact message semantics. [MAVLink offboard control](https://mavlink.io/en/services/offboard_control.html) [ArduPilot MAVLink interface](https://ardupilot.org/dev/docs/mavlink-commands.html)

---

## 10. Step-by-step implementation plan

Each phase has an output and an exit gate. Do not advance because a demo “looks promising.” Advance when the gate is recorded as passed.

Named tools in these phases instantiate the current leans in sections 6, 7, and 9. If measurements select another option, substitute its equivalent step while preserving the phase output, requirement IDs, and exit gate.

### Phase 0: Establish project controls

Tasks:

1. Create the repository structure.
2. Add Python dependency management, linting, and tests.
3. Add an experiment naming scheme such as `exp_YYYYMMDD_short_name`.
4. Create an experiment record containing git commit, config hash, dataset manifest hash, random seed, hardware, metrics, and artefact paths.
5. Create an issue tracker with milestones matching this plan.
6. Choose the first 10 target concepts.
7. Freeze the first command grammar.

Exit gate:

- A new contributor can find this specification, set up the desktop environment, run tests, and identify the current experiment.

### Phase 1: Build the desktop semantic baseline

Tasks:

1. Collect or assemble the v0 dataset.
2. Implement dataset manifests and scene-separated splits.
3. Run MobileCLIP2-S0 over validation images and the 10 target prompts.
4. Record top-1, top-3, confusion matrix, target/no-target performance, and latency.
5. Save embeddings and qualitative error examples.
6. Add prompt-template experiments such as `a photo of {label}` and `an aerial view of {label}`.

Exit gate:

- The teacher provides materially better-than-random target selection on held-out scenes.
- Failures are documented by class, scale, blur, and viewpoint.

### Phase 2: Train the first compact student

Tasks:

1. Implement a 64-dimensional MobileNetV2 0.35 student.
2. Distil normalised teacher embeddings using cosine and contrastive losses.
3. Compare 160×160 and 224×224 input.
4. Add hard-negative sampling.
5. Measure parameter count, multiply-accumulate estimate, checkpoint size, peak desktop inference memory, and accuracy.
6. Export a static-shape ONNX model early.

Initial targets:

- Model at or below 2 MB after INT8 quantisation
- At least 70% top-1 on the deliberately small v0 task, subject to dataset difficulty
- No unsupported ONNX operators

Exit gate:

- A reproducible checkpoint beats the fixed-vocabulary visual baseline or provides a clear semantic-generalisation advantage.
- ONNX output matches PyTorch output within a recorded tolerance.

### Phase 3: Establish constrained-device deployment before custom-model optimisation

Tasks:

1. Install and pin the selected device SDK and inference runtime.
2. Build and flash its camera example.
3. Run a known vendor-supported inference model.
4. If the ESP32-P4 lean is selected, run ESPDet-Pico from Espressif’s ESP-Detection project; otherwise run the closest documented reference model for the selected runtime.
5. Measure image capture rate, inference latency, peak memory, power, and temperature.
6. Run continuously for at least 30 minutes and check resets or allocation growth.

Espressif reports that its 0.36M-parameter ESPDet-Pico at 224×224 takes 51.4 ms on ESP32-P4 and 126.2 ms on ESP32-S3, including preprocessing and postprocessing. These numbers are comparison points until reproduced. [ESP-Detection](https://github.com/espressif/esp-detection)

Exit gate:

- Stable camera capture and known-model inference on the actual board
- A saved benchmark report with raw measurements

### Phase 4: Deploy the semantic student

Tasks:

1. Build a representative calibration set.
2. Quantise the ONNX student with ESP-PPQ.
3. Compare desktop FP32, desktop quantised, and device outputs.
4. Export `.espdl` and integrate it into firmware.
5. Implement normalisation and cosine scoring in fixed or quantised arithmetic.
6. Store command prototypes in flash.
7. Profile every stage: capture, resize, normalise, inference, similarity, output.
8. Tune internal SRAM allocation and PSRAM use.

Initial targets:

- At least 10 semantic evaluations per second
- No unexplained accuracy collapse after quantisation
- Stable peak memory with headroom for camera buffers, tracker, MAVLink, and logs
- AI compute power target below 2 W, measured rather than estimated

Exit gate:

- The board accepts a target ID or bounded text command and returns confidence plus target identity on live camera input.

### Phase 5: Add localisation

Tasks:

1. Implement proposal-then-classify on desktop video.
2. Evaluate detection AP, centre error, false acquisition rate, and no-target rejection.
3. Deploy a minimal proposal head or reuse an ESP-friendly detector.
4. Limit crop count and reuse backbone features where possible.
5. Compare against a dense semantic heatmap.

Exit gate:

- Live device output contains a useful target box or heatmap, not only a class label.
- Centre error is low enough for a gimbal or simulated yaw controller to converge.

### Phase 6: Add temporal tracking

Tasks:

1. Implement tracker baselines on recorded video.
2. Define acquisition, verification, lost-target, and reacquisition thresholds.
3. Run semantic inference less frequently while tracking at camera rate.
4. Test occlusion, crossing objects, blur, target exit, and re-entry.
5. Measure compute saved and identity-switch rate.

Exit gate:

- The tracker reduces semantic duty cycle without unacceptable identity switches.
- The system explicitly reports target loss instead of reusing stale coordinates indefinitely.

### Phase 7: Build the controller in simulation

Tasks:

1. Install the selected flight-stack simulator; the current lean is ArduPilot SITL.
2. Start a simulated quadcopter.
3. Implement the selected documented vehicle interface; the current lean is MAVLink heartbeat and telemetry ingestion.
4. Feed synthetic target-centre errors into the controller.
5. Send bounded velocity and yaw-rate setpoints.
6. Add command expiry, confidence decay, maximum acceleration, maximum speed, geofence, and pilot override.
7. Test dropped setpoints, stale telemetry, malformed commands, and simulated sensor failures.

Controller progression:

1. Yaw-only target centring
2. Yaw plus forward approach
3. Distance hold
4. Target following
5. Circle behaviour

Exit gate:

- Automated SITL tests prove convergence and safe timeout behaviour.
- No model output can bypass the safety limiter.

### Phase 8: Handheld and gimbal closed-loop testing

Tasks:

1. Mount the camera/compute assembly on a stick or two-axis gimbal.
2. Run live target acquisition and centring.
3. Measure reacquisition time and oscillation.
4. Walk the rig through different lighting and backgrounds.
5. Verify that target loss produces neutral or safe output.

Exit gate:

- Perception, tracking, state transitions, and controller intent work together without flight risk.

### Phase 9: Build and manually validate the aircraft

Tasks:

1. Assemble the standard frame, motors, ESC, FC, receiver, and battery system.
2. Inspect solder joints, continuity, polarity, prop direction, motor mapping, and fasteners.
3. Configure the selected flight-control firmware, calibrate sensors, and configure RC modes and failsafes.
4. Perform motor tests without propellers.
5. Perform restrained or controlled first hover with no AI control.
6. Tune and validate manual, altitude/position hold, return, and logging.
7. Measure baseline flight time and vibration before adding AI payload.

Exit gate:

- The aircraft is independently safe and stable under manual/autopilot control.
- A pilot can immediately override or disarm.

### Phase 10: Integrate compute, power, and communications

Tasks:

1. Mount the AI board and camera temporarily.
2. Power it from the dedicated regulated rail.
3. Test for brownouts and FC sensor noise across throttle range, initially without propellers where possible.
4. Connect the documented physical link and validate bidirectional interface heartbeats and telemetry.
5. Measure total payload mass, centre of gravity, current, temperature, and flight-time penalty.
6. Design and print the final mounts only after dimensions and thermal behaviour are known.

Exit gate:

- No compute reset, flight-controller reset, excessive sensor noise, thermal limit, or unsafe centre-of-gravity shift occurs during a controlled flight profile.

### Phase 11: Progressive autonomous flight tests

Use a safety cage or large clear controlled area, a spotter, conservative limits, and propeller removal whenever thrust is not required.

Progression:

1. AI observes and logs while pilot controls.
2. AI commands are displayed but not transmitted.
3. AI yaw control with pilot altitude/position control.
4. Short low-speed approach.
5. Full `find and approach` mission.
6. Target-loss and communications-loss injections.
7. Unseen target instances and unseen scenes.

Every level requires a preflight checklist, abort conditions, pilot override, and post-flight log review before advancing.

Exit gate:

- The first complete mission passes the predeclared success and safety criteria across repeated trials, not one demonstration.

### Phase 12: Add and evaluate the latent world model

Add temporal prediction only after the tracker/controller baseline is stable.

Tasks:

1. Log synchronised perception, vehicle state, action, and next-state data.
2. Train the small action-conditioned predictor.
3. Integrate candidate-action rollout.
4. Keep the deterministic safety shield unchanged.
5. Compare tracker-only control with tracker plus latent prediction.

Exit gate:

- The world model improves at least one predeclared outcome such as success, smoothness, reacquisition, or energy without increasing safety failures.

### Phase 13: Upper-bound comparison

Tasks:

1. Implement the same command and output interface on MaixCAM2 or RK3588.
2. Run MobileCLIP2 or a small VLM locally.
3. Use identical missions and metrics.
4. Compare capability, cost, power, mass, latency, and flight time.

Exit gate:

- The project can state precisely which capabilities the MCU preserves, loses, or performs more efficiently.

---

## 11. Testing strategy

### 11.1 Unit tests

Test:

- Command parsing
- Embedding normalisation and similarity
- Coordinate conventions
- Timestamp and stale-data handling
- State-machine transitions
- Control saturation
- MAVLink packing/unpacking
- Log encoding/decoding
- Config validation

### 11.2 Model tests

Test:

- Deterministic inference for a fixed seed/input
- PyTorch versus ONNX parity
- FP32 versus INT8 degradation
- Known, unknown, and no-target inputs
- Resolution and crop sensitivity
- Scene-separated validation
- Blur, noise, compression, brightness, rotation, and scale

### 11.3 Hardware-in-the-loop tests

Test:

- Camera disconnection
- SD-card failure or full card
- UART loss
- AI-board reset
- Flight-controller reboot detection
- BEC voltage sag
- Thermal soak
- RC override
- Battery failsafe

### 11.4 Flight acceptance metrics

Primary metrics:

- Mission success rate
- Collision/contact rate
- Pilot-intervention rate
- Target acquisition time
- Reacquisition time
- Target-centre error
- Stand-off distance error
- Completion time
- Energy used per successful mission

Efficiency metrics:

- Compute BOM cost
- Total vehicle BOM cost
- Compute mass
- Peak and average AI power
- Model flash size
- Peak SRAM/PSRAM
- Semantic latency and FPS
- Tracker latency and FPS
- Semantic duty cycle
- Flight-time penalty

Generalisation metrics:

- Seen class, seen environment
- Seen class, unseen environment
- Unseen instance of seen class
- Unseen attribute combination
- Unsupported class/no-target rejection

### 11.5 Required ablations

At minimum compare:

- Teacher versus student
- FP32 versus INT8
- 160 versus 224 input
- 16/32/64/128 embedding dimensions
- Semantic every frame versus semantic plus tracker
- Tracker-only versus tracker plus latent model
- Stored prompt template variants
- ESP32-P4 versus upper-bound board
- With versus without deterministic safety shield in simulation only

Never disable the safety shield during real flight to create an ablation.

---

## 12. Safety, security, and operational rules

### 12.1 Control safety

- AI never outputs motor commands.
- AI commands are bounded in speed, acceleration, altitude, range, and yaw rate.
- Setpoints expire quickly and require continuous refresh.
- Loss of AI heartbeat leads to hold, return, land, or pilot control according to the test configuration.
- RC override has priority over autonomy.
- Arming remains controlled by the flight stack and pilot workflow.
- Autonomous testing begins without propellers, then in simulation, then on a gimbal/rig, then with progressive flight authority.

Fallback order:

1. Reject an unsafe proposed action and choose a lower-speed admissible action.
2. If no useful action is admissible, command zero horizontal velocity and controlled yaw only when permitted.
3. If observation or command freshness fails briefly, hold using the flight controller.
4. If the failure persists and position estimation remains healthy, return or land according to the test plan.
5. If the autonomous stack cannot establish a safe response, yield to pilot override and the configured independent flight-controller failsafes.

The learned model may rank actions. It may not disable, weaken, or rewrite this fallback order at runtime.

### 12.2 Perception safety

- Low confidence cannot be converted into aggressive motion.
- A stale track cannot remain valid indefinitely.
- No-target and unsupported-target states are first-class outcomes.
- Semantic identity must be reverified after occlusion.
- Range and geofence constraints override semantic desire.

### 12.3 Data and privacy

- Avoid collecting identifiable people without consent.
- Document dataset licences and permitted uses.
- Do not publish home addresses, GPS traces, private network credentials, or identifiable flight footage without review.
- Encrypt or remove sensitive logs before sharing.

### 12.4 Regulatory work

Before outdoor operation, check the current rules for the operating country, location, aircraft mass, line-of-sight requirements, people/property separation, recording/privacy, and autonomous operation. Regulations change and must be reviewed again at flight time. This specification is not legal advice.

---

## 13. Logging and reproducibility

Each inference or control log should be reconstructable on desktop. Record:

- Monotonic timestamp
- Camera frame ID
- Command ID and target ID
- Semantic confidence and localisation
- Tracker confidence
- Vehicle telemetry snapshot
- Proposed action
- Safety-modified action
- State-machine state and transition reason
- Model version and config hash
- Dropped-frame and deadline counters

Do not synchronously write verbose JSON from the real-time loop. Buffer compact records and convert them after the run.

Each experiment report must include:

```text
experiment ID
question
hypothesis
git commit
hardware revision
firmware version
model hash
dataset manifest hash
configuration
protocol
raw artefacts
metrics
failures
conclusion
next decision
```

Video demonstrations supplement logs; they do not replace measured results.

---

## 14. Milestones and definition of done

### M0: Reproducible repository

Desktop environment, tests, config system, experiment records, and documentation exist.

### M1: Desktop semantic prototype

`run_target` accepts a video/image and supported target, then returns visibility, confidence, and localisation.

### M2: MCU semantic prototype

The selected constrained camera/compute device performs local target selection/localisation with recorded latency, memory, power, and accuracy. The current candidate is ESP32-P4-class hardware.

### M3: Simulated autonomous behaviour

The selected flight-stack simulator completes target-centring and approach behaviours with tested failure handling. The current candidate is ArduPilot SITL.

### M4: Stable manually flown aircraft

The physical quad is safe and stable before AI authority is enabled.

### M5: First language-conditioned flight

Bounded onboard command selects a target and drives a supervised autonomous behaviour.

### M6: Efficiency result

Repeated missions establish cost, power, model-size, latency, success, and flight-time results.

### M7: Research comparison

Required ablations and upper-bound comparison are complete, with all claims traceable to evidence.

The project is not “done” when one edited video works. Version 1 is done when another builder can reproduce the model, firmware, aircraft configuration, and benchmark protocol from released artefacts.

---

## 15. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Student loses too much semantic accuracy | Commands are unreliable | Narrow v0 vocabulary, improve teacher/data, use larger intermediate tier, measure graceful scaling |
| Selected runtime lacks a required model operator | Model cannot deploy | Export ONNX early, check the selected runtime’s operator support, maintain deployment tests from week one |
| External-memory latency dominates | Low FPS despite fitting | Use internal-memory planning, smaller resolution, fewer features, fused preprocessing, and per-layer profiling |
| Rolling-shutter blur | Target loss during motion | Short exposure, better lighting/lens, slower initial flights, blur augmentation, tracker verification |
| Text prototypes are too bounded | “Open vocabulary” claim is weak | Declare L1 honestly, evaluate L2/L3 separately, add onboard text encoder only when hardware permits |
| Tracker drifts | Wrong object followed | Periodic semantic verification, confidence decay, explicit lost state |
| Power noise resets compute or affects FC | Unsafe integration | Dedicated regulator, filtering, capacitors, wiring separation, load testing before props |
| Candidate-board availability changes | Schedule delay | Keep the compute options in section 6 behind the same software interfaces |
| Simulation overfits | Real flight fails | Handheld data, hardware camera tests, pilot-observe mode, progressive authority |
| Paper claims fail reproduction | Plan appears blocked | Treat TinyVLM as a hypothesis; own baseline and negative result remain valuable |
| Scope expands into a general VLA | Project never ships | Enforce v1 non-goals and milestone gates |

---

## 16. Purchase and CAD schedule

### Inventory before buying

1. Complete the bench inventory in section 4.6.
2. Identify which requirement-verification tasks existing equipment can perform.
3. List missing functions, not desired part numbers.
4. Compare two or three candidates for each missing function in section 6 or the BOM.
5. Record dated landed cost, measured/published mass, input requirements, interfaces, and software support.

### Source for immediate bench verification

- One constrained-compute camera platform; currently **leaning toward** an ESP32-P4 camera board
- Compatible RGB camera/lens and storage
- Calibrated power measurement equipment
- A regulator or bench supply only after PHY-004 is characterised
- One flight-control platform satisfying SAFE-001 through SAFE-004 and IF-001/IF-002; currently **leaning toward** an ArduPilot-compatible H7 controller
- Required development wiring and interface tools

### Buy after known-model device inference works

- Airframe selected from section 6.1 after PHY-001 through PHY-003 analysis
- Propulsion set selected from measured mass, thrust, efficiency, and electrical requirements
- Battery and charger compatible with the selected propulsion and safety procedures
- RC control link satisfying OPS-007 and SAFE-004
- Navigation sensors selected by required range, rate, field of view, interface, mass, power, and operating environment

These are standard parts and may be ordered earlier when shipping delays outweigh the risk.

### Begin CAD after

- Selected board and lens are physically present
- Power connector and control-interface routing are fixed
- Peak temperature is measured
- Desired camera angle is tested
- Payload mass and centre of gravity are known

CAD outputs should be mounts, guards, ducts, and strain relief. A custom airframe is outside version 1.

---

## 17. Immediate first 14 days

### Days 1–2

- Create repository skeleton and Python environment.
- Select the 10 target concepts.
- Define the L1 command grammar.
- Create dataset and experiment manifest schemas.
- Complete the section 4.6 bench inventory and source only missing bench functions.

### Days 3–4

- Collect the first real images and negative examples.
- Implement the selected teacher baseline; currently **leaning toward** MobileCLIP2-S0.
- Produce baseline metrics and an error gallery.

### Days 5–7

- Implement the selected compact student baseline; currently **leaning toward** a 64-dimensional MCU-friendly convolutional encoder.
- Train the first distillation baseline.
- Export ONNX immediately.
- Add PyTorch/ONNX parity tests.

### Days 8–10

- Install the selected constrained-compute SDK/runtime; the current lean is ESP-IDF plus ESP-DL.
- Build its firmware examples even if hardware has not arrived.
- Implement the desktop `run_target` interface over images and video.
- Start the selected flight-stack simulator; the current lean is ArduPilot SITL with Pymavlink.

### Days 11–12

- Implement synthetic target-error to yaw-rate control in SITL.
- Add setpoint expiry and safety saturation tests.
- Improve dataset coverage based on teacher/student failures.

### Days 13–14

- Quantise the first student using the selected deployment tool; the current lean is ESP-PPQ.
- Record FP32/INT8 accuracy and size.
- If hardware has arrived, run camera and known-model benchmarks.
- Publish the first experiment report and update this specification’s measured baselines.

At the end of day 14, the expected result is not a flying drone. It is a reproducible semantic model, an export path, a simulated safe controller, and a bench deployment either running or ready to flash.

---

## 18. Decision and leaning log

| ID | Type | Date | Decision or lean | Reason | Revisit trigger |
|---|---|---|---|---|---|
| D-001 | Scope decision | 2026-08-28 | Make cost and efficiency the central contribution | High-cost onboard VLMs do not test the intended resource limits | Evidence that constrained cost prevents any useful mission capability |
| L-001 | Current lean | 2026-08-28 | Leaning toward ESP32-P4 for constrained compute | Low prospective component cost, camera support, and an embedded inference ecosystem | Failure against PERF-001 through PERF-006 or a better measured candidate |
| D-002 | Safety architecture | 2026-08-28 | Separate semantic compute from stabilisation and motor output | A learned perception failure must not directly destabilise the aircraft | Not revisited for version 1 |
| L-002 | Current lean | 2026-08-28 | Leaning toward ArduPilot and MAVLink | SITL, Guided external control, and broad H7 support | Interface, board, or test limitation against the requirements |
| L-003 | Current lean | 2026-08-28 | Leaning toward a commodity 5-inch true-X frame | Expected replaceability and payload margin | PHY-001 through PHY-003 analysis favours another option |
| D-003 | Scope decision | 2026-08-28 | Begin at language level L1 | Fully onboard bounded commands avoid requiring a large text encoder | L1 milestone complete or vocabulary proves inadequate |
| L-004 | Current lean | 2026-08-28 | Leaning toward MobileCLIP2-S0 and a 64-dimensional convolutional student | Reproducible teacher and simple embedding contract | Another teacher/student pair wins the frozen benchmark and export checks |
| D-004 | Test sequencing | 2026-08-28 | Add learned world modelling only after tracker/controller baseline | Separates perception, control, and prediction failures | Phase 11 baseline complete |
| D-005 | Mechanical sequencing | 2026-08-28 | Delay CAD until bench measurements exist | Mount geometry, cooling, and mass are currently unknown | Compute hardware and PHY-003 are characterised |
| D-006 | Specification rule | 2026-08-28 | Separate part-independent requirements from design options and require verification for every requirement | Prevents preferred components from becoming unexamined constraints | Only if this document is replaced by another engineering-specification standard |
| D-007 | Scope decision | 2026-08-28 | Support `water bottle`, `black vehicle`, `bicycle`, `cardboard box`, and `sports ball` for Phase 2 headlines and the first mission; park the other five v0 names while retaining them in the vocabulary and parser | Phase 1 validation is badly unbalanced, thin classes cannot support claims, and Open Images does not verify colour-qualified names | New licensed, scene-separated evidence makes a parked class statistically defensible |

---

## 19. Open questions

These are intentionally unresolved:

1. Which exact ESP32-P4 camera board offers the best size, power, camera support, and availability?
2. Which H7 FC and ESC combination offers the best local cost and availability in Australia?
3. Does proposal-then-classify or a dense heatmap provide the best MCU localisation trade-off?
4. What semantic rate is needed once tracking is active?
5. Can a useful onboard text encoder fit within ESP32-P4 memory, or should arbitrary phrases remain L2?
6. Is a GRU latent predictor materially better than a calibrated kinematic predictor?
7. Which range sensors provide sufficient independent safety at acceptable cost and weight?
8. What exact v1 mission and environment provide a convincing but reproducible benchmark?

Resolve each through a time-bounded experiment or purchase comparison, then add a decision-log entry.

---

## 20. External references

Primary implementation and hardware references:

- [ESP32-P4 product specifications](https://www.espressif.com/en/products/socs/esp32-p4)
- [ESP-DL inference framework](https://github.com/espressif/esp-dl)
- [ESP-Detection and ESPDet-Pico](https://github.com/espressif/esp-detection)
- [MobileCLIP2 official implementation](https://github.com/apple/ml-mobileclip)
- [ArduPilot SITL](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html)
- [ArduPilot MAVLink interface](https://ardupilot.org/dev/docs/mavlink-commands.html)
- [MAVLink offboard-control protocol](https://mavlink.io/en/services/offboard_control.html)
- [ArduPilot Matek H743 board documentation](https://ardupilot.org/copter/docs/common-matekh743-wing.html)
- [MaixCAM Pro specifications](https://wiki.sipeed.com/hardware/en/maixcam/maixcam_pro.html)
- [MaixCAM2 specifications](https://wiki.sipeed.com/hardware/en/maixcam/maixcam2.html)

Research references to reproduce or compare:

- [TinyVLM: Zero-Shot Object Detection on Microcontrollers](https://arxiv.org/abs/2603.00136)
- [SINGER: An Onboard Generalist Vision-Language Navigation Policy for Drones](https://arxiv.org/abs/2509.18610)
- [AirHunt: VLM Semantics and Continuous Planning for Aerial Object Navigation](https://arxiv.org/abs/2601.12742)
- [VLA-AN: Efficient Onboard Vision-Language-Action Aerial Navigation](https://arxiv.org/abs/2512.15258)
- [FastVLM official implementation](https://github.com/apple/ml-fastvlm)
- [SmolVLM2-256M model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)

Re-check software versions, prices, board availability, regulations, and paper/code status before acting on them. The retrieval date for the current review is 28 August 2026.

---

## 21. Canonical next action

The next implementation task is:

> Create the repository skeleton, complete the bench inventory, choose the 10-target vocabulary, implement the selected desktop teacher baseline, and define a tested `TargetObservation` interface that can later be produced by recorded video or any constrained/upper-bound compute option.

MobileCLIP2-S0 and ESP32-P4 are current leans from section 6, not requirements. Before sourcing hardware, complete the bench inventory and compare candidates against section 4. Hardware is required for final deployment measurements, but it is not required to begin the model, interface, simulation, controller, test, or export work.

---

## Appendix A: Standard flight-test checklist

Adapt this checklist to the site and current test level. A checked item means it was verified for this flight, not assumed from a previous flight.

### Before travelling to the test site

- [ ] Test objective, success criteria, abort criteria, and allowed AI authority are written down.
- [ ] Current firmware, model, and configuration hashes are recorded.
- [ ] Simulation and bench tests required for this flight level pass.
- [ ] Batteries are undamaged, balanced, charged appropriately, and transported safely.
- [ ] Weather, airspace, site permission, people/property separation, and applicable rules are checked.
- [ ] Pilot, spotter, and test lead responsibilities are assigned.

### Aircraft inspection

- [ ] Frame, arms, fasteners, landing points, and mounts are secure.
- [ ] Propellers are undamaged, correctly oriented, and tightened only after non-thrust checks finish.
- [ ] Motor and ESC wiring is intact.
- [ ] Flight controller, receiver, GPS, camera, AI board, regulator, and antennas are secure.
- [ ] Centre of gravity is acceptable with the installed battery.
- [ ] No wire can enter a propeller or pull on a connector.
- [ ] SD cards have space and logging is enabled.

### Powered checks

- [ ] Correct model and configuration identifiers appear in telemetry.
- [ ] RC link, pilot override, mode switch, and disarm work.
- [ ] Battery voltage and current readings are plausible.
- [ ] Attitude, GPS/position, range, optical flow, camera, and AI heartbeat are healthy as required.
- [ ] Target command and perception output are correct while stationary.
- [ ] AI setpoints remain inhibited until the planned enable action.
- [ ] Geofence, speed, altitude, setpoint-age, and confidence limits match the test plan.
- [ ] Return/land behaviour and home position are configured.

### Immediately before takeoff

- [ ] Area and flight volume are clear.
- [ ] Spotter confirms readiness.
- [ ] Recording and logs are running.
- [ ] Pilot states the abort action aloud.
- [ ] Start in manual or standard autopilot control; enable AI only at the planned test point.

### After landing

- [ ] Disarm and disconnect the battery before approaching electronics or props.
- [ ] Record damage, heat, resets, warnings, and unexpected behaviour.
- [ ] Copy and hash flight-controller, AI, video, and power logs.
- [ ] Mark the test pass, fail, or invalid against predeclared criteria.
- [ ] Do not advance to the next authority level until failures are explained.

---

## Appendix B: Standard experiment report

```markdown
# Experiment <ID>: <title>

## Question
What single uncertainty does this experiment resolve?

## Hypothesis
State the expected measurable result before running the experiment.

## Configuration
- Git commit:
- Model/config hash:
- Dataset manifest hash:
- Hardware revision:
- Firmware versions:
- Random seeds:

## Protocol
List repeatable steps, sample count, environment, and exclusion rules.

## Acceptance criteria
State pass/fail thresholds before viewing the result.

## Results
Link raw logs and report summary metrics with uncertainty where possible.

## Failures and anomalies
Include resets, dropped frames, invalid trials, and protocol deviations.

## Conclusion
State what the evidence supports, does not support, and the next decision.
```
