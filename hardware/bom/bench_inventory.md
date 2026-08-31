# Bench inventory

Fill this inventory by photographing and physically inspecting the bench; do
not infer owned hardware from chat. Record exact part numbers and revisions,
weigh flight-bound items, and retain evidence before ordering duplicates.

## Planned flight-compute baseline

This is an intended purchase/configuration, not an assertion of owned hardware:

| Component | Selected baseline | Notes |
|---|---|---|
| Airframe | Self-assembled approximately 7-inch quadcopter | Complete and hover-tune without the perception payload first. |
| Flight controller | Pixhawk-class or SpeedyBee-class stack | ArduPilot or PX4 flies; payload has no control authority. |
| Perception payload | MaixCAM2 (Axera AX630C) | Integrated camera, Linux, NPU, storage, M12 lens, and 1/4-20 mount. |
| Payload allowance | Approximately 150–250 g all-up | Measure MaixCAM2, mount, BEC, wiring, and storage before flight. |
| Payload power | Dedicated fused 5 V BEC from flight LiPo | Do not share a thin FC 5 V rail. |
| Mount | Rubber-ball isolated nadir/belly mount | Keep clear of propeller disks and record centre-of-gravity shift. |
| Display | Optional | Strip or disable for flight unless measurements justify it. |

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
