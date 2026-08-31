# Bench inventory

Fill this inventory by photographing and physically inspecting the bench; do
not infer owned hardware from chat. Record exact part numbers and revisions,
weigh flight-bound items, and retain evidence before ordering duplicates.

## Planned budget flight-compute baseline

This is an intended purchase/configuration, not an assertion of owned hardware:

| Component | Selected baseline | Notes |
|---|---|---|
| Companion computer | Raspberry Pi 5, 4GB RAM | Lean headless camera/tracker/logging pipeline; 2GB is out of scope. |
| AI accelerator | Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L, INT8) | Budget vision-inference target for Phase 6. |
| Cooling | Raspberry Pi Active Cooler | Required for sustained benchmark and flight thermal testing. |
| Bench power | Official Raspberry Pi 27W USB-C power supply | Bench-only reference supply; flight regulator is separately specified and tested. |
| Initial storage | 64–128GB high-endurance microSD | Upgrade only after measured video/log retention requires it. |

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
