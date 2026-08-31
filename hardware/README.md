# Flight hardware and integration

## Flight baseline

The aircraft is a self-assembled approximately 7-inch quadcopter. A standard
Pixhawk-class or SpeedyBee-class flight-controller stack running ArduPilot or
PX4 performs stabilisation, motor control, pilot input, arming, failsafes, and
return-to-home.

The perception payload is MaixCAM2 (Axera AX630C): camera, Linux system, NPU,
storage, and M12 lens in one approximately 65 × 49 × 20 mm unit with a
1/4-20 mount. Its nominal board power is approximately 2.5 W. It detects,
tracks, records JSONL, and may record video. It has no command path to guided
mode or motor control in the MVP. The intended all-up perception payload
allowance is approximately 150–250 g; measure the actual mass of the MaixCAM2,
mount, BEC, wiring, and storage before flight.

## Power and mounting

```text
Flight LiPo ──> flight-controller/ESC power path ──> motors and flight control
     │
     └──> dedicated fused 5 V BEC ──> MaixCAM2
```

Do not power MaixCAM2 from a thin flight-controller 5 V rail. The first camera
position is a belly/nadir view to resemble the public aerial data. Use a
rubber-ball vibration-isolated mount, retain a clear view outside propeller
disks, and use the MaixCAM2 1/4-20 mount interface. A display/screen is
optional and should be removed or disabled for flight unless its value is
measured.

## Operational boundary

**The flight controller flies; the payload logs.** MVP communications are not
required: MaixCAM2 stores JSONL and video/log files locally, and clocks are
synchronised after landing. A later TELEM2 UART link may carry timestamps only;
it does not grant perception any control authority.

Follow the Phase 8 safety ladder in the implementation plan: tune without the
payload, check mass and centre of gravity, complete props-off and restrained
tests, then use short piloted logging-only flights.
