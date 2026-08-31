# Flight integration

The first flight MVP uses MaixCAM2 as an isolated perception-and-logging
payload. It reads its integrated camera, writes detection and tracking JSONL,
and may record video. It does not send movement commands to the flight
controller.

The flight controller independently handles stabilisation, pilot input, motors,
arming, and failsafes.

The planned aircraft is a self-assembled approximately 7-inch quadcopter with a
standard Pixhawk-class or SpeedyBee-class flight-controller stack. MaixCAM2 is
powered from the flight LiPo through a dedicated fused 5 V BEC, never a thin FC
5 V rail. It is belly/nadir mounted on vibration isolation and clear of the
propeller disks. See [hardware integration](../hardware/README.md) for the
payload boundary and safety ladder.
