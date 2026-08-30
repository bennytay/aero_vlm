# Flight integration

The first flight MVP runs perception as an isolated companion payload. It reads
the camera, writes detection and tracking logs, and may provide a ground preview.
It does not send movement commands to the flight controller.

The flight controller independently handles stabilisation, pilot input, motors,
arming, and failsafes.
