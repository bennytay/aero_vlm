# Flight integration

The first flight MVP runs perception as an isolated companion payload. It reads
the camera, writes detection and tracking logs, and may provide a ground preview.
It does not send movement commands to the flight controller.

The flight controller independently handles stabilisation, pilot input, motors,
arming, and failsafes.

The planned companion payload computer is a Raspberry Pi 5 with 4GB RAM and a
Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L, INT8). It must use active cooling and
be power-budgeted with its camera and storage before flight integration.
