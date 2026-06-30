"""Hardware abstraction layer.

Each peripheral is wrapped behind a small, independently testable class:

    l6470  — 3x stepper drivers over SPI0 daisy-chain (single CS), mode 3
    imu    — BNO055 9-axis IMU in UART mode (/dev/serial0)
    camera — Pi Camera Module V3 via picamera2 -> NumPy arrays for OpenCV

Implemented incrementally in milestone M1 (issues #5-#7).
"""
