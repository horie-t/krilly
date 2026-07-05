"""BNO055 9-axis IMU over **UART** (issue #6).

We use the BNO055 in UART mode (sensor PS1 tied high, wired to the Pi's
``/dev/serial0``) rather than I2C: the Pi's I2C controller does not handle the
BNO055's clock stretching reliably, and UART also sidesteps the Blinka/RP1 GPIO
friction on the Pi 5. The register protocol is implemented directly on top of
``pyserial`` so it is fully unit-testable with a fake serial port.

UART register protocol (BNO055 datasheet §4.4):
- Write:  ``AA 00 <reg> <len> <data...>`` -> ack ``EE 01`` (0x01 = success)
- Read:   ``AA 01 <reg> <len>``           -> ``BB <len> <data...>`` on success,
                                             ``EE <code>`` on failure
- ``EE 07`` (bus over-run / busy) is a known transient; reads are retried.

All multi-byte values are signed 16-bit little-endian. Default unit selection:
Euler 16 LSB/deg, gyro 16 LSB/deg-per-sec, quaternion 2^14 LSB/unit.
"""

from __future__ import annotations

import time

# --- registers (page 0) ----------------------------------------------------
CHIP_ID = 0x00
GYR_DATA_X_LSB = 0x14
EUL_HEADING_LSB = 0x1A
QUA_DATA_W_LSB = 0x20
CALIB_STAT = 0x35
UNIT_SEL = 0x3B
OPR_MODE = 0x3D
PWR_MODE = 0x3E
SYS_TRIGGER = 0x3F

CHIP_ID_VALUE = 0xA0

# operation modes
MODE_CONFIG = 0x00
MODE_NDOF = 0x0C  # 9-DOF fusion, absolute orientation

# scale factors for the default unit selection
_EULER_LSB_PER_DEG = 16.0
_GYRO_LSB_PER_DPS = 16.0
_QUAT_LSB = float(1 << 14)  # 16384

# UART protocol bytes
_START = 0xAA
_WRITE = 0x00
_READ = 0x01
_ACK = 0xEE
_READ_OK = 0xBB
_WRITE_SUCCESS = 0x01


def _s16(lo: int, hi: int) -> int:
    """Interpret two bytes as a signed 16-bit little-endian integer."""
    value = lo | (hi << 8)
    return value - 0x10000 if value & 0x8000 else value


class Bno055Imu:
    """BNO055 over UART.

    ``serial_obj`` may be injected (anything with ``write`` / ``read`` /
    ``reset_input_buffer`` / ``close``) for testing; otherwise a ``pyserial``
    port is opened on ``port``.
    """

    def __init__(
        self,
        port: str = "/dev/serial0",
        baudrate: int = 115200,
        timeout: float = 0.1,
        serial_obj=None,
    ) -> None:
        if serial_obj is None:
            import serial  # lazy import

            serial_obj = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self._ser = serial_obj

    # -- low-level register protocol ---------------------------------------
    def _write_register(self, reg: int, data: bytes, retries: int = 3) -> None:
        frame = bytes([_START, _WRITE, reg, len(data)]) + data
        resp = b""
        for _ in range(retries):
            self._ser.reset_input_buffer()
            self._ser.write(frame)
            resp = self._ser.read(2)
            if len(resp) == 2 and resp[0] == _ACK and resp[1] == _WRITE_SUCCESS:
                return
        raise IOError(f"BNO055 write reg 0x{reg:02X} failed (last resp={resp!r})")

    def _read_register(self, reg: int, length: int, retries: int = 5) -> bytes:
        frame = bytes([_START, _READ, reg, length])
        for _ in range(retries):
            self._ser.reset_input_buffer()
            self._ser.write(frame)
            header = self._ser.read(2)
            if len(header) == 2 and header[0] == _READ_OK and header[1] == length:
                data = self._ser.read(length)
                if len(data) == length:
                    return data
            # header[0] == _ACK (e.g. EE 07 bus over-run) or short read -> retry
        raise IOError(f"BNO055 read reg 0x{reg:02X} failed")

    def _read_vector(self, reg: int, count: int, scale: float) -> tuple[float, ...]:
        data = self._read_register(reg, count * 2)
        return tuple(_s16(data[2 * i], data[2 * i + 1]) / scale for i in range(count))

    # -- setup --------------------------------------------------------------
    def set_mode(self, mode: int) -> None:
        self._write_register(OPR_MODE, bytes([mode]))

    def begin(self, mode: int = MODE_NDOF, use_external_crystal: bool = False) -> None:
        """Verify the chip and enter an operating mode (NDOF by default).

        Set ``use_external_crystal=True`` only if the board has a 32.768 kHz
        crystal (most breakouts do); it improves heading stability but selecting
        it on a board without one leaves the device without a clock.
        """
        chip = self._read_register(CHIP_ID, 1)[0]
        if chip != CHIP_ID_VALUE:
            raise IOError(f"unexpected BNO055 chip id 0x{chip:02X} (expected 0xA0)")
        self.set_mode(MODE_CONFIG)
        time.sleep(0.025)  # operating -> CONFIG needs ~19 ms
        if use_external_crystal:
            self._write_register(SYS_TRIGGER, bytes([0x80]))
            time.sleep(0.01)
        self.set_mode(mode)
        time.sleep(0.02)  # CONFIG -> operating needs ~7 ms

    # -- fused / raw readings ----------------------------------------------
    @property
    def euler(self) -> tuple[float, float, float]:
        """Fused absolute orientation (heading, roll, pitch) in degrees."""
        return self._read_vector(EUL_HEADING_LSB, 3, _EULER_LSB_PER_DEG)  # type: ignore[return-value]

    @property
    def heading_deg(self) -> float:
        return self.euler[0]

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        """Fused orientation quaternion (w, x, y, z)."""
        return self._read_vector(QUA_DATA_W_LSB, 4, _QUAT_LSB)  # type: ignore[return-value]

    @property
    def gyro(self) -> tuple[float, float, float]:
        """Angular rate (x, y, z) in deg/s."""
        return self._read_vector(GYR_DATA_X_LSB, 3, _GYRO_LSB_PER_DPS)  # type: ignore[return-value]

    @property
    def calibration_status(self) -> tuple[int, int, int, int]:
        """Calibration levels (sys, gyro, accel, mag), each 0 (uncal) .. 3 (full)."""
        c = self._read_register(CALIB_STAT, 1)[0]
        return ((c >> 6) & 3, (c >> 4) & 3, (c >> 2) & 3, c & 3)

    def measure_gyro_bias(
        self, samples: int = 100, delay: float = 0.01
    ) -> tuple[float, float, float]:
        """Average the gyro output while the robot is held still (deg/s).

        Subtract this bias from later gyro readings in dead-reckoning. Keep the
        robot motionless for the duration.
        """
        sx = sy = sz = 0.0
        for _ in range(samples):
            x, y, z = self.gyro
            sx += x
            sy += y
            sz += z
            time.sleep(delay)
        return (sx / samples, sy / samples, sz / samples)

    def close(self) -> None:
        self._ser.close()

    def __enter__(self) -> "Bno055Imu":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
