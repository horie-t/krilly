"""BNO055 9-axis IMU over **I2C** (issue #6).

The board (Akizuki AE-BNO055-BO) ships in I2C mode (address 0x28) with no
jumper changes needed, so I2C avoids the tiny UART-select solder pads.

Clock-stretching note: the legacy Broadcom BSC I2C (Pi 1-4) mishandles the
BNO055's clock stretching, but the **Pi 5 uses RP1 (DesignWare) I2C which
handles it correctly** — 100 kHz has been verified stable on hardware, so no
bus-speed reduction is needed. We still retry every transfer on OSError as cheap
insurance; if a slower Pi ever mishandles reads, lower the bus speed via
``dtparam=i2c_arm_baudrate`` (see docs/setup-pi5.md).

Register protocol: standard I2C register read/write (write the register
pointer, repeated-start read N bytes). All multi-byte values are signed 16-bit
little-endian. Default units: Euler 16 LSB/deg, gyro 16 LSB/deg-per-sec,
quaternion 2^14 LSB/unit. All registers used are on page 0 (default).
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
DEFAULT_ADDRESS = 0x28  # AE-BNO055-BO factory default (J1 open)

# operation modes
MODE_CONFIG = 0x00
MODE_NDOF = 0x0C  # 9-DOF fusion, absolute orientation

# scale factors for the default unit selection
_EULER_LSB_PER_DEG = 16.0
_GYRO_LSB_PER_DPS = 16.0
_QUAT_LSB = float(1 << 14)  # 16384


def _s16(lo: int, hi: int) -> int:
    """Interpret two bytes as a signed 16-bit little-endian integer."""
    value = lo | (hi << 8)
    return value - 0x10000 if value & 0x8000 else value


class Bno055Imu:
    """BNO055 over I2C.

    ``bus`` may be injected (anything with ``read_i2c_block_data`` /
    ``write_i2c_block_data`` / ``close``, e.g. ``smbus2.SMBus``) for testing;
    otherwise an ``smbus2.SMBus`` is opened on ``bus_id``.
    """

    def __init__(
        self,
        address: int = DEFAULT_ADDRESS,
        bus_id: int = 1,
        bus=None,
        retries: int = 5,
    ) -> None:
        if bus is None:
            from smbus2 import SMBus  # lazy import

            bus = SMBus(bus_id)
        self._bus = bus
        self._addr = address
        self._retries = retries

    # -- low-level register access (retry on I2C errors / clock stretch) ----
    def _read_register(self, reg: int, length: int) -> bytes:
        last = None
        for _ in range(self._retries):
            try:
                return bytes(self._bus.read_i2c_block_data(self._addr, reg, length))
            except OSError as exc:  # bus error / clock-stretch corruption
                last = exc
        raise IOError(f"BNO055 I2C read reg 0x{reg:02X} failed: {last}")

    def _write_register(self, reg: int, data: bytes) -> None:
        last = None
        for _ in range(self._retries):
            try:
                self._bus.write_i2c_block_data(self._addr, reg, list(data))
                return
            except OSError as exc:
                last = exc
        raise IOError(f"BNO055 I2C write reg 0x{reg:02X} failed: {last}")

    def _read_vector(self, reg: int, count: int, scale: float) -> tuple[float, ...]:
        data = self._read_register(reg, count * 2)
        return tuple(_s16(data[2 * i], data[2 * i + 1]) / scale for i in range(count))

    # -- setup --------------------------------------------------------------
    def set_mode(self, mode: int) -> None:
        self._write_register(OPR_MODE, bytes([mode]))

    def begin(self, mode: int = MODE_NDOF, use_external_crystal: bool = False) -> None:
        """Verify the chip and enter an operating mode (NDOF by default).

        Set ``use_external_crystal=True`` if the board has a 32.768 kHz crystal
        (the AE-BNO055-BO does); it improves heading stability.
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
        self._bus.close()

    def __enter__(self) -> "Bno055Imu":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
