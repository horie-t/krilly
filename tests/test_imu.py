"""Unit tests for the BNO055 UART driver (no hardware).

A FakeSerial replays preloaded response bytes and records what was written,
so the register protocol framing and value parsing can be verified offline.
"""

import pytest

from krilly.hal import imu
from krilly.hal.imu import Bno055Imu


class FakeSerial:
    """FIFO fake: read() pops preloaded bytes; write() is recorded."""

    def __init__(self, rx=b""):
        self.rx = bytearray(rx)
        self.written = bytearray()

    def write(self, data):
        self.written += bytes(data)
        return len(data)

    def read(self, n):
        chunk = bytes(self.rx[:n])
        del self.rx[:n]
        return chunk

    def reset_input_buffer(self):
        pass

    def close(self):
        self.closed = True


def _euler_frame(*raw16):
    body = bytearray()
    for v in raw16:
        v &= 0xFFFF
        body += bytes([v & 0xFF, (v >> 8) & 0xFF])
    return bytes([0xBB, len(body)]) + body


# --- helpers ---------------------------------------------------------------
def test_s16_signed():
    assert imu._s16(0x00, 0x00) == 0
    assert imu._s16(0xFF, 0x7F) == 32767
    assert imu._s16(0x00, 0x80) == -32768
    assert imu._s16(0x38, 0xFF) == -200  # 0xFF38


# --- read protocol ---------------------------------------------------------
def test_read_register_success_and_request_frame():
    spi = FakeSerial(rx=bytes([0xBB, 0x01, 0xA0]))
    dev = Bno055Imu(serial_obj=spi)
    assert dev._read_register(imu.CHIP_ID, 1) == b"\xA0"
    assert spi.written == bytes([0xAA, 0x01, 0x00, 0x01])  # AA 01 <reg> <len>


def test_read_register_retries_on_bus_error_then_succeeds():
    # first EE 07 (bus over-run), then a good BB frame
    spi = FakeSerial(rx=bytes([0xEE, 0x07]) + bytes([0xBB, 0x01, 0x42]))
    dev = Bno055Imu(serial_obj=spi)
    assert dev._read_register(0x35, 1) == b"\x42"
    # two request frames were sent (retry)
    assert spi.written == bytes([0xAA, 0x01, 0x35, 0x01]) * 2


def test_read_register_raises_after_retries():
    spi = FakeSerial(rx=bytes([0xEE, 0x07]) * 5)
    dev = Bno055Imu(serial_obj=spi)
    with pytest.raises(IOError):
        dev._read_register(0x35, 1, retries=5)


# --- write protocol --------------------------------------------------------
def test_write_register_success_frame():
    spi = FakeSerial(rx=bytes([0xEE, 0x01]))
    dev = Bno055Imu(serial_obj=spi)
    dev.set_mode(imu.MODE_NDOF)
    assert spi.written == bytes([0xAA, 0x00, 0x3D, 0x01, 0x0C])


def test_write_register_raises_on_nack():
    spi = FakeSerial(rx=bytes([0xEE, 0x03]) * 3)  # non-success status
    dev = Bno055Imu(serial_obj=spi)
    with pytest.raises(IOError):
        dev._write_register(imu.OPR_MODE, b"\x0C")


# --- value parsing ---------------------------------------------------------
def test_euler_parsing_degrees():
    # heading=+180.0 (2880), roll=-10.0 (-160), pitch=+0.0
    spi = FakeSerial(rx=_euler_frame(2880, -160 & 0xFFFF, 0))
    dev = Bno055Imu(serial_obj=spi)
    h, r, p = dev.euler
    assert h == pytest.approx(180.0)
    assert r == pytest.approx(-10.0)
    assert p == pytest.approx(0.0)


def test_gyro_parsing_signed_dps():
    # x=+16 LSB -> +1.0 dps, y=-16 -> -1.0, z=1600 -> 100.0
    spi = FakeSerial(rx=_euler_frame(16, -16 & 0xFFFF, 1600))
    dev = Bno055Imu(serial_obj=spi)
    x, y, z = dev.gyro
    assert (x, y, z) == pytest.approx((1.0, -1.0, 100.0))


def test_calibration_status_bits():
    # sys=3, gyro=2, accel=1, mag=0 -> 0b11_10_01_00 = 0xE4
    spi = FakeSerial(rx=bytes([0xBB, 0x01, 0xE4]))
    dev = Bno055Imu(serial_obj=spi)
    assert dev.calibration_status == (3, 2, 1, 0)


def test_measure_gyro_bias_averages():
    # two gyro reads: (2,0,0) and (4,0,0) LSB -> avg 3 LSB /16
    frame = _euler_frame(2, 0, 0) + _euler_frame(4, 0, 0)
    dev = Bno055Imu(serial_obj=FakeSerial(rx=frame))
    bx, by, bz = dev.measure_gyro_bias(samples=2, delay=0)
    assert bx == pytest.approx(3 / 16.0)
    assert (by, bz) == pytest.approx((0.0, 0.0))


# --- begin -----------------------------------------------------------------
def test_begin_checks_chip_id():
    # chip id read returns wrong id
    spi = FakeSerial(rx=bytes([0xBB, 0x01, 0x00]))
    dev = Bno055Imu(serial_obj=spi)
    with pytest.raises(IOError):
        dev.begin()


def test_begin_success_sets_config_then_ndof():
    rx = (
        bytes([0xBB, 0x01, 0xA0])  # chip id OK
        + bytes([0xEE, 0x01])      # set_mode(CONFIG) ack
        + bytes([0xEE, 0x01])      # set_mode(NDOF) ack
    )
    spi = FakeSerial(rx=rx)
    dev = Bno055Imu(serial_obj=spi)
    dev.begin()
    # last write must be OPR_MODE = NDOF (0x0C)
    assert spi.written[-5:] == bytes([0xAA, 0x00, 0x3D, 0x01, 0x0C])


def test_context_manager_closes():
    spi = FakeSerial()
    with Bno055Imu(serial_obj=spi):
        pass
    assert getattr(spi, "closed", False) is True
