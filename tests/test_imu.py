"""BNO055 I2C ドライバのユニットテスト (ハードウェア不要)。

FakeI2CBus はあらかじめ設定したレジスタバイトを再生し、write を記録する。
これによりレジスタアクセス、エラー時のリトライ、値のパースをオフラインで検証できる。
"""

import pytest

from krilly.hal import imu
from krilly.hal.imu import Bno055Imu


class FakeI2CBus:
    def __init__(self, reads=None, fail_times=0):
        # reads: read_i2c_block_data が返す {reg: [ints]}
        self.reads = reads or {}
        self.writes = []          # (reg, [data]) のリスト
        self.fail_times = fail_times  # 先頭から OSError を投げる read/write の回数
        self.closed = False

    def read_i2c_block_data(self, addr, reg, length):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OSError("simulated clock-stretch error")
        if reg not in self.reads:
            raise AssertionError(f"unexpected read reg 0x{reg:02X}")
        return list(self.reads[reg][:length])

    def write_i2c_block_data(self, addr, reg, data):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OSError("simulated error")
        self.writes.append((reg, list(data)))

    def close(self):
        self.closed = True


def _vec(*raw16):
    out = []
    for v in raw16:
        v &= 0xFFFF
        out += [v & 0xFF, (v >> 8) & 0xFF]
    return out


# --- ヘルパー --------------------------------------------------------------
def test_s16_signed():
    assert imu._s16(0x00, 0x00) == 0
    assert imu._s16(0xFF, 0x7F) == 32767
    assert imu._s16(0x00, 0x80) == -32768
    assert imu._s16(0x38, 0xFF) == -200  # 0xFF38


# --- レジスタアクセス ------------------------------------------------------
def test_read_register_returns_bytes():
    bus = FakeI2CBus(reads={imu.CHIP_ID: [0xA0]})
    dev = Bno055Imu(bus=bus)
    assert dev._read_register(imu.CHIP_ID, 1) == b"\xA0"


def test_read_retries_on_oserror_then_succeeds():
    bus = FakeI2CBus(reads={imu.CALIB_STAT: [0x42]}, fail_times=2)
    dev = Bno055Imu(bus=bus, retries=5)
    assert dev._read_register(imu.CALIB_STAT, 1) == b"\x42"


def test_read_raises_after_retries_exhausted():
    bus = FakeI2CBus(reads={imu.CALIB_STAT: [0x42]}, fail_times=10)
    dev = Bno055Imu(bus=bus, retries=3)
    with pytest.raises(IOError):
        dev._read_register(imu.CALIB_STAT, 1)


def test_set_mode_writes_opr_mode():
    bus = FakeI2CBus()
    Bno055Imu(bus=bus).set_mode(imu.MODE_NDOF)
    assert bus.writes == [(imu.OPR_MODE, [imu.MODE_NDOF])]


def test_write_raises_after_retries():
    bus = FakeI2CBus(fail_times=10)
    dev = Bno055Imu(bus=bus, retries=3)
    with pytest.raises(IOError):
        dev._write_register(imu.OPR_MODE, b"\x0C")


# --- 値のパース ------------------------------------------------------------
def test_euler_parsing_degrees():
    bus = FakeI2CBus(reads={imu.EUL_HEADING_LSB: _vec(2880, -160 & 0xFFFF, 0)})
    h, r, p = Bno055Imu(bus=bus).euler
    assert h == pytest.approx(180.0)   # 2880 / 16
    assert r == pytest.approx(-10.0)
    assert p == pytest.approx(0.0)


def test_gyro_parsing_signed_dps():
    bus = FakeI2CBus(reads={imu.GYR_DATA_X_LSB: _vec(16, -16 & 0xFFFF, 1600)})
    x, y, z = Bno055Imu(bus=bus).gyro
    assert (x, y, z) == pytest.approx((1.0, -1.0, 100.0))


def test_quaternion_parsing():
    bus = FakeI2CBus(reads={imu.QUA_DATA_W_LSB: _vec(1 << 14, 0, 0, 0)})
    w, x, y, z = Bno055Imu(bus=bus).quaternion
    assert (w, x, y, z) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_calibration_status_bits():
    # sys=3, gyro=2, accel=1, mag=0 -> 0b11_10_01_00 = 0xE4
    bus = FakeI2CBus(reads={imu.CALIB_STAT: [0xE4]})
    assert Bno055Imu(bus=bus).calibration_status == (3, 2, 1, 0)


def test_measure_gyro_bias_averages():
    # 同じ reg を 2 回読む。FakeI2CBus は毎回同じフレームを返す
    bus = FakeI2CBus(reads={imu.GYR_DATA_X_LSB: _vec(48, 0, 0)})  # 48 LSB = 3.0 dps
    bx, by, bz = Bno055Imu(bus=bus).measure_gyro_bias(samples=4, delay=0)
    assert bx == pytest.approx(3.0)
    assert (by, bz) == pytest.approx((0.0, 0.0))


# --- begin -----------------------------------------------------------------
def test_begin_rejects_wrong_chip_id():
    bus = FakeI2CBus(reads={imu.CHIP_ID: [0x00]})
    with pytest.raises(IOError):
        Bno055Imu(bus=bus).begin()


def test_begin_sets_config_then_ndof():
    bus = FakeI2CBus(reads={imu.CHIP_ID: [imu.CHIP_ID_VALUE]})
    Bno055Imu(bus=bus).begin()
    assert bus.writes == [
        (imu.OPR_MODE, [imu.MODE_CONFIG]),
        (imu.OPR_MODE, [imu.MODE_NDOF]),
    ]


def test_begin_with_external_crystal_sets_sys_trigger():
    bus = FakeI2CBus(reads={imu.CHIP_ID: [imu.CHIP_ID_VALUE]})
    Bno055Imu(bus=bus).begin(use_external_crystal=True)
    assert bus.writes == [
        (imu.OPR_MODE, [imu.MODE_CONFIG]),
        (imu.SYS_TRIGGER, [0x80]),
        (imu.OPR_MODE, [imu.MODE_NDOF]),
    ]


def test_context_manager_closes():
    bus = FakeI2CBus()
    with Bno055Imu(bus=bus):
        pass
    assert bus.closed is True
