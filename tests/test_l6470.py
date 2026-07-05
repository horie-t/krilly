"""Unit tests for the L6470 driver: pure conversions + SPI command framing.

No hardware required — the SPI device is faked and we assert on the exact byte
stream, plus the datasheet register-conversion math.
"""

import math

import pytest

from krilly.hal import l6470
from krilly.hal.l6470 import FWD, REV, L6470, Reg


class FakeSpi:
    """Records every byte sent; returns ``read_values`` (default 0) on reads."""

    def __init__(self, read_values=None):
        self.sent = []
        self._reads = list(read_values or [])

    def xfer2(self, data):
        self.sent.extend(data)
        out = []
        for _ in data:
            out.append(self._reads.pop(0) if self._reads else 0)
        return out

    def close(self):
        self.closed = True


# --- conversions -----------------------------------------------------------
def test_run_register_known_value():
    # 400 step/s * (2^28 * 250ns) = 26843.55 -> 26844
    assert l6470.speed_to_run_register(400) == 26844


def test_run_register_round_trip():
    for s in (50, 400, 1000, 8000):
        reg = l6470.speed_to_run_register(s)
        assert math.isclose(l6470.run_register_to_speed(reg), s, rel_tol=1e-3)


def test_run_register_clamped_to_20_bits():
    assert l6470.speed_to_run_register(1e12) == (1 << 20) - 1
    assert l6470.speed_to_run_register(-5) == 0


def test_max_speed_register_known_value():
    # 400 step/s * (2^18 * 250ns) = 26.21 -> 26
    assert l6470.speed_to_max_speed_register(400) == 26


def test_accel_register_known_value():
    # 1000 step/s^2 * (2^40 * (250ns)^2) = 68.72 -> 69
    assert l6470.accel_to_register(1000) == 69


# --- command framing -------------------------------------------------------
def test_run_command_framing():
    spi = FakeSpi()
    drv = L6470(spi=spi)
    drv.run(FWD, 400)
    # RUN(0x50)|FWD(1) = 0x51, then 3-byte speed 26844 = 0x0068DC
    assert spi.sent == [0x51, 0x00, 0x68, 0xDC]


def test_run_reverse_opcode():
    spi = FakeSpi()
    L6470(spi=spi).run(REV, 100)
    assert spi.sent[0] == 0x50  # RUN | REV(0)


def test_set_param_single_byte():
    spi = FakeSpi()
    L6470(spi=spi).set_param(Reg.STEP_MODE, 0x04)
    # SET_PARAM(0x00) | STEP_MODE(0x16) = 0x16, payload 0x04
    assert spi.sent == [0x16, 0x04]


def test_set_param_multibyte_big_endian():
    spi = FakeSpi()
    L6470(spi=spi).set_param(Reg.ACC, 0x0123)  # ACC is 2 bytes
    assert spi.sent == [0x00 | Reg.ACC, 0x01, 0x23]


def test_get_status_reads_two_bytes():
    # The byte clocked out during the opcode transfer is a dummy; data follows.
    spi = FakeSpi(read_values=[0x00, 0xAB, 0xCD])
    drv = L6470(spi=spi)
    status = drv.get_status()
    assert spi.sent == [0xD0, 0x00, 0x00]  # GET_STATUS + 2 NOP
    assert status == 0xABCD


def test_get_param_big_endian():
    # Leading dummy for the opcode transfer, then 3 data bytes (ABS_POS).
    spi = FakeSpi(read_values=[0x00, 0x01, 0x02, 0x03])
    drv = L6470(spi=spi)
    val = drv.get_param(Reg.ABS_POS)
    assert spi.sent[0] == (0x20 | Reg.ABS_POS)  # GET_PARAM | reg
    assert val == 0x010203


def test_decode_status_no_fault():
    assert l6470.decode_status(0x7E50) == "フォールトなし"


def test_decode_status_no_comms():
    assert "通信不可" in l6470.decode_status(0x0000)
    assert "通信不可" in l6470.decode_status(0xFFFF)


def test_decode_status_uvlo_first_read_annotated():
    first = l6470.decode_status(0x7C03, first_read=True)
    normal = l6470.decode_status(0x7C03)
    assert "正常フラグ" in first          # power-up latch note
    assert "正常フラグ" not in normal
    assert "UVLO" in normal and "HiZ" in normal


def test_context_manager_puts_bridges_hiz_and_closes():
    spi = FakeSpi()
    with L6470(spi=spi) as drv:
        drv.run(FWD, 200)
    assert spi.sent[-1] == 0xA8  # HARD_HIZ on exit
    assert getattr(spi, "closed", False) is True
