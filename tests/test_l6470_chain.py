"""Unit tests for L6470 daisy-chain framing (no hardware).

We assert the exact per-pulse byte stream so the chain ordering and command
assembly are correct. A fake SPI records every xfer2 (one CS pulse) and can
return programmed response bytes.
"""

import pytest

from krilly.hal.l6470 import FWD, REV, Cmd, Reg, speed_to_run_register
from krilly.hal.l6470_chain import L6470Chain


class FakeSpi:
    def __init__(self, rx_queue=None):
        self.calls = []              # each entry = the byte list of one xfer2 (one CS pulse)
        self.rx_queue = list(rx_queue or [])

    def xfer2(self, data):
        self.calls.append(list(data))
        if self.rx_queue:
            return list(self.rx_queue.pop(0))
        return list(data)            # default: echo

    def close(self):
        self.closed = True


# --- pulse ordering --------------------------------------------------------
def test_pulse_reverses_tx_and_rx_when_index0_nearest_mosi():
    spi = FakeSpi()
    chain = L6470Chain(3, spi=spi)  # mosi_is_index0=True (default)
    out = chain._pulse([0x10, 0x20, 0x30])
    # index n-1 (nearest MISO) is clocked out first -> reversed tx
    assert spi.calls[-1] == [0x30, 0x20, 0x10]
    # rx mapped back to index order (echo -> same values)
    assert out == [0x10, 0x20, 0x30]


def test_pulse_no_reversal_when_index0_nearest_miso():
    spi = FakeSpi()
    chain = L6470Chain(3, spi=spi, mosi_is_index0=False)
    out = chain._pulse([0x10, 0x20, 0x30])
    assert spi.calls[-1] == [0x10, 0x20, 0x30]
    assert out == [0x10, 0x20, 0x30]


def test_pulse_wrong_length_raises():
    chain = L6470Chain(3, spi=FakeSpi())
    with pytest.raises(ValueError):
        chain._pulse([0x00, 0x00])


# --- command framing (use mosi_is_index0=False for direct index assertions) -
def test_broadcast_same_byte_to_all():
    spi = FakeSpi()
    L6470Chain(3, spi=spi, mosi_is_index0=False).broadcast(Cmd.SOFT_STOP)
    assert spi.calls == [[Cmd.SOFT_STOP] * 3]


def test_send_to_targets_one_device_others_nop():
    spi = FakeSpi()
    chain = L6470Chain(3, spi=spi, mosi_is_index0=False)
    chain.run(0, FWD, 400)  # RUN|FWD = 0x51, speed 26844 = 0x0068DC
    assert spi.calls == [
        [0x51, 0x00, 0x00],
        [0x00, 0x00, 0x00],
        [0x68, 0x00, 0x00],
        [0xDC, 0x00, 0x00],
    ]


def test_run_all_aligns_per_device_commands():
    spi = FakeSpi()
    chain = L6470Chain(2, spi=spi, mosi_is_index0=False)
    chain.run_all([FWD, REV], [400, 100])
    s0 = speed_to_run_register(400).to_bytes(3, "big")  # 0x0068DC
    s1 = speed_to_run_register(100).to_bytes(3, "big")  # 0x001A37
    assert spi.calls == [
        [0x51, 0x50],            # RUN|FWD, RUN|REV
        [s0[0], s1[0]],
        [s0[1], s1[1]],
        [s0[2], s1[2]],
    ]


def test_set_param_all_broadcasts_reg_and_value():
    spi = FakeSpi()
    L6470Chain(3, spi=spi, mosi_is_index0=False).set_param_all(Reg.STEP_MODE, 0x04)
    # SET_PARAM(0x00)|STEP_MODE(0x16) = 0x16, 1-byte payload
    assert spi.calls == [[0x16, 0x16, 0x16], [0x04, 0x04, 0x04]]


def test_get_status_all_parses_per_device():
    # 3 pulses: opcode-xfer (dummy), hi byte, lo byte
    spi = FakeSpi(rx_queue=[
        [0x00, 0x00, 0x00],       # dummy during opcode transfer
        [0x7C, 0x7E, 0x00],       # hi bytes for dev0/dev1/dev2
        [0x03, 0x50, 0x00],       # lo bytes
    ])
    chain = L6470Chain(3, spi=spi, mosi_is_index0=False)
    assert chain.get_status_all() == [0x7C03, 0x7E50, 0x0000]


def test_send_commands_length_mismatch_raises():
    chain = L6470Chain(2, spi=FakeSpi(), mosi_is_index0=False)
    with pytest.raises(ValueError):
        chain.send_commands([b"\x01\x02", b"\x03"])  # unequal lengths


def test_context_manager_hiz_all_and_close():
    spi = FakeSpi()
    with L6470Chain(2, spi=spi, mosi_is_index0=False):
        pass
    assert spi.calls[-1] == [Cmd.HARD_HIZ, Cmd.HARD_HIZ]
    assert getattr(spi, "closed", False) is True
