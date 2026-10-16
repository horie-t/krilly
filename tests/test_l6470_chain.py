"""L6470 デイジーチェーンのフレーミングのユニットテスト (ハードウェア不要)。

パルスごとの正確なバイト列を検証することで、チェーンの順序とコマンド
組み立てが正しいことを確認する。フェイク SPI は各 xfer2 (CS パルス 1 回分)
を記録し、あらかじめ設定した応答バイトを返せる。
"""

import pytest

from krilly.hal.l6470 import FWD, REV, Cmd, Reg, speed_to_run_register
from krilly.hal.l6470_chain import L6470Chain


class FakeSpi:
    def __init__(self, rx_queue=None):
        self.calls = []              # 各要素 = 1 回の xfer2 (CS パルス 1 回分) のバイトリスト
        self.rx_queue = list(rx_queue or [])

    def xfer2(self, data):
        self.calls.append(list(data))
        if self.rx_queue:
            return list(self.rx_queue.pop(0))
        return list(data)            # デフォルト: エコー

    def close(self):
        self.closed = True


# --- パルスの順序 ----------------------------------------------------------
def test_pulse_reverses_tx_and_rx_when_index0_nearest_mosi():
    spi = FakeSpi()
    chain = L6470Chain(3, spi=spi)  # mosi_is_index0=True (デフォルト)
    out = chain._pulse([0x10, 0x20, 0x30])
    # index n-1 (MISO に最も近い) が最初にクロックアウトされる -> tx は逆順
    assert spi.calls[-1] == [0x30, 0x20, 0x10]
    # rx は index 順にマッピングし直す (エコーなので同じ値)
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


# --- コマンドフレーミング (index を直接検証するため mosi_is_index0=False を使用) -
def test_broadcast_same_byte_to_all():
    spi = FakeSpi()
    L6470Chain(3, spi=spi, mosi_is_index0=False).broadcast(Cmd.SOFT_STOP)
    assert spi.calls == [[Cmd.SOFT_STOP] * 3]


def test_send_to_targets_one_device_others_nop():
    spi = FakeSpi()
    chain = L6470Chain(3, spi=spi, mosi_is_index0=False)
    chain.run(0, FWD, 400)  # RUN|FWD = 0x51、速度 26844 = 0x0068DC
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
        [0x51, 0x50],            # RUN|FWD、RUN|REV
        [s0[0], s1[0]],
        [s0[1], s1[1]],
        [s0[2], s1[2]],
    ]


def test_set_param_all_broadcasts_reg_and_value():
    spi = FakeSpi()
    L6470Chain(3, spi=spi, mosi_is_index0=False).set_param_all(Reg.STEP_MODE, 0x04)
    # SET_PARAM(0x00)|STEP_MODE(0x16) = 0x16、1 バイトのペイロード
    assert spi.calls == [[0x16, 0x16, 0x16], [0x04, 0x04, 0x04]]


def test_get_status_all_parses_per_device():
    # 3 パルス: オペコード転送 (ダミー)、上位バイト、下位バイト
    spi = FakeSpi(rx_queue=[
        [0x00, 0x00, 0x00],       # オペコード転送中のダミー
        [0x7C, 0x7E, 0x00],       # dev0/dev1/dev2 の上位バイト
        [0x03, 0x50, 0x00],       # 下位バイト
    ])
    chain = L6470Chain(3, spi=spi, mosi_is_index0=False)
    assert chain.get_status_all() == [0x7C03, 0x7E50, 0x0000]


def test_send_commands_length_mismatch_raises():
    chain = L6470Chain(2, spi=FakeSpi(), mosi_is_index0=False)
    with pytest.raises(ValueError):
        chain.send_commands([b"\x01\x02", b"\x03"])  # 長さが不一致


def test_context_manager_hiz_all_and_close():
    spi = FakeSpi()
    with L6470Chain(2, spi=spi, mosi_is_index0=False):
        pass
    assert spi.calls[-1] == [Cmd.HARD_HIZ, Cmd.HARD_HIZ]
    assert getattr(spi, "closed", False) is True


# --- SPI が途切れたときの解放 (#148) ----------------------------------------
class FlakyLink:
    """``dead_reads`` 回だけ STATUS が 0x0000 (応答なし) になり、その後は戻る。"""

    def __init__(self, chain, dead_reads):
        self.dead_reads = dead_reads
        self.hiz = 0
        self.sleeps = []
        chain.hard_hiz_all = self.hard_hiz_all
        chain.get_status_all = self.get_status_all

    def hard_hiz_all(self):
        self.hiz += 1

    def get_status_all(self):
        if self.dead_reads > 0:
            self.dead_reads -= 1
            return [0x0000, 0x0000, 0x0000]
        return [0x7E03, 0x7E03, 0x7E03]


def test_release_keeps_sending_hiz_until_the_link_comes_back():
    """途切れては戻る接触不良なので、戻った瞬間に解放が届くよう送り続ける。"""
    chain = L6470Chain(3, spi=FakeSpi())
    link = FlakyLink(chain, dead_reads=3)
    assert chain.release_until_alive(tries=10, sleep=link.sleeps.append) is True
    # 4 回目の読みで戻る。戻ったら念のためもう 1 回送る
    assert link.hiz == 5
    assert len(link.sleeps) == 3


def test_release_gives_up_when_the_link_never_comes_back():
    chain = L6470Chain(3, spi=FakeSpi())
    link = FlakyLink(chain, dead_reads=100)
    assert chain.release_until_alive(tries=4, sleep=link.sleeps.append) is False
    assert link.hiz == 4
    assert len(link.sleeps) == 3     # 最後の試行の後は待たない
