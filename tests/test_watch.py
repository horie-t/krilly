"""動作の途中の見張り (#153) と、止まったときの画像 (#153) のテスト。"""

import datetime
import math

import numpy as np
import pytest

from krilly.app.snapshot import save_snapshot, snapshot_path
from krilly.motion.watch import RotationWatch, motion_timeout_s


def test_a_single_spike_does_not_stop_the_machine():
    w = RotationWatch(math.radians(2.5), persist_ticks=3)
    assert not w.update(math.radians(3.0))
    assert not w.update(math.radians(1.0))       # 1 tick で戻った
    assert not w.update(math.radians(3.0))
    assert not w.update(math.radians(3.0))
    assert w.update(math.radians(-3.0))          # 3 回続いた (符号は問わない)
    assert w.peak_rad == pytest.approx(math.radians(3.0))


def test_reset_clears_the_peak_and_the_count():
    w = RotationWatch(math.radians(2.5), persist_ticks=2)
    w.update(math.radians(3.0))
    w.reset()
    assert w.peak_rad == 0.0
    assert not w.update(math.radians(3.0))


def test_the_timeout_scales_with_the_motion_instead_of_a_flat_10s():
    """以前は 1 セルに 10s、6 セルに 18s。見込み (実測 1 セル 1.1s) の 2 倍にする。"""
    assert motion_timeout_s(1) == pytest.approx(3.0)                 # 下限
    assert motion_timeout_s(6, corners=1) == pytest.approx(2 * (0.6 + 0.65 * 6 + 0.3))
    assert motion_timeout_s(6, corners=1) < 18.0
    assert motion_timeout_s(4) > 3.7                                 # 実測の 4 セル 3.7s より長い


def test_snapshot_names_are_dated_and_safe(tmp_path):
    now = datetime.datetime(2026, 11, 8, 12, 34, 56)
    p = snapshot_path("rotation guard/../x", now, tmp_path)
    assert p.parent == tmp_path
    assert p.name == "20261108_123456_rotation_guard_x.png"


def test_save_snapshot_writes_a_png(tmp_path):
    frame = np.zeros((72, 96, 3), dtype=np.uint8)
    p = save_snapshot(frame, "contact", tmp_path / "snaps")
    assert p is not None and p.exists() and p.suffix == ".png"


def test_save_snapshot_never_raises(tmp_path):
    assert save_snapshot(None, "contact", tmp_path) is None
    blocked = tmp_path / "file"
    blocked.write_text("x")                      # ディレクトリのはずの場所がファイル
    assert save_snapshot(np.zeros((4, 4, 3), np.uint8), "x", blocked / "sub") is None
