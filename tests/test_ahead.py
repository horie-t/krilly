"""前方の境目の見張り (#188) のテスト。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from krilly.app.ahead_watch import AheadWatch, AheadWatchConfig
from krilly.perception.ahead import ahead_profiles, boundary_fraction
from krilly.perception.wall_detect import CALIBRATED_GEOMETRY, DEFAULT_FRAME_SIZE

W, H = DEFAULT_FRAME_SIZE
G = CALIBRATED_GEOMETRY
RED = (0, 0, 255)


def frame_with_crossing_wall(ahead_mm: float) -> np.ndarray:
    """車体の前方 ``ahead_mm`` に、通り道を横切る壁の帯 (12 mm) を描く。"""
    img = np.zeros((H, W, 3), np.uint8)
    y = G.cell_center[1] - ahead_mm * G.px_per_mm_y
    img[int(y - 10):int(y + 10), :] = RED
    return img


def test_crossing_wall_is_read_at_its_distance_and_not_elsewhere():
    prof = ahead_profiles(frame_with_crossing_wall(150.0))
    assert boundary_fraction(prof, 0, 0.150) == pytest.approx(1.0)
    assert boundary_fraction(prof, 0, 0.090) == pytest.approx(0.0)    # 60 mm 手前には無い
    assert boundary_fraction(prof, 2, 0.150) == pytest.approx(0.0)    # 後ろには無い


def test_wall_hidden_by_the_robot_is_not_judged():
    prof = ahead_profiles(frame_with_crossing_wall(10.0))              # 機体の真上
    assert boundary_fraction(prof, 0, 0.010) is None


def watch_step(watch, wall_mm, x=0.0, ref_x=0.36):
    """東向き (φ=0) に進む機体。``wall_mm`` 先に壁の帯が写っている 1 フレーム。"""
    prof = ahead_profiles(frame_with_crossing_wall(wall_mm))
    return watch.check(prof, (x, 0.0, 0.0), (ref_x, 0.0, 0.0), 0, 0.0)


def test_wall_on_the_path_stops_after_two_frames():
    watch = AheadWatch()
    # 機体はセル中央 (x=0)、行き先は 2 セル先 → 90 mm と 270 mm 先の境目は通り道
    assert watch_step(watch, 90.0) is None              # 1 フレーム目はまだ
    alarm = watch_step(watch, 90.0)
    assert alarm is not None and alarm.distance_m == pytest.approx(0.09)
    assert watch.alarm is alarm


def test_wall_beyond_the_destination_is_ignored():
    watch = AheadWatch()
    # 行き先は 1 セル先 (x=0.18) → その向こうの境目 (270 mm 先) は見ない
    for _ in range(3):
        assert watch_step(watch, 270.0, ref_x=0.18) is None
    assert watch.alarm is None


def test_observe_only_records_but_does_not_ask_to_stop():
    from krilly.app.motion_fusion import MotionFusion
    from krilly.localization.delayed_fusion import FusionConfig
    from krilly.localization.estimator import DeadReckoning

    class FakeLogger:
        def __init__(self, *a, **k):
            self.frames = 0
        def start(self):
            pass
        def stop(self):
            pass

    watch = AheadWatch(AheadWatchConfig(observe_only=True))
    fuser = MotionFusion(None, None, DeadReckoning(), FusionConfig(), ahead=watch,
                         logger_factory=FakeLogger)
    watch_step(watch, 90.0)
    watch_step(watch, 90.0)
    assert watch.alarm is not None and fuser.ahead_stop is None
    watch.cfg = AheadWatchConfig(observe_only=False)
    assert fuser.ahead_stop is watch.alarm
