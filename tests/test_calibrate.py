"""キャリブレーションの純粋計算のユニットテスト。"""

import math

import pytest

from krilly.config import RobotConfig
from krilly.hal.l6470 import FWD, REV
from krilly.kinematics.kiwi import KiwiKinematics
from scripts.calibrate import (
    corrected_center_to_wheel,
    corrected_wheel_diameter,
    wheel_moves_for_body,
)

CFG = RobotConfig(
    wheel_diameter_m=0.048,
    wheel_count=3,
    center_to_wheel_m=0.05,
    steps_per_rev=200,
    microstep=16,
    wheel_angles_deg=[0.0, 120.0, 240.0],
)


def test_corrected_wheel_diameter_scales_by_measured_over_commanded():
    # 0.5m 指令で 0.475m しか進まなかった → 径を 0.95 倍に補正
    assert corrected_wheel_diameter(0.048, 0.5, 0.475) == pytest.approx(0.0456)


def test_corrected_wheel_diameter_no_error_keeps_value():
    assert corrected_wheel_diameter(0.048, 1.0, 1.0) == pytest.approx(0.048)


def test_corrected_center_to_wheel_scales_by_commanded_over_measured():
    # 720度 指令で 700度 しか回らなかった → L を 720/700 倍に補正 (不足回転→L大)
    assert corrected_center_to_wheel(0.05, 720, 700) == pytest.approx(0.05 * 720 / 700)


def test_corrected_center_to_wheel_no_error_keeps_value():
    assert corrected_center_to_wheel(0.05, 720, 720) == pytest.approx(0.05)


def test_wheel_moves_straight_front_wheel_idle():
    kin = KiwiKinematics(CFG)
    moves = wheel_moves_for_body(kin, 0.5, 0.0, 0.0)  # 前進 0.5m
    dirs = [d for d, _ in moves]
    micros = [m for _, m in moves]
    # 前進では M0 は転がらない、M1/M2 は逆向きで同じ量
    assert micros[0] == 0
    assert micros[1] == micros[2] > 0
    assert dirs[1] == REV and dirs[2] == FWD


def test_wheel_moves_rotation_all_equal_forward():
    kin = KiwiKinematics(CFG)
    moves = wheel_moves_for_body(kin, 0.0, 0.0, 2 * math.pi)  # 1回転(CCW)
    dirs = [d for d, _ in moves]
    micros = [m for _, m in moves]
    assert dirs == [FWD, FWD, FWD]           # +omega は 3輪とも正転
    assert micros[0] == micros[1] == micros[2] > 0
    # 1回転で各輪は L*2π だけ転がる
    expected = round(kin.distance_to_microsteps(CFG.center_to_wheel_m * 2 * math.pi))
    assert micros[0] == expected
