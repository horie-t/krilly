"""デッドレコニング推定器 DeadReckoning のユニットテスト。"""

import math

import pytest

from krilly.config import RobotConfig
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning

CFG = RobotConfig(
    wheel_diameter_m=0.048,
    wheel_count=3,
    center_to_wheel_m=0.05,
    steps_per_rev=200,
    microstep=16,
    wheel_angles_deg=[0.0, 120.0, 240.0],
)


@pytest.fixture
def kin():
    return KiwiKinematics(config=CFG)


@pytest.fixture
def est(kin):
    return DeadReckoning(kinematics=kin)


def _forward_wheels(kin, dist):
    """前進 dist[m] に対応する各輪の転がり距離。"""
    return kin.body_to_wheels(dist, 0.0, 0.0)


def _rotate_wheels(kin, dphi):
    return kin.body_to_wheels(0.0, 0.0, dphi)


# --- 基本 ------------------------------------------------------------------
def test_initial_pose_zero(est):
    assert est.pose == (0.0, 0.0, 0.0)


def test_straight_moves_along_x(est, kin):
    est.update_wheel_distances(*_forward_wheels(kin, 0.5))
    x, y, phi = est.pose
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.0)
    assert phi == pytest.approx(0.0)


def test_pure_rotation_changes_only_phi(est, kin):
    est.update_wheel_distances(*_rotate_wheels(kin, math.pi / 2))
    x, y, phi = est.pose
    assert phi == pytest.approx(math.pi / 2)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)


def test_forward_respects_heading(kin):
    # 先に +90° 向いてから前進すると、世界では +y へ進む
    est = DeadReckoning(kinematics=kin, phi=math.pi / 2)
    est.update_wheel_distances(*_forward_wheels(kin, 0.3))
    x, y, phi = est.pose
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.3)
    assert phi == pytest.approx(math.pi / 2)


def test_strafe_left_moves_along_y(est, kin):
    est.update_wheel_distances(*kin.body_to_wheels(0.0, 0.4, 0.0))
    x, y, _ = est.pose
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.4)


# --- 積分の分割不変性 (中点法) --------------------------------------------
def test_straight_split_matches_single(kin):
    single = DeadReckoning(kinematics=kin)
    single.update_wheel_distances(*_forward_wheels(kin, 1.0))
    split = DeadReckoning(kinematics=kin)
    for _ in range(10):
        split.update_wheel_distances(*_forward_wheels(kin, 0.1))
    assert split.pose == pytest.approx(single.pose)


# --- L字経路 (前進→左旋回90°→前進) ----------------------------------------
def test_l_shaped_path(kin):
    est = DeadReckoning(kinematics=kin)
    est.update_wheel_distances(*_forward_wheels(kin, 0.5))       # +x へ 0.5
    est.update_wheel_distances(*_rotate_wheels(kin, math.pi / 2))  # 左90°
    est.update_wheel_distances(*_forward_wheels(kin, 0.5))       # 世界 +y へ 0.5
    x, y, phi = est.pose
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5)
    assert phi == pytest.approx(math.pi / 2)


# --- 入力元別メソッド ------------------------------------------------------
def test_update_wheel_speeds_equivalent_to_distances(kin):
    a = DeadReckoning(kinematics=kin)
    b = DeadReckoning(kinematics=kin)
    wheel_mps = kin.body_to_wheels(0.2, 0.0, 0.0)
    a.update_wheel_speeds(wheel_mps, 0.5)
    b.update_wheel_distances(*(v * 0.5 for v in wheel_mps))
    assert a.pose == pytest.approx(b.pose)


def test_update_wheel_microsteps(kin):
    # 1回転ぶんのマイクロステップを各輪に与えると phi が 2π 増える
    micro = kin.distance_to_microsteps(CFG.center_to_wheel_m * 2 * math.pi)
    est = DeadReckoning(kinematics=kin)
    est.update_wheel_microsteps(micro, micro, micro)
    assert est.pose[2] == pytest.approx(2 * math.pi)


def test_reset(est, kin):
    est.update_wheel_distances(*_forward_wheels(kin, 1.0))
    est.reset(1.0, 2.0, 0.5)
    assert est.pose == (1.0, 2.0, 0.5)
