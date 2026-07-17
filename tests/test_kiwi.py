"""Unit tests for kiwi-drive kinematics and stepper conversion."""

import math

import pytest

from krilly.config import RobotConfig
from krilly.kinematics.kiwi import KiwiKinematics

# Fixed geometry so the tests do not depend on config/robot.yaml values.
CFG = RobotConfig(
    wheel_diameter_m=0.048,
    wheel_count=3,
    center_to_wheel_m=0.05,
    steps_per_rev=200,
    microstep=16,
    wheel_angles_deg=[90.0, 210.0, 330.0],
)
SQRT3_2 = math.sqrt(3) / 2


@pytest.fixture
def kin():
    return KiwiKinematics(config=CFG)


# --- inverse kinematics ----------------------------------------------------
def test_pure_forward(kin):
    # vx=1: v_i = -sin(theta_i) -> [-1, +0.5, +0.5]
    assert kin.body_to_wheels(1.0, 0.0, 0.0) == pytest.approx((-1.0, 0.5, 0.5))


def test_pure_left(kin):
    # vy=1: v_i = cos(theta_i) -> [0, -sqrt3/2, +sqrt3/2]
    assert kin.body_to_wheels(0.0, 1.0, 0.0) == pytest.approx((0.0, -SQRT3_2, SQRT3_2))


def test_pure_rotation(kin):
    # omega=1 rad/s: all wheels = L*omega
    assert kin.body_to_wheels(0.0, 0.0, 1.0) == pytest.approx((0.05, 0.05, 0.05))


# --- forward kinematics (round trip) ---------------------------------------
def test_round_trip_body_wheels_body(kin):
    for vx, vy, w in [(0.3, 0.0, 0.0), (0.0, -0.2, 0.0), (0.1, 0.15, 2.0), (-0.25, 0.05, -1.5)]:
        wheels = kin.body_to_wheels(vx, vy, w)
        assert kin.wheels_to_body(*wheels) == pytest.approx((vx, vy, w))


def test_all_wheels_equal_is_pure_rotation(kin):
    vx, vy, w = kin.wheels_to_body(0.05, 0.05, 0.05)
    assert (vx, vy) == pytest.approx((0.0, 0.0))
    assert w == pytest.approx(1.0)


# --- stepper conversion ----------------------------------------------------
def test_wheel_speed_to_full_step_hz(kin):
    # one wheel revolution/s = circumference m/s -> steps_per_rev full step/s
    circumference = math.pi * 0.048
    assert kin.wheel_speed_to_step_hz(circumference) == pytest.approx(200.0)


def test_step_hz_round_trip(kin):
    for v in (0.05, 0.2, 0.5):
        assert kin.step_hz_to_wheel_speed(kin.wheel_speed_to_step_hz(v)) == pytest.approx(v)


def test_distance_to_microsteps(kin):
    # one revolution = steps_per_rev * microstep microsteps = 200*16 = 3200
    circumference = math.pi * 0.048
    assert kin.distance_to_microsteps(circumference) == pytest.approx(3200.0)
    assert kin.microsteps_to_distance(3200.0) == pytest.approx(circumference)


def test_body_to_wheel_step_hz_matches_manual(kin):
    hz = kin.body_to_wheel_step_hz(0.2, 0.0, 0.0)
    expected = tuple(kin.wheel_speed_to_step_hz(v) for v in kin.body_to_wheels(0.2, 0.0, 0.0))
    assert hz == pytest.approx(expected)
