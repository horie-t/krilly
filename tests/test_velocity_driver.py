"""VelocityDriver (加減速ランプ付き速度指令) のユニットテスト。"""

import math

import pytest

from krilly.config import RobotConfig
from krilly.hal.l6470 import FWD, REV
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.motion.velocity_driver import RampLimits, VelocityDriver, _rate_limit

CFG = RobotConfig(
    wheel_diameter_m=0.048,
    wheel_count=3,
    center_to_wheel_m=0.05,
    steps_per_rev=200,
    microstep=16,
    wheel_angles_deg=[0.0, 120.0, 240.0],  # スポーク角 (実機校正済み #11)
)


class FakeChain:
    """run_all / soft_stop_all を記録するフェイク。"""

    def __init__(self):
        self.calls = []

    def run_all(self, directions, speeds):
        self.calls.append((list(directions), list(speeds)))

    def soft_stop_all(self):
        self.calls.append(("soft_stop",))


@pytest.fixture
def kin():
    return KiwiKinematics(config=CFG)


@pytest.fixture
def driver(kin):
    return VelocityDriver(FakeChain(), kinematics=kin)


# --- レートリミッタ --------------------------------------------------------
def test_rate_limit_up_down_and_clamp():
    assert _rate_limit(0.0, 1.0, 0.3) == pytest.approx(0.3)
    assert _rate_limit(1.0, 0.0, 0.3) == pytest.approx(0.7)
    assert _rate_limit(0.9, 1.0, 0.3) == pytest.approx(1.0)   # 目標を超えない
    assert _rate_limit(0.1, 0.0, 0.3) == pytest.approx(0.0)


# --- ランプ挙動 ------------------------------------------------------------
def test_update_ramps_instead_of_jumping(driver):
    # 既定の線形加速 0.5 m/s^2, dt=0.1 -> 1 tick 0.05 m/s まで
    driver.set_velocity(1.0, 0.0, 0.0)
    vx, vy, w = driver.update(0.1)
    assert (vx, vy, w) == pytest.approx((0.05, 0.0, 0.0))
    assert not driver.at_target()


def test_reaches_target_after_enough_updates(driver):
    driver.set_velocity(0.3, 0.0, 0.0)
    for _ in range(200):
        driver.update(0.1)
    assert driver.current_velocity == pytest.approx((0.3, 0.0, 0.0))
    assert driver.at_target()


def test_angular_ramp(driver):
    # 既定の角加速 5 rad/s^2, dt=0.1 -> 0.5 rad/s/tick
    driver.set_velocity(0.0, 0.0, 1.0)
    assert driver.update(0.1)[2] == pytest.approx(0.5)
    assert driver.update(0.1)[2] == pytest.approx(1.0)


def test_stop_targets_zero(driver):
    driver.set_velocity(0.2, 0.0, 0.0)
    for _ in range(100):
        driver.update(0.1)
    driver.stop()
    assert driver.target_velocity == (0.0, 0.0, 0.0)
    for _ in range(100):
        driver.update(0.1)
    assert driver.current_velocity == pytest.approx((0.0, 0.0, 0.0))


# --- 運動学との結線 --------------------------------------------------------
def test_command_matches_kinematics(driver, kin):
    driver.set_velocity(1.0, 0.0, 0.0)
    driver.update(0.1)  # current = (0.05, 0, 0)
    wheel_mps = kin.body_to_wheels(0.05, 0.0, 0.0)
    exp_dirs = [FWD if s >= 0 else REV for s in wheel_mps]
    exp_hz = [abs(kin.wheel_speed_to_step_hz(s)) for s in wheel_mps]
    dirs, hz = driver.chain.calls[-1]
    assert dirs == exp_dirs
    assert hz == pytest.approx(exp_hz)


def test_forward_wheel_directions(driver):
    # 前進(vx>0)では各輪の周速は (0, -, +) -> 向きは (FWD, REV, FWD)
    driver.set_velocity(1.0, 0.0, 0.0)
    for _ in range(100):
        driver.update(0.1)
    dirs, hz = driver.chain.calls[-1]
    assert dirs == [FWD, REV, FWD]
    assert all(h >= 0 for h in hz)   # run_all へ渡す速度は正の大きさ


def test_pure_rotation_all_same_speed_and_dir(driver):
    driver.set_velocity(0.0, 0.0, 2.0)
    for _ in range(100):
        driver.update(0.1)
    dirs, hz = driver.chain.calls[-1]
    assert dirs == [FWD, FWD, FWD]           # +omega=CCW は 3 輪とも正回転
    assert hz[1] == pytest.approx(hz[0])
    assert hz[2] == pytest.approx(hz[0])
