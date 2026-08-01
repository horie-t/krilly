"""cell_move_demo のシーケンス解析とトークン→プリミティブ対応のユニットテスト。"""

import math

import pytest

from krilly.config import MazeConfig, RobotConfig
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.motion.cell_motion import CellMotion
from krilly.motion.velocity_driver import VelocityDriver
from scripts.cell_move_demo import TOKENS, parse_seq, wrapped_deg

ROBOT = RobotConfig(
    wheel_diameter_m=0.048,
    wheel_count=3,
    center_to_wheel_m=0.05,
    steps_per_rev=200,
    microstep=16,
    wheel_angles_deg=[0.0, 120.0, 240.0],
)
MAZE = MazeConfig(
    grid_size=16,
    cell_pitch_m=0.180,
    wall_thickness_m=0.012,
    wall_height_m=0.050,
    goal_min=(7, 7),
    goal_max=(8, 8),
)


class FakeChain:
    def run_all(self, directions, speeds):
        pass

    def soft_stop_all(self):
        pass


@pytest.fixture
def motion():
    kin = KiwiKinematics(config=ROBOT)
    return CellMotion(VelocityDriver(FakeChain(), kinematics=kin), DeadReckoning(kin), maze=MAZE)


# --- シーケンス解析 --------------------------------------------------------
def test_parse_seq_accepts_both_notations():
    assert parse_seq("F,L,F") == ["F", "L", "F"]
    assert parse_seq("FLF") == ["F", "L", "F"]
    assert parse_seq("f, l ,r") == ["F", "L", "R"]


def test_wrapped_deg_normalises_accumulated_heading():
    # 推定φは積算値なので 360° を超える (4×90° 旋回で ≈ 358.8° = -1.2°)
    assert wrapped_deg(math.radians(358.76)) == pytest.approx(-1.24, abs=1e-9)
    assert wrapped_deg(math.radians(-90.0)) == pytest.approx(-90.0)
    assert wrapped_deg(math.radians(450.0)) == pytest.approx(90.0)


def test_parse_seq_rejects_unknown_token():
    with pytest.raises(SystemExit):
        parse_seq("F,X")


# --- トークン -> 基準姿勢の進み方 ------------------------------------------
@pytest.mark.parametrize(
    "token, expect_ref",
    [
        ("F", (0.180, 0.0, 0.0)),
        ("B", (-0.180, 0.0, 0.0)),
        ("L", (0.0, 0.0, math.pi / 2)),
        ("R", (0.0, 0.0, -math.pi / 2)),
        ("U", (0.0, 0.0, math.pi)),
    ],
)
def test_token_moves_reference_as_expected(motion, token, expect_ref):
    _label, start = TOKENS[token]
    start(motion)
    x, y, phi = motion.reference
    assert (x, y) == pytest.approx(expect_ref[:2])
    assert math.cos(phi - expect_ref[2]) == pytest.approx(1.0, abs=1e-12)  # ±π を同一視
