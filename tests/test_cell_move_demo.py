"""cell_move_demo のシーケンス解析とトークン→プリミティブ対応のユニットテスト。"""

import math

import pytest

from krilly.config import MazeConfig, RobotConfig
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.motion.cell_motion import CellMotion
from krilly.motion.velocity_driver import VelocityDriver
from scripts.cell_move_demo import Move, parse_seq, world_components, wrapped_deg

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
    assert parse_seq("F,L,F") == [Move("F"), Move("L"), Move("F")]
    assert parse_seq("FLF") == [Move("F"), Move("L"), Move("F")]
    assert parse_seq("f, l ,r") == [Move("F"), Move("L"), Move("R")]


def test_parse_seq_counts():
    """数字は「1 動作で何セル/何回転」。U は L2 の別名 (#21)。"""
    assert parse_seq("F4") == [Move("F", 4)]
    assert parse_seq("F4,L,B2") == [Move("F", 4), Move("L"), Move("B", 2)]
    assert parse_seq("U") == [Move("L", 2)]
    assert parse_seq("U2") == [Move("L", 4)]


def test_move_labels():
    assert Move("F", 4).label == "4セル前進"
    assert Move("R").label == "右90°"
    assert Move("L", 2).label == "左180°"
    assert Move("H", 15).label == "左へ15セル"
    assert Move("G").label == "右へ1セル"


@pytest.mark.parametrize("text", ["F0", "4F", "F,X", ""])
def test_parse_seq_rejects_bad_input(text):
    with pytest.raises(SystemExit):
        parse_seq(text)


def test_wrapped_deg_normalises_accumulated_heading():
    # 推定φは積算値なので 360° を超える (4×90° 旋回で ≈ 358.8° = -1.2°)
    assert wrapped_deg(math.radians(358.76)) == pytest.approx(-1.24, abs=1e-9)
    assert wrapped_deg(math.radians(-90.0)) == pytest.approx(-90.0)
    assert wrapped_deg(math.radians(450.0)) == pytest.approx(90.0)


# --- トークン -> 基準姿勢の進み方 ------------------------------------------
@pytest.mark.parametrize(
    "text, expect_ref",
    [
        ("F", (0.180, 0.0, 0.0)),
        ("B", (-0.180, 0.0, 0.0)),
        ("L", (0.0, 0.0, math.pi / 2)),
        ("R", (0.0, 0.0, -math.pi / 2)),
        ("U", (0.0, 0.0, math.pi)),
        ("F4", (0.720, 0.0, 0.0)),
        ("R2", (0.0, 0.0, math.pi)),
        # 平行移動: 機体は回らないので基準方位は 0 のまま (#76)
        ("H", (0.0, 0.180, 0.0)),
        ("G", (0.0, -0.180, 0.0)),
        ("H15", (0.0, 2.700, 0.0)),
    ],
)
def test_token_moves_reference_as_expected(motion, text, expect_ref):
    (move,) = parse_seq(text)
    move.start(motion)
    x, y, phi = motion.reference
    assert (x, y) == pytest.approx(expect_ref[:2])
    assert math.cos(phi - expect_ref[2]) == pytest.approx(1.0, abs=1e-12)  # ±π を同一視


# --- カメラ実測の格子成分 (#21) --------------------------------------------
def test_world_components_uses_the_base_pose_axes():
    """成分名は base_phi から見た前後/左右。迷路の東西南北ではない。"""
    # 基準と同じ向き: 前後・左右がそのまま
    assert world_components(0.01, 0.002, 0.0) == pytest.approx({"前後": 0.01, "左右": 0.002})
    # 基準から左 90° を向いている: 機体の前 = 基準の左
    assert world_components(0.01, 0.002, math.pi / 2) == pytest.approx(
        {"左右": 0.01, "前後": -0.002})
    # 基準から 180°: 両軸が反転する
    assert world_components(0.01, 0.002, math.pi) == pytest.approx(
        {"前後": -0.01, "左右": -0.002})


def test_world_components_base_phi_is_relative():
    """φ=0 始まりでない推定器でも、基準との相対だけで決まる。"""
    assert world_components(0.01, 0.002, math.pi / 2, math.pi / 2) == pytest.approx(
        {"前後": 0.01, "左右": 0.002})


def test_world_components_drops_unmeasured_axes():
    """測れなかった軸はキーごと落とす (差分で「測っていない方向」を混ぜないため)。"""
    assert world_components(0.01, None, 0.0) == {"前後": 0.01}
    assert world_components(None, 0.01, 0.0) == {"左右": 0.01}
    assert world_components(None, None, 0.0) == {}
