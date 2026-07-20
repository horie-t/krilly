"""格子スナップ補正 (GridCorrector / snap_to_grid) と位置補正のユニットテスト。"""

import pytest

from krilly.config import RobotConfig
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.localization.grid import GridCorrector, snap_to_grid

PITCH = 0.180

CFG = RobotConfig(
    wheel_diameter_m=0.048, wheel_count=3, center_to_wheel_m=0.05,
    steps_per_rev=200, microstep=16, wheel_angles_deg=[0.0, 120.0, 240.0],
)


@pytest.fixture
def est():
    return DeadReckoning(kinematics=KiwiKinematics(config=CFG))


# --- snap_to_grid ----------------------------------------------------------
def test_snap_to_nearest_line():
    assert snap_to_grid(0.17, PITCH) == pytest.approx(0.18)
    assert snap_to_grid(0.10, PITCH) == pytest.approx(0.18)   # 0.10 は 0.18 に近い
    assert snap_to_grid(0.08, PITCH) == pytest.approx(0.0)
    assert snap_to_grid(0.35, PITCH) == pytest.approx(0.36)


def test_snap_with_offset_cell_centers():
    # セル中心 (offset = pitch/2) 基準
    off = PITCH / 2
    assert snap_to_grid(0.10, PITCH, off) == pytest.approx(0.09)
    assert snap_to_grid(0.26, PITCH, off) == pytest.approx(0.27)


def test_snap_negative():
    assert snap_to_grid(-0.17, PITCH) == pytest.approx(-0.18)


# --- 位置補正メソッド ------------------------------------------------------
def test_correct_x_full_and_partial(est):
    est.reset(0.20, 0.0, 0.0)
    est.correct_x(0.18, weight=0.5)
    assert est.x == pytest.approx(0.19)      # 半分だけ
    est.correct_x(0.18, weight=1.0)
    assert est.x == pytest.approx(0.18)


def test_correct_position(est):
    est.reset(0.20, 0.35, 0.0)
    est.correct_position(0.18, 0.36, weight=1.0)
    assert (est.x, est.y) == pytest.approx((0.18, 0.36))


# --- GridCorrector ---------------------------------------------------------
def test_from_maze_uses_cell_pitch():
    from krilly.config import MazeConfig
    maze = MazeConfig(grid_size=16, cell_pitch_m=PITCH, wall_thickness_m=0.012,
                      wall_height_m=0.05, goal_min=(7, 7), goal_max=(8, 8))
    gc = GridCorrector.from_maze(maze)
    assert gc.pitch == PITCH


def test_apply_x_snaps_within_tolerance(est):
    est.reset(0.19, 0.0, 0.0)          # 0.18 グリッド線から +0.01
    gc = GridCorrector(PITCH)
    assert gc.apply_x(est, max_error=0.03) is True
    assert est.x == pytest.approx(0.18)


def test_apply_x_rejected_beyond_tolerance(est):
    est.reset(0.27, 0.0, 0.0)          # 最寄り線 0.36 まで 0.09 → tol超で棄却
    gc = GridCorrector(PITCH)
    assert gc.apply_x(est, max_error=0.03) is False
    assert est.x == pytest.approx(0.27)   # 変化なし


def test_apply_y_snaps(est):
    est.reset(0.0, 0.35, 0.0)
    gc = GridCorrector(PITCH)
    assert gc.apply_y(est, max_error=0.03) is True
    assert est.y == pytest.approx(0.36)


def test_residual():
    gc = GridCorrector(PITCH)
    assert gc.residual(0.19) == pytest.approx(0.01)
    assert gc.residual(0.18) == pytest.approx(0.0)


def test_invalid_pitch_raises():
    with pytest.raises(ValueError):
        GridCorrector(0.0)
