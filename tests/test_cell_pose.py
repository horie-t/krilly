"""セル内位置ずれの測定と補正 (cell_pose / apply_cell_offset #54) のユニットテスト。

合成フレーム (既知の位置に赤帯を描く) で「帯のずれ -> mm」を検証し、補正側は
推定器を直接動かして確認する。カメラ不要。
"""

import math

import numpy as np
import pytest

from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.localization.grid import apply_cell_offset
from krilly.perception.cell_pose import (
    PX_PER_MM_X,
    PX_PER_MM_Y,
    body_offset_to_world,
    cell_offset,
)
from krilly.perception.wall_detect import (
    BACK,
    CALIBRATED_BANDS,
    FRONT,
    LEFT,
    RIGHT,
    WallDetector,
    calibrated_config,
)

RED = (0, 0, 255)


def frame_with_bands(shift_x: int = 0, shift_y: int = 0, edges=(LEFT, RIGHT)) -> np.ndarray:
    """校正位置から (shift_x, shift_y) px ずらして赤帯を描いた合成フレーム。"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for edge in edges:
        lo, hi = CALIBRATED_BANDS[edge]
        if edge in (LEFT, RIGHT):
            img[100:400, lo + shift_x : hi + shift_x] = RED
        else:
            img[lo + shift_y : hi + shift_y, 180:400] = RED
    return img


@pytest.fixture
def detector():
    return WallDetector(calibrated_config())


# --- スケール --------------------------------------------------------------
def test_px_per_mm_from_known_geometry():
    # 対向する帯の間隔が 180mm。実測位置から 1.6-1.7 px/mm 程度になる
    assert 1.5 < PX_PER_MM_X < 1.8
    assert 1.5 < PX_PER_MM_Y < 1.8


# --- 測定 -----------------------------------------------------------------
def test_no_offset_when_bands_are_at_the_calibrated_position(detector):
    off = cell_offset(frame_with_bands(), detector)
    assert off.walls_x == 2 and off.walls_y == 0
    assert off.left_m == pytest.approx(0.0, abs=2e-3)
    assert off.forward_m is None            # 前後の壁が無いので測れない


def test_lateral_offset_is_recovered(detector):
    """帯が右へ 30px ずれていたら、機体は左へ約 18mm ずれている。"""
    off = cell_offset(frame_with_bands(shift_x=30), detector)
    assert off.forward_m is None
    assert off.left_m == pytest.approx(30 / PX_PER_MM_X / 1000.0, abs=2e-3)
    assert off.left_m > 0                    # 帯が +x にずれる = 機体は左へずれている


def test_negative_lateral_offset(detector):
    off = cell_offset(frame_with_bands(shift_x=-24), detector)
    assert off.left_m == pytest.approx(-24 / PX_PER_MM_X / 1000.0, abs=2e-3)


def test_forward_offset_from_horizontal_bands(detector):
    off = cell_offset(frame_with_bands(shift_y=20, edges=(FRONT, BACK)), detector)
    assert off.walls_y == 2 and off.walls_x == 0
    assert off.left_m is None
    assert off.forward_m == pytest.approx(20 / PX_PER_MM_Y / 1000.0, abs=2e-3)


def test_both_axes_when_all_four_walls_are_visible(detector):
    off = cell_offset(
        frame_with_bands(shift_x=16, shift_y=-12, edges=(LEFT, RIGHT, FRONT, BACK)),
        detector,
    )
    assert off.walls_x == 2 and off.walls_y == 2
    assert off.left_m == pytest.approx(16 / PX_PER_MM_X / 1000.0, abs=2e-3)
    assert off.forward_m == pytest.approx(-12 / PX_PER_MM_Y / 1000.0, abs=2e-3)
    assert off.measured


def test_single_wall_is_enough(detector):
    off = cell_offset(frame_with_bands(shift_x=30, edges=(RIGHT,)), detector)
    assert off.walls_x == 1
    assert off.left_m == pytest.approx(30 / PX_PER_MM_X / 1000.0, abs=3e-3)


def test_nothing_measurable_without_walls(detector):
    off = cell_offset(np.zeros((480, 640, 3), dtype=np.uint8), detector)
    assert not off.measured and off.walls_x == 0 and off.walls_y == 0


# --- 車体 -> 世界 ---------------------------------------------------------
@pytest.mark.parametrize(
    "phi_deg, expected",
    [
        (0.0, (0.01, 0.02)),      # 東向き: 前=+東, 左=+北
        (90.0, (-0.02, 0.01)),    # 北向き: 前=+北, 左=-東
        (180.0, (-0.01, -0.02)),
        (-90.0, (0.02, -0.01)),
    ],
)
def test_body_offset_to_world(phi_deg, expected):
    got = body_offset_to_world(0.01, 0.02, math.radians(phi_deg))
    assert got == pytest.approx(expected, abs=1e-9)


def test_body_offset_to_world_treats_none_as_zero():
    assert body_offset_to_world(None, 0.02, 0.0) == pytest.approx((0.0, 0.02))
    assert body_offset_to_world(0.01, None, 0.0) == pytest.approx((0.01, 0.0))


# --- 補正の適用 -----------------------------------------------------------
@pytest.fixture
def est():
    return DeadReckoning(KiwiKinematics())


def test_apply_cell_offset_pulls_the_estimate_to_the_measured_position(est):
    """北向きでセル中央 (0.18, 0.18)、実測「前へ 10mm・左へ 5mm ずれている」。"""
    est.reset(0.18, 0.18, math.pi / 2)
    assert apply_cell_offset(est, (0.18, 0.18), forward_m=0.010, left_m=-0.005)
    # 北向き: 前 = +Y, 左 = -X
    assert est.x == pytest.approx(0.18 + 0.005)
    assert est.y == pytest.approx(0.18 + 0.010)
    assert est.phi == pytest.approx(math.pi / 2)     # 方位は触らない


def test_apply_cell_offset_only_corrects_measured_axes(est):
    """測れなかった軸は動かさない (0 扱いで引っ張らない)。"""
    est.reset(0.18 + 0.02, 0.18 + 0.03, math.pi / 2)   # 東に20mm・北に30mmずれた推定
    assert apply_cell_offset(est, (0.18, 0.18), forward_m=None, left_m=0.0)
    # 北向きで左右 = 世界X のみ補正される
    assert est.x == pytest.approx(0.18)
    assert est.y == pytest.approx(0.18 + 0.03)        # 前後は測れていないので不変


def test_apply_cell_offset_east_facing_axes(est):
    est.reset(0.0, 0.0, 0.0)                           # 東向き
    apply_cell_offset(est, (0.0, 0.0), forward_m=0.012, left_m=None)
    assert est.x == pytest.approx(0.012)               # 前 = +X
    assert est.y == pytest.approx(0.0)


def test_apply_cell_offset_weight(est):
    est.reset(0.0, 0.0, 0.0)
    apply_cell_offset(est, (0.0, 0.0), forward_m=0.02, left_m=None, weight=0.5)
    assert est.x == pytest.approx(0.01)                # 半分だけ引き込む


def test_apply_cell_offset_rejects_large_corrections(est):
    est.reset(0.0, 0.0, 0.0)
    assert not apply_cell_offset(
        est, (0.0, 0.0), forward_m=0.20, left_m=None, max_error=0.05
    )
    assert est.pose == (0.0, 0.0, 0.0)                 # 棄却時は動かさない


def test_apply_cell_offset_without_measurement_is_a_noop(est):
    est.reset(0.05, 0.05, 0.0)
    assert not apply_cell_offset(est, (0.0, 0.0), forward_m=None, left_m=None)
    assert est.pose == (0.05, 0.05, 0.0)


def test_apply_cell_offset_uses_explicit_phi(est):
    est.reset(0.0, 0.0, 0.0)                           # 推定は東向きだが…
    apply_cell_offset(est, (0.0, 0.0), forward_m=0.01, left_m=None, phi=math.pi / 2)
    assert est.y == pytest.approx(0.01)                # 指定した北向きで解釈される
    assert est.x == pytest.approx(0.0)


def test_correction_removes_accumulated_drift(est):
    """オドメトリが「セル中央」と思っている状態で 20mm ずれを測ったら、その分動く。"""
    est.reset(0.36, 0.18, 0.0)                         # 東向き、セル中央だと思っている
    apply_cell_offset(est, (0.36, 0.18), forward_m=-0.020, left_m=0.008)
    assert est.x == pytest.approx(0.36 - 0.020)        # 実際は 20mm 手前だった
    assert est.y == pytest.approx(0.18 + 0.008)        # 8mm 左 (=北) にずれていた
