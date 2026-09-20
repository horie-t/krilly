"""格子から測るセル内位置 (#85) のテスト。

**なぜ要るか**: 実機が壁に食い込んで止まったとき、ROI はセル中央を前提に置いてあり
探索が ±40px = ±23.5mm しかないので帯を見失い、真は ``{E}`` だけなのに ``{N, E}`` と
読んで「迷子」でセッションが終わった。そのフレームを ``tests/data`` に置いてある。
"""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from krilly.localization.recovery import wall_contact
from krilly.motion.corner import corridor_clearance_m
from krilly.perception.lattice import (
    PITCH_MM,
    LatticeOffset,
    lattice_offset,
    offset_to_shift_px,
)
from krilly.perception.wall_detect import (
    CALIBRATED_GEOMETRY,
    DEFAULT_FRAME_SIZE,
    WallDetector,
    calibrated_config,
)

RED = (0, 0, 255)          # BGR
#: 実機が (6,6) の東壁に W2 を当てて止まったときのフレーム (#85)。
STUCK = Path("tests/data/stuck_against_east_wall.png")


def lattice_frame(left_mm: float = 0.0, forward_mm: float = 0.0,
                  yaw_deg: float = 0.0) -> np.ndarray:
    """機体が (left, forward) だけずれた位置に居るときの格子を描く。

    機体が左 (+y) へずれると世界の特徴は画像の +x へ動く (`cell_pose` と同じ約束)。
    """
    w, h = DEFAULT_FRAME_SIZE
    img = np.zeros((h, w, 3), dtype=np.uint8)
    half = 6 * CALIBRATED_GEOMETRY.px_per_mm_x            # 壁の厚み 12mm の半分
    pitch_x = PITCH_MM * CALIBRATED_GEOMETRY.px_per_mm_x
    pitch_y = PITCH_MM * CALIBRATED_GEOMETRY.px_per_mm_y
    dx = left_mm * CALIBRATED_GEOMETRY.px_per_mm_x
    dy = forward_mm * CALIBRATED_GEOMETRY.px_per_mm_y
    for k in (-2, -1, 0, 1, 2):
        x = w / 2 + dx + (k + 0.5) * pitch_x
        y = h / 2 + dy + (k + 0.5) * pitch_y
        if 0 <= x < w:
            img[:, int(x - half):int(x + half)] = RED
        if 0 <= y < h:
            img[int(y - half):int(y + half), :] = RED
    if yaw_deg:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), -yaw_deg, 1.0)
        img = cv2.warpAffine(img, m, (w, h))
    return img


@pytest.mark.parametrize("left_mm, forward_mm", [
    (0.0, 0.0), (28.0, 0.0), (-28.0, 0.0), (0.0, 12.0), (-40.0, 25.0),
])
def test_the_offset_is_recovered_from_the_lattice_alone(left_mm, forward_mm):
    """ROI をまったく使わずに、格子の位相だけでずれが出ること。"""
    got = lattice_offset(lattice_frame(left_mm, forward_mm))
    assert got.left_m * 1e3 == pytest.approx(left_mm, abs=1.5)
    assert got.forward_m * 1e3 == pytest.approx(forward_mm, abs=1.5)


def test_the_yaw_has_to_be_taken_out_first():
    """**傾いたまま位相を取ると潰れる。** 5° は 1 本の壁が端から端で 60px 動く量。"""
    frame = lattice_frame(28.0, 0.0, yaw_deg=-5.0)
    naive = lattice_offset(frame)
    aligned = lattice_offset(frame, math.radians(-5.0))
    assert aligned.left_m * 1e3 == pytest.approx(28.0, abs=2.0)
    assert aligned.left_confidence > naive.left_confidence


def test_red_that_is_not_on_the_lattice_shows_up_as_low_confidence():
    """迷路の外の赤 (#89 の定規袋) は**消えない** — 位相の揃い具合に出る。"""
    frame = lattice_frame(0.0, 0.0)
    clean = lattice_offset(frame)
    frame[40:400, 20:150] = RED              # 格子に乗らない大きな赤
    dirty = lattice_offset(frame)
    assert dirty.left_confidence < clean.left_confidence


def test_the_offset_folds_into_half_a_pitch():
    """測れるのは mod 180mm。機体はセルの中に居るので (-90, +90] で一意になる。"""
    got = lattice_offset(lattice_frame(PITCH_MM + 20.0, 0.0))
    assert got.left_m * 1e3 == pytest.approx(20.0, abs=1.5)


def test_shift_px_moves_the_rois_the_way_the_bands_moved():
    """機体が左へずれると帯は画像の +x へ動くので、ROI も +x へ動かす。"""
    g = CALIBRATED_GEOMETRY
    dx, dy = offset_to_shift_px(LatticeOffset(0.010, 0.020, 1.0, 1.0))
    assert dx == round(20.0 * g.px_per_mm_x) and dy == round(10.0 * g.px_per_mm_y)
    assert offset_to_shift_px(LatticeOffset(None, None, 0.0, 0.0)) == (0, 0)


# --- 実機のフレーム ---------------------------------------------------------
@pytest.mark.skipif(not STUCK.exists(), reason="フレームが無い")
def test_the_real_jam_is_measured_and_the_walls_read_correctly():
    """実機が東壁に食い込んで止まったフレーム (#85)。真の (6,6) の壁は **E だけ**。

    手で格子を測った値は 左右 -28mm / 前後 -6mm、軸角 -5.07°。ROI をそのままにすると
    ``{N, E}`` と読んで「迷子」になる — その幽霊の北壁が消えることを固定する。
    """
    img = cv2.imread(str(STUCK))
    yaw = math.radians(-5.07)
    off = lattice_offset(img, yaw)
    assert off.left_m * 1e3 == pytest.approx(-28.0, abs=2.0)
    assert off.forward_m * 1e3 == pytest.approx(-6.0, abs=4.0)
    assert off.left_confidence > 0.9

    det = WallDetector(calibrated_config(neighbors=False))
    def verdict(shift):
        got = {k: v[0] for k, v in det.measure(img, shift).items()}
        return {n for n, f in got.items() if f >= det.cfg.threshold_for(n)}

    assert verdict((0, 0)) == {"front", "right"}          # 幽霊の北壁が生える
    assert verdict(offset_to_shift_px(off)) == {"right"}  # 真のパターン


@pytest.mark.skipif(not STUCK.exists(), reason="フレームが無い")
def test_the_real_jam_is_reported_as_contact_not_as_lost():
    """**居るセルが分かっても、壁に食い込んだまま走り出してはいけない。**"""
    img = cv2.imread(str(STUCK))
    off = lattice_offset(img, math.radians(-5.07))
    det = WallDetector(calibrated_config(neighbors=False))
    got = det.measure(img, offset_to_shift_px(off))
    walls = {n: v[0] >= det.cfg.threshold_for(n) for n, v in got.items()}
    said = wall_contact(off, walls, corridor_clearance_m())
    assert said is not None and "接触している" in said
    # 寄っていない側に壁があっても接触とは言わない
    assert wall_contact(off, {"left": True, "right": False},
                        corridor_clearance_m()) is None
