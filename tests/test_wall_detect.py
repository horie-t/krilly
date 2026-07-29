"""カメラからのセル壁判定 (wall_detect) のユニットテスト。"""

import numpy as np
import pytest

from krilly.perception.wall_detect import (
    BACK,
    FRONT,
    LEFT,
    RIGHT,
    Roi,
    WallDetector,
    WallDetectorConfig,
    body_walls_to_maze,
    default_rois,
    roi_red_fraction,
    update_maze_walls,
)
from krilly.solver.maze import Direction, Maze


def _blank_bgr(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _fill_red(img, roi: Roi):
    img[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = (0, 0, 255)  # BGR red


# --- roi_red_fraction ------------------------------------------------------
def test_roi_red_fraction():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 255                     # 20x20 = 400 px 赤
    assert roi_red_fraction(mask, Roi(10, 10, 20, 20)) == pytest.approx(1.0)
    assert roi_red_fraction(mask, Roi(50, 50, 20, 20)) == pytest.approx(0.0)
    assert roi_red_fraction(mask, Roi(0, 0, 40, 20)) == pytest.approx(200 / 800)


def test_default_rois_positions():
    rois = default_rois(640, 480, thickness=70, span=0.5)
    assert set(rois) == {FRONT, BACK, LEFT, RIGHT}
    assert rois[FRONT].y == 0
    assert rois[BACK].y == 480 - 70
    assert rois[LEFT].x == 0
    assert rois[RIGHT].x == 640 - 70


# --- WallDetector ----------------------------------------------------------
def test_detect_wall_only_in_front():
    rois = default_rois(640, 480)
    img = _blank_bgr()
    _fill_red(img, rois[FRONT])                  # 前方 ROI だけ赤
    det = WallDetector(WallDetectorConfig(rois=rois))
    walls = det.detect(img)
    assert walls[FRONT] is True
    assert walls[BACK] is False
    assert walls[LEFT] is False
    assert walls[RIGHT] is False


def test_exclude_region_removes_red():
    # BACK ROI を赤で埋めるが、そこを exclude 矩形で除外 -> 壁なし判定
    rois = default_rois(640, 480)
    img = _blank_bgr()
    _fill_red(img, rois[BACK])
    det = WallDetector(WallDetectorConfig(rois=rois, exclude=[rois[BACK]]))
    assert det.detect(img)[BACK] is False


def test_detect_below_threshold_is_no_wall():
    rois = default_rois(640, 480)
    img = _blank_bgr()
    # FRONT ROI のごく一部だけ赤 (閾値未満)
    r = rois[FRONT]
    img[r.y : r.y + 3, r.x : r.x + 5] = (0, 0, 255)
    det = WallDetector(WallDetectorConfig(rois=rois, threshold=0.15))
    assert det.detect(img)[FRONT] is False


# --- 機体相対 -> 迷路方角 --------------------------------------------------
def test_body_to_maze_facing_north():
    walls = body_walls_to_maze(
        {FRONT: True, BACK: False, LEFT: True, RIGHT: False}, Direction.N
    )
    assert walls[Direction.N] is True    # front
    assert walls[Direction.S] is False   # back
    assert walls[Direction.W] is True    # left
    assert walls[Direction.E] is False   # right


def test_body_to_maze_facing_east():
    walls = body_walls_to_maze(
        {FRONT: True, BACK: False, LEFT: True, RIGHT: False}, Direction.E
    )
    assert walls[Direction.E] is True    # front
    assert walls[Direction.W] is False   # back
    assert walls[Direction.N] is True    # left
    assert walls[Direction.S] is False   # right


# --- Maze への反映 ---------------------------------------------------------
def test_update_maze_walls_sets_and_shares():
    m = Maze(4)
    update_maze_walls(m, 1, 1, {Direction.N: True, Direction.E: True,
                                Direction.S: False, Direction.W: False})
    assert m.has_wall(1, 1, Direction.N) is True
    assert m.has_wall(1, 2, Direction.S) is True     # 共有エッジ
    assert m.has_wall(1, 1, Direction.E) is True
    assert m.has_wall(1, 1, Direction.S) is False
