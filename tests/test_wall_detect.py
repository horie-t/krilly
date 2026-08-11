"""カメラからのセル壁判定 (wall_detect) のユニットテスト。"""

import numpy as np
import pytest

from krilly.perception.red_wall import red_mask
from krilly.perception.wall_detect import (
    BACK,
    BODY_DIRS,
    CALIBRATED_BANDS,
    CALIBRATED_RED,
    FRONT,
    LEFT,
    RIGHT,
    Roi,
    WallDetector,
    WallDetectorConfig,
    best_roi_red_fraction,
    body_walls_to_maze,
    calibrated_config,
    calibrated_rois,
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


def test_calibrated_config():
    rois = calibrated_rois()
    assert set(rois) == set(BODY_DIRS)
    cfg = calibrated_config()
    assert set(cfg.rois) == set(BODY_DIRS)
    # 右壁が白飛びして彩度が落ちるため、赤しきい値は既定より緩い (#56)
    assert cfg.red.s_min <= 50 and cfg.red.v_min < 70


def _frame_with_vertical_band(x: int, w: int = 26) -> np.ndarray:
    """指定の列位置に赤い縦帯 (壁上面) を描いた合成フレーム。"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:400, x : x + w] = (0, 0, 255)
    return img


def test_best_roi_red_fraction_finds_a_shifted_band():
    """ROI からずれた帯でも、探索すれば見つかりオフセットも分かる (#56)。"""
    roi = calibrated_rois()[LEFT]
    shifted = _frame_with_vertical_band(roi.x + 30)
    mask = red_mask(shifted, CALIBRATED_RED)
    fixed, off0 = best_roi_red_fraction(mask, roi, vertical=True, search_px=0)
    found, off = best_roi_red_fraction(mask, roi, vertical=True, search_px=40)
    assert off0 == 0
    assert found > fixed                      # 探索すれば拾える
    # 帯 (幅26) の中心は ROI 中心から +21px。平坦部の中心を返すのでこれに一致する
    assert 15 <= off <= 27


def test_search_does_not_pick_up_the_opposite_wall():
    """探索範囲を広げても、反対側の壁の帯 (292px 先) は拾わない。"""
    rois = calibrated_rois()
    only_right = _frame_with_vertical_band(rois[RIGHT].x)
    mask = red_mask(only_right, CALIBRATED_RED)
    left, _ = best_roi_red_fraction(mask, rois[LEFT], vertical=True, search_px=40)
    right, _ = best_roi_red_fraction(mask, rois[RIGHT], vertical=True, search_px=40)
    assert left == pytest.approx(0.0)
    assert right > 0.4


def test_detector_search_px_zero_keeps_fixed_roi_behaviour():
    rois = calibrated_rois()
    shifted = _frame_with_vertical_band(rois[LEFT].x + 30)
    fixed = WallDetector(WallDetectorConfig(rois=rois, red=CALIBRATED_RED, search_px=0))
    searching = WallDetector(
        WallDetectorConfig(rois=rois, red=CALIBRATED_RED, search_px=40)
    )
    assert fixed.red_fractions(shifted)[LEFT] < searching.red_fractions(shifted)[LEFT]
    assert searching.band_offsets(shifted)[LEFT] != 0


def test_calibrated_config_searches_for_the_band():
    cfg = calibrated_config()
    # 実機のセル内位置ずれは最大 ~25mm ≒ 40px (#56) なので、それを吸収できる幅
    assert cfg.search_px >= 40
    # 全セル調査の分離ギャップ (0.033..0.102) の内側
    assert 0.04 <= cfg.threshold <= 0.10


def test_calibrated_rois_contain_the_measured_wall_bands():
    """ROI が実測した赤帯を内側に含むこと (#56 の ROI ずれの回帰テスト)。

    帯から外れた分だけ赤割合が下がり、しきい値を割ると壁を見落として衝突する。
    """
    rois = calibrated_rois()
    for edge, (lo, hi) in CALIBRATED_BANDS.items():
        roi = rois[edge]
        start, length = (roi.y, roi.h) if edge in (FRONT, BACK) else (roi.x, roi.w)
        assert start <= lo and hi <= start + length, (
            f"{edge}: ROI {start}..{start + length} が帯 {lo}..{hi} を含んでいない"
        )


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


def test_per_edge_threshold_override():
    """辺別しきい値 (#65): BACK はケーブルの偽帯 (<=0.12) と実壁 (>=0.51) を分離する。"""
    cfg = calibrated_config()
    assert cfg.threshold_for(BACK) == pytest.approx(0.25)
    for edge in (FRONT, LEFT, RIGHT):
        assert cfg.threshold_for(edge) == pytest.approx(cfg.threshold)
    # detect() が辺別しきい値を使うこと: BACK に薄い赤 (0.12 相当) を置いても壁なし
    rois = calibrated_rois()
    img = _blank_bgr()
    r = rois[BACK]
    img[r.y : r.y + int(r.h * 0.12) + 1, r.x : r.x + r.w] = (0, 0, 255)
    det = WallDetector(WallDetectorConfig(rois=rois, threshold=0.08,
                                          thresholds={BACK: 0.25}, search_px=0))
    assert det.detect(img)[BACK] is False
