"""カメラからのセル壁判定 (wall_detect) のユニットテスト。"""

import numpy as np
import pytest

from krilly.perception.cell_pose import cell_offset
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
    body_edge_for,
    body_walls_to_maze,
    calibrated_config,
    maze_walls_to_body,
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
    fixed, off0, _ = best_roi_red_fraction(mask, roi, vertical=True, search_px=0)
    found, off, saturated = best_roi_red_fraction(mask, roi, vertical=True, search_px=40)
    assert off0 == 0
    assert found > fixed                      # 探索すれば拾える
    # 帯 (幅26) の中心は ROI 中心から +21px。平坦部の中心を返すのでこれに一致する
    assert 15 <= off <= 27
    assert not saturated                      # 端で頭打ちにはなっていない


def test_search_does_not_pick_up_the_opposite_wall():
    """探索範囲を広げても、反対側の壁の帯 (292px 先) は拾わない。"""
    rois = calibrated_rois()
    only_right = _frame_with_vertical_band(rois[RIGHT].x)
    mask = red_mask(only_right, CALIBRATED_RED)
    left, _, _ = best_roi_red_fraction(mask, rois[LEFT], vertical=True, search_px=40)
    right, _, _ = best_roi_red_fraction(mask, rois[RIGHT], vertical=True, search_px=40)
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


# --- 帯探索がフレーム端で頭打ちになる場合 (#21) -----------------------------
def _frame_with_horizontal_band(y: int, h: int = 26) -> np.ndarray:
    """指定の行位置に赤い横帯 (前後の壁上面) を描いた合成フレーム。"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[y : y + h, 100:540] = (0, 0, 255)
    return img


def test_back_search_saturates_at_the_bottom_of_the_frame():
    """BACK ROI は下端まで +25px しか動かせない。その先の帯は頭打ちになる (#21)。

    実機で機体を 20mm (=34px) 前進させたのに読みは 7mm しか動かなかった事例。
    頭打ちの値は「小さめのもっともらしいずれ」に化けるので、飽和を知らせる。
    """
    roi = calibrated_rois()[BACK]
    limit = 480 - roi.h                          # ROI 開始位置の上限
    far = _frame_with_horizontal_band(roi.y + 34)   # 34px 先 = 20mm 前進相当
    mask = red_mask(far, CALIBRATED_RED)
    _fraction, off, saturated = best_roi_red_fraction(
        mask, roi, vertical=False, search_px=40)
    assert saturated
    assert roi.y + off == limit or off % 2 == 0   # 端まで寄せて打ち切られている
    assert off < 34                               # 本当のずれより小さく出る


def test_saturated_edge_is_dropped_from_the_position_measurement():
    """飽和した辺は位置測定に使わない (中央寄りに居ると誤解させないため)。"""
    roi = calibrated_rois()[BACK]
    frame = _frame_with_horizontal_band(roi.y + 34)
    det = WallDetector(calibrated_config())
    off = cell_offset(frame, det)
    assert BACK in off.saturated
    assert off.forward_m is None                  # 前後は測れなかった扱いになる
    assert off.walls_y == 0


def test_unsaturated_band_still_measures():
    """フレーム端から遠い側 (後退方向) は従来どおり測れる。"""
    roi = calibrated_rois()[BACK]
    frame = _frame_with_horizontal_band(roi.y - 20)
    det = WallDetector(calibrated_config())
    off = cell_offset(frame, det)
    assert off.saturated == ()
    assert off.forward_m is not None and off.forward_m < 0


# --- 進行方角 -> 機体の辺 (#76) --------------------------------------------
def test_body_edge_for_is_the_identity_when_facing_north():
    """向きを北に固定すると N→FRONT / S→BACK / W→LEFT / E→RIGHT の定数写像になる。"""
    assert body_edge_for(Direction.N, Direction.N) == FRONT
    assert body_edge_for(Direction.S, Direction.N) == BACK
    assert body_edge_for(Direction.W, Direction.N) == LEFT
    assert body_edge_for(Direction.E, Direction.N) == RIGHT


def test_body_edge_for_matches_maze_walls_to_body():
    """maze_walls_to_body と同じ写像であること (辺だけを取り出した版)。"""
    for facing in Direction:
        walls = {d: (d is Direction.N) for d in Direction}   # 北だけ壁
        body = maze_walls_to_body(walls, facing)
        for d in Direction:
            edge = body_edge_for(d, facing)
            assert body[edge] == walls[d], (facing, d, edge)


# --- 帯の位置測定 (#87 の画角変更に伴う再校正で使う) -------------------------
def test_band_positions_recovers_synthetic_bands():
    """行/列プロファイルから 4 辺の帯の位置を取り出せること。"""
    import numpy as np

    from krilly.perception.wall_detect import band_positions

    mask = np.zeros((480, 640), np.uint8)
    mask[120:143, 100:540] = 255      # FRONT (行)
    mask[425:448, 100:540] = 255      # BACK
    mask[150:400, 134:156] = 255      # LEFT (列)
    mask[150:400, 438:462] = 255      # RIGHT
    got = band_positions(mask)
    assert got[FRONT] == (120, 142)
    assert got[BACK] == (425, 447)
    assert got[LEFT] == (134, 155)
    assert got[RIGHT] == (438, 461)


def test_band_positions_skips_edges_with_no_wall():
    """壁の無い辺は帯が出ないので結果に入らない。"""
    import numpy as np

    from krilly.perception.wall_detect import band_positions

    mask = np.zeros((480, 640), np.uint8)
    mask[120:143, 100:540] = 255      # FRONT だけ
    got = band_positions(mask)
    assert FRONT in got
    assert BACK not in got and LEFT not in got and RIGHT not in got


def test_band_positions_finds_the_calibrated_bands():
    """校正済みの帯位置を入れたら、その位置が返ること (回帰)。"""
    import numpy as np

    from krilly.perception.wall_detect import CALIBRATED_BANDS, band_positions

    mask = np.zeros((480, 640), np.uint8)
    for edge in (FRONT, BACK):
        lo, hi = CALIBRATED_BANDS[edge]
        mask[lo:hi + 1, 100:540] = 255
    for edge in (LEFT, RIGHT):
        lo, hi = CALIBRATED_BANDS[edge]
        mask[150:400, lo:hi + 1] = 255
    got = band_positions(mask)
    for edge in (FRONT, BACK, LEFT, RIGHT):
        assert got[edge] == CALIBRATED_BANDS[edge], edge


# --- 色相の 2 帯を分けて取る (#87 の調べもの用) -----------------------------
def test_red_mask_parts_splits_by_hue_band():
    """h1 (オレンジ寄り) と h2 (マゼンタ寄り) が分かれて返ること。"""
    import cv2
    import numpy as np

    from krilly.perception.red_wall import RedDetectorConfig, red_mask_parts

    cfg = RedDetectorConfig(morph_kernel=0)
    img = np.zeros((40, 40, 3), np.uint8)
    # 上半分を H=5 (h1 帯)、下半分を H=170 (h2 帯) の彩度・明度が十分な色で塗る
    hsv = np.zeros((40, 40, 3), np.uint8)
    hsv[:20] = (5, 200, 200)
    hsv[20:] = (170, 200, 200)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    h1, h2 = red_mask_parts(img, cfg)
    assert (h1[:20] > 0).all() and (h1[20:] == 0).all()
    assert (h2[20:] > 0).all() and (h2[:20] == 0).all()


def test_red_mask_parts_agrees_with_red_mask_without_morphology():
    """形態学処理を切れば h1 | h2 は red_mask と一致すること。

    掛けると 2 帯の境目で結果がずれるので、判定には red_mask を使う (docstring 参照)。
    """
    import numpy as np

    from krilly.perception.red_wall import RedDetectorConfig, red_mask, red_mask_parts

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    cfg = RedDetectorConfig(morph_kernel=0)
    h1, h2 = red_mask_parts(img, cfg)
    assert ((h1 | h2) == red_mask(img, cfg)).all()


def test_red_mask_parts_finds_the_orange_side_separately():
    """壁の色相 (h2) と、オレンジ寄りの異物 (h1) を区別できること。

    #87 でリボンケーブルの残留を特定した使い方。単色のマスクでは分からなかった。
    """
    import cv2
    import numpy as np

    from krilly.perception.red_wall import RedDetectorConfig, red_mask_parts

    hsv = np.zeros((30, 60, 3), np.uint8)
    hsv[:, :30] = (170, 200, 200)   # 壁 (h2)
    hsv[:, 30:] = (7, 90, 60)       # 暗いオレンジ寄りの異物 (h1)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    h1, h2 = red_mask_parts(img, RedDetectorConfig(morph_kernel=0, s_min=50, v_min=40))
    assert (h2[:, :30] > 0).all() and (h2[:, 30:] == 0).all()
    assert (h1[:, 30:] > 0).all() and (h1[:, :30] == 0).all()


# --- 幾何からの校正 (#88) ---------------------------------------------------
def test_geometry_reproduces_the_measured_band_centres():
    """壁は必ずセル中心から半ピッチなので、帯の位置は幾何から出る。

    実測 (:data:`CALIBRATED_BANDS`) と一致しなければ、幾何の起こし方が間違っている。
    """
    from krilly.perception.wall_detect import CALIBRATED_GEOMETRY

    for edge in (FRONT, BACK, LEFT, RIGHT):
        lo, hi = CALIBRATED_BANDS[edge]
        assert CALIBRATED_GEOMETRY.band_center(edge) == pytest.approx((lo + hi) / 2.0)


def test_geometry_matches_the_hand_tuned_rois():
    """幾何 + mm の形から作った ROI が、手で追い込んだ画素値と 1px 以内で一致すること。"""
    hand = {FRONT: (230, 105, 180, 50), BACK: (262, 417, 90, 38),
            LEFT: (121, 172, 46, 215), RIGHT: (426, 172, 46, 215)}
    got = calibrated_rois()
    for edge, (x, y, w, h) in hand.items():
        r = got[edge]
        assert abs(r.x - x) <= 1 and abs(r.y - y) <= 1, edge
        assert (r.w, r.h) == (w, h), edge


def test_geometry_keeps_the_roi_shape_when_the_field_of_view_changes():
    """画角を 1.5 倍・出力も 1.5 倍にすると、px/mm が同じなので**形は据え置き**になる。

    #88 の移行がこの性質に乗っている。位置だけが変わる。
    """
    from krilly.perception.wall_detect import CALIBRATED_GEOMETRY, CameraGeometry

    g = CALIBRATED_GEOMETRY
    # 光軸は常に画像中心にあり、セル中心の光軸からのずれ [mm] は取付で決まる固定量。
    # px/mm が同じなら、そのずれの**画素数も同じ**。だから新しいセル中心は
    # 「新しい画像中心 + 元のずれ」で予測できる (1.5 倍にはならない)。
    off = (g.cell_center[0] - g.width / 2, g.cell_center[1] - g.height / 2)
    wide = CameraGeometry(width=960, height=720,
                          cell_center=(960 / 2 + off[0], 720 / 2 + off[1]),
                          px_per_mm_x=g.px_per_mm_x, px_per_mm_y=g.px_per_mm_y)
    a, b = calibrated_rois(), calibrated_rois(wide)
    for edge in (FRONT, BACK, LEFT, RIGHT):
        assert (b[edge].w, b[edge].h) == (a[edge].w, a[edge].h), edge   # 形は同じ
    # 帯とセル中心の距離 (= 半ピッチ x px/mm) は変わらない
    for edge in (FRONT, BACK, LEFT, RIGHT):
        base = 0 if edge in (LEFT, RIGHT) else 1
        assert (wide.band_center(edge) - wide.cell_center[base]) == pytest.approx(
            g.band_center(edge) - g.cell_center[base]), edge


def test_geometry_scales_the_roi_when_only_the_field_of_view_changes():
    """出力を据え置きで画角だけ広げると px/mm が下がり、**ROI は小さくなる**。

    分解能が落ちるので壁帯も細くなる。移行では出力も一緒に上げてこれを避ける。
    """
    from krilly.perception.wall_detect import CALIBRATED_GEOMETRY, CameraGeometry

    g = CALIBRATED_GEOMETRY
    coarse = CameraGeometry(width=640, height=480, cell_center=g.cell_center,
                            px_per_mm_x=g.px_per_mm_x / 1.5,
                            px_per_mm_y=g.px_per_mm_y / 1.5)
    a, b = calibrated_rois(), calibrated_rois(coarse)
    assert b[LEFT].w < a[LEFT].w and b[FRONT].h < a[FRONT].h


def test_geometry_from_bands_round_trips():
    from krilly.perception.wall_detect import CameraGeometry

    bands = {FRONT: (200, 224), BACK: (500, 524), LEFT: (300, 324), RIGHT: (600, 624)}
    g = CameraGeometry.from_bands(bands, 960, 720)
    assert g.cell_center == pytest.approx((462.0, 362.0))
    assert g.px_per_mm_x == pytest.approx(300 / 180.0)
    for edge in bands:
        lo, hi = bands[edge]
        assert g.band_center(edge) == pytest.approx((lo + hi) / 2.0)


# --- 解像度と ROI の対応 (#88) -----------------------------------------------
def test_measure_rejects_a_frame_of_the_wrong_size():
    """撮影サイズと ROI の校正サイズが違ったら止まること。

    黙って動くと ROI が帯から外れ、**壁ありでも赤割合が出ずに機体が壁へ突っ込む**
    (#56 で実際に起きた失敗)。回り道より衝突の方が高くつくので、ここは止める。
    """
    det = WallDetector(calibrated_config())          # 640x480 で校正
    wrong = np.zeros((720, 960, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="校正サイズ"):
        det.measure(wrong)


def test_measure_accepts_the_matching_size():
    det = WallDetector(calibrated_config())
    det.measure(np.zeros((480, 640, 3), dtype=np.uint8))   # 例外が出なければよい


def test_config_without_frame_size_skips_the_check():
    """合成フレームのテスト用に、検算を切れること。"""
    det = WallDetector(WallDetectorConfig(rois=calibrated_rois(), red=CALIBRATED_RED))
    det.measure(np.zeros((720, 960, 3), dtype=np.uint8))


def test_geometry_for_refuses_an_uncalibrated_size():
    """校正していない解像度は**黙って既定へ落とさず**エラーにすること。"""
    from krilly.perception.wall_detect import geometry_for

    assert geometry_for((640, 480)) is not None
    with pytest.raises(KeyError, match="校正済み幾何が無い"):
        geometry_for((960, 720))
