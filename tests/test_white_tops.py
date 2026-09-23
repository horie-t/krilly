"""始点・終点の白/黄の壁上面を読む (#125) — 壁判定・位置補正への組み込み。

実機フレームは ``tests/data/white_tops`` (黒い床・EV -2・緑のフェルトで覆った機体、
同じセル)。``wy_a`` は front/right が白・back/left が黄、``wy_b`` はその逆、
``on_0w_2s`` は柱だけで機体を光沢の強い右へ 15mm ずらしたもの (壁なしで一番
危ない形)、``on_4w_2`` は赤い壁 4 枚、``wy_a_s`` は白い壁を front/right の 2 枚だけ
立てて機体を右へ ~13mm ずらしたもの (白い帯と床の光沢が同じ ROI に並ぶ)。
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from krilly.perception.cell_pose import OFFSET_MIN_FRACTION, cell_offset
from krilly.perception.wall_detect import (
    BODY_DIRS,
    NEIGHBOR_CLEAR_MAX_FRACTION,
    PATH_BLOCK_MIN_FRACTION,
    RED,
    WHITE_YELLOW,
    WallDetector,
    calibrated_config,
)

DATA = Path("tests/data/white_tops")


def _frame(name: str) -> np.ndarray:
    frame = cv2.imread(str(DATA / f"{name}.png"))
    assert frame is not None, name
    return frame


def _synthetic_white_walls() -> np.ndarray:
    """校正済み ROI の中央に、幅 20px の白い帯を 4 辺とも描いた黒いフレーム。"""
    cfg = calibrated_config()
    w, h = cfg.frame_size
    img = np.full((h, w, 3), 35, dtype=np.uint8)
    for edge in BODY_DIRS:
        r = cfg.rois[edge]
        if cfg.target(edge).vertical:
            cx = r.x + r.w // 2
            img[r.y:r.y + r.h, cx - 10:cx + 10] = 190
        else:
            cy = r.y + r.h // 2
            img[cy - 10:cy + 10, r.x:r.x + r.w] = 190
    return img


def test_white_tops_are_off_by_default():
    """既定は従来どおり赤だけ。**木の床では白/黄を使えない**ので、明示しない限り入らない。"""
    assert calibrated_config().white_yellow is None
    walls = WallDetector(calibrated_config()).detect(_frame("wy_a"))
    assert not any(walls.values())


def test_white_walls_are_walls_when_enabled():
    det = WallDetector(calibrated_config(white_tops=True))
    measured = det.measure(_synthetic_white_walls())
    for edge in BODY_DIRS:
        assert measured[edge][0] >= det.cfg.threshold_for(edge)
        assert measured[edge].source == WHITE_YELLOW


def test_white_bands_do_not_move_the_position_estimate():
    """白/黄の帯のずれは検証していないので、位置補正は**測れない**扱いにする。"""
    det = WallDetector(calibrated_config(white_tops=True))
    off = cell_offset(_synthetic_white_walls(), det)
    assert not off.measured
    assert (off.walls_x, off.walls_y) == (0, 0)
    assert det.lateral_shift_px(det.measure(_synthetic_white_walls())) is None


def test_readings_still_unpack_as_plain_tuples():
    """既存の呼び出し側は ``fraction, offset, saturated = measured[edge]`` のまま動く。"""
    reading = WallDetector(calibrated_config(white_tops=True)).measure(
        _synthetic_white_walls())["front"]
    fraction, offset, saturated = reading
    assert fraction > 0.3 and isinstance(offset, int) and saturated is False
    assert reading == (fraction, offset, saturated)


@pytest.mark.parametrize("name", ["wy_a", "wy_b"])
def test_real_white_and_yellow_walls_are_read_on_every_edge(name):
    """実測の最小は 0.46。壁判定・進路確認・(赤なら) 位置補正のどの下限も超える。"""
    det = WallDetector(calibrated_config(white_tops=True))
    measured = det.measure(_frame(name))
    for edge in BODY_DIRS:
        fraction = measured[edge][0]
        assert fraction >= max(OFFSET_MIN_FRACTION, PATH_BLOCK_MIN_FRACTION), (edge, fraction)
        assert measured[edge].source == WHITE_YELLOW


def test_glossy_floor_next_to_the_machine_is_not_a_wall():
    """柱だけ・光沢の強い側へずらした位置。**明るさだけで判定したら 0.245 で壁だった。**

    隣セルの「壁なし」に使える上限 (0.04) まで下回ることを求める。
    """
    det = WallDetector(calibrated_config(white_tops=True))
    measured = det.measure(_frame("on_0w_2s"))
    for edge in BODY_DIRS:
        assert measured[edge][0] <= NEIGHBOR_CLEAR_MAX_FRACTION, (edge, measured[edge])


def test_red_walls_read_exactly_as_before():
    """赤い壁は赤で読み、割合もずれも白/黄を入れる前と同じ (位置補正が変わらない)。"""
    frame = _frame("on_4w_2")
    red_only = WallDetector(calibrated_config()).measure(frame)
    both = WallDetector(calibrated_config(white_tops=True)).measure(frame)
    for edge in BODY_DIRS:
        assert both[edge].source == RED
        assert tuple(both[edge]) == tuple(red_only[edge])


def test_white_walls_next_to_the_glossy_floor_with_the_machine_off_centre():
    """白い壁 2 枚 + 壁なし 2 辺を、光沢の強い右へずらした位置で。"""
    det = WallDetector(calibrated_config(white_tops=True, neighbors=True))
    measured = det.measure(_frame("wy_a_s"))
    assert det.detect(_frame("wy_a_s")) == {"front": True, "back": False,
                                             "left": False, "right": True}
    for edge in ("front", "right"):
        assert measured[edge][0] >= PATH_BLOCK_MIN_FRACTION
    for edge in ("back", "left"):
        assert measured[edge][0] <= NEIGHBOR_CLEAR_MAX_FRACTION


def test_far_neighbour_edges_stay_undecided_when_only_white_bands_are_seen():
    """機体の左右のずれは赤い帯からしか測らないので、白い帯しか無いセルでは
    遠い側の辺を「壁なし」と言い切れない (#89 の飽和対策がそのまま効く)。"""
    det = WallDetector(calibrated_config(white_tops=True, neighbors=True))
    measured = det.measure(_frame("wy_a_s"))
    assert det.lateral_shift_px(measured) is None
    assert "right" not in det.neighbor_walls(measured).get("right", {})


# --- 3x3 の実走 (mazes/white_goal3.txt) で見つかった 3 つ -------------------------
# goal3_run_003 = (1,2): 東がゴールの西の白い壁、西は開いていて床板の継ぎ目が写る。
# goal3_run_002 = (0,2): 右の隣セル (1,2) の遠い側 = 同じ白い壁。
# goal3_run_000 = (0,0): 南の外周の赤い壁の外が木の床。


def test_a_bluish_white_wall_is_still_white():
    """AWB のロックで白が S 64-73 と写った。s_max 60 では 0.05 と読んで突っ込んだ。"""
    det = WallDetector(calibrated_config(white_tops=True))
    reading = det.measure(_frame("goal3_run_003"))["right"]
    assert reading[0] >= PATH_BLOCK_MIN_FRACTION and reading.source == WHITE_YELLOW


def test_a_floor_seam_is_not_a_wall():
    """幅 3px の継ぎ目はトップハットにも白にも入る。細い線は opening で落とす。"""
    det = WallDetector(calibrated_config(white_tops=True))
    assert det.measure(_frame("goal3_run_003"))["left"][0] <= NEIGHBOR_CLEAR_MAX_FRACTION


def test_the_white_goal_wall_is_seen_one_cell_early_as_a_far_neighbour_band():
    det = WallDetector(calibrated_config(white_tops=True, neighbors=True))
    measured = det.measure(_frame("goal3_run_002"))
    assert measured["right:right"][0] >= det.cfg.threshold_for("right:right")
    assert det.neighbor_walls(measured)["right"]["right"] is True


def test_a_confident_red_wall_stays_red_even_with_yellow_wood_beyond_it():
    """探索範囲が外周の外の木の床 (黄 0.999) にはみ出しても、赤で壁と言えるなら赤。"""
    frame = _frame("goal3_run_000")
    both = WallDetector(calibrated_config(white_tops=True)).measure(frame)
    red_only = WallDetector(calibrated_config()).measure(frame)
    assert both["back"].source == RED
    assert tuple(both["back"]) == tuple(red_only["back"])
