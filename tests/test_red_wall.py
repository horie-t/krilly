"""赤い壁上端の検出のユニットテスト (合成画像を使用、カメラ不要)。"""

import cv2
import numpy as np

from krilly.perception.red_wall import (
    RedDetectorConfig,
    annotate,
    detect_red_regions,
    red_mask,
)


def _blank(h=120, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _hsv_bgr(h, s, v):
    """HSV 3 要素 (OpenCV の値域) から単一の BGR 色を生成する。"""
    px = np.array([[[h, s, v]]], dtype=np.uint8)
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


def test_red_mask_flags_red_not_black():
    img = _blank()
    cv2.rectangle(img, (40, 30), (80, 70), (0, 0, 255), -1)  # BGR の赤
    mask = red_mask(img)
    assert mask[50, 60] == 255      # 赤の内側
    assert mask[5, 5] == 0          # 黒い背景


def test_detect_single_red_region_centroid():
    img = _blank()
    cv2.rectangle(img, (40, 30), (80, 70), (0, 0, 255), -1)
    regions = detect_red_regions(img)
    assert len(regions) == 1
    r = regions[0]
    assert abs(r.cx - 60) <= 2      # 矩形の中心 x
    assert abs(r.cy - 50) <= 2      # 矩形の中心 y
    assert r.area > 1000


def test_detects_red_at_high_hue_end():
    # hue ~175 もまだ赤であり、2 つ目の範囲で検出される必要がある
    color = _hsv_bgr(175, 220, 220)
    img = _blank()
    cv2.rectangle(img, (30, 30), (90, 90), color, -1)
    assert len(detect_red_regions(img)) == 1


def test_blue_and_green_not_detected():
    img = _blank()
    cv2.rectangle(img, (10, 10), (50, 50), (255, 0, 0), -1)   # 青
    cv2.rectangle(img, (90, 60), (140, 100), (0, 255, 0), -1)  # 緑
    assert detect_red_regions(img) == []


def test_min_area_filters_small_specks():
    img = _blank()
    cv2.rectangle(img, (50, 50), (53, 53), (0, 0, 255), -1)   # 約 3x3 の赤
    assert detect_red_regions(img, RedDetectorConfig(min_area=100)) == []


def test_two_regions_sorted_largest_first():
    img = _blank()
    cv2.rectangle(img, (10, 10), (30, 30), (0, 0, 255), -1)    # 小
    cv2.rectangle(img, (80, 40), (140, 100), (0, 0, 255), -1)  # 大
    regions = detect_red_regions(img)
    assert len(regions) == 2
    assert regions[0].area > regions[1].area


def test_annotate_returns_same_shape_copy():
    img = _blank()
    cv2.rectangle(img, (40, 30), (80, 70), (0, 0, 255), -1)
    regions = detect_red_regions(img)
    out = annotate(img, regions)
    assert out.shape == img.shape
    assert out is not img
    # どこかに緑の矩形 (0,255,0) が描画されている
    assert (out[:, :, 1] == 255).sum() > 0


# --- red_breakdown: マスクがどの条件で画素を落としたか (#78) ----------------

def _patch(h, s, v, size=20):
    """指定 HSV 一色の小片。"""
    return np.full((size, size, 3), _hsv_bgr(h, s, v), dtype=np.uint8)


def test_breakdown_blames_saturation_for_a_washed_out_wall():
    """#56 の白飛びした右壁 (S=46) は「彩度で落ちた」と出ること。"""
    from krilly.perception.red_wall import red_breakdown
    from krilly.perception.wall_detect import CALIBRATED_RED

    got = red_breakdown(_patch(0, 46, 200), CALIBRATED_RED)
    assert got.accepted == 0 and got.lost_to_s == 400
    assert "彩度" in got.reason and got.hsv_lost[1] == 46


def test_breakdown_blames_hue_for_a_wall_that_drifted_magenta():
    """#65 の H=141-155 へ流れた壁上面。現行なら拾え、h2_lo=160 なら色相で落ちる。"""
    from krilly.perception.red_wall import red_breakdown
    from krilly.perception.wall_detect import CALIBRATED_RED

    patch = _patch(150, 150, 200)
    assert red_breakdown(patch, CALIBRATED_RED).accepted == 400
    strict = RedDetectorConfig(h2_lo=160, s_min=50, v_min=40)
    got = red_breakdown(patch, strict)
    assert got.accepted == 0 and got.hue_out == 400 and "色相" in got.reason


def test_black_pixels_are_colorless_not_a_hue_match():
    """**黒の H は 0** なので、素直に数えると赤の下側の帯に入ってしまう。

    それを「色相は合うが彩度で落ちた」と数えると、s_min を下げれば拾えるように
    見える — 実際には拾うものが無い。無彩色として別に数えること。
    """
    from krilly.perception.red_wall import red_breakdown
    from krilly.perception.wall_detect import CALIBRATED_RED

    got = red_breakdown(_blank(20, 20), CALIBRATED_RED)
    assert got.colorless == 400 and got.lost_to_s == 0 and got.hue_out == 0
    assert "遮蔽" in got.reason           # 落ちた赤は無い = 写っていない


def test_breakdown_fraction_matches_the_red_area():
    from krilly.perception.red_wall import red_breakdown
    from krilly.perception.wall_detect import CALIBRATED_RED

    img = _blank(20, 20)
    img[:10] = (0, 0, 255)
    got = red_breakdown(img, CALIBRATED_RED)
    assert got.fraction == 0.5 and got.total == 400


# --- 赤マスクの色相下限 h2_lo (#65 -> #78) ---------------------------------

def test_the_hue_band_reaches_the_palest_wall_tops():
    """**淡く写る壁ほど色相がマゼンタ側へ寄る** (#78、実測 90 本の帯)。

    彩度は 70-213 の連続分布で、色相はそれに連動する (S 70 の帯は H 131、
    S 198 の帯は H 169)。h2_lo=140 では一番淡い帯の画素の 81% が外れ、赤割合は
    0.092 = しきい値の 1.1 倍しか残っていなかった。
    """
    from krilly.perception.wall_detect import CALIBRATED_RED

    palest = _patch(131, 70, 190)                 # 実測で一番淡かった帯
    strongest = _patch(169, 198, 134)
    assert red_mask(palest, CALIBRATED_RED).all()
    assert red_mask(strongest, CALIBRATED_RED).all()
    # 以前の 140 では一番淡い帯が丸ごと落ちていた
    old = RedDetectorConfig(h2_lo=140, s_min=50, v_min=40)
    assert not red_mask(palest, old).any()
    assert red_mask(strongest, old).all()


def test_the_hue_band_still_excludes_the_blue_wiring():
    """**下限を下げすぎない根拠は機体の青配線** (#78)。

    FRONT / LEFT の ROI は機体が写る領域と重なるので、配線の色は他人事ではない。
    実測では配線の有彩色画素の 63% が H 100-115 に集中し、120-140 には 1% しか
    無い。h2_lo を 115 まで下げると壁なしの辺が 0.000 -> 0.010 と反応し始めるので、
    125 は飽和点であると同時に漏れ点まで 10 の余裕がある位置。
    """
    from krilly.perception.wall_detect import CALIBRATED_RED

    assert 120 < CALIBRATED_RED.h2_lo <= 130
    for hue in (100, 110, 115, 120):                  # 配線が集中する色相
        assert not red_mask(_patch(hue, 150, 200), CALIBRATED_RED).any()
