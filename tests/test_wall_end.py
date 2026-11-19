"""壁の帯の切れ目から位置を読む (#189) のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from krilly.perception.wall_detect import (
    CALIBRATED_BANDS,
    CALIBRATED_GEOMETRY,
    DEFAULT_FRAME_SIZE,
    LEFT,
    RIGHT,
)
from krilly.perception.wall_end import BandEdge, find_band_edges, offset_from_edge

W, H = DEFAULT_FRAME_SIZE
G = CALIBRATED_GEOMETRY
RED = (0, 0, 255)
GREEN = (60, 200, 30)        # テープの色 (BGR)


def side_wall_frame(gaps_mm, band=LEFT, blur_px=0, tape_rows=()):
    """縦の帯 ``band`` を描き、車体の前後 (mm、セル中央の写る位置から +前) の区間を抜く。"""
    img = np.zeros((H, W, 3), np.uint8)
    lo, hi = CALIBRATED_BANDS[band]
    img[:, lo:hi] = RED
    for a, b in gaps_mm:
        # +前 = 画像の上 (y 小)
        y0 = int(round(G.cell_center[1] - b * G.px_per_mm_y))
        y1 = int(round(G.cell_center[1] - a * G.px_per_mm_y))
        img[max(0, y0):max(0, y1), lo:hi] = 0
    for y in tape_rows:
        img[y - 6:y + 6, lo - 5:hi + 5] = GREEN
    if blur_px:
        import cv2
        k = np.ones((blur_px, 1), np.float32) / blur_px
        img = cv2.filter2D(img, -1, k)
    return img


def test_edges_of_a_gap_are_found_with_subpixel_position():
    edges = find_band_edges(side_wall_frame([(40.0, 120.0)]), bands=(LEFT,))
    assert len(edges) == 2
    near, far = sorted(edges, key=lambda e: e.offset_mm)
    assert near.offset_mm == pytest.approx(40.0, abs=0.8) and not near.red_on_plus
    assert far.offset_mm == pytest.approx(120.0, abs=0.8) and far.red_on_plus
    assert near.axis == "x"


def test_motion_blur_keeps_the_half_maximum_position():
    edges = find_band_edges(side_wall_frame([(40.0, 120.0)], blur_px=15), bands=(LEFT,))
    assert sorted(round(e.offset_mm) for e in edges) == [40, 120]


def test_tape_crossing_is_not_an_edge():
    y_tape = int(G.cell_center[1] - 150 * G.px_per_mm_y)
    edges = find_band_edges(side_wall_frame([], tape_rows=(y_tape,)), bands=(LEFT,))
    assert edges == []


def test_continuous_wall_has_no_edges():
    assert find_band_edges(side_wall_frame([]), bands=(LEFT, RIGHT)) == []


def test_offset_from_edge_uses_the_nearest_boundary():
    # 機体はセル中央から +7 mm 前。前方の境目 (+90) の柱から先が切れ目 → 縁は +90+6
    edge = BandEdge(LEFT, "x", offset_mm=90 + 6.0 - 7.0, red_on_plus=False, pos_px=0.0)
    assert offset_from_edge(edge, estimated_offset_mm=0.0) == pytest.approx(7.0)
    # 推定が 30 mm ずれていても同じ境目を選ぶ
    assert offset_from_edge(edge, estimated_offset_mm=-30.0) == pytest.approx(7.0)
    # 後方の境目 (-90) の手前で壁が始まる (+ 側が赤) → 縁は -90-6
    edge2 = BandEdge(LEFT, "x", offset_mm=-90 - 6.0 - 3.0, red_on_plus=True, pos_px=0.0)
    assert offset_from_edge(edge2, estimated_offset_mm=0.0) == pytest.approx(3.0)


def test_shadowed_wall_is_not_a_gap():
    """2026-11-19 の衝突の原因: 壁の上面が影で暗くなっただけ (赤のまま) なら切れ目ではない。"""
    img = side_wall_frame([])
    lo, hi = CALIBRATED_BANDS[LEFT]
    y0 = int(G.cell_center[1] - 120 * G.px_per_mm_y)
    y1 = int(G.cell_center[1] + 40 * G.px_per_mm_y)
    img[y0:y1, lo:hi] = (0, 0, 95)            # 赤さは 255 -> 95 (半分以下)、色は赤のまま
    assert find_band_edges(img, bands=(LEFT,)) == []


WHITE = (235, 235, 235)


def gap_with_post(post_color, glare_mm=0.0):
    """+前の 30..198 mm (境目 +6 から次の境目 −6) が切れ目。境目 (+24 mm と仮定) の柱の色を変える。

    機体はセル中央から +66 mm (境目 = 90 → 機体から見て +24 mm)。
    """
    img = side_wall_frame([(30.0, 198.0)])
    lo, hi = CALIBRATED_BANDS[LEFT]
    def rows(a, b):
        return (int(round(G.cell_center[1] - b * G.px_per_mm_y)),
                int(round(G.cell_center[1] - a * G.px_per_mm_y)))
    if post_color == "white":
        y0, y1 = rows(18.0, 30.0)         # 柱 (境目 24 ± 6)
        img[y0:y1, lo:hi] = WHITE
    if glare_mm:
        y0, y1 = rows(30.0, 30.0 + glare_mm)
        img[y0:y1, lo:hi] = WHITE
    return img


def test_red_post_edge_is_beyond_the_post():
    edges = [e for e in find_band_edges(gap_with_post("red"), bands=(LEFT,))
             if not e.red_on_plus]
    assert len(edges) == 1 and edges[0].post == "red"
    assert edges[0].offset_mm == pytest.approx(30.0, abs=0.8)
    # 機体はセル中央から +66 mm: 境目 90 の柱の向こう側 (96) が縁
    assert offset_from_edge(edges[0], estimated_offset_mm=60.0) == pytest.approx(66.0, abs=0.8)


def test_white_post_edge_is_before_the_post():
    edges = [e for e in find_band_edges(gap_with_post("white"), bands=(LEFT,))
             if not e.red_on_plus]
    assert len(edges) == 1 and edges[0].post == "white"
    assert edges[0].offset_mm == pytest.approx(18.0, abs=0.8)   # 赤は柱の手前 (84) で終わる
    assert offset_from_edge(edges[0], estimated_offset_mm=60.0) == pytest.approx(66.0, abs=0.8)


def test_long_white_run_is_glare_not_a_post():
    edges = [e for e in find_band_edges(gap_with_post("red", glare_mm=40.0), bands=(LEFT,))
             if not e.red_on_plus]
    assert edges == []


def test_venue_post_colour_can_be_fixed():
    from krilly.perception.wall_end import EdgeConfig

    img = gap_with_post("white")
    assert [e for e in find_band_edges(img, bands=(LEFT,), config=EdgeConfig(post_top="red"))
            if not e.red_on_plus] == []
