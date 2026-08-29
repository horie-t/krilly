"""校正データ (labels.csv) の記録と解析のテスト (issue #78)。

実機のフレームは要らない。ここで固定したいのは「測定値と地図から正しいラベルが
貼れるか」と「余裕の数え方」であって、カメラの性能ではない。
"""

import pytest

from krilly.perception.survey import (
    CSV_FIELDS,
    EdgeStats,
    FrameRecord,
    SurveyRow,
    compare,
    confusion,
    format_confusion,
    format_report,
    label_rows,
    label_run,
    read_rows,
    slot_wall,
    summarize,
    write_rows,
)
from krilly.solver.maze import Direction, Maze


def measured(**fractions):
    """``measure()`` の戻り値の形 (赤割合, 帯のずれ, 飽和) を組む。"""
    return {slot: (value, 0, False) for slot, value in fractions.items()}


def maze_with(size=5, walls=()):
    maze = Maze(size)
    maze.set_outer_walls()
    for x, y, d in walls:
        maze.set_wall(x, y, d, True)
    return maze


# --- ラベル付け ------------------------------------------------------------

def test_self_edges_are_labelled_through_the_facing():
    """自セルの 4 辺は機体の向きで迷路方角へ写して正解を引く。"""
    maze = maze_with(walls=[(1, 1, Direction.N)])
    rows = label_rows("f.png", (1, 1), Direction.N,
                      measured(front=0.44, back=0.02, left=0.01, right=0.03), maze)
    truth = {r.edge: r.wall for r in rows}
    assert truth == {"front": True, "back": False, "left": False, "right": False}
    # 東を向けば同じ壁は左に見える
    rows = label_rows("f.png", (1, 1), Direction.E,
                      measured(front=0.44, back=0.02, left=0.01, right=0.03), maze)
    assert {r.edge: r.wall for r in rows}["left"] is True


def test_the_outer_wall_behind_the_start_is_a_real_label():
    """スタートで後ろに見えるのは外周壁。正解は「壁あり」で、行として残す。"""
    rows = label_rows("f.png", (0, 0), Direction.N,
                      measured(front=0.0, back=0.53, left=0.48, right=0.0),
                      maze_with())
    assert {r.edge: r.wall for r in rows}["back"] is True


def test_neighbor_slots_are_labelled_on_the_neighbor_cell():
    """``left:front`` は**左隣のセル**の北の壁 (北向きのとき)。"""
    maze = maze_with(walls=[(0, 1, Direction.N)])
    rows = label_rows("f.png", (1, 1), Direction.N,
                      measured(front=0.0, back=0.0, left=0.0, right=0.0,
                               **{"left:front": 0.47}), maze)
    assert {r.edge: r.wall for r in rows}["left:front"] is True


def test_slots_looking_outside_the_maze_have_no_label_and_are_dropped():
    """迷路の外を見たスロットには正解が無い。**捨てる** (#89 の赤い定規袋の一件)。

    落とさずに「壁なし」と書くと、迷路の外に置かれた赤い物が「誤検出」として
    集計に混ざる — 見るべきは迷路の中の分離なのに。
    """
    maze = maze_with(size=5)
    rows = label_rows("f.png", (4, 0), Direction.N,
                      measured(front=0.0, back=0.0, left=0.0, right=0.0,
                               **{"right:right": 0.97, "right:front": 0.10}), maze)
    assert [r.edge for r in rows] == ["front", "back", "left", "right"]
    assert slot_wall("right:right", (4, 0), Direction.N, maze) is None
    assert slot_wall("left:front", (4, 0), Direction.N, maze) == ((3, 0), Direction.N)


def test_label_run_counts_what_it_dropped():
    maze = maze_with()
    records = [FrameRecord("a.png", (4, 0), Direction.N,
                           measured(front=0.0, back=0.0, left=0.0, right=0.0,
                                    **{"right:right": 0.97}))]
    rows, skipped = label_run(records, maze)
    assert len(rows) == 4 and skipped == 1


def test_labels_follow_the_map_they_are_given():
    """**同じフレームでも、渡す地図が違えばラベルは違う。**

    走行後の学習地図で貼るか、既知形状で貼るかを選べる意味がここにある。
    学習地図で貼ると「毎回見落としている壁」は壁なしと記録され、誤判定 0 に見える。
    """
    learned = maze_with()                                   # 壁を見落とした地図
    truth = maze_with(walls=[(1, 1, Direction.N)])
    frame = measured(front=0.06, back=0.0, left=0.0, right=0.0)
    assert label_rows("f.png", (1, 1), Direction.N, frame, learned)[0].wall is False
    assert label_rows("f.png", (1, 1), Direction.N, frame, truth)[0].wall is True


# --- CSV -------------------------------------------------------------------

def test_csv_round_trip_keeps_every_field(tmp_path):
    rows = [SurveyRow("a.png", 1, 2, "N", "front", True, 0.4321, 3.5, None),
            SurveyRow("a.png", 1, 2, "N", "left:back", False, 0.0, 3.5, -2.0)]
    path = tmp_path / "labels.csv"
    assert write_rows(path, rows) == 2
    back = read_rows(path)
    assert back == rows
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(CSV_FIELDS)


def test_append_does_not_repeat_the_header(tmp_path):
    path = tmp_path / "labels.csv"
    row = SurveyRow("a.png", 0, 0, "N", "front", True, 0.4)
    write_rows(path, [row], append=True)
    write_rows(path, [row], append=True)
    assert path.read_text(encoding="utf-8").count("file,x,y") == 1
    assert len(read_rows(path)) == 2


# --- 分布と余裕 ------------------------------------------------------------

def rows_for(edge, walls, clears):
    return ([SurveyRow(f"w{i}.png", 0, 0, "N", edge, True, v)
             for i, v in enumerate(walls)]
            + [SurveyRow(f"c{i}.png", 0, 0, "N", edge, False, v)
               for i, v in enumerate(clears)])


def test_separation_and_ratio_are_read_off_the_extremes():
    """分離も余裕倍率も**端の 2 点**で決まる。分布の真ん中は run を終わらせない。"""
    rows = rows_for("front", walls=[0.44, 0.11, 0.55], clears=[0.0, 0.02])
    stats = summarize(rows, lambda edge: 0.08)["front"]
    assert stats.weakest_wall.fraction == pytest.approx(0.11)
    assert stats.strongest_clear.fraction == pytest.approx(0.02)
    assert stats.separation == pytest.approx(0.09)
    assert stats.ratio == pytest.approx(0.11 / 0.08)          # #23 の 1.4 倍
    assert stats.headroom == pytest.approx(0.06)
    assert stats.quantiles(True) == pytest.approx((0.11, 0.44, 0.55))


def test_zero_errors_can_still_be_a_thin_margin():
    """**誤判定 0 は「余裕がある」を意味しない。** #78 が測りたいのはここ。"""
    stats = summarize(rows_for("back", walls=[0.11], clears=[0.0]),
                      lambda edge: 0.08)["back"]
    assert not stats.misses and not stats.false_walls
    assert stats.ratio < 1.5


def test_misses_and_false_walls_are_counted_separately():
    """見落とし (衝突) と誤検出 (回り道) は等価ではないので分けて数える。"""
    rows = rows_for("back", walls=[0.24, 0.51], clears=[0.30, 0.01])
    stats = summarize(rows, lambda edge: 0.25)["back"]
    assert [r.fraction for r in stats.misses] == [0.24]
    assert [r.fraction for r in stats.false_walls] == [0.30]
    assert confusion({"back": stats}) == (1, 1, 1, 1)


def test_the_threshold_used_is_the_one_the_machine_uses():
    """辺別しきい値をそのまま渡せること (実機と違う基準で読んでも意味がない)。"""
    from krilly.perception.wall_detect import calibrated_config

    rows = rows_for("front", [0.10], [0.0]) + rows_for("back", [0.10], [0.0])
    stats = summarize(rows, calibrated_config().threshold_for)
    assert stats["front"].threshold == pytest.approx(0.08)
    assert stats["back"].threshold == pytest.approx(0.08)     # #88 で 0.25 から下げた


def test_edges_come_out_in_the_order_the_machine_names_them():
    rows = (rows_for("right", [0.4], []) + rows_for("front", [0.4], [])
            + rows_for("left:back", [0.4], []) + rows_for("back", [0.4], []))
    assert list(summarize(rows, lambda e: 0.08)) == [
        "front", "back", "right", "left:back"]


def test_an_edge_with_only_one_class_has_no_separation():
    """壁ばかり (または開ばかり) の辺では分離は決まらない。0 ではなく None。"""
    stats = summarize(rows_for("front", [0.4, 0.5], []), lambda e: 0.08)["front"]
    assert stats.separation is None and stats.headroom is None
    assert stats.quantiles(False) is None


# --- 報告 ------------------------------------------------------------------

def test_the_report_names_the_frame_of_every_miss():
    """見落としは**どのフレームか**まで出す。次に見るのは画像だから。"""
    rows = rows_for("back", walls=[0.24], clears=[0.0])
    lines = "\n".join(format_report(summarize(rows, lambda e: 0.25)))
    assert "w0.png" in lines and "見落とし" in lines
    assert "ラベル 2 個" in lines


def test_the_confusion_matrix_labels_which_error_is_which():
    lines = "\n".join(format_confusion(
        summarize(rows_for("front", [0.4], [0.0]), lambda e: 0.08)))
    assert "衝突" in lines and "回り道" in lines


def test_compare_shows_how_much_the_weakest_wall_moved():
    """照明を変えた 2 回の比較で読むのは**弱壁の変化**。"""
    before = summarize(rows_for("front", [0.44], [0.0]), lambda e: 0.08)
    after = summarize(rows_for("front", [0.11], [0.0]), lambda e: 0.08)
    lines = "\n".join(compare(before, after))
    assert "-0.330" in lines


def test_compare_survives_an_edge_missing_from_one_run():
    """走行が違えば見た辺も違う。片側にしか無い辺で落ちないこと。"""
    before = summarize(rows_for("front", [0.4], [0.0]), lambda e: 0.08)
    after = summarize(rows_for("left:front", [0.4], [0.0]), lambda e: 0.08)
    assert len(compare(before, after)) == 3                   # 見出し + 2 辺


def test_edge_stats_is_constructible_without_rows():
    """空でも壊れないこと (辺を 1 度も見なかった走行がありうる)。"""
    stats = EdgeStats("front", 0.08, [], [])
    assert stats.ratio is None and stats.separation is None
    assert format_report({"front": stats})


# --- スクリプト: 保存済みフレームの一括再判定 (wall_detect --batch) ---------

def _frame_with_walls(edges):
    """校正済みの帯の位置に赤帯を描いた合成フレーム (壁ありの辺だけ)。"""
    import numpy as np

    from krilly.perception.wall_detect import (
        CALIBRATED_BANDS,
        DEFAULT_FRAME_SIZE,
        LEFT,
        RIGHT,
    )

    w, h = DEFAULT_FRAME_SIZE
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for edge in edges:
        lo, hi = CALIBRATED_BANDS[edge]
        if edge in (LEFT, RIGHT):
            img[:, lo:hi] = (0, 0, 255)
        else:
            img[lo:hi, :] = (0, 0, 255)
    return img


def test_batch_reevaluation_re_measures_saved_frames(tmp_path):
    """保存済みフレームを別設定で測り直せること (#78 の受け入れ基準 2)。

    正解ラベルは CSV のものをそのまま使い、**赤割合だけ**が新しくなる。
    """
    import cv2

    from krilly.perception.wall_detect import CALIBRATED_RED
    from scripts.wall_detect import batch_reevaluate

    cv2.imwrite(str(tmp_path / "run_001.png"), _frame_with_walls(["front", "left"]))
    write_rows(tmp_path / "run_labels.csv", [
        SurveyRow("run_001.png", 0, 0, "N", "front", True, 0.0),   # 記録は 0 (嘘)
        SurveyRow("run_001.png", 0, 0, "N", "back", False, 0.0),
        SurveyRow("run_001.png", 0, 0, "N", "left", True, 0.0),
        SurveyRow("run_001.png", 0, 0, "N", "right", False, 0.0),
    ])
    out = tmp_path / "redo_labels.csv"
    batch_reevaluate(str(tmp_path), None, CALIBRATED_RED, None, str(out))

    redone = {r.edge: r for r in read_rows(out)}
    assert redone["front"].fraction > 0.3 and redone["left"].fraction > 0.3
    assert redone["back"].fraction == 0.0 and redone["right"].fraction == 0.0
    assert redone["front"].wall is True                    # 正解は書き換えない
    # 再判定した値なら front/left は壁と判定される (記録の 0.0 では見落としだった)
    stats = summarize(read_rows(out), lambda e: 0.08)
    assert not stats["front"].misses and not stats["left"].misses


def test_batch_reevaluation_survives_a_missing_frame(tmp_path, caplog):
    """CSV にあってフレームが無い行は飛ばす (途中で消した画像があっても止まらない)。"""
    import cv2

    from krilly.perception.wall_detect import CALIBRATED_RED
    from scripts.wall_detect import batch_reevaluate

    cv2.imwrite(str(tmp_path / "run_001.png"), _frame_with_walls(["front"]))
    write_rows(tmp_path / "run_labels.csv", [
        SurveyRow("run_001.png", 0, 0, "N", "front", True, 0.4),
        SurveyRow("gone.png", 0, 0, "N", "front", True, 0.4),
    ])
    out = tmp_path / "redo_labels.csv"
    batch_reevaluate(str(tmp_path), None, CALIBRATED_RED, None, str(out))
    assert [r.file for r in read_rows(out)] == ["run_001.png"]
