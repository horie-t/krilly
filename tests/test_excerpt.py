"""大会迷路からの切り出しと「実機でしか試せない度」の指標 (#85 / #23) のテスト。"""

import pytest

from krilly.sim import open_maze
from krilly.sim.excerpt import (
    correctable,
    excerpt,
    excerpts,
    longest_blind_cross_run,
    longest_blind_run,
    pieces_needed,
)
from krilly.solver.maze import Direction, Maze


def test_pieces_needed_counts_the_goal_centre_out():
    """壁は ``(N+1)^2``、柱は**そこから 1 本少ない**。

    格子点は ``(N+1)^2`` あるが、2x2 のゴールの中央には柱を立てない
    (NTF クラシック競技規定 9:「迷路の終点となる4区画内には壁や柱は存在しない。」)。
    """
    for n in (5, 6, 7, 8, 16):
        walls, posts = pieces_needed(n)
        assert walls == 4 * n + (n - 1) ** 2 == (n + 1) ** 2
        assert posts == walls - 1
        assert pieces_needed(n, goal_2x2=False)[1] == walls
    assert pieces_needed(8) == (81, 80)      # 手持ち 80 枚 / 85 本で組める


def test_excerpt_copies_the_window_and_closes_the_outside():
    source = open_maze(16)
    source.set_wall(5, 5, Direction.N)
    source.set_wall(6, 7, Direction.E)
    out = excerpt(source, 4, 4, 8)
    assert out.size == 8
    assert out.has_wall(1, 1, Direction.N)      # (5,5) -> (1,1)
    assert out.has_wall(2, 3, Direction.E)      # (6,7) -> (2,3)
    # 外周は閉じる (窓の中では壁が無かった辺も)
    for i in range(8):
        assert out.has_wall(i, 7, Direction.N) and out.has_wall(i, 0, Direction.S)
        assert out.has_wall(0, i, Direction.W) and out.has_wall(7, i, Direction.E)


def test_excerpt_keeps_the_start_and_the_two_by_two_goal():
    """偶数サイズならゴールは中央 2x2 (本番と同じ形)。奇数だと 1 セルになる。"""
    out = excerpt(open_maze(16), 0, 0, 8)
    assert out.start == (0, 0)
    assert set(out.goal_cells()) == {(3, 3), (3, 4), (4, 3), (4, 4)}
    assert len(excerpt(open_maze(16), 0, 0, 7).goal_cells()) == 1


def test_excerpts_covers_every_window():
    assert len(list(excerpts(open_maze(16), 8))) == 9 * 9
    assert len(list(excerpts(open_maze(16), 16))) == 1


def test_correctable_needs_a_wall_on_that_axis():
    """壁の無い辺には赤帯が無いので、その軸は測れない。"""
    maze = open_maze(5)                          # 外周だけ
    assert correctable(maze, (2, 2)) == (False, False)   # 真ん中は四辺とも壁なし
    assert correctable(maze, (0, 2)) == (True, False)    # 西が外周
    assert correctable(maze, (2, 0)) == (False, True)    # 南が外周
    assert correctable(maze, (0, 0)) == (True, True)


def test_longest_blind_run_counts_consecutive_cells():
    maze = open_maze(5)
    maze.set_wall(2, 2, Direction.E)
    path = [(0, 1), (1, 1), (2, 1), (3, 1), (2, 2)]
    # (0,1) は外周の西壁があるので測れる -> 連続は (1,1)(2,1)(3,1) の 3。
    # (2,2) には東壁を立てたのでそこで切れる。
    assert longest_blind_run(maze, path, 0) == 3
    assert longest_blind_run(maze, path, 1) == 5      # 南北はどこにも無い


def test_the_cross_run_ignores_the_along_axis():
    """**危ないのは直交だけ。** 進行方向の欠落は距離が狂うだけで壁には当たらない。

    東西に一直線の経路では、東西の壁が無くても「進行方向」なので数えない。
    数えるのは南北の壁 (= その経路にとっての直交) の有無。
    """
    maze = open_maze(5)
    east = [(0, 2), (1, 2), (2, 2), (3, 2)]          # 東へ 3 セル
    # 進行方向 (東西) は外周の西壁がある (0,2) 以外どこも測れない
    assert longest_blind_run(maze, east, 0) == 3
    assert longest_blind_cross_run(maze, east) == 3  # 直交 = 南北も無いので全部盲目
    # 南北の壁を足すと直交だけが解消する
    for x in (1, 2, 3):
        maze.set_wall(x, 2, Direction.N)
    assert longest_blind_run(maze, east, 0) == 3     # 東西は相変わらず無い
    assert longest_blind_cross_run(maze, east) == 0  # 直交は毎セル測れる


def test_the_cross_run_switches_axis_with_the_travel_direction():
    """南北へ進むセルでは東西の壁が、東西へ進むセルでは南北の壁が「直交」。"""
    maze = open_maze(5)
    for y in (1, 2, 3):
        maze.set_wall(2, y, Direction.W)             # 南北の廊下に西壁だけ立てる
    north = [(2, 0), (2, 1), (2, 2), (2, 3)]
    assert longest_blind_cross_run(maze, north) == 0     # 東西の壁があるので測れる
    east = [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert longest_blind_cross_run(maze, east) > 0       # 南北の壁は無い


def test_the_practice_mazes_have_no_blind_cross_run():
    """**5x5 では横方向の補正が毎セル入っていた** — だから 8x8 に意味がある (#23)。"""
    from pathlib import Path

    from krilly.strategy.shortest_path import shortest_path

    for name in ("practice5", "practice8"):
        maze = Maze.from_ascii(Path(f"mazes/{name}.txt").read_text(encoding="utf-8"))
        known = {(x, y) for x in range(maze.size) for y in range(maze.size)}
        path = shortest_path(maze, maze.start, start_facing=Direction.N, known=known)
        assert longest_blind_cross_run(maze, path) == 0, name


def test_the_chosen_excerpt_is_buildable_and_harder():
    """選んだ 8x8 が手持ち (壁 80 / 柱 85) で組め、5x5 に無い要素を含むこと。"""
    from pathlib import Path

    from krilly.sim.check import check_maze, wall_counts

    maze = Maze.from_ascii(Path("mazes/excerpt8.txt").read_text(encoding="utf-8"))
    assert check_maze(maze).ok
    assert wall_counts(maze).total <= 80
    assert (maze.size + 1) ** 2 <= 85
    # 実際に走る経路は**探索で分かった範囲**で引かれるので、全知の経路とは違いうる。
    # 指標はスクリプトと同じ計算 (measure) で見る。
    from scripts.maze_excerpt import measure

    metrics = measure(maze)
    assert metrics.blind_cross >= 2      # 5x5 と practice8 は 0
    assert metrics.no_wall_cells >= 3    # 赤帯が 1 本も写らないセル (5x5 は 0)
    assert metrics.path_cells >= 20
