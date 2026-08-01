"""flood-fill (距離計算・1手選択 #18) のユニットテスト。"""

import pytest

from krilly.solver.maze import Direction, Maze
from krilly.strategy.flood_fill import (
    UNREACHABLE,
    accessible_directions,
    flood_fill,
    next_direction,
)


def open_maze(size: int = 5) -> Maze:
    """外周のみ壁の迷路 (内部は未探索 = 壁なし)。"""
    m = Maze(size)
    m.set_outer_walls()
    return m


# --- 距離 -----------------------------------------------------------------
def test_flood_fill_on_open_maze_is_manhattan_distance():
    m = open_maze(5)                      # 5x5 -> ゴールは中央 (2,2) の 1 セル
    assert m.goal_cells() == [(2, 2)]
    dist = flood_fill(m)
    assert dist[2][2] == 0
    assert dist[0][0] == 4                # |0-2| + |0-2|
    assert dist[4][0] == 4
    assert dist[2][0] == 2
    for x in range(5):
        for y in range(5):
            assert dist[x][y] == abs(x - 2) + abs(y - 2)


def test_flood_fill_goal_area_is_all_zero():
    m = open_maze(4)                      # 4x4 -> ゴールは中央 2x2
    assert set(m.goal_cells()) == {(1, 1), (1, 2), (2, 1), (2, 2)}
    dist = flood_fill(m)
    for g in m.goal_cells():
        assert dist[g[0]][g[1]] == 0
    assert dist[0][0] == 2


def test_flood_fill_explicit_goals():
    m = open_maze(3)
    dist = flood_fill(m, goals=[(0, 0)])
    assert dist[0][0] == 0
    assert dist[2][2] == 4


def test_flood_fill_detours_around_walls():
    m = open_maze(3)
    # 下段 (y=0) の北側を 2 セル分塞ぐと、東へ回り込むしかなくなり距離が伸びる
    m.set_wall(0, 0, Direction.N)
    m.set_wall(1, 0, Direction.N)
    dist = flood_fill(m, goals=[(0, 0)])
    assert dist[1][0] == 1                 # 東隣はそのまま
    # (1,1) へは (0,0)->(1,0)->(2,0)->(2,1)->(1,1) の迂回で 4 (直線なら 2)
    assert dist[1][1] == 4
    assert dist[0][1] == 5                 # さらに 1 つ西


def test_flood_fill_marks_isolated_cells_unreachable():
    m = open_maze(3)
    for d in Direction:                    # (0,0) を四方から囲む
        m.set_wall(0, 0, d)
    dist = flood_fill(m)
    assert dist[0][0] == UNREACHABLE
    assert dist[1][1] == 0                 # 3x3 のゴールは中央 (1,1)


# --- 通行可能な方角 --------------------------------------------------------
def test_accessible_directions_excludes_walls_and_outside():
    m = open_maze(3)
    assert set(accessible_directions(m, 0, 0)) == {Direction.N, Direction.E}
    m.set_wall(0, 0, Direction.N)
    assert accessible_directions(m, 0, 0) == [Direction.E]


# --- 1手選択 ---------------------------------------------------------------
def test_next_direction_follows_the_gradient():
    m = open_maze(5)
    dist = flood_fill(m)
    # (0,0) から中央 (2,2) へ: 北向きなら直進 (N) を選ぶ (距離は同じで旋回コスト最小)
    assert next_direction(m, (0, 0), Direction.N, dist) is Direction.N
    # 東向きなら直進 (E)
    assert next_direction(m, (0, 0), Direction.E, dist) is Direction.E


def test_next_direction_prefers_straight_over_turning():
    m = open_maze(5)
    dist = flood_fill(m)
    # (2,0) は N も E/W も距離が違う: N=1 が最小なので必ず N
    assert next_direction(m, (2, 0), Direction.E, dist) is Direction.N
    # (1,1) からは N(1,2)=1 と E(2,1)=1 が同距離 -> 向いている方を選ぶ
    assert next_direction(m, (1, 1), Direction.N, dist) is Direction.N
    assert next_direction(m, (1, 1), Direction.E, dist) is Direction.E


def test_next_direction_prefers_unvisited_on_ties():
    m = open_maze(5)
    dist = flood_fill(m)
    # (1,1) から N(1,2) と E(2,1) は同距離。N を訪問済みにすると E を選ぶ
    assert next_direction(m, (1, 1), Direction.N, dist, visited={(1, 1), (1, 2)}) is Direction.E


def test_next_direction_avoids_walls():
    m = open_maze(5)
    m.set_wall(0, 0, Direction.N)
    dist = flood_fill(m)
    assert next_direction(m, (0, 0), Direction.N, dist) is Direction.E


def test_next_direction_none_when_isolated():
    m = open_maze(3)
    for d in Direction:
        m.set_wall(0, 0, d)
    dist = flood_fill(m)
    assert next_direction(m, (0, 0), Direction.N, dist) is None


def test_next_direction_none_when_goal_walled_off():
    """ゴールが完全に塞がれていれば、どこからも到達不能で None。"""
    m = open_maze(3)
    for d in Direction:
        m.set_wall(1, 1, d)               # 中央 (=ゴール) を囲む
    dist = flood_fill(m)
    assert dist[0][0] == UNREACHABLE
    assert next_direction(m, (0, 0), Direction.N, dist) is None


def test_16x16_default_maze_distance_from_start():
    m = Maze(16)
    m.set_outer_walls()
    dist = flood_fill(m)
    # ゴールは中央 2x2 (7,7)-(8,8)。壁なしならスタート (0,0) からの距離は 7+7
    assert dist[0][0] == 14
    assert dist[15][15] == 14
