"""迷路モデル (Maze / Direction) のユニットテスト。"""

import pytest

from krilly.config import MazeConfig
from krilly.solver.maze import Direction, Maze


# --- Direction -------------------------------------------------------------
def test_direction_deltas():
    assert Direction.N.delta == (0, 1)
    assert Direction.E.delta == (1, 0)
    assert Direction.S.delta == (0, -1)
    assert Direction.W.delta == (-1, 0)


def test_direction_opposite():
    assert Direction.N.opposite == Direction.S
    assert Direction.E.opposite == Direction.W
    assert Direction.S.opposite == Direction.N
    assert Direction.W.opposite == Direction.E


# --- 基本 ------------------------------------------------------------------
def test_new_maze_has_no_interior_walls():
    m = Maze(16)
    assert m.size == 16
    assert not m.has_wall(5, 5, Direction.N)
    assert m.open_neighbors(5, 5) == [(5, 6), (6, 5), (5, 4), (4, 5)]


def test_in_bounds():
    m = Maze(16)
    assert m.in_bounds(0, 0) and m.in_bounds(15, 15)
    assert not m.in_bounds(-1, 0)
    assert not m.in_bounds(16, 0)


def test_neighbor():
    m = Maze(16)
    assert m.neighbor(3, 3, Direction.N) == (3, 4)
    assert m.neighbor(3, 3, Direction.W) == (2, 3)


# --- 共有エッジ ------------------------------------------------------------
def test_east_wall_shared_with_neighbor_west():
    m = Maze(16)
    m.set_wall(1, 1, Direction.E)
    assert m.has_wall(1, 1, Direction.E) is True
    assert m.has_wall(2, 1, Direction.W) is True    # 隣が同じ壁を見る


def test_north_wall_shared_with_neighbor_south():
    m = Maze(16)
    m.set_wall(3, 4, Direction.N)
    assert m.has_wall(3, 4, Direction.N) is True
    assert m.has_wall(3, 5, Direction.S) is True


def test_clear_wall():
    m = Maze(16)
    m.set_wall(2, 2, Direction.E, True)
    m.set_wall(2, 2, Direction.E, False)
    assert m.has_wall(2, 2, Direction.E) is False
    assert m.has_wall(3, 2, Direction.W) is False


def test_has_wall_out_of_bounds_raises():
    m = Maze(4)
    with pytest.raises(IndexError):
        m.has_wall(4, 0, Direction.N)


# --- 外周壁 ----------------------------------------------------------------
def test_outer_walls():
    m = Maze(16)
    m.set_outer_walls()
    assert m.has_wall(0, 0, Direction.W) is True
    assert m.has_wall(0, 0, Direction.S) is True
    assert m.has_wall(15, 15, Direction.N) is True
    assert m.has_wall(15, 15, Direction.E) is True
    # 内部は壁なし
    assert m.has_wall(5, 5, Direction.N) is False


def test_open_neighbors_corner_with_outer_walls():
    m = Maze(16)
    m.set_outer_walls()
    assert sorted(m.open_neighbors(0, 0)) == [(0, 1), (1, 0)]


# --- ゴール / スタート -----------------------------------------------------
def test_default_goal_center_2x2():
    m = Maze(16)
    assert m.start == (0, 0)
    assert sorted(m.goal_cells()) == [(7, 7), (7, 8), (8, 7), (8, 8)]
    assert m.is_goal(7, 8) is True
    assert m.is_goal(0, 0) is False


def test_from_config():
    cfg = MazeConfig(grid_size=16, cell_pitch_m=0.18, wall_thickness_m=0.012,
                     wall_height_m=0.05, goal_min=(7, 7), goal_max=(8, 8))
    m = Maze.from_config(cfg)
    assert m.size == 16
    assert m.is_goal(8, 8) is True


# --- ASCII -----------------------------------------------------------------
def test_to_ascii_shape():
    m = Maze(4)
    m.set_outer_walls()
    text = m.to_ascii()
    lines = text.splitlines()
    assert len(lines) == 2 * 4 + 1        # 各行2本 + 最下辺
    assert all(line[0] in "+|" for line in lines)
    assert "---" in lines[0]              # 北端に壁
