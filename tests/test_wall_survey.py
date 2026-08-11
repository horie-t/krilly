"""既知形状の巡回 (wall_survey #56) と ASCII 迷路の読み込みのユニットテスト。"""

import pytest

from krilly.perception.wall_detect import (
    BACK,
    FRONT,
    LEFT,
    RIGHT,
    body_walls_to_maze,
    maze_walls_to_body,
)
from krilly.solver.maze import Direction, Maze
from scripts.wall_survey import plan_tour

# 実機の練習迷路 (3x3)。中央 (1,1) は東からしか入れない。
PRACTICE = """
+-+-+-+
|     |
+ +-+ +
| |   |
+ +-+ +
| |   |
+-+-+-+
"""


# --- ASCII の読み込み ------------------------------------------------------
def test_from_ascii_parses_the_practice_maze():
    m = Maze.from_ascii(PRACTICE)
    assert m.size == 3
    # 外周
    assert m.has_wall(0, 0, Direction.S) and m.has_wall(0, 0, Direction.W)
    assert m.has_wall(2, 2, Direction.N) and m.has_wall(2, 2, Direction.E)
    # 西列と中央列を隔てる縦壁 (y=0,1 のみ。y=2 は開いている)
    assert m.has_wall(0, 0, Direction.E)
    assert m.has_wall(0, 1, Direction.E)
    assert not m.has_wall(0, 2, Direction.E)
    # 中央セルは北と南が塞がれ、東だけ開いている
    assert m.has_wall(1, 1, Direction.N)
    assert m.has_wall(1, 1, Direction.S)
    assert m.has_wall(1, 1, Direction.W)
    assert not m.has_wall(1, 1, Direction.E)


def test_from_ascii_round_trips_to_ascii():
    m = Maze.from_ascii(PRACTICE)
    assert Maze.from_ascii(m.to_ascii()).to_ascii() == m.to_ascii()


def test_from_ascii_round_trips_a_16x16_maze():
    m = Maze(16)
    m.set_outer_walls()
    m.set_wall(3, 4, Direction.N)
    m.set_wall(7, 7, Direction.W)
    m.set_wall(15, 0, Direction.N)
    assert Maze.from_ascii(m.to_ascii()).to_ascii() == m.to_ascii()


def test_from_ascii_rejects_malformed_text():
    with pytest.raises(ValueError):
        Maze.from_ascii("+-+\n|  \n")           # 行数が偶数
    with pytest.raises(ValueError):
        Maze.from_ascii("+-+-+\n|   |\n+-+-+")  # '+' の数と行数が不整合


# --- 機体相対 <-> 迷路方角の往復 -------------------------------------------
@pytest.mark.parametrize("facing", list(Direction))
def test_body_maze_wall_mapping_round_trips(facing):
    body = {FRONT: True, BACK: False, LEFT: True, RIGHT: False}
    assert maze_walls_to_body(body_walls_to_maze(body, facing), facing) == body


def test_maze_walls_to_body_uses_the_facing():
    m = Maze.from_ascii(PRACTICE)
    walls = {d: m.has_wall(0, 1, d) for d in Direction}   # (0,1): W と E が壁
    north = maze_walls_to_body(walls, Direction.N)
    assert north == {FRONT: False, BACK: False, LEFT: True, RIGHT: True}
    east = maze_walls_to_body(walls, Direction.E)
    assert east == {FRONT: True, BACK: True, LEFT: False, RIGHT: False}


# --- 巡回順路 --------------------------------------------------------------
def test_plan_tour_visits_every_cell_without_crossing_walls():
    m = Maze.from_ascii(PRACTICE)
    tour = plan_tour(m)
    cell, facing = (0, 0), Direction.N
    visited = {cell}
    for step in tour:
        facing = step.direction
        assert not m.has_wall(*cell, facing), f"{cell} から {facing.name} は壁"
        assert m.neighbor(*cell, facing) == step.to_cell
        cell = step.to_cell
        visited.add(cell)
    assert visited == {(x, y) for x in range(3) for y in range(3)}


def test_plan_tour_turns_are_quarter_turns():
    tour = plan_tour(Maze.from_ascii(PRACTICE))
    assert all(step.turn in (-1, 0, 1, 2) for step in tour)


def test_plan_tour_on_open_maze_is_efficient():
    m = Maze(3)
    m.set_outer_walls()
    tour = plan_tour(m)
    # 9 セルを回るので最低 8 手。無駄な往復が入っていないこと
    assert 8 <= len(tour) <= 12


def test_plan_tour_skips_isolated_cells():
    m = Maze(3)
    m.set_outer_walls()
    for d in Direction:
        m.set_wall(2, 2, d)                  # (2,2) を隔離
    tour = plan_tour(m)
    cells = {(0, 0)} | {s.to_cell for s in tour}
    assert (2, 2) not in cells
    assert len(cells) == 8


# --- 手動 survey (survey_shot #65) ------------------------------------------
def test_parse_pose_accepts_both_notations():
    from scripts.survey_shot import parse_pose
    assert parse_pose("0 0 N", 5) == (0, 0, Direction.N)
    assert parse_pose("4,3,e", 5) == (4, 3, Direction.E)
    assert parse_pose("2  1  w", 5) == (2, 1, Direction.W)


def test_parse_pose_rejects_bad_input():
    from scripts.survey_shot import parse_pose
    assert parse_pose("5 0 N", 5) is None      # 範囲外
    assert parse_pose("0 0 X", 5) is None      # 不正な向き
    assert parse_pose("0 0", 5) is None        # 要素不足
    assert parse_pose("a b N", 5) is None


def test_next_shot_number_skips_existing(tmp_path):
    from scripts.survey_shot import next_shot_number
    assert next_shot_number(tmp_path, "shot") == 1
    (tmp_path / "shot07_c00_N.png").touch()
    (tmp_path / "shot12_c11_E.png").touch()
    assert next_shot_number(tmp_path, "shot") == 13
