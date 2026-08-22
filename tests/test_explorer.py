"""探索ランの進行管理 (Explorer #18) のユニットテスト。

Explorer は離散の世界だけを扱うので、「真の迷路」を用意して壁観測を模擬すれば、
探索ラン全体をハードウェア無しでシミュレートできる。
"""

import math

import pytest

from krilly.config import MazeConfig
from krilly.perception.wall_detect import BACK, FRONT, LEFT, RIGHT
from krilly.sim import open_maze, sense, sense_neighbors   # カメラの代役 (#77)
from krilly.solver.maze import Direction, Maze
from krilly.strategy.explorer import (
    Explorer,
    Step,
    Unreachable,
    cell_center,
    heading_rad,
    quarter_turns,
)


def run_search(truth: Maze, max_steps: int = 500) -> Explorer:
    """真の迷路 ``truth`` を相手に探索ランを回し、ゴール到達した Explorer を返す。"""
    ex = Explorer(open_maze(truth.size))
    for _ in range(max_steps):
        ex.observe(sense(truth, ex.cell, ex.facing))
        step = ex.plan()
        if step is None:
            return ex
        ex.advance(step)
    pytest.fail(f"{max_steps} 手でゴールに到達しなかった (現在 {ex.cell})")


# --- 座標・方角の対応 ------------------------------------------------------
def test_heading_rad_matches_maze_directions():
    assert heading_rad(Direction.E) == pytest.approx(0.0)
    assert heading_rad(Direction.N) == pytest.approx(math.pi / 2)
    assert heading_rad(Direction.S) == pytest.approx(-math.pi / 2)
    assert heading_rad(Direction.W) == pytest.approx(math.pi)


def test_cell_center_uses_the_maze_pitch():
    pitch = MazeConfig(16, 0.180, 0.012, 0.050, (7, 7), (8, 8)).cell_pitch_m
    assert cell_center((0, 0), pitch) == pytest.approx((0.0, 0.0))
    assert cell_center((2, 3), pitch) == pytest.approx((0.360, 0.540))


def test_quarter_turns_takes_the_shortest_rotation():
    assert quarter_turns(Direction.N, Direction.N) == 0
    assert quarter_turns(Direction.N, Direction.W) == 1    # 左 90°
    assert quarter_turns(Direction.N, Direction.E) == -1   # 右 90°
    assert quarter_turns(Direction.N, Direction.S) == 2    # 180°
    assert quarter_turns(Direction.W, Direction.N) == -1
    assert quarter_turns(Direction.E, Direction.N) == 1


# --- 観測 -----------------------------------------------------------------
def test_observe_maps_body_walls_to_maze_directions():
    ex = Explorer(open_maze(5), cell=(2, 2), facing=Direction.N)
    walls = ex.observe({FRONT: True, BACK: False, LEFT: True, RIGHT: False})
    assert walls == {
        Direction.N: True,     # 前 = 北
        Direction.S: False,    # 後 = 南
        Direction.W: True,     # 左 = 西
        Direction.E: False,    # 右 = 東
    }
    assert ex.maze.has_wall(2, 2, Direction.N)
    assert ex.maze.has_wall(2, 2, Direction.W)
    assert not ex.maze.has_wall(2, 2, Direction.E)
    # 共有エッジなので隣接セルからも同じ壁が見える
    assert ex.maze.has_wall(2, 3, Direction.S)
    assert ex.maze.has_wall(1, 2, Direction.E)


def test_observe_never_clears_outer_walls():
    """外周は既知。カメラが「壁なし」と誤判定しても消してはいけない (実機で発生)。"""
    ex = Explorer(open_maze(3))                     # (0,0) 北向き: 南と西が外周
    ex.observe({FRONT: False, BACK: False, LEFT: False, RIGHT: False})
    assert ex.maze.has_wall(0, 0, Direction.S)      # 外周は残る
    assert ex.maze.has_wall(0, 0, Direction.W)
    assert not ex.maze.has_wall(0, 0, Direction.N)  # 内側の辺は観測どおり
    assert ex.conflicts == 0                        # 外周は矛盾として数えない


def test_observe_keeps_known_walls_and_counts_conflicts():
    ex = Explorer(open_maze(5), cell=(2, 2), facing=Direction.N)
    ex.observe({FRONT: True, BACK: False, LEFT: False, RIGHT: False})
    assert ex.maze.has_wall(2, 2, Direction.N)
    # 2 回目に前方を「壁なし」と判定しても消さず、矛盾として数える
    ex.observe({FRONT: False, BACK: False, LEFT: False, RIGHT: False})
    assert ex.maze.has_wall(2, 2, Direction.N)
    assert ex.conflicts == 1


def test_observe_can_clear_walls_when_sticky_is_off():
    ex = Explorer(open_maze(5), cell=(2, 2), facing=Direction.N, sticky_walls=False)
    ex.observe({FRONT: True, BACK: False, LEFT: False, RIGHT: False})
    ex.observe({FRONT: False, BACK: False, LEFT: False, RIGHT: False})
    assert not ex.maze.has_wall(2, 2, Direction.N)
    assert ex.conflicts == 1                        # 数えるのは sticky でなくても同じ


def test_observe_is_relative_to_facing():
    ex = Explorer(open_maze(5), cell=(2, 2), facing=Direction.E)
    ex.observe({FRONT: True, BACK: False, LEFT: False, RIGHT: True})
    assert ex.maze.has_wall(2, 2, Direction.E)     # 前 = 東
    assert ex.maze.has_wall(2, 2, Direction.S)     # 右 = 南
    assert not ex.maze.has_wall(2, 2, Direction.N)


# --- 計画 -----------------------------------------------------------------
def test_plan_moves_toward_the_goal():
    ex = Explorer(open_maze(5))                     # (0,0) 北向き、ゴール (2,2)
    step = ex.plan()
    assert step == Step(direction=Direction.N, to_cell=(0, 1))


def test_plan_turns_when_the_front_is_walled():
    ex = Explorer(open_maze(5))
    ex.observe({FRONT: True, BACK: True, LEFT: True, RIGHT: False})   # 東しか開いていない
    step = ex.plan()
    assert step == Step(direction=Direction.E, to_cell=(1, 0))


def test_plan_returns_none_at_the_goal():
    ex = Explorer(open_maze(5), cell=(2, 2))
    assert ex.at_goal
    assert ex.plan() is None


def test_plan_raises_when_the_goal_is_walled_off():
    ex = Explorer(open_maze(3))
    for d in Direction:
        ex.maze.set_wall(1, 1, d)                   # ゴール (1,1) を囲む
    with pytest.raises(Unreachable):
        ex.plan()


def test_advance_updates_cell_and_visited():
    ex = Explorer(open_maze(5))
    ex.advance(Step(direction=Direction.E, to_cell=(1, 0)))
    assert ex.cell == (1, 0)
    assert ex.travel is Direction.E
    assert ex.visited == {(0, 0), (1, 0)}
    assert ex.steps == 1


def test_holonomic_advance_keeps_the_facing():
    """旋回レス走行では機体の向きは変わらない (#76)。"""
    ex = Explorer(open_maze(5))                       # holonomic=True が既定
    ex.advance(Step(direction=Direction.E, to_cell=(1, 0)))
    assert ex.facing is Direction.N                   # 北を向いたまま
    assert ex.travel is Direction.E


def test_turning_mode_updates_the_facing():
    """旧モード (旋回する) では機体の向きが進行方角になる。"""
    ex = Explorer(open_maze(5), holonomic=False)
    ex.advance(Step(direction=Direction.E, to_cell=(1, 0)))
    assert ex.facing is Direction.E
    assert ex.travel is Direction.E


def test_holonomic_tie_break_prefers_keeping_the_travel_axis():
    """同距離なら進行軸を変えない方を選ぶ (機体の向きは固定なので travel で判断)。"""
    ex = Explorer(open_maze(5))
    ex.advance(Step(direction=Direction.E, to_cell=(1, 0)))   # 東へ進んだ直後
    # (1,0) からゴール (2,2) へは東も北も同距離。直前と同じ東を選ぶ
    assert ex.plan().direction is Direction.E


def test_goal_distance_and_progress_text():
    ex = Explorer(open_maze(5))
    assert ex.goal_distance() == 4                  # (0,0) -> (2,2) 壁なし
    text = ex.progress()
    assert "セル=(0, 0)" in text and "向き=N" in text and "ゴール距離=4" in text


# --- 探索ラン全体のシミュレーション ---------------------------------------
def test_search_run_reaches_goal_in_an_open_maze():
    ex = run_search(open_maze(5))
    assert ex.at_goal and ex.cell == (2, 2)
    assert ex.steps == 4                            # 壁なしなら最短 (マンハッタン距離)


def test_search_run_reaches_goal_with_a_wall_in_the_way():
    truth = open_maze(5)
    for y in range(0, 4):                           # x=0/1 の間を y=0..3 で塞ぐ壁
        truth.set_wall(1, y, Direction.W)
    ex = run_search(truth)
    assert ex.at_goal
    # 壁を見つけてから迂回するので最短 (4) より手数が増える
    assert ex.steps > 4
    # 見つけた壁は迷路に反映されている
    assert ex.maze.has_wall(1, 0, Direction.W)


def test_search_run_solves_a_spiral_16x16():
    """行き止まりを含む蛇行迷路 (16x16) でもゴールへ到達する。"""
    truth = open_maze(16)
    # y=1..14 の各行に、交互に東西どちらかを開けた横壁を作る (蛇行させる)
    for y in range(1, 15):
        for x in range(16):
            open_at = 0 if y % 2 else 15           # 通り抜けられる列を交互に
            if x != open_at:
                truth.set_wall(x, y, Direction.N)
    ex = run_search(truth, max_steps=2000)
    assert ex.at_goal
    assert len(ex.visited) > 16                     # それなりに開拓している


def test_search_run_detects_unreachable_goal():
    truth = open_maze(5)
    for d in Direction:
        truth.set_wall(2, 2, d)                     # ゴールを完全に封鎖
    ex = Explorer(open_maze(5))
    with pytest.raises(Unreachable):
        for _ in range(200):
            ex.observe(sense(truth, ex.cell, ex.facing))
            step = ex.plan()
            if step is None:
                pytest.fail("封鎖されたゴールに到達したことになっている")
            ex.advance(step)


# --- 隣のセルを読む / 止まらずに通過する (#89) --------------------------------
def _spiral_truth() -> Maze:
    """テスト用の 5x5。壁がそこそこ有り、直進が続く区間もある形。"""
    truth = Maze(5)
    truth.set_outer_walls()
    for x in range(4):
        truth.set_wall(x, 1, Direction.N)
    truth.set_wall(4, 1, Direction.W)
    truth.set_wall(1, 3, Direction.E)
    return truth


def test_observe_marks_the_cell_known_and_the_neighbours_partly():
    """自セルを 1 回見れば 4 壁が確定する。隣は 1 辺だけなので確定しない。"""
    truth = open_maze(4)
    ex = Explorer(open_maze(4))
    ex.observe(sense(truth, (0, 0), Direction.N))
    assert ex.is_known((0, 0))
    assert not ex.is_known((1, 0)) and not ex.is_known((0, 1))
    # 共有辺なので、隣のセルの「こちら向きの 1 辺」は確定している
    assert ex.observed[(1, 0)] == {Direction.W}


def test_observe_with_neighbours_makes_the_side_cells_known():
    truth = _spiral_truth()
    ex = Explorer(open_maze(5), cell=(2, 0))
    ex.observe(sense(truth, (2, 0), Direction.N),
               sense_neighbors(truth, (2, 0), Direction.N))
    assert ex.is_known((1, 0)) and ex.is_known((3, 0))     # 東西の隣は 4 壁とも確定
    assert not ex.is_known((2, 1))                          # 北の隣は 1 辺だけ
    assert ex.maze.has_wall(1, 0, Direction.N) is truth.has_wall(1, 0, Direction.N)


def test_observe_ignores_neighbours_outside_the_maze():
    truth = _spiral_truth()
    ex = Explorer(open_maze(5))          # スタート (0,0) の西は迷路の外
    ex.observe(sense(truth, (0, 0), Direction.N),
               sense_neighbors(truth, (0, 0), Direction.N))
    assert ex.is_known((1, 0))
    assert (-1, 0) not in ex.known and (-1, 0) not in ex.observed


def test_partial_neighbour_observation_leaves_the_cell_unknown():
    """確定しなかった辺 (キーが無い) があれば、そのセルは通過対象にならない。"""
    truth = _spiral_truth()
    ex = Explorer(open_maze(5), cell=(2, 0))
    walls = sense_neighbors(truth, (2, 0), Direction.N)
    del walls[LEFT][FRONT]               # 隣の奥の壁が読めなかった状況
    ex.observe(sense(truth, (2, 0), Direction.N), walls)
    assert not ex.is_known((1, 0))
    assert ex.is_known((3, 0))


def test_plan_leg_returns_one_step_by_default():
    truth = _spiral_truth()
    ex = Explorer(open_maze(5))
    ex.observe(sense(truth, ex.cell, ex.facing))
    assert len(ex.plan_leg()) == 1


def test_plan_leg_passes_through_a_known_cell_in_the_same_direction():
    """通過するセルの 4 壁が確定していれば、2 セルを 1 動作にまとめる。"""
    truth = _spiral_truth()
    ex = Explorer(open_maze(5), travel=Direction.E)
    ex.observe(sense(truth, ex.cell, ex.facing),
               sense_neighbors(truth, ex.cell, ex.facing))
    steps = ex.plan_leg(2)
    assert [s.direction for s in steps] == [Direction.E, Direction.E]
    assert steps[-1].to_cell == (2, 0)      # (1,0) は隣として読んだので通過できる


def test_plan_leg_stops_where_it_would_change_direction():
    """通過するセルが既知でも、そこで曲がるなら止まる (旋回レスでも減速は要る)。"""
    truth = _spiral_truth()
    truth.set_wall(1, 0, Direction.E)       # (1,0) から先は北へしか行けない
    ex = Explorer(open_maze(5), travel=Direction.E)
    ex.observe(sense(truth, ex.cell, ex.facing),
               sense_neighbors(truth, ex.cell, ex.facing))
    assert ex.is_known((1, 0))
    steps = ex.plan_leg(4)
    assert [s.direction for s in steps] == [Direction.E]


def test_plan_leg_stops_at_the_goal():
    """ゴールセルは通過しない (そこで走行が終わる)。"""
    truth = open_maze(5)
    ex = Explorer(open_maze(5), cell=(0, 2), travel=Direction.E)
    ex.observe(sense(truth, ex.cell, ex.facing),
               sense_neighbors(truth, ex.cell, ex.facing))
    steps = ex.plan_leg(4)
    assert [s.to_cell for s in steps] == [(1, 2), (2, 2)]
    assert steps[-1].to_cell == ex.maze.goal_cells()[0]


def test_plan_leg_never_passes_through_an_unobserved_cell():
    """隣を読まなければ、進行先は未確定なので 1 セルずつしか進まない。"""
    truth = _spiral_truth()
    ex = Explorer(open_maze(5))
    for _ in range(6):
        ex.observe(sense(truth, ex.cell, ex.facing))
        steps = ex.plan_leg(4)
        for step in steps:
            assert ex.is_known(step.to_cell) or step is steps[-1]
        for step in steps:
            ex.advance(step)


def test_neighbour_sensing_reaches_the_goal_with_fewer_stops():
    """隣を読んで通過すると、同じ迷路を**より少ない停止**でゴールできる。"""

    def search(neighbors: bool, max_cells: int) -> tuple[int, Explorer]:
        truth = _spiral_truth()
        ex = Explorer(open_maze(5))
        stops = 0
        for _ in range(200):
            ex.observe(sense(truth, ex.cell, ex.facing),
                       sense_neighbors(truth, ex.cell, ex.facing) if neighbors else None)
            steps = ex.plan_leg(max_cells)
            if not steps:
                return stops, ex
            stops += 1
            for step in steps:
                ex.advance(step)
        pytest.fail("ゴールに到達しなかった")

    base, ex_base = search(False, 1)
    fast, ex_fast = search(True, 2)
    assert fast < base
    # 通過したセルの壁も観測済みなので、地図は真の迷路と食い違わない
    truth = _spiral_truth()
    for cell in ex_fast.known:
        for d in Direction:
            assert ex_fast.maze.has_wall(*cell, d) is truth.has_wall(*cell, d)
    assert ex_fast.known >= ex_fast.visited
