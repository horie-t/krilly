"""走行の状態機械 (RunManager #20) のユニットテスト。

探索は tests.test_explorer のシミュレーション (真の迷路を相手に走らせる) で行い、
その結果 (壁情報 + visited) の上で 復帰 → 最速 の差配をテストする。時刻は
now を渡すだけなので、時間経過は数値で自由に作れる。
"""

import math

import pytest

from krilly.app.run_manager import RunManager, RunPhase, facing_after
from krilly.solver.maze import Direction
from krilly.strategy.explorer import quarter_turns
from krilly.strategy.shortest_path import Leg, path_to_legs, shortest_path
from tests.test_explorer import open_maze, run_search


def explored(size: int = 5):
    """壁ありの迷路を探索し終えた Explorer を作る (ゴール到達済み)。"""
    truth = open_maze(size)
    for y in range(0, size - 1):                 # x=0/1 の間に縦壁 -> 迂回が必要
        truth.set_wall(1, y, Direction.W)
    return run_search(truth)


@pytest.fixture
def manager():
    return RunManager(explored())


# --- facing_after -----------------------------------------------------------
def test_facing_after_folds_turns():
    legs = [Leg(turn=0, cells=2), Leg(turn=-1, cells=2)]   # 直進 -> 右90°
    assert facing_after(legs, Direction.N) is Direction.E
    assert facing_after([Leg(2, 1)], Direction.N) is Direction.S
    assert facing_after([], Direction.W) is Direction.W


# --- 基本の遷移 --------------------------------------------------------------
def test_full_cycle_search_return_speed(manager):
    ex = manager.explorer
    assert manager.phase is RunPhase.WAIT
    manager.start_search(now=0.0)
    assert manager.phase is RunPhase.SEARCH and manager.runs_used == 1

    home = manager.goal_reached(now=60.0, cell=ex.cell, facing=ex.facing)
    assert manager.phase is RunPhase.RETURN_HOME
    assert home is not None
    # 復帰経路は探索済みセルだけで組める (歩いてきた道があるので必ず存在する)
    facing_home = facing_after(home, ex.facing)

    speed = manager.home_reached(now=120.0, facing=facing_home)
    assert manager.phase is RunPhase.SPEED and manager.runs_used == 2
    # 最速経路は shortest_path (known=visited) と一致する
    path = shortest_path(ex.maze, ex.maze.start, known=ex.visited,
                         start_facing=facing_home)
    assert speed == path_to_legs(path, facing_home)


def test_repeats_until_max_runs(manager):
    ex = manager.explorer
    manager.start_search(0.0)
    now, cell, facing = 60.0, ex.cell, ex.facing
    while manager.phase is not RunPhase.FINISHED:
        home = manager.goal_reached(now, cell, facing)
        if home is None:
            break
        facing = facing_after(home, facing)
        now += 30.0
        speed = manager.home_reached(now, facing)
        if speed is None:
            break
        facing = facing_after(speed, facing)
        cell = ex.maze.goal_cells()[0]
        now += 30.0
    assert manager.runs_used == 5                 # 探索 1 + 最速 4
    assert manager.phase is RunPhase.FINISHED


def test_fifth_run_goal_finishes_without_returning(manager):
    """5 走目のゴール後は、時間が残っていても復帰しない (走行が残っていない)。"""
    ex = manager.explorer
    manager.start_search(0.0)
    manager.runs_used = 5                          # 5 走目の途中とする
    manager.phase = RunPhase.SPEED
    assert manager.goal_reached(10.0, ex.maze.goal_cells()[0], Direction.N) is None
    assert manager.phase is RunPhase.FINISHED


# --- 時間予算 ----------------------------------------------------------------
def test_time_budget_blocks_late_speed_run(manager):
    """残り時間が (復帰+最速)×安全率 に足りなければゴールで終了する。"""
    ex = manager.explorer
    manager.start_search(0.0)
    late = manager.time_limit_s - 5.0              # 残り 5 秒
    assert manager.goal_reached(late, ex.cell, ex.facing) is None
    assert manager.phase is RunPhase.FINISHED


def test_time_budget_blocks_at_home_too(manager):
    """復帰中に時間を使いすぎたら、スタートに着いても出発しない。"""
    ex = manager.explorer
    manager.start_search(0.0)
    home = manager.goal_reached(30.0, ex.cell, ex.facing)
    assert home is not None
    facing = facing_after(home, ex.facing)
    late = manager.time_limit_s - 1.0
    assert manager.home_reached(late, facing) is None
    assert manager.phase is RunPhase.FINISHED
    assert manager.runs_used == 1                  # 出発していないので増えない


def test_estimate_uses_measured_times(manager):
    legs = [Leg(0, 3), Leg(-1, 2)]                 # 5 セル + 旋回 1 回
    assert manager.estimate_s(legs) == pytest.approx(5 * 1.75 + 1 * 1.64)


def test_elapsed_starts_at_first_departure(manager):
    assert manager.elapsed_s(100.0) == 0.0         # 開始前は 0
    manager.start_search(100.0)
    assert manager.elapsed_s(160.0) == pytest.approx(60.0)
    assert manager.remaining_s(160.0) == pytest.approx(540.0)


# --- 経路 ---------------------------------------------------------------------
def test_speed_route_only_uses_visited_cells(manager):
    ex = manager.explorer
    legs = manager.speed_legs(Direction.N)
    assert legs is not None
    # Leg を辿ってセル列を復元し、全部 visited であること
    cell, facing = ex.maze.start, Direction.N
    for leg in legs:
        facing = Direction((facing - leg.turn) % 4)
        for _ in range(leg.cells):
            cell = ex.maze.neighbor(*cell, facing)
            assert cell in ex.visited
    assert ex.maze.is_goal(*cell)


def test_return_legs_from_goal(manager):
    ex = manager.explorer
    legs = manager.return_legs(ex.cell, ex.facing)
    assert legs is not None
    cell, facing = ex.cell, ex.facing
    for leg in legs:
        facing = Direction((facing - leg.turn) % 4)
        for _ in range(leg.cells):
            cell = ex.maze.neighbor(*cell, facing)
    assert cell == ex.maze.start


# --- 誤用ガード ----------------------------------------------------------------
def test_event_guards(manager):
    with pytest.raises(RuntimeError):
        manager.goal_reached(0.0, (0, 0), Direction.N)   # 走行していない
    manager.start_search(0.0)
    with pytest.raises(RuntimeError):
        manager.start_search(1.0)                        # 二重開始
    with pytest.raises(RuntimeError):
        manager.home_reached(1.0, Direction.N)           # 復帰中ではない


def test_abort_finishes(manager):
    manager.start_search(0.0)
    manager.abort()
    assert manager.phase is RunPhase.FINISHED


def test_summary_text(manager):
    manager.start_search(0.0)
    text = manager.summary(30.0)
    assert "走行 1/5" in text and "経過 30s" in text and "search" in text
