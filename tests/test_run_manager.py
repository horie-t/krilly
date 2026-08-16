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
from krilly.strategy.shortest_path import Leg, path_to_legs, shortest_path, turns_in
from tests.test_explorer import open_maze, run_search


def explored(size: int = 5):
    """壁ありの迷路を探索し終えた Explorer を作る (ゴール到達済み)。"""
    truth = open_maze(size)
    for y in range(0, size - 1):                 # x=0/1 の間に縦壁 -> 迂回が必要
        truth.set_wall(1, y, Direction.W)
    return run_search(truth)


@pytest.fixture
def explorer():
    return explored()


@pytest.fixture
def manager(explorer):
    return RunManager(explorer)


# --- facing_after -----------------------------------------------------------
def test_facing_after_is_unchanged_when_holonomic():
    """旋回レス走行では機体は回らないので、走り終えても向きは同じ (#76)。"""
    legs = [Leg(Direction.N, 2), Leg(Direction.E, 2)]
    assert facing_after(legs, Direction.N) is Direction.N
    assert facing_after(legs, Direction.W) is Direction.W


def test_facing_after_follows_the_last_leg_when_turning():
    """旋回する走り方では最後の区間の方角を向いて終わる。"""
    legs = [Leg(Direction.N, 2), Leg(Direction.E, 2)]
    assert facing_after(legs, Direction.N, holonomic=False) is Direction.E
    assert facing_after([], Direction.W, holonomic=False) is Direction.W


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
    assert speed == path_to_legs(path)


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
    """セル数に加えて**区間の本数**も数える (ランプの固定費)。旋回レスでは旋回は乗らない。"""
    legs = [Leg(Direction.N, 3), Leg(Direction.E, 2)]      # 5 セル / 区間 2 本
    assert manager.estimate_s(legs) == pytest.approx(
        3 * manager.cell_time_s + 2 * manager.lateral_cell_time_s
        + 2 * manager.straight_time_s
    )


def test_estimate_adds_turns_only_in_the_turning_mode(explorer):
    """旋回する走り方では旋回の時間が乗る (#76 の退避路)。"""
    legs = [Leg(Direction.N, 3), Leg(Direction.E, 2)]      # 北 -> 東 = 90° 1 回
    # 旋回する走り方は常に前後軸で進むので、比較のため東西も前後と同じ時間に揃える
    # (既定は東西 0.75s / 南北 0.73s と少し違う)。
    turning = RunManager(explorer, holonomic=False, lateral_cell_time_s=0.73)
    assert turning.estimate_s(legs, Direction.N) - turning.turn_time_s == pytest.approx(
        RunManager(explorer, lateral_cell_time_s=0.73).estimate_s(legs)
    )


def test_estimate_prefers_fewer_straights_for_the_same_cells(manager):
    """同じ 4 セルでも、区間が細切れなほど見積もりは大きい (固定費が効く)。"""
    one_leg = manager.estimate_s([Leg(Direction.N, 4)])
    zigzag = [Leg(Direction.N, 1), Leg(Direction.N, 1),
              Leg(Direction.N, 1), Leg(Direction.N, 1)]
    assert manager.estimate_s(zigzag) > one_leg
    assert manager.estimate_s(zigzag) - one_leg == pytest.approx(
        3 * manager.straight_time_s)


# 実機の最速ラン: 20 セル / 区間 12 本 / 旋回 13 回 (北向きスタート)
MEASURED_LEGS = [Leg(Direction.S, 3), Leg(Direction.W, 1), Leg(Direction.S, 1),
                 Leg(Direction.W, 3), Leg(Direction.N, 2), Leg(Direction.E, 1),
                 Leg(Direction.N, 1), Leg(Direction.W, 1), Leg(Direction.N, 1),
                 Leg(Direction.E, 3), Leg(Direction.S, 2), Leg(Direction.W, 1)]


def test_estimate_matches_the_measured_speed_run(explorer):
    """3 項モデル (セル数・区間の本数・旋回回数) が実測に合う。

    M5 #20 の速度設定 (v=0.12, omega=1.0) でこの経路は実測 70.9s だった。定数はその後
    #21 で速くした側へ更新したので、モデルの検証は当時の定数と**旋回する走り方**で行う。
    セル数と旋回数だけの 2 項モデルでは同じ経路が 25% 低く出ていた。
    """
    assert sum(leg.cells for leg in MEASURED_LEGS) == 20
    assert len(MEASURED_LEGS) == 12
    assert turns_in(MEASURED_LEGS, Direction.N) == 13
    m5 = RunManager(explorer, cell_time_s=1.50, straight_time_s=0.70, turn_time_s=2.56,
                    holonomic=False)
    assert m5.estimate_s(MEASURED_LEGS, Direction.N) == pytest.approx(70.9, abs=1.5)


def test_holonomic_estimate_is_about_half_of_the_turning_one(explorer, manager):
    """旋回をやめると同じ経路の見積もりが半分近くになる (#76 の狙い)。"""
    turning = RunManager(explorer, holonomic=False)
    ratio = manager.estimate_s(MEASURED_LEGS) / turning.estimate_s(MEASURED_LEGS,
                                                                   Direction.N)
    assert 0.40 < ratio < 0.60


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
    cell = ex.maze.start
    for leg in legs:
        for _ in range(leg.cells):
            cell = ex.maze.neighbor(*cell, leg.direction)
            assert cell in ex.visited
    assert ex.maze.is_goal(*cell)


def test_return_legs_from_goal(manager):
    ex = manager.explorer
    legs = manager.return_legs(ex.cell, ex.facing)
    assert legs is not None
    cell = ex.cell
    for leg in legs:
        for _ in range(leg.cells):
            cell = ex.maze.neighbor(*cell, leg.direction)
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
