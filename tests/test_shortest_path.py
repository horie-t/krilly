"""最短経路 (shortest_path #19) のユニットテスト。

探索ラン (#18) と違い、**未探索セルは通さない**のが最速ランの前提。合成した迷路と、
シミュレートした探索ランの結果 (visited) の両方で確認する。
"""

import pytest

from krilly.solver.maze import Direction, Maze
from krilly.strategy.explorer import Explorer
from krilly.strategy.shortest_path import (
    DEFAULT_COST,
    LEGACY_COST,
    Leg,
    MoveCost,
    describe_legs,
    direction_between,
    path_cost,
    path_to_legs,
    route,
    shortest_path,
    turns_in,
)
from tests.test_explorer import open_maze, run_search, sense


def path_turns(path, start_facing=Direction.N) -> int:
    return turns_in(path_to_legs(path), start_facing)


def crosses_no_wall(maze: Maze, path) -> bool:
    """経路が壁を横切っていないか (妥当性チェック)。"""
    return all(
        not maze.has_wall(*path[i], direction_between(path[i], path[i + 1]))
        for i in range(len(path) - 1)
    )


# --- 方角 -----------------------------------------------------------------
def test_direction_between():
    assert direction_between((1, 1), (1, 2)) is Direction.N
    assert direction_between((1, 1), (2, 1)) is Direction.E
    assert direction_between((1, 1), (1, 0)) is Direction.S
    assert direction_between((1, 1), (0, 1)) is Direction.W
    with pytest.raises(ValueError):
        direction_between((0, 0), (2, 2))


# --- 基本 -----------------------------------------------------------------
def test_shortest_path_in_open_maze_is_cell_count_optimal():
    m = open_maze(5)                                  # ゴールは中央 (2,2)
    path = shortest_path(m, (0, 0))
    assert path[0] == (0, 0) and path[-1] == (2, 2)
    assert len(path) - 1 == 4                          # マンハッタン距離
    assert crosses_no_wall(m, path)


def test_shortest_path_minimises_turns_with_turn_cost():
    """壁なしなら同じセル数の経路が多数ある。旋回コストで曲がりの少ない方を選ぶ。"""
    m = open_maze(5)
    # 北向きスタート: 北へ2 -> 東へ2 なら旋回 1 回で足りる
    cost = MoveCost(leg=0.0, turn=1.0)
    assert path_turns(shortest_path(m, (0, 0), cost=cost)) == 1
    # 旋回も区間の固定費も 0 ならセル数だけの最短 (曲がり方は問わない) になる
    flat = MoveCost(cell_ew=1.0, leg=0.0, turn=0.0)   # 軸差も消して 1 セル = 1.0 にする
    assert len(shortest_path(m, (0, 0), cost=flat)) - 1 == 4


def test_leg_cost_prefers_fewer_straight_runs():
    """旋回レスでも「区間の固定費」があるので、曲がりの少ない経路を選ぶ (#76)。"""
    m = open_maze(5)
    # 既定 (leg=1.04, turn=0) でも、区間が 2 本で済む L 字が選ばれる
    assert len(path_to_legs(shortest_path(m, (0, 0)))) == 2


def test_legacy_cost_reproduces_the_turning_route():
    """LEGACY_COST は「セル数 + turn_cost x 旋回回数」と厳密に一致する (#76 の後方互換)。"""
    m = open_maze(5)
    path = shortest_path(m, (0, 0), cost=LEGACY_COST)
    legs = path_to_legs(path)
    expected = (len(path) - 1) + LEGACY_COST.turn * turns_in(legs)
    assert path_cost(path, Direction.N, LEGACY_COST) == pytest.approx(expected)


def test_shortest_path_start_on_goal():
    m = open_maze(5)
    assert shortest_path(m, (2, 2)) == [(2, 2)]
    assert path_to_legs(shortest_path(m, (2, 2))) == []


def test_shortest_path_uses_maze_start_and_goals_by_default():
    m = open_maze(5)
    assert shortest_path(m)[0] == m.start
    assert shortest_path(m)[-1] in m.goal_cells()
    assert shortest_path(m, goals=[(4, 4)])[-1] == (4, 4)


def test_shortest_path_detours_around_walls():
    m = open_maze(5)
    for y in range(0, 4):                              # x=0/1 の間に縦壁
        m.set_wall(1, y, Direction.W)
    path = shortest_path(m, (0, 0))
    assert crosses_no_wall(m, path)
    assert len(path) - 1 > 4                           # 迂回で伸びる
    assert (0, 4) in path                              # 北端を回り込む


def test_shortest_path_returns_empty_when_unreachable():
    m = open_maze(5)
    for d in Direction:
        m.set_wall(2, 2, d)                            # ゴールを封鎖
    assert shortest_path(m, (0, 0)) == []


# --- 未探索セルの扱い (最速ランの本質) ------------------------------------
def test_shortest_path_refuses_unknown_cells():
    """known を渡すと、そのセルしか通らない (未探索は壁が未確定なので通さない)。"""
    m = open_maze(5)
    known = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)}   # 北へ2 -> 東へ2 の L 字だけ既知
    path = shortest_path(m, (0, 0), known=known)
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    assert all(c in known for c in path)


def test_shortest_path_empty_when_known_cells_do_not_reach_goal():
    m = open_maze(5)
    assert shortest_path(m, (0, 0), known={(0, 0), (0, 1)}) == []


def test_shortest_path_ignores_open_shortcut_outside_known():
    """既知セルの外に近道があっても使わない (壁を見ていないので走れない)。"""
    m = open_maze(5)
    known = {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}   # 東へ2 -> 北へ2
    path = shortest_path(m, (0, 0), known=known)
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]


# --- 区間への圧縮 (最速ランが使う形) --------------------------------------
def test_path_to_legs_run_length_encodes_straights():
    path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    assert path_to_legs(path) == [Leg(Direction.N, 2), Leg(Direction.E, 2)]


def test_path_to_legs_does_not_depend_on_the_facing():
    """区間は「どちらへ何セル」なので、機体の向きは関係しない (#76)。"""
    path = [(0, 0), (1, 0), (2, 0)]
    assert path_to_legs(path) == [Leg(Direction.E, 2)]


def test_path_to_legs_handles_a_zigzag():
    path = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
    assert path_to_legs(path) == [
        Leg(Direction.N, 1), Leg(Direction.E, 1), Leg(Direction.N, 1), Leg(Direction.E, 1),
    ]


def test_turns_in_counts_quarter_turns_from_the_start_facing():
    legs = [Leg(Direction.N, 2), Leg(Direction.E, 2)]
    assert turns_in(legs, Direction.N) == 1        # 北のまま -> 東へ 90°
    assert turns_in(legs, Direction.E) == 2        # 東 -> 北 -> 東
    assert turns_in([], Direction.N) == 0


def test_path_cost_matches_what_the_search_minimises():
    path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]   # 北へ2 -> 東へ2
    flat = MoveCost(cell_ew=1.0, leg=0.0, turn=0.0)   # 軸差も消して 1 セル = 1.0 にする
    assert path_cost(path, Direction.N, flat) == pytest.approx(4.0)
    turning = MoveCost(cell_ew=1.0, leg=0.0, turn=1.0)
    assert path_cost(path, Direction.N, turning) == pytest.approx(5.0)   # 旋回1回
    # 旋回レス既定: 南北 2 セル + 東西 2 セル x 1.03 + 区間 2 本 x 1.10
    assert path_cost(path, Direction.N, DEFAULT_COST) == pytest.approx(
        2 * 1.0 + 2 * 1.03 + 2 * 1.10
    )
    assert path_cost([(0, 0)]) == pytest.approx(0.0)


def test_direction_split_cost_biases_the_route():
    """南北と東西でコストが違えば、安い軸を長く使う経路を選ぶ (#76 の横移動用)。"""
    m = open_maze(5)
    cheap_ns = MoveCost(cell_ns=1.0, cell_ew=3.0, leg=0.0, turn=0.0)
    path = shortest_path(m, (0, 0), [(2, 2)], cost=cheap_ns)
    legs = path_to_legs(path)
    # 東西が高いので、東西の移動は最小限 (2 セル) のまま。経路長は変わらないが
    # コストの内訳は南北優先になる
    assert sum(l.cells for l in legs if l.direction in (Direction.E, Direction.W)) == 2


def test_route_composes_path_and_legs():
    m = open_maze(5)
    assert route(m, (0, 0)) == path_to_legs(shortest_path(m, (0, 0)))


def test_describe_legs_text():
    legs = [Leg(Direction.N, 2), Leg(Direction.E, 3), Leg(Direction.S, 1)]
    assert describe_legs(legs) == "Nへ2 -> Eへ3 -> Sへ1"
    assert describe_legs([]) == "(移動なし)"


# --- 探索ランの結果と組み合わせる -----------------------------------------
def test_route_from_a_simulated_search_run():
    """探索ランで得た壁情報 + visited から、妥当な最速経路が出る。"""
    truth = open_maze(5)
    for y in range(0, 4):
        truth.set_wall(1, y, Direction.W)
    ex = run_search(truth)
    path = shortest_path(ex.maze, ex.maze.start, known=ex.visited)
    assert path and path[-1] in ex.maze.goal_cells()
    assert crosses_no_wall(ex.maze, path)              # 判明した壁を横切らない
    assert crosses_no_wall(truth, path)               # 真の迷路でも壁を横切らない
    assert all(c in ex.visited for c in path)          # 未探索セルを通らない
    # 探索は行き止まりに寄ることもあるので、最速経路は探索の手数以下になる
    assert len(path) - 1 <= ex.steps


def test_route_from_search_run_on_16x16_serpentine():
    truth = open_maze(16)
    for y in range(1, 15):
        for x in range(16):
            open_at = 0 if y % 2 else 15
            if x != open_at:
                truth.set_wall(x, y, Direction.N)
    ex = run_search(truth, max_steps=2000)
    path = shortest_path(ex.maze, ex.maze.start, known=ex.visited)
    assert path and path[-1] in ex.maze.goal_cells()
    assert crosses_no_wall(truth, path)
    assert len(path) - 1 <= ex.steps
    legs = path_to_legs(path)
    assert sum(leg.cells for leg in legs) == len(path) - 1   # 区間の合計 = セル数


def test_known_restriction_matters_after_a_partial_search():
    """探索直後は未探索セルが残るので、known 付きの方がコストが高い (安全側)。"""
    truth = open_maze(9)
    for y in range(0, 8):
        truth.set_wall(1, y, Direction.W)
    ex = run_search(truth, max_steps=2000)
    safe = shortest_path(ex.maze, ex.maze.start, known=ex.visited)
    optimistic = shortest_path(ex.maze, ex.maze.start)      # 未探索も通れると仮定
    assert path_cost(safe) >= path_cost(optimistic)
    assert len(ex.visited) < ex.maze.size ** 2              # 未探索セルが残っている
