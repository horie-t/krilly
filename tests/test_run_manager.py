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
    # (既定は東西 0.60s / 南北 0.61s と少し違う)。
    same = RunManager(explorer).cell_time_s
    turning = RunManager(explorer, holonomic=False, lateral_cell_time_s=same)
    assert turning.estimate_s(legs, Direction.N) - turning.turn_time_s == pytest.approx(
        RunManager(explorer, lateral_cell_time_s=same).estimate_s(legs)
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
    # 持ち時間はクラシック競技規定の 7 分 (420s)
    assert manager.remaining_s(160.0) == pytest.approx(360.0)


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


# --- 競技規定 (NTF クラシックマウス競技規定) ---------------------------------
def test_time_limit_follows_the_classic_rules(explorer):
    """持ち時間は 7 分 (規定 3-6)。10 分は別競技 (旧ハーフサイズ) の規定。"""
    assert RunManager(explorer).time_limit_s == 420.0
    assert RunManager(explorer).max_runs == 5


def test_restart_dwell_is_charged_before_a_speed_run(explorer):
    """規定 3-4: 始点に戻って自動再スタートするなら 2 秒以上停止する。

    削れない時間なので、見積もりにも予算判断にも入っていなければならない。
    """
    mgr = RunManager(explorer)
    assert mgr.restart_dwell_s >= 2.0
    mgr.start_search(0.0)
    legs = mgr.speed_legs(Direction.N)
    need = mgr.time_margin * mgr.estimate_s(legs)
    mgr.goal_reached(0.0, explorer.cell, explorer.facing)
    # 停止の分だけ足りないときは走行を始めない
    late = mgr.time_limit_s - need - mgr.restart_dwell_s / 2
    assert mgr.home_reached(late, Direction.N) is None


# --- 止まらずに繋ぐぶんの見積もり (#80) --------------------------------------
#: 5x5 の最速経路 (#80 の対照セッション実測: 区間ごとに止まると 25.0s、
#: 2 本ずつ繋いで 20.0s。同じ日・同じ電池で連続して測った 4 本ずつの平均)。
MEASURED_5X5 = [
    Leg(Direction.N, 3), Leg(Direction.E, 1), Leg(Direction.N, 1), Leg(Direction.E, 3),
    Leg(Direction.S, 2), Leg(Direction.W, 1), Leg(Direction.S, 1), Leg(Direction.E, 1),
    Leg(Direction.S, 1), Leg(Direction.W, 3), Leg(Direction.N, 2), Leg(Direction.E, 1),
]


def test_motions_counts_chunks_not_legs(manager):
    """固定費は区間ではなく**動作**に付く (#80)。"""
    assert manager.motions(MEASURED_5X5) == 12          # 既定は区間ごとに停止
    for size, expected in ((2, 6), (3, 4), (4, 3), (5, 3), (12, 1), (99, 1)):
        manager.chain_legs = size
        assert manager.motions(MEASURED_5X5) == expected, size
    manager.chain_legs = 0                              # 不正値は 1 として扱う
    assert manager.motions(MEASURED_5X5) == 12


def test_the_estimate_matches_the_measured_chained_speed_run(explorer):
    """実測との突き合わせ: 5x5 の最速経路 (20 セル / 12 区間)。

    区間ごとに停止 25.0s / 2 本ずつ繋いで 20.0s (いずれも実機、同じ日・同じ電池)。
    ここが合っていないと、7 分の予算判断が「走れる走行を断る」方へ狂う。

    この対 (#80) は **v=0.24 m/s** の機体で測ったもの。既定は #103 で 0.30 m/s 用に
    更新したので、当時の定数を明示的に渡してモデルの検証だけを行う
    (:func:`test_estimate_matches_the_measured_speed_run` が M5 の定数でやるのと同じ)。
    """
    m = RunManager(explorer, cell_time_s=0.74, lateral_cell_time_s=0.76,
                   straight_time_s=0.83)
    assert m.estimate_s(MEASURED_5X5) == pytest.approx(25.0, abs=0.3)
    m.chain_legs = 2
    assert m.estimate_s(MEASURED_5X5) == pytest.approx(20.0, abs=0.3)


def test_chaining_never_makes_the_estimate_longer(manager):
    base = manager.estimate_s(MEASURED_5X5)
    for size in range(2, 13):
        manager.chain_legs = size
        assert manager.estimate_s(MEASURED_5X5) <= base


def test_the_default_chain_legs_errs_on_the_safe_side(manager):
    """**渡し忘れたときに見積もりが短くならない**こと (#80)。

    実機は 2 本ずつ繋いで走るが、ここの既定を 2 にすると、繋がない呼び出し側が
    渡し忘れたときに見積もりが短い方へ外れ、7 分の予算判断が「終われない走行を
    始める」side へ倒れる。1 なら外れ方は「走れる走行を断る」側で済む。
    """
    assert manager.chain_legs == 1
    assert manager.motions(MEASURED_5X5) == len(MEASURED_5X5)


# --- 安全率 (#87) ----------------------------------------------------------

def test_the_margin_is_the_one_the_contest_mazes_argued_for(manager):
    """既定は 1.2。**1.5 は崖の上に乗っていた** (#87)。

    大会迷路 31 面で、1.4 へ下げるだけで 4 面が最速ランを走れるようになり、
    1.2 では実機が見積もりより 20% 遅くても 7 分を超える面はゼロだった
    (実測の見積もり誤差は 8x8 で 0.5%)。1.0 は 10% 遅いだけで超過するので行き過ぎ。
    """
    assert manager.time_margin == 1.2


def test_a_lower_margin_accepts_a_run_the_old_one_refused(explorer):
    """安全率が守っているのは「始めた走行を終えられるか」だけ。

    1.5 では断る残り時間でも 1.2 なら出る — 規定 3-1 では記録は**最速の 1 走行**で、
    時間切れの最速ランは時間を失うだけなので、**断る方が高くつく。**
    """
    legs = [Leg(Direction.N, 8), Leg(Direction.E, 8)]
    bold, safe = RunManager(explorer), RunManager(explorer, time_margin=1.5)
    need = bold.estimate_s(legs)
    # 1.2 倍ぶんは足りるが 1.5 倍には足りない残り時間
    assert bold.time_margin * need < 1.35 * need < safe.time_margin * need


# --- 区間長の上限 (#85) -----------------------------------------------------
def test_the_route_is_capped_and_the_estimate_counts_the_extra_motions():
    """長い直進を割ると**動作が増え、見積もりもそのぶん伸びる**こと (#85)。

    割った区間を繋いでしまうと停止も補正も増えないので、:func:`chunk_legs` は
    同じ方角の境目で必ず切る。見積もりがそれを数えていないと、実機より短い時間を
    信じて走行を始めてしまう。
    """
    from krilly.solver.maze import Maze
    from krilly.strategy.explorer import Explorer

    maze = Maze(16)                      # 壁なし = スタートからゴールまで直進できる
    maze.set_outer_walls()
    ex = Explorer(maze, cell=(0, 0))
    ex.known.update({(x, y) for x in range(16) for y in range(16)})

    free = RunManager(ex, chain_legs=2, max_leg_cells=0)
    capped = RunManager(ex, chain_legs=2, max_leg_cells=4)
    long_legs = free.speed_legs(Direction.N)
    short_legs = capped.speed_legs(Direction.N)
    assert max(leg.cells for leg in long_legs) > 4
    assert max(leg.cells for leg in short_legs) <= 4
    assert (sum(leg.cells for leg in short_legs)
            == sum(leg.cells for leg in long_legs))          # 距離は同じ
    assert capped.motions(short_legs) > free.motions(long_legs)
    assert capped.estimate_s(short_legs) > free.estimate_s(long_legs)


def test_the_cap_costs_nothing_on_the_board_that_is_on_the_floor():
    """床の 8x8 (excerpt8_2015) は最長区間が 4 セルなので、上限 4 はタダ (#85)。

    **上限の値段は盤面で全く違う**: ここでは 0 秒だが、16x16 の大会迷路は最短経路の
    最長区間の中央値が 11.5 セルあるので、同じ上限が最速ランを 48 本 -> 44 本に減らす
    (:mod:`tests.test_sim` 側で固定してある)。
    """
    from pathlib import Path

    from krilly.solver.maze import Maze
    from krilly.strategy.explorer import Explorer

    truth = Maze.from_ascii(
        Path("mazes/excerpt8_2015.txt").read_text(encoding="utf-8"))
    ex = Explorer(truth, cell=truth.start)
    ex.known.update({(x, y) for x in range(truth.size) for y in range(truth.size)})
    free = RunManager(ex, chain_legs=2, max_leg_cells=0)
    capped = RunManager(ex, chain_legs=2, max_leg_cells=4)
    assert capped.speed_legs(Direction.N) == free.speed_legs(Direction.N)
    assert capped.estimate_s(capped.speed_legs(Direction.N)) == pytest.approx(
        free.estimate_s(free.speed_legs(Direction.N)))


# --- 補正できない始点・終点 (#126) -----------------------------------------------
# 位置補正はまとまりの先頭でしか入らない。始点・終点の壁が白/黄だとそこで補正が
# 入らず、補正の無い区間が「前の走行の最後のまとまり + 次の最初のまとまり」に伸びる。

from krilly.strategy.shortest_path import chunk_legs, walk_legs  # noqa: E402

N1, E2, N3, W1 = (Leg(Direction.N, 1), Leg(Direction.E, 2),
                  Leg(Direction.N, 3), Leg(Direction.W, 1))


def test_chunk_legs_can_shorten_only_the_first_or_last_chunk():
    legs = [N1, E2, N3, W1]
    assert chunk_legs(legs, 2) == [[N1, E2], [N3, W1]]
    assert chunk_legs(legs, 2, head=1) == [[N1], [E2, N3], [W1]]
    assert chunk_legs(legs, 2, tail=1) == [[N1, E2], [N3], [W1]]
    assert chunk_legs(legs, 2, head=1, tail=1) == [[N1], [E2, N3], [W1]]
    assert chunk_legs([N1, E2], 2, head=1, tail=1) == [[N1], [E2]]


def test_walk_legs_ends_where_the_legs_end():
    assert walk_legs((0, 0), [N1, E2, N3, W1]) == (1, 4)


def test_an_unfixable_endpoint_keeps_the_blind_stretch_within_chain_legs(explorer):
    mgr = RunManager(explorer, chain_legs=2)
    start = explorer.maze.start
    legs = [N1, E2, N3, W1]
    end = walk_legs(start, legs)
    assert mgr.chunks(legs, start) == [[N1, E2], [N3, W1]]
    mgr.mark_fix(start, False)
    assert mgr.chunks(legs, start)[0] == [N1]            # 始点を出る側
    mgr.mark_fix(end, False)
    assert mgr.chunks(legs, start)[-1] == [W1]           # 行き先に入る側
    mgr.mark_fix(start, True)
    mgr.mark_fix(end, True)
    assert mgr.chunks(legs, start) == [[N1, E2], [N3, W1]]


def test_the_estimate_pays_for_the_extra_stops(explorer):
    """見積もりも実機と同じ切り方で数える (でないと予算判断が甘くなる)。"""
    mgr = RunManager(explorer, chain_legs=2)
    start = explorer.maze.start
    legs = [N1, E2, N3, W1]
    before = mgr.estimate_s(legs, origin=start)
    mgr.mark_fix(start, False)
    assert mgr.motions(legs, start) == 3
    assert mgr.estimate_s(legs, origin=start) == pytest.approx(before + mgr.straight_time_s)
    # 出発点を渡さない見積もりは従来どおり (端点を気にしない)
    assert mgr.motions(legs) == 2


def test_stopping_at_every_leg_is_unaffected(explorer):
    mgr = RunManager(explorer, chain_legs=1)
    mgr.mark_fix(explorer.maze.start, False)
    assert mgr.chunks([N1, E2], explorer.maze.start) == [[N1], [E2]]
