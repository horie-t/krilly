"""探索 → 復帰 → 最速 xN を一本で回す統合シミュレータ (issue #77)。

:class:`krilly.app.run_manager.RunManager` は ``now`` を渡される純ロジックなので、
時計を実測由来の見積もりで進めれば**7 分 5 走のセッション全体が実機なしで回る**。

**このシミュレータが検証するもの**: 探索の完走、判明した地図と真の迷路の一致、
予算の管理 (いつ次の走行を始め、いつ打ち切るか)、経路計画。

**検証しないもの**: カメラの見落とし・誤検出、姿勢の誤差、車輪の滑り。壁観測は
:func:`krilly.sim.sense.sense` が真の迷路から作るので常に正しい。そこは実機 (#23)
の担当。

**時計の作り方に注意**: 既定では経過時間も ``RunManager.estimate_s`` で作るので、
見積もりは定義上ぴったり当たる。それでは予算判断の余裕を試せないので
``actual_scale`` で「実際は見積もりの N 倍かかる」状況を作れる。``time_margin``
(既定 1.5) が妥当かはこれで確かめる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krilly.app.run_manager import RunManager, RunPhase, facing_after
from krilly.sim.check import map_agrees
from krilly.sim.generate import open_maze
from krilly.sim.sense import sense, sense_neighbors
from krilly.solver.maze import Direction, Maze
from krilly.strategy.explorer import Explorer, Unreachable
from krilly.strategy.shortest_path import DEFAULT_COST, Leg, MoveCost, walk_legs


@dataclass(frozen=True)
class RunRecord:
    """1 走行 (探索 / 復帰 / 最速) の記録。"""

    phase: RunPhase
    started_s: float
    duration_s: float
    cells: int
    legs: int

    def describe(self) -> str:
        return (f"{self.phase.value:<8} {self.duration_s:6.1f}s "
                f"({self.cells} セル / {self.legs} 区間)")


@dataclass
class SessionResult:
    """セッション 1 回分の結果。"""

    truth: Maze
    explorer: Explorer
    records: list[RunRecord] = field(default_factory=list)
    elapsed_s: float = 0.0
    runs_used: int = 0
    reached_goal: bool = False
    mismatches: list[str] = field(default_factory=list)
    #: 途中で止まった理由 (完走なら None)。
    aborted: str | None = None
    #: 持ち時間 [s] (:attr:`ok` の判定に使う)。
    limit_s: float = field(default=420.0, repr=False)

    @property
    def search(self) -> RunRecord | None:
        return next((r for r in self.records if r.phase is RunPhase.SEARCH), None)

    @property
    def speed_runs(self) -> list[RunRecord]:
        return [r for r in self.records if r.phase is RunPhase.SPEED]

    @property
    def best_speed_s(self) -> float | None:
        runs = self.speed_runs
        return min(r.duration_s for r in runs) if runs else None

    @property
    def ok(self) -> bool:
        """ゴールに着き、地図が一致し、予算内に収まったか。"""
        return (self.reached_goal and not self.mismatches and self.aborted is None
                and self.elapsed_s <= self.limit_s)

    def describe(self) -> str:
        lines = [f"{self.truth.size}x{self.truth.size} "
                 f"経過 {self.elapsed_s:.1f}s / 走行 {self.runs_used} 回"]
        lines += ["  " + r.describe() for r in self.records]
        if self.mismatches:
            lines.append(f"  [誤り] 地図が真の迷路と {len(self.mismatches)} 箇所違う: "
                         f"{self.mismatches[0]}")
        if self.aborted:
            lines.append(f"  [中断] {self.aborted}")
        return "\n".join(lines)


_walk = walk_legs


def simulate_session(
    truth: Maze,
    *,
    holonomic: bool = True,
    cost: MoveCost = DEFAULT_COST,
    time_limit_s: float = 420.0,
    max_runs: int = 5,
    time_margin: float = 1.2,
    start_facing: Direction = Direction.N,
    search_step_overhead_s: float = 0.0,
    actual_scale: float = 1.0,
    max_steps: int = 5000,
    neighbor_sensing: bool = True,
    pass_cells: int = 2,
    max_leg_cells: int = 4,
    chain_legs: int = 2,
    times: dict[str, float] | None = None,
) -> SessionResult:
    """真の迷路 ``truth`` を相手に 7 分 5 走 (クラシック競技規定) のセッションを丸ごと回す。

    ``search_step_overhead_s`` は探索の**停止 1 回あたり**の追加時間。ただし
    ``straight_time_s`` (0.83s) には既に停止の 0.44s が入っているので、**既定の 0 のまま
    でよい**。8x8 の実測で確かめてある: 南北 10 + 東西 12 セル + 20 停止 の見積もり
    33.1s に対し実測 33.3s (0.5%)。ここに 0.5s などを足すと停止を二重に数えることになり、
    26% 過大に出る。実測から求めるなら :func:`fit_search_overhead` を使う。

    ``chain_legs`` は最速・復帰で止まらずに繋ぐ区間の本数 (#80)。固定費は動作に
    付くので、繋ぐと見積もりが縮む。

    ``neighbor_sensing`` は左右の隣セルまで読むか (#89)。``pass_cells`` は**探索で**
    止まらずに続けて通過してよいセル数の上限。両方そろって初めて停止回数が減る
    (隣を読むと進行先のセルが「4 壁とも既知」になり、通過してよくなる)。

    ``max_leg_cells`` は**最速・復帰の 1 区間**の長さの上限 (#85)。別物なので混ぜない
    こと — ``pass_cells`` は「探索で何セル覗いてから止まるか」、``max_leg_cells`` は
    「補正なしで何セル走り切ってよいか」。前者は速さの話、後者は壁に擦るかの話。

    ``actual_scale`` は「実際は見積もりの何倍かかるか」。1.0 なら見積もりが定義上
    ぴったり当たるので、予算判断の余裕を試すには 1.2-1.5 を入れる。

    ``time_margin`` は :class:`RunManager` の安全率。走行を始めるかの判断だけに効く
    (小さくすると際どい走行にも出るようになる)。

    ``times`` は :class:`RunManager` の時間定数の上書き
    (``cell_time_s`` / ``lateral_cell_time_s`` / ``straight_time_s`` / ``turn_time_s``)。
    **「機体が速くなったら何面走れるか」を測るためのもの** (#87)。速度を上げても
    固定費 (停止 0.44s + カメラ) は縮まないので、**全部を同じ比率で割ってはいけない** —
    セル時間だけを速度比で割り、ランプの分だけ固定費に足すのが実態に近い。
    """
    learned = open_maze(truth.size)
    learned.start = truth.start
    learned.set_goal(truth.goal_min, truth.goal_max)   # ゴール位置は競技前から既知
    ex = Explorer(learned, cell=truth.start, facing=start_facing,
                  travel=start_facing, holonomic=holonomic)
    mgr = RunManager(ex, holonomic=holonomic, cost=cost, time_limit_s=time_limit_s,
                     max_runs=max_runs, time_margin=time_margin,
                     chain_legs=1 if not holonomic else chain_legs,
                     max_leg_cells=max_leg_cells,
                     **(times or {}))
    result = SessionResult(truth=truth, explorer=ex)
    result.limit_s = time_limit_s

    def leg_time(legs: list[Leg], facing: Direction) -> float:
        return mgr.estimate_s(legs, facing) * actual_scale

    now = 0.0
    mgr.start_search(now)
    started = now
    cells = 0

    # --- 探索ラン -----------------------------------------------------------
    stops = 0
    for _ in range(max_steps):
        ex.observe(sense(truth, ex.cell, ex.facing),
                   sense_neighbors(truth, ex.cell, ex.facing) if neighbor_sensing else None)
        try:
            steps = ex.plan_leg(pass_cells)
        except Unreachable as exc:
            result.aborted = f"探索中に到達不能: {exc}"
            mgr.abort()
            break
        if not steps:
            result.reached_goal = True
            break
        # 1 区間 = 1 停止。止まらずに通過したセルではカメラを見ないので、
        # 壁判定の時間 (search_step_overhead_s) も掛からない。
        leg = Leg(steps[0].direction, len(steps))
        now += (mgr.estimate_s([leg], ex.facing) + search_step_overhead_s) * actual_scale
        for step in steps:
            ex.advance(step)
        cells += len(steps)
        stops += 1
    else:
        result.aborted = f"{max_steps} 手でゴールに到達しなかった (現在 {ex.cell})"
        mgr.abort()

    result.records.append(RunRecord(RunPhase.SEARCH, started, now - started, cells, stops))
    result.mismatches = map_agrees(truth, ex.maze, ex.visited)

    # --- 復帰 → 最速 の繰り返し ---------------------------------------------
    cell, facing = ex.cell, ex.facing
    while result.reached_goal and mgr.phase not in (RunPhase.FINISHED,):
        home = mgr.goal_reached(now, cell, facing)
        if home is None:
            break
        started, dt = now, leg_time(home, facing)
        now += dt
        result.records.append(RunRecord(
            RunPhase.RETURN_HOME, started, dt, sum(g.cells for g in home), len(home)))
        cell, facing = truth.start, facing_after(home, facing, holonomic)

        legs = mgr.home_reached(now, facing)
        if legs is None:
            break
        now += mgr.restart_dwell_s      # 競技規定 3-4: 始点で 2 秒以上停止してから再出発
        started, dt = now, leg_time(legs, facing)
        now += dt
        result.records.append(RunRecord(
            RunPhase.SPEED, started, dt, sum(g.cells for g in legs), len(legs)))
        cell, facing = _walk(truth.start, legs), facing_after(legs, facing, holonomic)

    result.elapsed_s = now
    result.runs_used = mgr.runs_used
    return result


def fit_search_overhead(truth: Maze, measured_s: float, **kwargs) -> float:
    """探索ランの実測時間から**停止 1 回あたり**の追加時間を求める。

    移動そのものの時間は最速ランの実測定数から出るので、残りを停止回数で割れば
    「壁判定 + 位置補正 + 進路チェック」に費やしている時間になる。ASCII に
    書き起こした迷路で ``search_run`` を回した実測があるときに使う。
    """
    kwargs.pop("search_step_overhead_s", None)
    base = simulate_session(truth, search_step_overhead_s=0.0, **kwargs)
    rec = base.search
    if rec is None or rec.legs == 0:
        raise ValueError("探索の記録が無い (迷路かパラメータを確認)")
    return (measured_s - rec.duration_s) / rec.legs
