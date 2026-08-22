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
from krilly.strategy.shortest_path import DEFAULT_COST, Leg, MoveCost


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


def _walk(start: tuple[int, int], legs: list[Leg]) -> tuple[int, int]:
    """``legs`` を実行し終えたときのセル。"""
    x, y = start
    for leg in legs:
        dx, dy = leg.direction.delta
        x, y = x + dx * leg.cells, y + dy * leg.cells
    return (x, y)


def simulate_session(
    truth: Maze,
    *,
    holonomic: bool = True,
    cost: MoveCost = DEFAULT_COST,
    time_limit_s: float = 420.0,
    max_runs: int = 5,
    time_margin: float = 1.5,
    start_facing: Direction = Direction.N,
    search_step_overhead_s: float = 0.0,
    actual_scale: float = 1.0,
    max_steps: int = 5000,
    neighbor_sensing: bool = False,
    max_leg_cells: int = 1,
) -> SessionResult:
    """真の迷路 ``truth`` を相手に 7 分 5 走 (クラシック競技規定) のセッションを丸ごと回す。

    ``search_step_overhead_s`` は探索の**停止 1 回あたり**の追加時間。最速ランの実測から
    出た区間の固定費には**壁判定と位置補正の時間が入っていない**ので、探索はそのぶん
    遅い。カメラを見るのは止まったときだけなので、セル数ではなく停止回数に比例する。
    実測があれば :func:`fit_search_overhead` で求められる。

    ``neighbor_sensing`` は左右の隣セルまで読むか (#89)。``max_leg_cells`` は
    止まらずに続けて通過してよいセル数の上限。両方そろって初めて停止回数が減る
    (隣を読むと進行先のセルが「4 壁とも既知」になり、通過してよくなる)。

    ``actual_scale`` は「実際は見積もりの何倍かかるか」。1.0 なら見積もりが定義上
    ぴったり当たるので、予算判断の余裕を試すには 1.2-1.5 を入れる。

    ``time_margin`` は :class:`RunManager` の安全率。走行を始めるかの判断だけに効く
    (小さくすると際どい走行にも出るようになる)。
    """
    learned = open_maze(truth.size)
    learned.start = truth.start
    learned.set_goal(truth.goal_min, truth.goal_max)   # ゴール位置は競技前から既知
    ex = Explorer(learned, cell=truth.start, facing=start_facing,
                  travel=start_facing, holonomic=holonomic)
    mgr = RunManager(ex, holonomic=holonomic, cost=cost, time_limit_s=time_limit_s,
                     max_runs=max_runs, time_margin=time_margin)
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
            steps = ex.plan_leg(max_leg_cells)
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
