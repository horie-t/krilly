"""走行の状態機械: 待機 → 探索 → 復帰 → 最速 → … (issue #20)。

クラシック競技の制約 (**持ち時間 7 分・走行は最大 5 回**) の下で、いつ次の走行を
始めてよいかを決める純ロジック。

持ち時間は NTF のクラシックマウス競技規定による: 「マイクロマウスは 7 分間の持ち時間を
有し」「この間 5 回までの走行をすることができる」。ただし「特に必要と認められた競技会に
ついては、持ち時間を 5 分、走行回数を 5 回とすることがある」ので、**大会ごとに確認して
``time_limit_s`` を設定すること**。10 分はマイクロマウス競技 (旧ハーフサイズ、9cm 区画・
最大 32x32) の規定で、本機の出る競技のものではない。「走行」はスタート区画を出発してゴールを目指す
試行と数える (探索ランも 1 走行。ゴールからスタートへの**復帰は走行に数えない**)。

方針:

- 1 走目は必ず探索 (:class:`krilly.strategy.explorer.Explorer`)。ゴール到達で
  壁情報が確定した経路が手に入る。
- 以降は 復帰 → 最速 (確定した壁情報のみで :func:`shortest_path`) を繰り返す。
- **次の走行を始めるのは、残り時間が (復帰+最速) の見積もりに間に合うときだけ**。
  見積もりは実測由来の定数 (下記) に安全率を掛ける。途中で時間切れになるより、
  確実に完走できる走行だけを始める方が良い。

時刻は ``now`` [s] (monotonic) を呼び出し側が渡す。内部で時計を読まないので、
シミュレーションで任意の時間経過をテストできる。ハードウェアには触らない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from krilly.solver.maze import Direction
from krilly.strategy.explorer import Explorer
from krilly.strategy.shortest_path import (
    DEFAULT_COST,
    Leg,
    MoveCost,
    path_to_legs,
    shortest_path,
    turns_in,
)


class RunPhase(Enum):
    """走行の状態。"""

    WAIT = "wait"              # スタート区画で待機 (開始前)
    SEARCH = "search"          # 探索ラン (1 走目)
    RETURN_HOME = "return"     # ゴール -> スタートへ復帰 (走行に数えない)
    SPEED = "speed"            # 最速ラン (確定した壁情報のみで走る)
    FINISHED = "finished"      # 終了 (時間切れ / 5 走消化 / 続行不能)


def facing_after(legs: list[Leg], facing: Direction, holonomic: bool = True) -> Direction:
    """``legs`` を実行し終えたときの機体の向き。

    旋回レス走行 (#76) では機体は回らないので ``facing`` のまま。旋回する走り方では
    最後の区間の方角を向いている。
    """
    if holonomic or not legs:
        return facing
    return legs[-1].direction


@dataclass
class RunManager:
    """7 分・5 走の予算内で 探索 → (復帰 → 最速)×N を差配する。

    使い方 (駆動側は各イベントで 1 回ずつ呼ぶ):

    1. ``start_search(now)`` — 探索ランの出発 (1 走目)。
    2. ゴール到達で ``goal_reached(now, cell, facing)`` — 復帰経路 (Leg 列) が返れば
       :attr:`RunPhase.RETURN_HOME`、None なら :attr:`RunPhase.FINISHED` (そこで停止)。
    3. スタート到着で ``home_reached(now, facing)`` — 最速経路が返れば
       :attr:`RunPhase.SPEED` (走行数 +1)、None なら FINISHED。
    4. 以降 2. と 3. を繰り返す。
    """

    explorer: Explorer
    #: 持ち時間 [s]。クラシック競技規定の 7 分。大会によっては 5 分 (300s) なので、
    #: **競技会ごとの発表を確認して設定すること**。
    time_limit_s: float = 420.0
    max_runs: int = 5                  # 最大走行回数
    # 旋回レス走行 (#76) の 5x5 通しラン実測。speed_run が末尾に軸ごとの当てはめを出す:
    #   南北 (機体の前後軸) 1セル 0.73s + 区間 0.39s
    #   東西 (機体の左右軸) 1セル 0.75s + 区間 0.35s
    #   動作あたりの停止 0.44s (最速ラン実測 24.6s - 動作時間の合計 19.3s を 12 区間で割る)
    # 南北と東西の差は 1 セルで 15ms しかない (横移動は前進より 4% 高い輪速を要するが、
    # MAX_SPEED に収まっているので速度は同じ)。
    # 旋回する走り方 (#21 の実測) は 1セル 0.75s / 区間 0.36s / 90°旋回 1.76s。
    # **速度設定を変えたら speed_run の末尾に出る実測表で測り直すこと。**
    cell_time_s: float = 0.73          # 南北 1 セルあたり増分
    straight_time_s: float = 0.81      # 区間 1 本あたりの固定費 (ランプ + 整定 + 撮影 + 停止)
    turn_time_s: float = 2.18          # 90° 旋回 1 回 (旋回する走り方のみ)
    time_margin: float = 1.5           # 見積もりに掛ける安全率
    #: スタート区画へ戻ってから再スタートするまでの停止時間 [s]。
    #: **競技規定 3-4 の要求**: 「マイクロマウスが始点に戻り、自動的に再スタートする
    #: 場合、始点において 2 秒以上停止しなければならない」。守らないと走行が無効に
    #: なりうるので、これは最適化の対象ではない。持ち時間からは確実に引かれるので
    #: 見積もりにも入れる。
    restart_dwell_s: float = 2.0
    #: 東西 (旋回レスでは機体の左右軸) へ 1 セル進む時間。
    lateral_cell_time_s: float = 0.75
    #: True なら機体を旋回させない (#76)。旋回の時間は見積もりに入らない。
    holonomic: bool = True
    cost: MoveCost = DEFAULT_COST

    phase: RunPhase = field(default=RunPhase.WAIT, init=False)
    runs_used: int = field(default=0, init=False)
    started_at: float | None = field(default=None, init=False)

    # -- 時間 -----------------------------------------------------------------
    def elapsed_s(self, now: float) -> float:
        """1 走目の出発からの経過時間 (開始前は 0)。"""
        return 0.0 if self.started_at is None else now - self.started_at

    def remaining_s(self, now: float) -> float:
        return self.time_limit_s - self.elapsed_s(now)

    def estimate_s(self, legs: list[Leg], facing: Direction = Direction.N) -> float:
        """Leg 列の所要時間の見積もり (安全率は掛けない素の値)。

        セル数だけでなく**区間の本数**も数える。連続直進はランプの固定費を償却するので、
        同じセル数でも区間が細切れなほど時間がかかる。

        セルは進行軸で分ける。旋回レス走行 (#76) では機体の向きが固定なので、南北は
        機体の前後軸・東西は左右軸の移動になり、所要時間が違いうる。旋回する走り方
        (``holonomic=False``) では常に前を向いて進むので両者は同じで、代わりに旋回の
        時間が乗る。
        """
        ns = sum(leg.cells for leg in legs if leg.direction in (Direction.N, Direction.S))
        ew = sum(leg.cells for leg in legs if leg.direction in (Direction.E, Direction.W))
        # 旋回レスでは東西が機体の左右軸 (横移動) になる。旋回するなら常に前を向いて
        # 進むので、東西も南北も同じ時間。
        ew_time = self.lateral_cell_time_s if self.holonomic else self.cell_time_s
        total = ns * self.cell_time_s + ew * ew_time + len(legs) * self.straight_time_s
        if not self.holonomic:
            total += turns_in(legs, facing) * self.turn_time_s
        return total

    # -- 経路 -----------------------------------------------------------------
    def _route(
        self, start: tuple[int, int], goals: list[tuple[int, int]], facing: Direction
    ) -> list[Leg] | None:
        """観測済みセルだけを通る最小コスト経路。繋がっていなければ None。"""
        path = shortest_path(
            self.explorer.maze, start, goals,
            start_facing=facing, known=self.explorer.visited, cost=self.cost,
        )
        return path_to_legs(path) if path else None

    def speed_legs(self, facing: Direction) -> list[Leg] | None:
        """スタート -> ゴールの最速経路 (スタート区画で ``facing`` を向いている前提)。"""
        maze = self.explorer.maze
        return self._route(maze.start, maze.goal_cells(), facing)

    def return_legs(self, cell: tuple[int, int], facing: Direction) -> list[Leg] | None:
        """現在地 -> スタートの復帰経路。"""
        return self._route(cell, [self.explorer.maze.start], facing)

    # -- イベント ---------------------------------------------------------------
    def start_search(self, now: float) -> None:
        """探索ラン (1 走目) の出発。WAIT 以外から呼ぶのは誤り。"""
        if self.phase is not RunPhase.WAIT:
            raise RuntimeError(f"探索は WAIT からのみ開始できる (現在 {self.phase})")
        self.started_at = now
        self.runs_used = 1
        self.phase = RunPhase.SEARCH

    def goal_reached(
        self, now: float, cell: tuple[int, int], facing: Direction
    ) -> list[Leg] | None:
        """ゴール到達。次に回す価値があれば復帰経路を返す (無ければ FINISHED)。

        「価値がある」= 走行回数が残っており、かつ 復帰 + 次の最速 の見積もり
        (安全率込み) が残り時間に収まること。満たさなければゴールに留まって終了する
        (中途半端に走り出して時間切れになるより良い)。
        """
        if self.phase not in (RunPhase.SEARCH, RunPhase.SPEED):
            raise RuntimeError(f"goal_reached は走行中のみ (現在 {self.phase})")
        if self.runs_used < self.max_runs:
            home = self.return_legs(cell, facing)
            if home is not None:
                next_speed = self.speed_legs(facing_after(home, facing))
                if next_speed is not None:
                    budget = self.restart_dwell_s + self.time_margin * (
                        self.estimate_s(home) + self.estimate_s(next_speed)
                    )
                    if self.remaining_s(now) >= budget:
                        self.phase = RunPhase.RETURN_HOME
                        return home
        self.phase = RunPhase.FINISHED
        return None

    def home_reached(self, now: float, facing: Direction) -> list[Leg] | None:
        """スタート到着。最速ランを始められれば経路を返す (走行数 +1)。"""
        if self.phase is not RunPhase.RETURN_HOME:
            raise RuntimeError(f"home_reached は復帰中のみ (現在 {self.phase})")
        legs = self.speed_legs(facing)
        if (
            legs is not None
            and self.runs_used < self.max_runs
            and self.remaining_s(now) >= (self.restart_dwell_s
                                          + self.time_margin * self.estimate_s(legs))
        ):
            self.runs_used += 1
            self.phase = RunPhase.SPEED
            return legs
        self.phase = RunPhase.FINISHED
        return None

    def abort(self) -> None:
        """続行不能 (安全チェック・到達不能など)。以後は FINISHED。"""
        self.phase = RunPhase.FINISHED

    def summary(self, now: float) -> str:
        return (
            f"走行 {self.runs_used}/{self.max_runs} 回 / "
            f"経過 {self.elapsed_s(now):.0f}s / 残り {self.remaining_s(now):.0f}s / "
            f"状態 {self.phase.value}"
        )
