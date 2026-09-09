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
    # 旋回レス走行 (#76) の実測。speed_run が末尾に軸ごとの当てはめを出す。
    # 値は **#103 の 8x8 対照セッション** (同じ日・同じ電池で v=0.24 と v=0.30 m/s を
    # 続けて 5 走ずつ / chain-legs 1 / 最速経路は 南北 10 + 東西 10 セル / 区間 15 本)
    # の **0.30 m/s 側**:
    #   南北 (機体の前後軸) 1セル 0.61s / 東西 (機体の左右軸) 1セル 0.60s
    #   動作あたりの固定費 0.90s (最速ラン実測 25.60s - セル分 12.10s を 15 動作で割る)
    # 0.24 m/s 側は 0.77 / 0.75 / 0.825s。**この 2 点は台形モデルに合う**ので外挿できる:
    # セル時間は速度比ぴったり (0.77 -> 0.61 = -21%、0.24/0.30 = 0.80)、固定費の増分は
    # ランプの超過 v/2*(1/accel + 1/decel) の差 +0.071s に対し実測 +0.08 / +0.07s。
    # つまり**増えたぶんは全部ランプで、停止 + 撮影の 0.55s は速度に依存しない**。
    # 南北と東西の差は 1 セルで 10ms しかない (横移動は前進より 4% 高い輪速を要するが、
    # MAX_SPEED に収まっているので速度は同じ)。
    # **固定費が「区間」ではなく「動作」に付くこと**は #80 の対で確かめている
    # (どちらも 0.24 m/s の 5x5、20 セル / 区間 12 本):
    #   chain 1: 25.0s - セル分 15.0s = 10.0s / 12 動作 = 0.83s
    #   chain 2: 20.0s - セル分 15.0s =  5.0s /  6 動作 = 0.83s
    # 旋回する走り方 (#21 の実測) は 1セル 0.75s / 区間 0.36s / 90°旋回 1.76s。
    # **速度設定を変えたら speed_run の末尾に出る実測表で測り直すこと。**
    # 見積もりは機体の速度と対でしか意味を持たない: ここが 0.30 用なのに 0.24 で走ると
    # 見積もりが 7% 低く出て、**終われない走行を始める**向きに外れる (逆は断る側)。
    cell_time_s: float = 0.61          # 南北 1 セルあたり増分
    straight_time_s: float = 0.90      # 動作 1 回あたりの固定費 (ランプ + 整定 + 撮影 + 停止)
    turn_time_s: float = 2.18          # 90° 旋回 1 回 (旋回する走り方のみ)
    #: 見積もりに掛ける安全率。**1.5 から 1.2 へ下げた** (#87)。
    #:
    #: これは「始めた走行を時間内に終えられるか」だけを守っている。守りすぎると
    #: **走れたはずの最速ランを断る**方に外れ、規定 3-1 では記録は最速の 1 走行
    #: なので、時間切れの最速ランは時間を失うだけ (それまでの記録は残る) —
    #: **断る方が高くつく。**
    #:
    #: 大会迷路 31 面 (#84) を安全率 x「実機が見積もりの何倍かかるか」で掃引した結果。
    #: 「最速 0 本」= 最速ランを 1 本も走れない面、「総走行」= 31 面の最速ラン合計、
    #: 「超過」= 7 分を超えた面:
    #:
    #:   実機/見積もり     1.0 倍          1.1 倍          1.2 倍          1.4 倍
    #:   安全率      最速0本 総 超過  最速0本 総 超過  最速0本 総 超過  最速0本 総 超過
    #:    1.5          7/30  30  0     8/30  27  0     9/30  24  0    12/30  19  0
    #:    1.4          3/30  37  0     7/30  28  0     8/30  26  0    10/30  21  0
    #:    1.2          3/30  38  0     3/30  36  0     4/30  31  0     9/30  22  1
    #:    1.1          2/30  42  0     3/30  37  0     4/30  31  0     9/30  23  2
    #:    1.0          0/30  46  0     2/30  39  2     4/30  34  3     9/30  23  2
    #:
    #: 再現: ``maze_sim --maze mazes/contest/*.txt --budget-sweep``
    #: (31 面のうち健全性チェックを通る 30 面)。
    #:
    #: **1.2 は 1.5 をあらゆる列で上回る**: 走れない面が減り、走行数が増え、しかも
    #: 実機が見積もりより 20% 遅くても 7 分を超えた面はゼロ。実測の見積もり誤差は
    #: 8x8 で **0.5%** (探索 33.3s 対 33.1s、最速 21.7s 対 21.6s) なので 20% は 40 倍の余裕。
    #: 1.0 まで下げると 10% 遅いだけで 2 面が超過するので、そこは行き過ぎ。
    #:
    #: **1.5 は崖の上に乗っていた**: 1.4 へ下げるだけで 4 面 (2011 exp 予選 /
    #: 2013 exp 決勝 / 2018 / 2022) が走れるようになる。1.2 で残る 3 面
    #: (2014/2015/2017 の exp 決勝) は探索そのものが長すぎる面で、安全率では届かない。
    time_margin: float = 1.2
    #: スタート区画へ戻ってから再スタートするまでの停止時間 [s]。
    #: **競技規定 3-4 の要求**: 「マイクロマウスが始点に戻り、自動的に再スタートする
    #: 場合、始点において 2 秒以上停止しなければならない」。守らないと走行が無効に
    #: なりうるので、これは最適化の対象ではない。持ち時間からは確実に引かれるので
    #: 見積もりにも入れる。
    restart_dwell_s: float = 2.0
    #: 東西 (旋回レスでは機体の左右軸) へ 1 セル進む時間。
    lateral_cell_time_s: float = 0.60
    #: True なら機体を旋回させない (#76)。旋回の時間は見積もりに入らない。
    holonomic: bool = True
    #: 止まらずに繋ぐ区間の本数の上限 (#80)。**固定費は区間ではなく「動作」に付く**
    #: ので、繋げばそのぶん見積もりが縮む。1 なら従来 (区間ごとに停止)。
    #: 5x5 実測: 20 セル / 12 区間を 2 本ずつ繋いで 6 動作、20.0s (見積 20.0s)。
    #: 同じ経路を区間ごとに止まると 25.0s なので **-20%**。
    #:
    #: **既定が 1 なのは安全側だから。** 実機は 2 で走る (``speed_run --chain-legs``)
    #: が、ここを 2 にすると「繋がない呼び出し側が渡し忘れた」ときに見積もりが
    #: **短い方へ**外れ、予算判断が終われない走行を始めてしまう。1 なら外れ方は
    #: 「走れる走行を断る」側になる。渡し忘れが高くつかない向きに倒しておく。
    chain_legs: int = 1
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

        セル数だけでなく**動作の回数**も数える (:meth:`motions`)。連続直進はランプの
        固定費を償却するので、同じセル数でも動作が細切れなほど時間がかかる。
        ``chain_legs`` > 1 ならコーナーを丸めて繋ぐぶん動作が減る (#80)。

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
        total = ns * self.cell_time_s + ew * ew_time + self.motions(legs) * self.straight_time_s
        if not self.holonomic:
            total += turns_in(legs, facing) * self.turn_time_s
        return total

    def motions(self, legs: list[Leg]) -> int:
        """``legs`` を実行するのに必要な**動作の回数** (#80)。

        固定費 (ランプ + 整定 + 撮影 + 停止) は区間ではなく動作に付く。コーナーを
        丸めて繋げば、区間の本数はそのままでも動作は減る。実測でも繋いだ動作の
        所要は「セル数 × 1 セルの時間 + 固定費 1 回分」で、コーナーの有無で変わらない
        (コーナーのブレンドは、どのみち必要な減速を次の区間の加速と重ねただけだから)。
        """
        size = max(1, self.chain_legs)
        return -(-len(legs) // size)          # 切り上げ

    # -- 経路 -----------------------------------------------------------------
    def _route(
        self, start: tuple[int, int], goals: list[tuple[int, int]], facing: Direction
    ) -> list[Leg] | None:
        """観測済みセルだけを通る最小コスト経路。繋がっていなければ None。

        通してよいのは **4 壁が観測で確定したセル** (:attr:`Explorer.known`)。
        通ったセルは必ずそこに入るので ``visited`` はその部分集合だが、隣のセルを
        読めば (#89) **入っていないセルも確定する**ので、そのぶん経路の選択肢が増える。
        未確定のセルを最速で走るのは、見ていない壁に突っ込むということ。
        """
        path = shortest_path(
            self.explorer.maze, start, goals,
            start_facing=facing, known=self.explorer.known, cost=self.cost,
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
