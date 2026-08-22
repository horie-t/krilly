"""探索ランの進行管理: 壁の観測 → flood-fill → 次の 1 手 (issue #18)。

:mod:`krilly.strategy.flood_fill` の距離計算と、実機の動作プリミティブ
(:class:`krilly.motion.cell_motion.CellMotion` の 90°ターン + 1セル前進) の間を
つなぐ層。**離散の世界 (セル座標・方角) だけを扱い、ハードウェアには触らない**ので、
迷路を与えれば探索そのものをオフPiでシミュレートしてテストできる。

1 ステップの流れ:

1. :meth:`Explorer.observe` — カメラ判定の機体相対の壁を迷路方角へ写して Maze に反映
2. :meth:`Explorer.plan` — flood-fill を塗り直し、次に進む方角と旋回量を返す
3. 呼び出し側が旋回 + 1セル前進を実行
4. :meth:`Explorer.advance` — 内部のセル・方角を進める

ゴール到達で :meth:`plan` は None を返す。**探索の完了 (最短経路の確定) や
スタートへの復帰は本 issue の範囲外**で、#19 (最短経路) / #20 (状態機械) で扱う。

迷路グリッドと世界座標の対応 (docs/coordinate-frames.md):
セル (x, y) の中心は世界座標 ``(x * pitch, y * pitch)``、方角 N/E/S/W の機体方位 φ は
N=+90° / E=0° / S=-90° / W=180°。スタートは (0, 0) で北向き = φ=+90°。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from krilly.perception.wall_detect import maze_direction_for
from krilly.solver.maze import Direction, Maze
from krilly.strategy.flood_fill import UNREACHABLE, flood_fill, next_direction

# 方角 -> 機体方位 φ [rad] (迷路 東=+x/北=+y、車体 +x前/+y左、+ω=CCW)
_HEADING = {
    Direction.N: math.pi / 2,
    Direction.E: 0.0,
    Direction.S: -math.pi / 2,
    Direction.W: math.pi,
}


def heading_rad(d: Direction) -> float:
    """方角に対応する機体方位 φ [rad] (N=+90°, E=0°, S=-90°, W=180°)。"""
    return _HEADING[d]


def cell_center(cell: tuple[int, int], pitch: float) -> tuple[float, float]:
    """セル (x, y) の中心の世界座標 [m] (スタートセル (0,0) の中心を原点とする)。"""
    return (cell[0] * pitch, cell[1] * pitch)


def quarter_turns(facing: Direction, target: Direction) -> int:
    """``facing`` から ``target`` へ向くのに必要な 90° 旋回数 (+CCW / -CW)。

    戻り値は -1, 0, +1, 2 のいずれか (180° は +2 = 左に 2 回で表す)。
    """
    cw = (target - facing) % 4          # 時計回りに何回で合うか (0..3)
    return {0: 0, 1: -1, 2: 2, 3: 1}[cw]


class Unreachable(RuntimeError):
    """観測した壁により、現在セルからゴールへ到達できない。"""


@dataclass(frozen=True)
class Step:
    """次の 1 手 (``direction`` の方角へ 1 セル進む)。

    旋回する走り方では「その方角を向いてから前進」、旋回レス走行 (#76) では
    「向きを変えずにその方角へ平行移動」。**旋回量は持たない** — 同じ名前で
    「機体の旋回量」と「進行軸の変更量」の両方を表すと、走り方を混ぜたときに
    符号は合うのに意味が違うという最悪のバグになる。旋回量が要る側 (旧モードの
    実行と見積もり) は :func:`quarter_turns` で方角から計算する。
    """

    direction: Direction   # 進む方角
    to_cell: tuple[int, int]   # 進んだ後のセル


@dataclass
class Explorer:
    """flood-fill 探索ランの状態 (迷路・現在セル・向き・訪問済み)。"""

    maze: Maze
    cell: tuple[int, int] = (0, 0)
    facing: Direction = Direction.N
    #: 直前に進んだ方角。旋回レス走行 (#76) では機体の向きが変わらないので、
    #: 「進行軸を変えない方を優先する」タイブレークにはこちらを使う。
    travel: Direction = Direction.N
    #: True なら機体を旋回させない (向きは facing に固定されたまま平行移動する)。
    holonomic: bool = True
    visited: set[tuple[int, int]] = field(default_factory=set)
    steps: int = 0
    # 一度観測した壁は「無し」の観測で消さない。壁の見落とし (偽陰性) は壁に突っ込む
    # 事故になり、偽陽性 (回り道が増えるだけ) より高くつくため既定で有効。
    sticky_walls: bool = True
    conflicts: int = 0   # 既知の壁を「無し」と観測した回数 (姿勢誤差・誤検出の目安)
    #: 4 壁すべてが観測で確定したセル (#89)。**未観測の壁を持つセルは入らない**ので、
    #: 「止まらずに通過してよいか」と「最速ランで通してよいか」の判断に使える。
    #: 自セルを観測すると隣接 4 セルの 1 辺ずつも確定するので、``visited`` より広い。
    known: set[tuple[int, int]] = field(default_factory=set)
    #: セル -> 観測で確定した辺の集合 (:attr:`known` の内訳)。
    observed: dict[tuple[int, int], set[Direction]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.visited.add(self.cell)

    # -- 観測 ---------------------------------------------------------------
    def observe(self, walls_body: dict[str, bool],
                neighbors: dict[str, dict[str, bool]] | None = None
                ) -> dict[Direction, bool]:
        """機体相対の壁有無 (front/back/left/right) を現在セルに反映する。

        戻り値は迷路方角に写した壁有無 (カメラが何と言ったかのログ用)。共有エッジ
        なので隣接セルにも入る。反映には 2 つの安全弁がある:

        - **外周は書かない**: 迷路の外へ出る辺は既知なので、カメラ判定で消さない
          (実機では自機のリボンケーブルや影で外周を「壁なし」と誤判定しうる)
        - ``sticky_walls`` が真なら、既知の壁を「無し」と観測しても消さずに
          :attr:`conflicts` を数える (姿勢がずれている / 誤検出しているサイン)

        ``neighbors`` は隣のセルの壁 (#89)。``{機体から見た隣の向き: そのセルの
        機体相対の壁}`` で、全画素モードでは左右の隣セルの 4 壁まで読める
        (:meth:`krilly.perception.wall_detect.WallDetector.neighbor_walls`)。
        **確定しなかった辺はキーごと落とすこと** — 「壁なし」として渡すと、その
        セルが :attr:`known` に化けて通過対象になってしまう。
        """
        walls_maze = self._apply(self.cell, walls_body)
        for side, walls in (neighbors or {}).items():
            cell = self.maze.neighbor(*self.cell, maze_direction_for(side, self.facing))
            if self.maze.in_bounds(*cell):
                self._apply(cell, walls)
        return walls_maze

    def _apply(self, cell: tuple[int, int],
               walls_body: dict[str, bool]) -> dict[Direction, bool]:
        """1 セル分の機体相対の壁観測を迷路へ反映する (部分的な観測も受け付ける)。"""
        walls_maze = {maze_direction_for(edge, self.facing): present
                      for edge, present in walls_body.items()}
        x, y = cell
        for d, present in walls_maze.items():
            self._mark_observed(cell, d)
            if not self.maze.in_bounds(*self.maze.neighbor(x, y, d)):
                continue                      # 外周は既知なので触らない
            if not present and self.maze.has_wall(x, y, d):
                self.conflicts += 1
                if self.sticky_walls:
                    continue
            self.maze.set_wall(x, y, d, present)
        return walls_maze

    def _mark_observed(self, cell: tuple[int, int], d: Direction) -> None:
        """辺 ``d`` を観測済みにする。壁は共有なので**隣のセルの裏側も**確定する。"""
        for c, side in ((cell, d), (self.maze.neighbor(*cell, d), d.opposite)):
            if not self.maze.in_bounds(*c):
                continue
            edges = self.observed.setdefault(c, set())
            edges.add(side)
            if self._fully_observed(c, edges):
                self.known.add(c)

    def _fully_observed(self, cell: tuple[int, int], edges: set[Direction]) -> bool:
        """4 辺すべてが分かっているか。迷路の外へ出る辺は観測不要 (既知)。"""
        return all(d in edges or not self.maze.in_bounds(*self.maze.neighbor(*cell, d))
                   for d in Direction)

    def is_known(self, cell: tuple[int, int]) -> bool:
        """そのセルの 4 壁がすべて観測で確定しているか (#89)。"""
        return cell in self.known

    # -- 計画 ---------------------------------------------------------------
    @property
    def at_goal(self) -> bool:
        return self.maze.is_goal(*self.cell)

    def distances(self) -> list[list[int]]:
        """現在の壁情報での flood-fill 距離 (未探索は通行可と楽観視)。"""
        return flood_fill(self.maze)

    def plan(self) -> Step | None:
        """次の 1 手を返す。ゴール到達なら None。到達不能なら Unreachable。"""
        if self.at_goal:
            return None
        return self._plan_from(self.cell, self.travel, self.visited)

    def _plan_from(self, cell: tuple[int, int], travel: Direction,
                   visited: set[tuple[int, int]]) -> Step:
        """``cell`` に居るものとして 1 手を選ぶ (先読み用に状態を引数で受ける)。"""
        dist = self.distances()
        # タイブレークの基準: 旋回レスなら「直前に進んだ方角」、旋回するなら「機体の向き」。
        # facing 固定のまま facing を渡すと、迷路と無関係に北を最優先する静的な偏りになる。
        reference = travel if self.holonomic else self.facing
        d = next_direction(self.maze, cell, reference, dist, visited)
        if d is None:
            raise Unreachable(
                f"セル {cell} からゴールへ到達できない "
                f"(距離={dist[cell[0]][cell[1]]})"
            )
        return Step(d, self.maze.neighbor(*cell, d))

    def plan_leg(self, max_cells: int = 1) -> list[Step]:
        """**止まらずに続けて実行できる 1 区間**を返す (#89)。ゴール到達なら空。

        1 手目は :meth:`plan` と同じ。2 手目以降は、通過するセルが

        1. :attr:`known` (4 壁が観測済み) — 通っても新しく見えるものが無く、かつ
           **その先へ抜ける辺が「壁なし」と観測されている**のが保証される
        2. ゴールでない (ゴールでは止まる)
        3. そこからの 1 手が**同じ方角** (向きを変えるなら結局止まる)

        を満たす限り伸ばす。``max_cells`` が上限 (1 なら従来どおり 1 セルずつ)。

        通過したセルの隣は観測できないので**情報は減る**が、大会迷路 31 面の
        シミュレーションでは経路長はほとんど変わらず、停止回数だけが減る。

        上限を設ける理由は 2 つ: :class:`~krilly.motion.cell_motion.CellMotion` の
        1 動作が長くなるほど誤差が乗ること、位置補正が停止時にしか入らないこと。
        """
        if self.at_goal:
            return []
        steps = [self._plan_from(self.cell, self.travel, self.visited)]
        visited = set(self.visited)
        while len(steps) < max_cells:
            cell = steps[-1].to_cell
            visited.add(cell)
            if not self.is_known(cell) or self.maze.is_goal(*cell):
                break
            try:
                nxt = self._plan_from(cell, steps[-1].direction, visited)
            except Unreachable:
                break
            if nxt.direction is not steps[-1].direction:
                break
            steps.append(nxt)
        return steps

    # -- 前進 ---------------------------------------------------------------
    def advance(self, step: Step) -> tuple[int, int]:
        """1 手を実行したものとして内部状態を進める。新しいセルを返す。"""
        self.travel = step.direction
        if not self.holonomic:
            self.facing = step.direction   # 旋回する走り方だけ機体の向きが変わる
        self.cell = step.to_cell
        self.visited.add(self.cell)
        self.steps += 1
        return self.cell

    # -- 進捗 ---------------------------------------------------------------
    def goal_distance(self) -> int:
        """現在セルからゴールまでの flood-fill 距離 (到達不能なら UNREACHABLE)。"""
        return self.distances()[self.cell[0]][self.cell[1]]

    def progress(self) -> str:
        """ログ用の 1 行サマリ。"""
        d = self.goal_distance()
        dist_text = "到達不能" if d >= UNREACHABLE else f"{d}"
        return (
            f"step={self.steps} セル={self.cell} 向き={self.facing.name} "
            f"ゴール距離={dist_text} 訪問={len(self.visited)}"
        )
