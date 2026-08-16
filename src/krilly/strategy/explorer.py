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

from krilly.perception.wall_detect import body_walls_to_maze
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

    def __post_init__(self) -> None:
        self.visited.add(self.cell)

    # -- 観測 ---------------------------------------------------------------
    def observe(self, walls_body: dict[str, bool]) -> dict[Direction, bool]:
        """機体相対の壁有無 (front/back/left/right) を現在セルに反映する。

        戻り値は迷路方角に写した壁有無 (カメラが何と言ったかのログ用)。共有エッジ
        なので隣接セルにも入る。反映には 2 つの安全弁がある:

        - **外周は書かない**: 迷路の外へ出る辺は既知なので、カメラ判定で消さない
          (実機では自機のリボンケーブルや影で外周を「壁なし」と誤判定しうる)
        - ``sticky_walls`` が真なら、既知の壁を「無し」と観測しても消さずに
          :attr:`conflicts` を数える (姿勢がずれている / 誤検出しているサイン)
        """
        walls_maze = body_walls_to_maze(walls_body, self.facing)
        x, y = self.cell
        for d, present in walls_maze.items():
            if not self.maze.in_bounds(*self.maze.neighbor(x, y, d)):
                continue                      # 外周は既知なので触らない
            if not present and self.maze.has_wall(x, y, d):
                self.conflicts += 1
                if self.sticky_walls:
                    continue
            self.maze.set_wall(x, y, d, present)
        return walls_maze

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
        dist = self.distances()
        # タイブレークの基準: 旋回レスなら「直前に進んだ方角」、旋回するなら「機体の向き」。
        # facing 固定のまま facing を渡すと、迷路と無関係に北を最優先する静的な偏りになる。
        reference = self.travel if self.holonomic else self.facing
        d = next_direction(self.maze, self.cell, reference, dist, self.visited)
        if d is None:
            raise Unreachable(
                f"セル {self.cell} からゴールへ到達できない "
                f"(距離={dist[self.cell[0]][self.cell[1]]})"
            )
        return Step(d, self.maze.neighbor(*self.cell, d))

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
