"""大会迷路から小さい盤面を切り出し、実機で試す価値を測る (issue #85 / #23)。

**16x16 を組まなくてよい理由**: カメラは 1 セルが画角いっぱいに写り、``CellMotion`` は
1 セル移動と複数セルの区間しか知らない。効くのは**局所の壁パターンとその並び**であって
16x16 という全体の形ではない。だから #84 で書き起こした実迷路から難しい部分を切り出し、
手持ちの壁と柱で組める大きさにすればよい。

切り出しは窓を当てるだけだが、**外周を閉じると迷路の性質が変わる**ので、そのままでは
使えない。到達不能なセルや壁の付かない柱ができるため、:func:`~krilly.sim.check.check_maze`
で弾く必要がある。

測る指標は「5x5 では一度も踏んでいない状況」に対応させてある (#85):

- **補正が入らないセル**: 旋回レス走行では機体は常に北を向くので、LEFT/RIGHT の ROI が
  東西の壁 → **東西 (X) の位置補正**、FRONT/BACK が南北の壁 → **南北 (Y) の位置補正**を
  与える。壁の無い辺には赤帯が無いので、その軸は補正できない。
- **それが経路上で連続する長さ**: 1 セルなら次で取り返せるが、続くとその間の横方向は
  完全にオドメトリ任せになる。ここが実機でしか試せない部分。
- **四辺とも壁が無いセル**: カメラに赤帯が 1 本も写らない。壁判定も位置補正も何も
  できないセルを通れるか。
"""

from __future__ import annotations

from dataclasses import dataclass

from krilly.sim.check import goal_entrances, goal_interior_walls
from krilly.solver.maze import Direction, Maze

def pieces_needed(size: int, goal_2x2: bool = True) -> tuple[int, int]:
    """``size``×``size`` を**どんなレイアウトでも**組むのに要る (壁, 柱) の数。

    壁は 外周 ``4N`` + 内壁の上限 ``(N-1)^2`` = ``(N+1)^2``。柱は格子点の数が
    同じ ``(N+1)^2`` だが、**2x2 のゴールの中央には柱を立てない**ので 1 本少ない
    (NTF クラシック競技規定 9:「迷路の終点となる4区画内には壁や柱は存在しない。」)。
    """
    corners = (size + 1) ** 2
    return (corners, corners - (1 if goal_2x2 else 0))


def excerpt(source: Maze, x0: int, y0: int, size: int) -> Maze:
    """``source`` の (x0, y0) から ``size``×``size`` を切り出す。

    切り出した窓の内壁をそのまま写し、外周を閉じる。スタートは (0,0)、ゴールは
    ``Maze`` の既定 (偶数サイズなら中央 2x2)。
    """
    out = Maze(size)
    for i in range(size):
        for j in range(size):
            for d in Direction:
                if source.has_wall(x0 + i, y0 + j, d):
                    out.set_wall(i, j, d)
    out.set_outer_walls()
    return out


def goal_variants(maze: Maze):
    """ゴール区画を**競技の形**にした迷路を、入口の選び方ごとに返す。

    切り出しは「元の迷路では普通のセルだった場所」をゴールと宣言するので、そのままでは
    内側に壁が残り入口も複数になる。競技のゴールは **2x2 が 1 つの開いた区画で、入口は
    1 つ** (大会迷路 31 面のうち 23 面が 1 つ、残りも 2-3 つ)。

    内側の壁を外し、開いている外周の辺を 1 つだけ残して他を閉じる。どれを残すかで
    迷路の難しさが変わるので、**選ぶのは呼び出し側の仕事**にしてある。
    """
    goals = set(maze.goal_cells())
    base = Maze(maze.size)
    base.start = maze.start
    base.set_goal(maze.goal_min, maze.goal_max)
    for x in range(maze.size):
        for y in range(maze.size):
            for d in Direction:
                if maze.has_wall(x, y, d):
                    base.set_wall(x, y, d)
    for cell in goals:                      # ゴールの内側を開ける
        for d in Direction:
            if base.neighbor(*cell, d) in goals:
                base.set_wall(*cell, d, False)
    doors = goal_entrances(base)
    for keep in doors:
        out = Maze(maze.size)
        out.start = base.start
        out.set_goal(base.goal_min, base.goal_max)
        for x in range(maze.size):
            for y in range(maze.size):
                for d in Direction:
                    if base.has_wall(x, y, d):
                        out.set_wall(x, y, d)
        for cell, d in doors:               # 残す 1 つ以外を閉じる
            if (cell, d) != keep:
                out.set_wall(cell[0], cell[1], d)
        yield out


def excerpts(source: Maze, size: int, step: int = 1):
    """``source`` に窓を総当たりで当てて切り出す。``(x0, y0, Maze)`` を返す。"""
    for x0 in range(0, source.size - size + 1, step):
        for y0 in range(0, source.size - size + 1, step):
            yield (x0, y0, excerpt(source, x0, y0, size))


def correctable(maze: Maze, cell: tuple[int, int]) -> tuple[bool, bool]:
    """そのセルで (東西の位置補正ができるか, 南北ができるか)。

    壁が 1 枚でもあればその軸は測れる (対向 2 枚なら回転成分も落ちるが、
    ここでは有無だけを見る)。
    """
    x, y = cell
    return (maze.has_wall(x, y, Direction.E) or maze.has_wall(x, y, Direction.W),
            maze.has_wall(x, y, Direction.N) or maze.has_wall(x, y, Direction.S))


def longest_blind_run(maze: Maze, path: list[tuple[int, int]], axis: int) -> int:
    """``path`` 上で、その軸の補正が入らないセルが**連続する**最大数。

    ``axis`` は 0 = 東西 (X)、1 = 南北 (Y)。1 セルなら次のセルで取り返せるが、
    続くとその間はオドメトリ任せになる。
    """
    best = run = 0
    for cell in path:
        run = 0 if correctable(maze, cell)[axis] else run + 1
        best = max(best, run)
    return best


def longest_blind_cross_run(maze: Maze, path: list[tuple[int, int]]) -> int:
    """経路上で**進行方向と直交する向き**の補正が入らない連続セル数。

    **危ないのはこちらだけ。** 「補正が入らない」には性質の違う 2 つが混ざっている:

    - 進行方向 (along) に補正が入らない → 距離が狂う。止まる位置がずれるが、
      次に直交する壁を見れば取り返せるし、そもそも壁には当たらない
    - **直交方向 (cross) に補正が入らない → 横に流れる。廊下の余裕は片側 21.4mm
      しかないので、これが続くと壁に当たる**

    南北へ進むセルでは東西の壁が、東西へ進むセルでは南北の壁が「直交」にあたる。
    シミュレーションは誤差を持たないのでここを再現できない。**実機でしか試せない。**
    """
    best = run = 0
    for previous, cell in zip(path, path[1:]):
        vertical = previous[0] == cell[0]        # 南北へ進んだ = 直交は東西 (X)
        run = 0 if correctable(maze, cell)[0 if vertical else 1] else run + 1
        best = max(best, run)
    return best


@dataclass(frozen=True)
class Difficulty:
    """実機で試す価値を表す指標 (大きいほど厳しい)。"""

    walls: int
    posts: int             # 実際に立てる柱の本数 (ゴール中央の 1 本を含まない)
    search_steps: int          # 探索の手数
    path_cells: int            # 最速経路のセル数
    legs: int                  # 最速経路の区間数
    longest_leg: int           # 最長の直進 [セル]
    blind_x: int               # 経路上で東西の補正が入らない連続セル数
    blind_y: int               # 同 南北
    blind_cross: int           # **進行方向と直交する**補正が入らない連続 (危険なのはこれ)
    no_wall_cells: int         # 四辺とも壁が無いセル
    chained_motions: int = 0   # 25 走セッションで得られる連結動作の本数 (:attr:`exposure`)

    def describe(self) -> str:
        return (f"壁 {self.walls} 柱 {self.posts} / 探索 {self.search_steps} 手 / "
                f"最速 {self.path_cells} セル {self.legs} 区間 (最長 {self.longest_leg}) / "
                f"補正なし連続 直交{self.blind_cross} (X{self.blind_x} Y{self.blind_y}) / "
                f"無壁セル {self.no_wall_cells} / 連結動作 {self.chained_motions}")

    @property
    def score(self) -> tuple:
        """並べ替え用。**進行方向と直交する補正が入らない連続**を最優先にする。

        速さの検証はシミュレーションでできる。実機でしか分からないのは誤差の蓄積で、
        しかも走行を終わらせるのは横方向の流れ (廊下の余裕は片側 21.4mm) なので、
        そこが長いレイアウトほど価値がある。along 方向の欠落は距離が狂うだけで
        壁には当たらないため、優先度を落とす。
        """
        return (self.blind_cross, self.no_wall_cells, self.search_steps,
                self.path_cells)

    @property
    def exposure(self) -> tuple:
        """並べ替え用その 2: **中断の発生率を測るための露出**が多い順 (#85)。

        :attr:`score` とは目的が違う。あちらは「1 回の走行でどこが壊れうるか」だが、
        こちらは「**1 セッションで何サンプル採れるか**」。中断のガードが発火するのは
        コーナーを含む動作なので、サンプル数は連結動作の本数で決まる — 探索の動作は
        1 区間ずつでコーナーが無いので数に入らず、**稼げるのは復帰と最速だけ**。

        だから欲しいのは「曲がりが多く直進が短い」盤面になる。同じセル数でも区間が
        細かいほど束ねる数が増える。セッションの所要時間は同時に伸びるので、
        同点なら短い方を採る。
        """
        return (self.chained_motions, self.legs, self.path_cells)
