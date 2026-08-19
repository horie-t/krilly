"""迷路モデル (16x16、辺を共有する壁) (issue #15).

壁は**共有エッジ表現**で保持する。隣り合う 2 セルは同じ壁の実体を参照するため、
「セルAの東壁」と「セルBの西壁」が食い違うことがない。

座標系: セル (x, y)。x=列(東が+)、y=行(北が+)。スタートは (0, 0) で北向き、
ゴールは中央 2x2 (MazeConfig の goal_min..goal_max)。docs/coordinate-frames.md の
車体座標 (+x前/+y左) とは別に、迷路グリッドはこの (東=+x, 北=+y) を用いる。

内部表現:
- ``_vwall[x][y]``: セル (x, y) の**西**辺の縦壁 (= (x-1, y) の東辺)。x=0..size。
- ``_hwall[x][y]``: セル (x, y) の**南**辺の横壁 (= (x, y-1) の北辺)。y=0..size。
"""

from __future__ import annotations

from enum import IntEnum

from krilly.config import MazeConfig


class Direction(IntEnum):
    """方角。N=北, E=東, S=南, W=西 (時計回り)。"""

    N = 0
    E = 1
    S = 2
    W = 3

    @property
    def delta(self) -> tuple[int, int]:
        """(dx, dy)。N=(0,+1), E=(+1,0), S=(0,-1), W=(-1,0)。"""
        return _DELTA[self]

    @property
    def opposite(self) -> "Direction":
        return Direction((self + 2) % 4)


_DELTA = {
    Direction.N: (0, 1),
    Direction.E: (1, 0),
    Direction.S: (0, -1),
    Direction.W: (-1, 0),
}


class Maze:
    """N×N の迷路。壁は共有エッジで保持する。"""

    def __init__(self, size: int = 16) -> None:
        if size < 1:
            raise ValueError("size must be >= 1")
        self.size = size
        self.start: tuple[int, int] = (0, 0)
        self._goal_min = ((size - 1) // 2, (size - 1) // 2)
        self._goal_max = (size // 2, size // 2)
        # 縦壁: x=0..size (size+1 列) × y=0..size-1
        self._vwall = [[False] * size for _ in range(size + 1)]
        # 横壁: x=0..size-1 × y=0..size (size+1 行)
        self._hwall = [[False] * (size + 1) for _ in range(size)]

    @classmethod
    def from_config(cls, maze: MazeConfig) -> "Maze":
        m = cls(maze.grid_size)
        m.set_goal(tuple(maze.goal_min), tuple(maze.goal_max))  # type: ignore[arg-type]
        return m

    @classmethod
    def from_ascii(cls, text: str) -> "Maze":
        """ASCII 表記から迷路を作る (:meth:`to_ascii` の逆)。

        セル幅は任意 (``+-+`` でも ``+---+`` でもよい)。行数は 2N+1 で、上の行が
        北。既知形状を手書きで与えたいとき (校正・テスト・デバッグ) に使う。

        セルの内側の ``S`` / ``G`` はスタート / ゴールを表す (#77)。``G`` が 1 つでも
        あればゴールはその**外接矩形**になり、矩形が ``G`` で埋まっていなければ
        エラーにする (ゴール領域は矩形と決まっているため)。どちらも無ければ
        スタート (0,0) ・ゴール中央 2x2 の既定のまま。

        例::

            +-+-+-+
            |     |
            + +-+ +
            | |   |
            + +-+ +
            | |   |
            +-+-+-+
        """
        lines = [ln.rstrip() for ln in text.strip("\n").splitlines()]
        if len(lines) < 3 or len(lines) % 2 == 0:
            raise ValueError("ASCII 迷路の行数は 2N+1 でなければならない")
        size = len(lines) // 2
        plus = [i for i, ch in enumerate(lines[0]) if ch == "+"]
        if len(plus) != size + 1:
            raise ValueError(f"1 行目の '+' が {len(plus)} 個 (期待 {size + 1} 個)")

        def at(row: int, col: int) -> str:
            line = lines[row]
            return line[col] if col < len(line) else " "

        def segment(row: int, x: int) -> str:
            return "".join(at(row, c) for c in range(plus[x] + 1, plus[x + 1]))

        maze = cls(size)
        goals: list[tuple[int, int]] = []
        for row in range(size):
            y = size - 1 - row                  # 上の行が北 (y が大きい)
            for x in range(size):
                if "-" in segment(2 * row, x):
                    maze.set_wall(x, y, Direction.N)
                if "-" in segment(2 * row + 2, x):
                    maze.set_wall(x, y, Direction.S)
                if at(2 * row + 1, plus[x]) == "|":
                    maze.set_wall(x, y, Direction.W)
                if at(2 * row + 1, plus[x + 1]) == "|":
                    maze.set_wall(x, y, Direction.E)
                inside = segment(2 * row + 1, x)
                if "S" in inside:
                    maze.start = (x, y)
                if "G" in inside:
                    goals.append((x, y))
        if goals:
            lo = (min(g[0] for g in goals), min(g[1] for g in goals))
            hi = (max(g[0] for g in goals), max(g[1] for g in goals))
            span = (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1)
            if span != len(goals):
                raise ValueError(
                    f"ゴールは矩形でなければならない ('G' が {len(goals)} 個、"
                    f"外接矩形 {lo}..{hi} は {span} セル)"
                )
            maze.set_goal(lo, hi)
        return maze

    # -- 範囲・近傍 ---------------------------------------------------------
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def neighbor(self, x: int, y: int, d: Direction) -> tuple[int, int]:
        dx, dy = d.delta
        return (x + dx, y + dy)

    # -- 壁 (共有エッジ) ----------------------------------------------------
    def has_wall(self, x: int, y: int, d: Direction) -> bool:
        if not self.in_bounds(x, y):
            raise IndexError((x, y))
        if d == Direction.N:
            return self._hwall[x][y + 1]
        if d == Direction.S:
            return self._hwall[x][y]
        if d == Direction.E:
            return self._vwall[x + 1][y]
        return self._vwall[x][y]  # W

    def set_wall(self, x: int, y: int, d: Direction, present: bool = True) -> None:
        """セル (x, y) の d 方向の壁を設定する。隣接セルにも自動的に共有される。"""
        if not self.in_bounds(x, y):
            raise IndexError((x, y))
        if d == Direction.N:
            self._hwall[x][y + 1] = present
        elif d == Direction.S:
            self._hwall[x][y] = present
        elif d == Direction.E:
            self._vwall[x + 1][y] = present
        else:  # W
            self._vwall[x][y] = present

    def set_outer_walls(self) -> None:
        """迷路外周をすべて壁にする。"""
        for y in range(self.size):
            self._vwall[0][y] = True          # 西端
            self._vwall[self.size][y] = True   # 東端
        for x in range(self.size):
            self._hwall[x][0] = True           # 南端
            self._hwall[x][self.size] = True    # 北端

    def open_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        """壁が無く進入できる隣接セルの一覧。"""
        out = []
        for d in Direction:
            if not self.has_wall(x, y, d):
                nx, ny = self.neighbor(x, y, d)
                if self.in_bounds(nx, ny):
                    out.append((nx, ny))
        return out

    # -- ゴール -------------------------------------------------------------
    @property
    def goal_min(self) -> tuple[int, int]:
        return self._goal_min

    @property
    def goal_max(self) -> tuple[int, int]:
        return self._goal_max

    def set_goal(self, lo: tuple[int, int], hi: tuple[int, int]) -> None:
        """ゴール領域 (両端含む矩形) を設定する。

        既定は中央 2x2 (本番設定) だが、対角ゴール ``set_goal((n-1, n-1), (n-1, n-1))``
        のように経路長を最大化するストレステストにも使う (#77)。
        """
        for c in (lo, hi):
            if not self.in_bounds(*c):
                raise IndexError(c)
        if lo[0] > hi[0] or lo[1] > hi[1]:
            raise ValueError(f"ゴール矩形の下限が上限を超えている ({lo} .. {hi})")
        self._goal_min, self._goal_max = lo, hi

    def goal_cells(self) -> list[tuple[int, int]]:
        return [
            (x, y)
            for x in range(self._goal_min[0], self._goal_max[0] + 1)
            for y in range(self._goal_min[1], self._goal_max[1] + 1)
        ]

    def is_goal(self, x: int, y: int) -> bool:
        return (self._goal_min[0] <= x <= self._goal_max[0]
                and self._goal_min[1] <= y <= self._goal_max[1])

    # -- デバッグ表示 -------------------------------------------------------
    def to_ascii(self, markers: bool = True) -> str:
        """迷路を ASCII で表示 (北が上)。壁 '---'/'|'、格子点 '+'。

        ``markers`` でスタート ``S`` / ゴール ``G`` をセルの内側に書く。
        :meth:`from_ascii` が読み戻すので往復できる。
        """
        lines = []
        for y in range(self.size - 1, -1, -1):
            top = "+"
            for x in range(self.size):
                top += ("---" if self.has_wall(x, y, Direction.N) else "   ") + "+"
            lines.append(top)
            mid = "|" if self.has_wall(0, y, Direction.W) else " "
            for x in range(self.size):
                mark = " "
                if markers:
                    if (x, y) == self.start:
                        mark = "S"
                    elif self.is_goal(x, y):
                        mark = "G"
                mid += f" {mark} " + ("|" if self.has_wall(x, y, Direction.E) else " ")
            lines.append(mid)
        bottom = "+"
        for x in range(self.size):
            bottom += ("---" if self.has_wall(x, 0, Direction.S) else "   ") + "+"
        lines.append(bottom)
        return "\n".join(lines)
