"""迷路の健全性チェックと壁の枚数 (issue #77)。

2 つの用途がある:

1. **書き起こしの検証** — 大会記録集の画像から起こした迷路は読み違いを含む前提で扱う
   (16x16 は壁の置き場所が 544 箇所ある)。実際の大会迷路は必ず解けるので、
   **到達不能が出たらそれは迷路の性質ではなく書き起こしの誤り**と分かる。
2. **実機の準備** — 手持ちの壁で組めるかを、組む前に判定する (#23)。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from krilly.solver.maze import Direction, Maze


# --- 壁の枚数 ---------------------------------------------------------------
@dataclass(frozen=True)
class WallCounts:
    """迷路 1 面に要る壁の枚数と、その理論上の上下限。

    上下限は設計の好みではなく制約から出る (N は 1 辺のセル数):

    - **内壁の上限 ``(N-1)^2``** — 全セルが到達可能なら開いている内壁が最低 ``N^2-1``
      本必要 (全域木)。内壁の置き場所 ``2N(N+1) - 4N`` から引くとこうなる。
      完全に囲まれたセルがあると連結成分 1 つにつき 1 枚だけ上限が上がる。
    - **内壁の下限 ``ceil((N-1)^2/2)``** — 「柱には必ず 1 枚以上の壁が接する」を
      満たす最小の辺被覆 (内側の柱 ``(N-1)^2`` 本を 1 枚 2 本ずつ覆う)。

    ``outer + inner_max == (N+1)^2`` = 柱の本数、というきれいな関係がある。
    """

    size: int
    outer: int          # 外周にある壁の枚数 (完全なら 4N)
    inner: int          # 内壁の枚数
    posts: int          # 柱の本数 (N+1)^2

    @property
    def total(self) -> int:
        return self.outer + self.inner

    @property
    def inner_slots(self) -> int:
        """内壁の置き場所の総数。"""
        n = self.size
        return 2 * n * (n + 1) - 4 * n

    @property
    def inner_max(self) -> int:
        n = self.size
        return (n - 1) ** 2

    @property
    def inner_min(self) -> int:
        n = self.size
        return math.ceil((n - 1) ** 2 / 2)

    def describe(self) -> str:
        return (
            f"{self.size}x{self.size}: 壁 {self.total} 枚 "
            f"(外周 {self.outer} / 内壁 {self.inner}、内壁の枠 {self.inner_slots}、"
            f"内壁の上限 {self.inner_max} / 下限 {self.inner_min})、柱 {self.posts} 本"
        )


def wall_counts(maze: Maze) -> WallCounts:
    """迷路に立っている壁を数える。"""
    n = maze.size
    outer = sum(maze.has_wall(x, 0, Direction.S) for x in range(n))
    outer += sum(maze.has_wall(x, n - 1, Direction.N) for x in range(n))
    outer += sum(maze.has_wall(0, y, Direction.W) for y in range(n))
    outer += sum(maze.has_wall(n - 1, y, Direction.E) for y in range(n))
    inner = sum(maze.has_wall(x - 1, y, Direction.E)
                for x in range(1, n) for y in range(n))          # 縦の内壁
    inner += sum(maze.has_wall(x, y - 1, Direction.N)
                 for x in range(n) for y in range(1, n))         # 横の内壁
    return WallCounts(size=n, outer=outer, inner=inner, posts=(n + 1) ** 2)


# --- 柱 ---------------------------------------------------------------------
def posts_without_wall(maze: Maze) -> list[tuple[int, int]]:
    """壁が 1 枚も接していない**内側の柱**の一覧 (公式規則では存在してはいけない)。

    柱 (i, j) は 1 <= i, j <= N-1。外周の柱は外壁が接するので常に条件を満たす。
    機体にとっては「柱が壁の手がかりを与えない」= カメラが赤い柱だけを見て
    セルの向きを誤る余地になる。
    """
    n = maze.size
    out = []
    for i in range(1, n):
        for j in range(1, n):
            touching = (
                maze.has_wall(i - 1, j - 1, Direction.E)   # 柱の南の縦壁
                or maze.has_wall(i - 1, j, Direction.E)    # 柱の北の縦壁
                or maze.has_wall(i - 1, j - 1, Direction.N)  # 柱の西の横壁
                or maze.has_wall(i, j - 1, Direction.N)    # 柱の東の横壁
            )
            if not touching:
                out.append((i, j))
    return out


# --- 到達性 -----------------------------------------------------------------
def reachable_cells(maze: Maze, start: tuple[int, int] | None = None) -> set[tuple[int, int]]:
    """``start`` から壁を越えずに行けるセルの集合。"""
    src = start if start is not None else maze.start
    seen = {src}
    queue = deque([src])
    while queue:
        cell = queue.popleft()
        for nxt in maze.open_neighbors(*cell):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


# --- 総合レポート -----------------------------------------------------------
@dataclass
class MazeReport:
    """健全性チェックの結果。``errors`` が空なら迷路として成立している。"""

    counts: WallCounts
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def describe(self) -> str:
        lines = [self.counts.describe()]
        lines += [f"  [誤り] {e}" for e in self.errors]
        lines += [f"  [注意] {w}" for w in self.warnings]
        if self.ok and not self.warnings:
            lines.append("  問題なし")
        return "\n".join(lines)


def check_maze(maze: Maze, wall_budget: int | None = None) -> MazeReport:
    """迷路の健全性を機械的に調べる。

    **誤り** (迷路として成立しない。書き起こしなら読み違いを疑う):

    - 外周が閉じていない
    - スタートからゴールへ到達できない

    **注意** (成立はするが意図しない可能性がある):

    - スタートから到達できないセルがある (大会迷路にも稀にあるので誤りにはしない)
    - 壁が 1 枚も接していない内側の柱がある (公式規則違反)
    - 四方を壁で囲まれたセルがある
    - ``wall_budget`` を渡すと、手持ちの壁で組めるかを判定する (#23)
    """
    n = maze.size
    counts = wall_counts(maze)
    report = MazeReport(counts=counts)

    missing = counts.outer < 4 * n
    if missing:
        report.errors.append(f"外周が閉じていない (外壁 {counts.outer}/{4 * n} 枚)")

    reachable = reachable_cells(maze)
    goals = maze.goal_cells()
    if not any(g in reachable for g in goals):
        report.errors.append(
            f"スタート {maze.start} からゴール {goals[0]}..{goals[-1]} へ到達できない"
        )

    stranded = n * n - len(reachable)
    if stranded:
        report.warnings.append(f"スタートから到達できないセルが {stranded} 個ある")

    bare = posts_without_wall(maze)
    if bare:
        report.warnings.append(
            f"壁が接していない柱が {len(bare)} 本ある (公式規則違反): {bare[:5]}"
            + (" ..." if len(bare) > 5 else "")
        )

    boxed = [(x, y) for x in range(n) for y in range(n)
             if all(maze.has_wall(x, y, d) for d in Direction)]
    if boxed:
        report.warnings.append(f"四方を壁で囲まれたセルが {len(boxed)} 個ある: {boxed[:5]}")

    if wall_budget is not None and counts.total > wall_budget:
        report.warnings.append(
            f"手持ちの壁 {wall_budget} 枚では {counts.total - wall_budget} 枚足りない"
        )
    return report


# --- 探索結果の照合 ---------------------------------------------------------
def map_agrees(
    truth: Maze, learned: Maze, cells: set[tuple[int, int]] | None = None
) -> list[str]:
    """``cells`` の範囲で判明した地図が真の迷路と一致するかを調べ、不一致を返す。

    範囲を限るのが要点で、**探索は訪問していないセルの壁を知らない** (未知は
    ``False`` = 開いていると楽観的に扱われる)。既定は探索が訪れたセル
    (``Explorer.visited``) を渡すこと。全セルを渡すと未訪問セルが必ず不一致になる。
    """
    if truth.size != learned.size:
        return [f"迷路の大きさが違う (真 {truth.size} / 判明 {learned.size})"]
    target = cells if cells is not None else {
        (x, y) for x in range(truth.size) for y in range(truth.size)
    }
    out = []
    for x, y in sorted(target):
        for d in Direction:
            want, got = truth.has_wall(x, y, d), learned.has_wall(x, y, d)
            if want != got:
                out.append(
                    f"({x},{y}) の {d.name}: 真 {'壁あり' if want else '壁なし'} / "
                    f"判明 {'壁あり' if got else '壁なし'}"
                )
    return out
