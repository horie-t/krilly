"""迷路の生成 (issue #77)。

経路計画のバグは**実際の大会迷路より、意地悪な合成迷路の方が安く炙り出せる**ので、
ランダム生成に加えて極端なパターンを用意する。

- :func:`open_maze` — 全面開放。タイブレークの決定性と最短経路の素の性能を見る
- :func:`serpentine_maze` — 長い一本道。経路長が最大 (N^2 セル) になる
- :func:`comb_maze` — 袋小路だらけ。探索が無駄足を踏む形
- :func:`random_maze` — 全域木 + ループ。大会迷路に近い密度
- :func:`seal_goal` — ゴール封鎖。``Unreachable`` の扱いを確かめる

生成した迷路は :func:`krilly.sim.check.check_maze` を通ること。
"""

from __future__ import annotations

import random

from krilly.solver.maze import Direction, Maze

# 内壁の識別子。("V", x, y) = セル (x-1,y) と (x,y) の間の縦壁 (1 <= x <= N-1)、
# ("H", x, y) = セル (x,y-1) と (x,y) の間の横壁 (1 <= y <= N-1)。
Edge = tuple[str, int, int]


def _interior_edges(n: int) -> list[Edge]:
    edges: list[Edge] = [("V", x, y) for x in range(1, n) for y in range(n)]
    edges += [("H", x, y) for x in range(n) for y in range(1, n)]
    return edges


def _set_edge(maze: Maze, e: Edge, present: bool) -> None:
    kind, x, y = e
    if kind == "V":
        maze.set_wall(x - 1, y, Direction.E, present)
    else:
        maze.set_wall(x, y - 1, Direction.N, present)


def _has_edge(maze: Maze, e: Edge) -> bool:
    kind, x, y = e
    return (maze.has_wall(x - 1, y, Direction.E) if kind == "V"
            else maze.has_wall(x, y - 1, Direction.N))


def _posts_of(e: Edge) -> tuple[tuple[int, int], tuple[int, int]]:
    """壁の両端の柱 (柱の座標は 0..N)。"""
    kind, x, y = e
    return ((x, y), (x, y + 1)) if kind == "V" else ((x, y), (x + 1, y))


def _edges_at_post(post: tuple[int, int]) -> list[Edge]:
    i, j = post
    return [("V", i, j - 1), ("V", i, j), ("H", i - 1, j), ("H", i, j)]


def _post_still_covered(maze: Maze, n: int, post: tuple[int, int]) -> bool:
    """内側の柱に壁が 1 枚以上残っているか (外周の柱は外壁があるので常に真)。"""
    i, j = post
    if not (1 <= i <= n - 1 and 1 <= j <= n - 1):
        return True
    return any(_has_edge(maze, e) for e in _edges_at_post(post))


def walled_maze(size: int) -> Maze:
    """外周も内壁もすべて立っている迷路 (掘り進める元になる)。"""
    m = Maze(size)
    m.set_outer_walls()
    for e in _interior_edges(size):
        _set_edge(m, e, True)
    return m


def open_maze(size: int) -> Maze:
    """外周だけの迷路 (内壁なし)。"""
    m = Maze(size)
    m.set_outer_walls()
    return m


def random_maze(
    size: int, seed: int = 0, loop_ratio: float = 0.10, keep_posts: bool = True
) -> Maze:
    """全域木を掘ってから壁を抜いてループを作る。

    ``loop_ratio`` は「内壁の置き場所のうち何割を抜こうと試みるか」。0 なら完全迷路
    (内壁が理論上限の ``(N-1)^2`` 枚)、大きいほど疎になる。``keep_posts`` を立てると
    「柱には必ず 1 枚以上の壁が接する」を守るので、抜ける枚数には下限がある。

    **既定を 0.10 にしてあるのは、疎な迷路は探索の試験にならないため。** 16x16 で
    種を 8 通り振った実測 (内壁の上限は 225 枚):

    ====================  ==========  ==================
    ``loop_ratio``        内壁の枚数  探索の手数 (中央値)
    ====================  ==========  ==================
    0.00 (完全迷路)       225         26-168 (76)
    0.05                  204-210     18-94 (25)
    **0.10 (既定)**       190-195     14-37 (24)
    0.20                  164-176     14-24 (17)
    0.35                  142-152     14-19 (**14**)
    ====================  ==========  ==================

    0.35 では中央値が 14 = スタートからゴールへの最短手数そのもので、**一度も
    行き止まりに当たらない**。それでは探索の論理を何も試していない。0.10 前後が
    実際の大会迷路の密度 (内壁 180-220 枚程度) に近く、行き止まりも踏む。

    なお乱数で作った迷路は**設計された大会迷路より易しい**。難所を意図的に置いた
    本物は :func:`krilly.solver.maze.Maze.from_ascii` で書き起こして使うこと。
    """
    rng = random.Random(seed)
    maze = walled_maze(size)

    # 深さ優先で全域木を掘る (全セルが到達可能になることが保証される)
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        x, y = stack[-1]
        nbrs = [d for d in Direction
                if maze.in_bounds(*maze.neighbor(x, y, d))
                and maze.neighbor(x, y, d) not in seen]
        if not nbrs:
            stack.pop()
            continue
        d = rng.choice(nbrs)
        maze.set_wall(x, y, d, False)
        nxt = maze.neighbor(x, y, d)
        seen.add(nxt)
        stack.append(nxt)

    # ループ用に壁を抜く。柱の条件は抜く壁の両端だけ見ればよい (他の柱は影響を受けない)
    standing = [e for e in _interior_edges(size) if _has_edge(maze, e)]
    rng.shuffle(standing)
    for e in standing[: int(len(_interior_edges(size)) * loop_ratio)]:
        _set_edge(maze, e, False)
        if keep_posts and not all(_post_still_covered(maze, size, p) for p in _posts_of(e)):
            _set_edge(maze, e, True)
    return maze


def serpentine_maze(size: int) -> Maze:
    """長い一本道 (蛇行)。全 N^2 セルを 1 本の経路で繋ぐ最長経路の迷路。

    行ごとに東西へ折り返す。行の間の横壁は折り返す端の 1 箇所だけ開ける。
    最短経路が最長になるので、持ち時間と経路長の見積もりの上限側を突く。
    """
    maze = walled_maze(size)
    for y in range(size):
        for x in range(size - 1):
            maze.set_wall(x, y, Direction.E, False)       # 行の中は東西に開通
        if y < size - 1:
            turn = size - 1 if y % 2 == 0 else 0          # 折り返す端で北へ抜ける
            maze.set_wall(turn, y, Direction.N, False)
    return maze


def comb_maze(size: int) -> Maze:
    """袋小路だらけの櫛形。南端が幹線で、そこから北へ伸びる歯はすべて行き止まり。

    探索が歯を 1 本ずつ入っては戻る形になるので、探索の手数と持ち時間を突く。
    """
    maze = walled_maze(size)
    for x in range(size - 1):
        maze.set_wall(x, 0, Direction.E, False)           # 南端の幹線
    for x in range(size):
        for y in range(size - 1):
            maze.set_wall(x, y, Direction.N, False)       # 北へ伸びる歯
    return maze


def seal_goal(maze: Maze) -> Maze:
    """ゴール領域を壁で完全に囲む (破壊的)。到達不能の扱いを確かめるため。"""
    goals = set(maze.goal_cells())
    for x, y in goals:
        for d in Direction:
            if maze.neighbor(x, y, d) not in goals:
                maze.set_wall(x, y, d, True)
    return maze


def diagonal_goal(maze: Maze) -> Maze:
    """ゴールをスタートの対角 1 セルに移す (破壊的)。

    中央 2x2 は本番設定なのでリハーサルに必須だが、対角ゴールは**経路長を最大化しつつ
    進入方向が限られる**ストレステストになる (#77)。
    """
    n = maze.size
    maze.set_goal((n - 1, n - 1), (n - 1, n - 1))
    return maze
