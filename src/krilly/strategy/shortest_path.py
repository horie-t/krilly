"""確定した壁情報から最速ランの最短経路を求める (issue #19)。

探索ラン (#18) と最速ランでは、**未知セルの扱いが逆**であることに注意する。

- 探索 (`flood_fill`): 未知セルは「壁なし = 通れる」と**楽観視**する。行ってみて
  壁を見つけたら塗り直せばよいので、開拓が進む。
- 最速 (本モジュール): 未知セルは**通さない**。壁を見ていないセルを最速で走るのは
  見えない壁に突っ込む事故になる。``known`` に観測済みセル
  (:attr:`krilly.strategy.explorer.Explorer.visited`) を渡すことで表現する。

また、最短は**セル数だけでは決まらない**。実機の実測 (5x5 通しラン、--omega 1.0) では

    直進 1 セルあたり ≈ 1.50 s / 90° ターン ≈ 2.56 s (整定・停止込み)

で、**旋回 1 回は 1.7 セル分の時間**を食う。そのためセル数の BFS ではなく
**(セル, 向き) を状態にした Dijkstra** で「セル数 + turn_cost × 旋回回数」を
最小化する (``turn_cost`` の既定 1.0 はこの実測に由来)。``turn_cost=0`` にすれば
純粋なセル数最短 (BFS 相当) になる。

結果は :func:`path_to_legs` で **旋回 + 直進セル数** に run-length 圧縮できる。
最速ランの走行状態機械 (#20) はこの形 (連続直進の長さが分かる形) を必要とする。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import groupby
from typing import Iterable

from krilly.solver.maze import Direction, Maze
from krilly.strategy.explorer import quarter_turns
from krilly.strategy.flood_fill import accessible_directions

# 実機実測 (#21 の採用点): 直進 1 セルあたり 0.75s / 90°ターン 2.18s -> 旋回 1 回 ≈ 2.9 セル。
# 直進を速くしたぶん旋回が相対的に高くつくようになった (#20 時点は 1.7)。
# 速度設定 (CellMotionConfig / --omega) を変えたら測り直すこと。
DEFAULT_TURN_COST = 2.9


@dataclass(frozen=True)
class Leg:
    """経路の 1 区間: ``turn`` だけ旋回してから ``cells`` セル直進する。"""

    turn: int      # 90°単位の旋回量 (+CCW / -CW / 2=180°)
    cells: int     # 直進するセル数 (>= 1)


def direction_between(a: tuple[int, int], b: tuple[int, int]) -> Direction:
    """隣接セル ``a`` -> ``b`` の方角。隣接していなければ ValueError。"""
    delta = (b[0] - a[0], b[1] - a[1])
    for d in Direction:
        if d.delta == delta:
            return d
    raise ValueError(f"{a} と {b} は隣接していない")


def shortest_path(
    maze: Maze,
    start: tuple[int, int] | None = None,
    goals: Iterable[tuple[int, int]] | None = None,
    *,
    start_facing: Direction = Direction.N,
    known: Iterable[tuple[int, int]] | None = None,
    turn_cost: float = DEFAULT_TURN_COST,
) -> list[tuple[int, int]]:
    """``start`` からゴールまでの最小コスト経路をセル列で返す。無ければ空リスト。

    コストは ``セル数 + turn_cost × 旋回回数`` (旋回回数は 90° を 1、180° を 2)。
    ``known`` を渡すと**そのセルしか通らない** (未探索セルは壁が未確定なので
    最速ランでは通さない)。戻り値は ``[start, ..., goal]`` で、start がゴールの
    場合は ``[start]``。
    """
    goal_set = set(goals) if goals is not None else set(maze.goal_cells())
    known_set = set(known) if known is not None else None
    origin = start if start is not None else maze.start
    if origin in goal_set:
        return [origin]

    best: dict[tuple[tuple[int, int], Direction], float] = {(origin, start_facing): 0.0}
    prev: dict[tuple[tuple[int, int], Direction], tuple[tuple[int, int], Direction]] = {}
    heap: list[tuple[float, tuple[int, int], Direction]] = [(0.0, origin, start_facing)]
    while heap:
        cost, cell, facing = heapq.heappop(heap)
        if cost > best.get((cell, facing), math.inf):
            continue                       # 既に更新された古いエントリ
        if cell in goal_set:
            return _reconstruct(prev, (origin, start_facing), (cell, facing))
        for d in accessible_directions(maze, *cell):
            nxt = maze.neighbor(*cell, d)
            if known_set is not None and nxt not in known_set:
                continue                   # 壁が未確定のセルは通さない
            nxt_cost = cost + 1.0 + turn_cost * abs(quarter_turns(facing, d))
            if nxt_cost < best.get((nxt, d), math.inf):
                best[(nxt, d)] = nxt_cost
                prev[(nxt, d)] = (cell, facing)
                heapq.heappush(heap, (nxt_cost, nxt, d))
    return []


def _reconstruct(prev, origin, node) -> list[tuple[int, int]]:
    """(セル, 向き) の来歴からセル列を復元する。"""
    cells = [node[0]]
    while node != origin:
        node = prev[node]
        cells.append(node[0])
    cells.reverse()
    return cells


def path_to_legs(
    path: list[tuple[int, int]], start_facing: Direction = Direction.N
) -> list[Leg]:
    """セル列を「旋回 + 直進セル数」の区間列へ run-length 圧縮する。"""
    if len(path) < 2:
        return []
    dirs = [direction_between(path[i], path[i + 1]) for i in range(len(path) - 1)]
    legs = []
    facing = start_facing
    for d, group in groupby(dirs):
        legs.append(Leg(quarter_turns(facing, d), sum(1 for _ in group)))
        facing = d
    return legs


def path_cost(
    path: list[tuple[int, int]],
    start_facing: Direction = Direction.N,
    turn_cost: float = DEFAULT_TURN_COST,
) -> float:
    """経路のコスト (セル数 + turn_cost × 旋回回数)。"""
    legs = path_to_legs(path, start_facing)
    return (len(path) - 1) + turn_cost * sum(abs(leg.turn) for leg in legs)


def route(
    maze: Maze,
    start: tuple[int, int] | None = None,
    goals: Iterable[tuple[int, int]] | None = None,
    *,
    start_facing: Direction = Direction.N,
    known: Iterable[tuple[int, int]] | None = None,
    turn_cost: float = DEFAULT_TURN_COST,
) -> list[Leg]:
    """便利版: 最短経路を求めて区間列 (旋回 + 直進) にして返す。無ければ空リスト。"""
    path = shortest_path(
        maze, start, goals, start_facing=start_facing, known=known, turn_cost=turn_cost
    )
    return path_to_legs(path, start_facing)


def describe_legs(legs: list[Leg]) -> str:
    """ログ用の 1 行表記 例: "直進2 -> 右90° -> 直進3"。"""
    if not legs:
        return "(移動なし)"
    names = {0: None, 1: "左90°", -1: "右90°", 2: "180°"}
    parts = []
    for leg in legs:
        turn = names.get(leg.turn, f"{leg.turn * 90}°")
        if turn:
            parts.append(turn)
        parts.append(f"直進{leg.cells}")
    return " -> ".join(parts)
