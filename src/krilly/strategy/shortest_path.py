"""確定した壁情報から最速ランの最短経路を求める (issue #19)。

探索ラン (#18) と最速ランでは、**未知セルの扱いが逆**であることに注意する。

- 探索 (`flood_fill`): 未知セルは「壁なし = 通れる」と**楽観視**する。行ってみて
  壁を見つけたら塗り直せばよいので、開拓が進む。
- 最速 (本モジュール): 未知セルは**通さない**。壁を見ていないセルを最速で走るのは
  見えない壁に突っ込む事故になる。``known`` に観測済みセル
  (:attr:`krilly.strategy.explorer.Explorer.visited`) を渡すことで表現する。

また、最短は**セル数だけでは決まらない**。動作には「1 セル進む」以外のコストがある:

- **区間ごとの固定費** (``MoveCost.leg``): 加減速のランプ・整定・停止。連続直進なら
  1 回で済むので、セルをまとめるほど得になる。
- **旋回** (``MoveCost.turn``): 機体を回す時間。**ホロノミック走行 (#76) では 0**。

そのためセル数の BFS ではなく **(セル, 直前の進行方角) を状態にした Dijkstra** で
総コストを最小化する。状態の第 2 要素は旋回する走り方では「機体の向き」と一致するので、
``MoveCost`` を差し替えるだけで**両方の走り方を同じ Dijkstra で表現できる**
(:data:`LEGACY_COST` が旋回あり、:data:`DEFAULT_COST` が旋回レス)。

結果は :func:`path_to_legs` で **方角 + セル数** に run-length 圧縮できる。
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
class MoveCost:
    """1 セル進むコストの内訳 (単位は「直進 1 セル」)。

    実測の時間から比で作る。既定は**旋回レス走行** (#76) の 5x5 通しラン実測:
    区間の固定費 0.81s / 南北 1 セル 0.73s = 1.10、東西 0.75s / 0.73s = 1.03。
    """

    cell_ns: float = 1.0    # 南北 (機体の前後軸) へ 1 セル
    cell_ew: float = 1.03   # 東西 (機体の左右軸) へ 1 セル (実測 0.75s / 0.73s)
    leg: float = 1.10       # 直進区間 1 本あたりの固定費 (ランプ + 整定 + 撮影 + 停止)
    turn: float = 0.0       # 機体旋回 90° 1 回 (旋回レスでは 0)


#: 旋回レス走行 (#76) のコスト。
DEFAULT_COST = MoveCost()

#: 旋回して前進する従来の走り方のコスト。現行の実装と厳密に一致する。
LEGACY_COST = MoveCost(cell_ns=1.0, cell_ew=1.0, leg=0.0, turn=DEFAULT_TURN_COST)


@dataclass(frozen=True)
class Leg:
    """経路の 1 区間: ``direction`` の方角へ ``cells`` セル進む。

    旋回する走り方では「その方角を向いてから直進する」、旋回レス走行 (#76) では
    「向きを変えずにその方角へ平行移動する」。**どちらへ何セル進むか**という情報は
    共通なので、この表現は両方で使える。
    """

    direction: Direction   # 進む方角
    cells: int             # 進むセル数 (>= 1)


def direction_between(a: tuple[int, int], b: tuple[int, int]) -> Direction:
    """隣接セル ``a`` -> ``b`` の方角。隣接していなければ ValueError。"""
    delta = (b[0] - a[0], b[1] - a[1])
    for d in Direction:
        if d.delta == delta:
            return d
    raise ValueError(f"{a} と {b} は隣接していない")


def _step_cost(
    cost: MoveCost, prev: Direction | None, d: Direction, start_facing: Direction
) -> float:
    """``prev`` の方角から ``d`` へ 1 セル進むコスト。

    ``prev`` が None なのは出発点だけ。そのとき区間の固定費は課金し (止まっている状態
    から動き出すので)、旋回量は ``start_facing`` から測る (機体は既にその向きを向いている)。
    """
    step = cost.cell_ns if d in (Direction.N, Direction.S) else cost.cell_ew
    if d != prev:
        step += cost.leg
    if cost.turn:
        step += cost.turn * abs(quarter_turns(start_facing if prev is None else prev, d))
    return step


def shortest_path(
    maze: Maze,
    start: tuple[int, int] | None = None,
    goals: Iterable[tuple[int, int]] | None = None,
    *,
    start_facing: Direction = Direction.N,
    known: Iterable[tuple[int, int]] | None = None,
    cost: MoveCost = DEFAULT_COST,
) -> list[tuple[int, int]]:
    """``start`` からゴールまでの最小コスト経路をセル列で返す。無ければ空リスト。

    状態は **(セル, 直前の進行方角)**。旋回する走り方 (``cost=LEGACY_COST``) では
    第 2 要素が機体の向きと一致するので、同じ Dijkstra が両方の走り方を表現する。

    ``known`` を渡すと**そのセルしか通らない** (未探索セルは壁が未確定なので最速ランでは
    通さない)。戻り値は ``[start, ..., goal]`` で、start がゴールの場合は ``[start]``。
    """
    goal_set = set(goals) if goals is not None else set(maze.goal_cells())
    known_set = set(known) if known is not None else None
    origin = start if start is not None else maze.start
    if origin in goal_set:
        return [origin]

    Node = tuple[tuple[int, int], Direction | None]
    source: Node = (origin, None)
    best: dict[Node, float] = {source: 0.0}
    prev_of: dict[Node, Node] = {}
    # heap の要素に通し番号を挟む。挟まないとコストとセルが同値のとき
    # None と Direction を比較して TypeError になる (同コスト経路が多い迷路で再現する)。
    counter = 0
    heap: list[tuple[float, int, tuple[int, int], Direction | None]] = [
        (0.0, counter, origin, None)
    ]
    while heap:
        total, _seq, cell, came_from = heapq.heappop(heap)
        node: Node = (cell, came_from)
        if total > best.get(node, math.inf):
            continue                       # 既に更新された古いエントリ
        if cell in goal_set:
            return _reconstruct(prev_of, source, node)
        for d in accessible_directions(maze, *cell):
            nxt = maze.neighbor(*cell, d)
            if known_set is not None and nxt not in known_set:
                continue                   # 壁が未確定のセルは通さない
            nxt_cost = total + _step_cost(cost, came_from, d, start_facing)
            if nxt_cost < best.get((nxt, d), math.inf):
                best[(nxt, d)] = nxt_cost
                prev_of[(nxt, d)] = node
                counter += 1
                heapq.heappush(heap, (nxt_cost, counter, nxt, d))
    return []


def _reconstruct(prev, origin, node) -> list[tuple[int, int]]:
    """(セル, 直前の進行方角) の来歴からセル列を復元する。"""
    cells = [node[0]]
    while node != origin:
        node = prev[node]
        cells.append(node[0])
    cells.reverse()
    return cells


def path_to_legs(path: list[tuple[int, int]]) -> list[Leg]:
    """セル列を「方角 + セル数」の区間列へ run-length 圧縮する。"""
    if len(path) < 2:
        return []
    dirs = [direction_between(path[i], path[i + 1]) for i in range(len(path) - 1)]
    return [Leg(d, sum(1 for _ in group)) for d, group in groupby(dirs)]


def path_cost(
    path: list[tuple[int, int]],
    start_facing: Direction = Direction.N,
    cost: MoveCost = DEFAULT_COST,
) -> float:
    """経路のコスト (:func:`shortest_path` が最小化しているものと同じ式)。"""
    if len(path) < 2:
        return 0.0
    dirs = [direction_between(path[i], path[i + 1]) for i in range(len(path) - 1)]
    total = 0.0
    prev: Direction | None = None
    for d in dirs:
        total += _step_cost(cost, prev, d, start_facing)
        prev = d
    return total


def route(
    maze: Maze,
    start: tuple[int, int] | None = None,
    goals: Iterable[tuple[int, int]] | None = None,
    *,
    start_facing: Direction = Direction.N,
    known: Iterable[tuple[int, int]] | None = None,
    cost: MoveCost = DEFAULT_COST,
) -> list[Leg]:
    """便利版: 最短経路を求めて区間列 (方角 + セル数) にして返す。無ければ空リスト。"""
    path = shortest_path(
        maze, start, goals, start_facing=start_facing, known=known, cost=cost
    )
    return path_to_legs(path)


def describe_legs(legs: list[Leg]) -> str:
    """ログ用の 1 行表記 例: "北へ2 -> 東へ3"。"""
    if not legs:
        return "(移動なし)"
    return " -> ".join(f"{leg.direction.name}へ{leg.cells}" for leg in legs)


def turns_in(legs: list[Leg], start_facing: Direction = Direction.N) -> int:
    """区間列に含まれる 90° 旋回の回数 (旋回する走り方の見積もり用)。

    ``Leg`` は方角しか持たないので、旋回量は隣り合う区間の方角の差から出す。
    旋回レス走行ではこの値は時間に効かない。
    """
    total = 0
    facing = start_facing
    for leg in legs:
        total += abs(quarter_turns(facing, leg.direction))
        facing = leg.direction
    return total
