"""flood-fill によるゴールまでの距離計算と 1 手選択 (issue #18)。

マイクロマウスの探索は **flood-fill**（ゴール集合からの 4 近傍 BFS）で行う。
壁が無い辺だけを通行可として距離を塗り、現在セルから**距離が小さくなる隣接セル**へ
1 セルずつ進む。壁は進むたびに観測して :class:`Maze` に反映し、そのたびに塗り直す。

未探索セルは :class:`Maze` 上で「壁なし」なので、そのまま**楽観的に通行可**として
扱われる (未知を通れると仮定 → 実際に行って壁を見つけたら塗り直す)。これがいわゆる
modified flood fill で、ゴールへ向かう過程で必要なセルだけを自然に開拓できる。

壁が全く無い迷路ではゴールまでの距離はマンハッタン距離に一致する (4 近傍 BFS)。

純ロジックのみ (Maze への参照だけ) なので、ハードウェア無しでテストできる。
"""

from __future__ import annotations

from collections import deque

from krilly.solver.maze import Direction, Maze

# 到達不能セルの距離 (壁で完全に隔離されている場合)
UNREACHABLE = 1 << 30

# 旋回コスト: 直進 0 / 左右 1 / 反転 2 (同距離の候補を選ぶときのタイブレーク)
_TURN_COST = {0: 0, 1: 1, 3: 1, 2: 2}


def flood_fill(
    maze: Maze, goals: list[tuple[int, int]] | None = None
) -> list[list[int]]:
    """ゴール集合からの BFS 距離を ``dist[x][y]`` で返す (到達不能は UNREACHABLE)。

    ``goals`` 省略時は ``maze.goal_cells()`` (中央 2x2)。壁の無い辺のみを辿るので、
    未探索セル (壁なし) は通行可として楽観視される。
    """
    size = maze.size
    dist = [[UNREACHABLE] * size for _ in range(size)]
    queue: deque[tuple[int, int]] = deque()
    for gx, gy in goals if goals is not None else maze.goal_cells():
        if maze.in_bounds(gx, gy) and dist[gx][gy] != 0:
            dist[gx][gy] = 0
            queue.append((gx, gy))
    while queue:
        x, y = queue.popleft()
        d = dist[x][y] + 1
        for direction in Direction:
            if maze.has_wall(x, y, direction):
                continue
            nx, ny = maze.neighbor(x, y, direction)
            if not maze.in_bounds(nx, ny) or dist[nx][ny] <= d:
                continue
            dist[nx][ny] = d
            queue.append((nx, ny))
    return dist


def accessible_directions(maze: Maze, x: int, y: int) -> list[Direction]:
    """セル (x, y) から壁なしで出られる方角 (迷路の外へ出る辺は除く)。"""
    return [
        d
        for d in Direction
        if not maze.has_wall(x, y, d) and maze.in_bounds(*maze.neighbor(x, y, d))
    ]


def next_direction(
    maze: Maze,
    cell: tuple[int, int],
    facing: Direction,
    dist: list[list[int]],
    visited: set[tuple[int, int]] | None = None,
) -> Direction | None:
    """距離が最小になる隣接セルへの方角を選ぶ。候補が無ければ None。

    同じ距離の候補が複数あるときのタイブレークは、順に

    1. **未訪問セル優先** (``visited`` を渡した場合)。壁情報を増やす方に賭ける
    2. **旋回コストが小さい方** (直進 0 < 左右 1 < 反転 2)。旋回は時間と誤差の元
    3. 方角の N/E/S/W 順 (決定的にするため)

    現在セルが到達不能 (壁で隔離) の場合も None を返す。

    注意: ゴール上 (距離 0) で呼ぶとゴールから離れる方角を返す。ゴール判定は呼び出し側
    (:class:`krilly.strategy.explorer.Explorer`) で先に行う。
    """
    x, y = cell
    if dist[x][y] >= UNREACHABLE:
        return None
    best: tuple[int, int, int, Direction] | None = None
    for d in accessible_directions(maze, x, y):
        nx, ny = maze.neighbor(x, y, d)
        nd = dist[nx][ny]
        if nd >= UNREACHABLE:
            continue
        unvisited_rank = 0 if (visited is not None and (nx, ny) not in visited) else 1
        key = (nd, unvisited_rank, _TURN_COST[(d - facing) % 4], d)
        if best is None or key < best:
            best = key
    return best[3] if best is not None else None
