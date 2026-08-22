"""真の迷路から壁観測を作る (実機カメラの代役) (issue #77)。

:meth:`krilly.perception.wall_detect.WallDetector.measure` と同じ形の辞書を返すので、
:meth:`krilly.strategy.explorer.Explorer.observe` にそのまま渡せる。
``body_walls_to_maze`` の逆写像であり、両者が食い違うと探索が壊れるため
``tests/test_sim.py`` で往復を検証している。

**この代役が模擬しないもの**: 見落とし・誤検出・飽和・姿勢ずれ。つまりカメラ由来の
誤差は一切入らない。そこは実機 (#23) の担当で、ここで検証するのは離散層の論理だけ。
"""

from __future__ import annotations

from krilly.perception.wall_detect import (
    BACK,
    FRONT,
    LEFT,
    NEIGHBOR_SIDES,
    RIGHT,
    maze_direction_for,
)
from krilly.solver.maze import Direction, Maze


def sense(truth: Maze, cell: tuple[int, int], facing: Direction) -> dict[str, bool]:
    """``cell`` で ``facing`` を向いているときに見える機体相対の壁有無。

    LEFT = facing の反時計回り、RIGHT = 時計回り。旋回レス走行 (#76) では facing は
    常に北なので、この写像は定数になる (FRONT→N / BACK→S / LEFT→W / RIGHT→E)。
    """
    x, y = cell
    return {
        FRONT: truth.has_wall(x, y, facing),
        BACK: truth.has_wall(x, y, Direction((facing + 2) % 4)),
        LEFT: truth.has_wall(x, y, Direction((facing - 1) % 4)),
        RIGHT: truth.has_wall(x, y, Direction((facing + 1) % 4)),
    }


def sense_neighbors(
    truth: Maze, cell: tuple[int, int], facing: Direction,
    sides: tuple[str, ...] = NEIGHBOR_SIDES,
) -> dict[str, dict[str, bool]]:
    """左右の隣セルの壁 (#89)。:meth:`Explorer.observe` の ``neighbors`` にそのまま渡せる。

    全画素モード (#88) では左右の隣セルの奥の壁まで画角に入るので、そのセルの 4 壁が
    すべて分かる。迷路の外にはみ出す側は返さない。

    **実機との違いに注意**: 実機の
    :meth:`~krilly.perception.wall_detect.WallDetector.neighbor_walls` は確信の
    持てない辺を落とすので、辺が欠けた辞書が返る。ここは常に 4 辺そろう「上限の
    性能」なので、停止回数の削減もここで出る値が上限になる。
    """
    out: dict[str, dict[str, bool]] = {}
    for side in sides:
        x, y = truth.neighbor(*cell, maze_direction_for(side, facing))
        if truth.in_bounds(x, y):
            out[side] = sense(truth, (x, y), facing)
    return out
