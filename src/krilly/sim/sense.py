"""真の迷路から壁観測を作る (実機カメラの代役) (issue #77)。

:meth:`krilly.perception.wall_detect.WallDetector.measure` と同じ形の辞書を返すので、
:meth:`krilly.strategy.explorer.Explorer.observe` にそのまま渡せる。
``body_walls_to_maze`` の逆写像であり、両者が食い違うと探索が壊れるため
``tests/test_sim.py`` で往復を検証している。

**この代役が模擬しないもの**: 見落とし・誤検出・飽和・姿勢ずれ。つまりカメラ由来の
誤差は一切入らない。そこは実機 (#23) の担当で、ここで検証するのは離散層の論理だけ。
"""

from __future__ import annotations

from krilly.perception.wall_detect import BACK, FRONT, LEFT, RIGHT
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
