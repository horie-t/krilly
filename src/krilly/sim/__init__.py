"""迷路シミュレーション基盤 (issue #77)。

探索の完走・地図の一致・10 分 5 走の予算は**実機なしで検証できる**。離散層
(:mod:`krilly.solver`, :mod:`krilly.strategy`, :mod:`krilly.app`) はハードウェアに
触らないので、真の迷路を用意して壁観測を差し替えれば、セッション全体をそのまま回せる。
これにより実機の役割を「誤差の蓄積」と「カメラ判定」に絞れる (#23)。

構成:

- :mod:`~krilly.sim.sense` — 真の迷路から機体相対の壁を返す (実機カメラの代役)
- :mod:`~krilly.sim.generate` — 迷路の生成 (ランダム + 意地悪なパターン)
- :mod:`~krilly.sim.check` — 健全性チェックと壁の枚数 (書き起こしの検証・実機の準備)
- :mod:`~krilly.sim.session` — 探索 → 復帰 → 最速 xN を一本で回す統合シミュレータ
"""

from krilly.sim.check import MazeReport, check_maze, map_agrees, wall_counts
from krilly.sim.generate import (
    comb_maze,
    diagonal_goal,
    open_maze,
    random_maze,
    seal_goal,
    serpentine_maze,
    walled_maze,
)
from krilly.sim.sense import sense
from krilly.sim.session import RunRecord, SessionResult, fit_search_overhead, simulate_session

__all__ = [
    "MazeReport", "check_maze", "map_agrees", "wall_counts",
    "comb_maze", "diagonal_goal", "open_maze", "random_maze", "seal_goal",
    "serpentine_maze", "walled_maze",
    "sense",
    "RunRecord", "SessionResult", "fit_search_overhead", "simulate_session",
]
