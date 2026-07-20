"""自己位置推定: デッドレコニング + ジャイロ方位 + カメラによるグリッド補正。

状態量は [X, Y, phi]。M3 (issues #12-#14) で実装する。
"""

from .estimator import DeadReckoning

__all__ = ["DeadReckoning"]
