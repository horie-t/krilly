"""迷路の既知格子への X/Y スナップ補正 (issue #14).

マイクロマウス迷路の壁・格子点は 180mm ピッチの既知グリッド上にある。
下向きカメラが赤い壁上面 (またはセル境界) を検出したとき、その方向の推定座標を
最寄りのグリッド線へスナップすることで、デッドレコニングの並進ドリフトを
リセットできる。

本モジュールは**格子スナップの純ロジック**を提供する。カメラ画素から地面座標
への投影 (取付高さ・画角の外部パラメータ) は M4 #16 で確定させ、その出力
「どの軸のグリッド線を、どの推定座標付近で観測したか」を本補正に渡す想定。

グリッド線の位置は ``offset + n * pitch`` (n は整数)。壁中心を原点に取るなら
offset=0、セル中心基準なら offset=pitch/2 を使う。
"""

from __future__ import annotations

from krilly.config import MazeConfig
from krilly.localization.estimator import DeadReckoning


def snap_to_grid(value: float, pitch: float, offset: float = 0.0) -> float:
    """``value`` を最寄りのグリッド線 (offset + n*pitch) にスナップして返す。"""
    return offset + round((value - offset) / pitch) * pitch


class GridCorrector:
    """推定位置を既知グリッドへスナップして補正する。"""

    def __init__(self, pitch: float, offset: float = 0.0) -> None:
        if pitch <= 0:
            raise ValueError("pitch must be > 0")
        self.pitch = pitch
        self.offset = offset

    @classmethod
    def from_maze(cls, maze: MazeConfig, offset: float = 0.0) -> "GridCorrector":
        """MazeConfig の cell_pitch_m からコレクタを作る。"""
        return cls(maze.cell_pitch_m, offset)

    def snap(self, value: float) -> float:
        return snap_to_grid(value, self.pitch, self.offset)

    def residual(self, value: float) -> float:
        """最寄りグリッド線からのズレ (value - snap(value))。"""
        return value - self.snap(value)

    def apply_x(
        self, est: DeadReckoning, weight: float = 1.0, max_error: float | None = None
    ) -> bool:
        """est.x を最寄りグリッド線へスナップ。max_error 超なら無視し False。

        max_error は誤検出ガード (グリッド線から離れすぎた観測は棄却する)。
        """
        target = self.snap(est.x)
        if max_error is not None and abs(est.x - target) > max_error:
            return False
        est.correct_x(target, weight)
        return True

    def apply_y(
        self, est: DeadReckoning, weight: float = 1.0, max_error: float | None = None
    ) -> bool:
        """est.y を最寄りグリッド線へスナップ。max_error 超なら無視し False。"""
        target = self.snap(est.y)
        if max_error is not None and abs(est.y - target) > max_error:
            return False
        est.correct_y(target, weight)
        return True
