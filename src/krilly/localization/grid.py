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

import math

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


def apply_axis_heading(
    est: DeadReckoning,
    axis_rad: float,
    weight: float = 1.0,
    max_error: float | None = math.radians(15.0),
) -> bool:
    """カメラで測った迷路軸からの方位ずれで、推定方位を補正する。

    ``axis_rad`` は :func:`krilly.perception.axis_yaw.axis_yaw` が返す
    **(-45°, 45°] に折り返した軸角** (+ = 機体が CCW にずれている)。迷路走行中の
    機体の向きは常に 90° の倍数なので、この値がそのまま「最寄りの軸からのずれ」
    = 絶対方位になる。

    方位はジャイロで閉じているが、その基準は**走行開始時の機体の向き**であって
    迷路の軸ではない。1 走行あたり 0.3-0.5° ずつ基準がずれ、それが次の走行の基準に
    なるので、走行を重ねるほど迷路軸から傾く (実機で 4 セルあたり 10mm 横に流れ、
    壁を押した)。位置の絶対補正 (:func:`apply_cell_offset`) と対になる、方位側の
    絶対補正がこれ。

    推定 φ から測定値を引いた値を 90° 格子へスナップし、測定値を足し戻したものを
    目標とする。``max_error`` (既定 15°) を超える補正は誤測定として棄却する。
    """
    quarter = math.pi / 2.0
    target = snap_to_grid(est.phi - axis_rad, quarter) + axis_rad
    if max_error is not None and abs(target - est.phi) > max_error:
        return False
    est.correct_heading(target, weight)
    return True


def apply_cell_offset(
    est: DeadReckoning,
    cell_center: tuple[float, float],
    forward_m: float | None,
    left_m: float | None,
    phi: float | None = None,
    weight: float = 1.0,
    max_error: float | None = 0.05,
) -> bool:
    """カメラで測ったセル内のずれで推定位置を補正する (issue #54)。

    ``forward_m`` / ``left_m`` は :func:`krilly.perception.cell_pose.cell_offset` が
    返す**車体フレーム**のずれ (None の軸は補正しない)。``cell_center`` は理想格子上の
    セル中央の世界座標、``phi`` は機体方位 (省略時は ``est.phi``)。

    「真の位置 = セル中央 + 回転(ずれ)」を推定値へ引き込む。並進のオドメトリは
    絶対の拠り所を持たないので、これが唯一の絶対補正になる。

    補正は**測れた車体軸に沿った成分だけ**に掛ける。壁が無い軸は帯が無くてずれが
    分からないので、そこを 0 扱いで補正すると測ってもいない方向へ引っ張ってしまう。
    ``max_error`` を超える補正量は誤測定として棄却し False を返す (既定 50mm)。
    """
    if forward_m is None and left_m is None:
        return False
    heading = est.phi if phi is None else phi
    c, s = math.cos(heading), math.sin(heading)
    cx, cy = cell_center
    axes = []
    if forward_m is not None:
        axes.append(((c, s), forward_m))         # 車体 +x (前) の世界方向
    if left_m is not None:
        axes.append(((-s, c), left_m))           # 車体 +y (左) の世界方向
    deltas = []
    for (ux, uy), offset in axes:
        current = (est.x - cx) * ux + (est.y - cy) * uy   # いまの推定のずれ
        error = offset - current                          # その軸の補正量
        if max_error is not None and abs(error) > max_error:
            return False
        deltas.append(((ux, uy), error))
    for (ux, uy), error in deltas:
        est.x += weight * error * ux
        est.y += weight * error * uy
    return True
