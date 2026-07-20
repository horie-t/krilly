"""デッドレコニング (ステップ積算) による自己位置推定 (issue #12).

各輪の転がり距離デルタを正運動学 :meth:`KiwiKinematics.wheels_to_body` で
車体フレームの微小変位 (dx, dy, dφ) に変換し、現在の姿勢 φ で世界フレームへ
回転して状態 ``[X, Y, φ]`` に積分する (回転は中点法で近似)。

距離デルタの入力元は本クラスから切り離してある:
- ``update_wheel_speeds``: 指令した各輪速度[m/s] × dt (オープンループ, #12 の既定)
- ``update_wheel_microsteps``: 各輪のマイクロステップ数デルタ (将来 ABS_POS 読み戻し用)
- ``update_wheel_distances``: 各輪の転がり距離[m]デルタ (最も汎用)

姿勢 φ の誤差はスリップで蓄積しやすいため、後段で BNO055 ジャイロ (#13) と
カメラ格子補正 (#14) で補正する前提。純ロジックなのでハードウェア無しでテスト可能。
"""

from __future__ import annotations

import math

from krilly.kinematics.kiwi import KiwiKinematics


class DeadReckoning:
    """ステップ積算で世界座標の姿勢 [X, Y, φ] を推定する。"""

    def __init__(
        self,
        kinematics: KiwiKinematics | None = None,
        x: float = 0.0,
        y: float = 0.0,
        phi: float = 0.0,
    ) -> None:
        self.kin = kinematics or KiwiKinematics()
        self.x = x
        self.y = y
        self.phi = phi

    @property
    def pose(self) -> tuple[float, float, float]:
        """世界フレームの推定姿勢 (X[m], Y[m], φ[rad])。"""
        return (self.x, self.y, self.phi)

    def reset(self, x: float = 0.0, y: float = 0.0, phi: float = 0.0) -> None:
        self.x, self.y, self.phi = x, y, phi

    # -- 更新 (汎用: 各輪の転がり距離デルタ) --------------------------------
    def update_wheel_distances(self, d0: float, d1: float, d2: float) -> tuple[float, float, float]:
        """各輪の転がり距離デルタ[m](符号付き)から姿勢を1ステップ積分する。"""
        # 車体フレームの微小変位 (前進dx, 左dy, 回転dφ)
        dx_b, dy_b, dphi = self.kin.wheels_to_body(d0, d1, d2)
        # 中点姿勢で世界フレームへ回転 (小さな回転中の並進を近似)
        phi_mid = self.phi + dphi / 2.0
        cos_m, sin_m = math.cos(phi_mid), math.sin(phi_mid)
        self.x += dx_b * cos_m - dy_b * sin_m
        self.y += dx_b * sin_m + dy_b * cos_m
        self.phi += dphi
        return self.pose

    # -- 更新 (入力元別の便利メソッド) --------------------------------------
    def update_wheel_speeds(
        self, wheel_mps: tuple[float, float, float], dt: float
    ) -> tuple[float, float, float]:
        """各輪速度[m/s]×dt を距離デルタとして積分する (指令ベースのオープンループ)。"""
        return self.update_wheel_distances(*(v * dt for v in wheel_mps))

    def update_wheel_microsteps(
        self, m0: float, m1: float, m2: float
    ) -> tuple[float, float, float]:
        """各輪のマイクロステップ数デルタから積分する。"""
        d2m = self.kin.microsteps_to_distance
        return self.update_wheel_distances(d2m(m0), d2m(m1), d2m(m2))
