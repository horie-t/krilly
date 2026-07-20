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


def _wrap_angle(a: float) -> float:
    """角度を (-π, π] に正規化する。"""
    return (a + math.pi) % (2 * math.pi) - math.pi


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

    # -- 積分の共通処理 -----------------------------------------------------
    def _integrate(self, dx_b: float, dy_b: float, dphi: float) -> tuple[float, float, float]:
        """車体フレームの微小変位を中点姿勢で世界フレームへ回転し積分する。"""
        phi_mid = self.phi + dphi / 2.0  # 回転中の並進を中点姿勢で近似
        cos_m, sin_m = math.cos(phi_mid), math.sin(phi_mid)
        self.x += dx_b * cos_m - dy_b * sin_m
        self.y += dx_b * sin_m + dy_b * cos_m
        self.phi += dphi
        return self.pose

    # -- 更新 (汎用: 各輪の転がり距離デルタ) --------------------------------
    def update_wheel_distances(self, d0: float, d1: float, d2: float) -> tuple[float, float, float]:
        """各輪の転がり距離デルタ[m](符号付き)から姿勢を1ステップ積分する。"""
        dx_b, dy_b, dphi = self.kin.wheels_to_body(d0, d1, d2)
        return self._integrate(dx_b, dy_b, dphi)

    # -- 更新 (ジャイロ融合: 姿勢はジャイロ、並進は車輪) #13 -----------------
    def update_with_gyro(
        self, d0: float, d1: float, d2: float, gyro_dphi: float
    ) -> tuple[float, float, float]:
        """並進は車輪から、回転 dφ は**ジャイロ**から取って積分する。

        ステッパの脱調・スリップは並進より回転に効きやすいため、回転を
        スリップしないジャイロに委ねてφ誤差の蓄積を抑える (#13)。
        ``gyro_dphi`` はこのステップのジャイロ積分値[rad] (+ω=CCW, バイアス減算済)。
        """
        dx_b, dy_b, _dphi_odom = self.kin.wheels_to_body(d0, d1, d2)
        return self._integrate(dx_b, dy_b, gyro_dphi)

    def update_with_gyro_rate(
        self,
        wheel_mps: tuple[float, float, float],
        gyro_rate: float,
        dt: float,
    ) -> tuple[float, float, float]:
        """便利版: 各輪速度[m/s]と ジャイロ角速度[rad/s] を dt で積分する。"""
        return self.update_with_gyro(*(v * dt for v in wheel_mps), gyro_rate * dt)

    # -- 絶対方位による緩やかな補正 (相補フィルタの低域側) #13 --------------
    def correct_heading(self, phi_ref: float, weight: float = 1.0) -> float:
        """絶対方位 ``phi_ref``[rad] へ向けて φ を緩やかに補正する。

        ジャイロは長期的にバイアスでドリフトするため、信頼できる絶対方位
        (BNO055 融合 heading や、後段 #14 の迷路軸に整列した方位) が得られた
        ときに ``weight`` (0..1) の割合で引き込む。角度差は最短方向で扱う。
        """
        err = _wrap_angle(phi_ref - self.phi)
        self.phi += weight * err
        return self.phi

    # -- 位置の補正 (カメラ格子スナップの受け皿) #14 ------------------------
    def correct_x(self, x_ref: float, weight: float = 1.0) -> float:
        """世界座標 X を ``x_ref`` へ weight の割合で引き込む。"""
        self.x += weight * (x_ref - self.x)
        return self.x

    def correct_y(self, y_ref: float, weight: float = 1.0) -> float:
        """世界座標 Y を ``y_ref`` へ weight の割合で引き込む。"""
        self.y += weight * (y_ref - self.y)
        return self.y

    def correct_position(
        self, x_ref: float, y_ref: float, weight: float = 1.0
    ) -> tuple[float, float]:
        """(X, Y) を参照位置へ weight の割合で引き込む。"""
        return (self.correct_x(x_ref, weight), self.correct_y(y_ref, weight))

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
