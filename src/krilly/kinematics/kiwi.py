"""kiwi-drive (3輪オムニホイール) の運動学と輪速 <-> ステッパ変換。

規約は docs/coordinate-frames.md と config/robot.yaml に従う:
- ボディ座標系: +x が前方、+y が左方、+z が上方 (右手系)、+omega は反時計回り (CCW)。
- ``wheel_angles_deg`` は各輪の駆動方向角 theta_i (スポーク方向 + 90 度) であり、
  デフォルトは M0(前) / M1(後左) / M2(後右) に対して [90, 210, 330]。

逆運動学 (ボディ速度 -> 各輪の接地面速度)、各輪 i について:

    v_i = -sin(theta_i) * vx + cos(theta_i) * vy + L * omega

すなわち row_i = [-sin theta_i, cos theta_i, L] とした ``v = J @ [vx, vy, omega]``。
順運動学は ``J^-1 @ v`` (対称配置では J は正則で逆行列が存在する)。

ステッパ変換 — L6470 が用いる 2 種類の単位に注意:
- **Run (速度)**: L6470 の速度レジスタは **フルステップ/s** 単位 (マイクロステップは
  内部で適用され、指令速度をスケールしない)。よって ``wheel_speed_to_step_hz`` は
  フルステップ/s を返す。
- **Move / odometry (位置)**: 距離は (STEP_MODE に応じた) **マイクロステップ** 単位で
  数えるため、位置推定・デッドレコニングには ``distance_to_microsteps`` を使う。
"""

from __future__ import annotations

import math

import numpy as np

from krilly.config import RobotConfig, load_robot_config


class KiwiKinematics:
    """指定した車体形状に対する kiwi-drive の順運動学・逆運動学。"""

    def __init__(self, config: RobotConfig | None = None) -> None:
        self.cfg = config or load_robot_config()
        L = self.cfg.center_to_wheel_m
        thetas = [math.radians(a) for a in self.cfg.wheel_angles_deg]
        self._J = np.array(
            [[-math.sin(t), math.cos(t), L] for t in thetas], dtype=float
        )
        self._J_inv = np.linalg.inv(self._J)
        self._m_per_fullstep = self.cfg.wheel_circumference_m / self.cfg.steps_per_rev
        self._m_per_microstep = self.cfg.metres_per_microstep

    # -- 運動学 -------------------------------------------------------------
    def body_to_wheels(
        self, vx: float, vy: float, omega: float
    ) -> tuple[float, float, float]:
        """ボディ速度 (m/s, m/s, rad/s) -> 各輪の接地面速度 (m/s)。"""
        v = self._J @ np.array([vx, vy, omega], dtype=float)
        return (float(v[0]), float(v[1]), float(v[2]))

    def wheels_to_body(
        self, v0: float, v1: float, v2: float
    ) -> tuple[float, float, float]:
        """各輪の接地面速度 (m/s) -> ボディ速度 (vx, vy, omega)。"""
        b = self._J_inv @ np.array([v0, v1, v2], dtype=float)
        return (float(b[0]), float(b[1]), float(b[2]))

    # -- ステッパ変換 -------------------------------------------------------
    def wheel_speed_to_step_hz(self, v_mps: float) -> float:
        """各輪の接地面速度 (m/s) -> L6470 の Run 速度 (フルステップ/s)。"""
        return v_mps / self._m_per_fullstep

    def step_hz_to_wheel_speed(self, step_hz: float) -> float:
        """L6470 の Run 速度 (フルステップ/s) -> 各輪の接地面速度 (m/s)。"""
        return step_hz * self._m_per_fullstep

    def distance_to_microsteps(self, distance_m: float) -> float:
        """車輪の転がり距離 (m) -> マイクロステップ数 (Move / odometry 用)。"""
        return distance_m / self._m_per_microstep

    def microsteps_to_distance(self, microsteps: float) -> float:
        """マイクロステップ数 -> 車輪の転がり距離 (m)。"""
        return microsteps * self._m_per_microstep

    # -- 補助メソッド -------------------------------------------------------
    def body_to_wheel_step_hz(
        self, vx: float, vy: float, omega: float
    ) -> tuple[float, float, float]:
        """ボディ速度 -> 各輪の L6470 Run 速度 (フルステップ/s)。"""
        return tuple(  # type: ignore[return-value]
            self.wheel_speed_to_step_hz(v) for v in self.body_to_wheels(vx, vy, omega)
        )
