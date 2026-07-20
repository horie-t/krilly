"""加減速ランプ付きの速度指令ドライバ (issue #9)。

ボディ速度指令 (vx, vy, omega) を **レート制限 (台形加速)** で目標へ滑らかに
追従させ、各制御 tick で kiwi 運動学により 3 輪の Run 速度へ変換し、
:meth:`L6470Chain.run_all` で **3 輪同時**に指令する。

加速中も 3 輪の速度比が保たれるため、直進・横移動・その場回転が経路を崩さず
出せる。L6470 の内部 ACC/DEC だけに任せると各輪が独立にランプして速度比が崩れ、
加速中に経路がずれてしまうため、ソフト側で協調ランプをかけるのが本モジュールの
役割である。

``update(dt)`` は純粋な計算のみ (time.sleep を含まない) で、外部の制御ループから
一定レートで呼び出す前提。ハードウェアはコンストラクタに注入された chain 経由で
のみ触るため、フェイク chain でユニットテストできる。
"""

from __future__ import annotations

from dataclasses import dataclass

from krilly.hal.l6470_chain import FWD, REV
from krilly.kinematics.kiwi import KiwiKinematics


@dataclass(frozen=True)
class RampLimits:
    """加減速の上限 (脱調を防ぐため保守的な既定値)。"""

    max_linear_accel_mps2: float = 0.5      # vx, vy の加速度上限 [m/s^2]
    max_angular_accel_radps2: float = 5.0   # omega の角加速度上限 [rad/s^2]


def _rate_limit(current: float, target: float, max_delta: float) -> float:
    """``current`` を ``target`` へ 1 tick あたり最大 ``max_delta`` だけ近づける。"""
    if target > current:
        return min(current + max_delta, target)
    return max(current - max_delta, target)


class VelocityDriver:
    """ボディ速度指令をランプさせて 3 輪 L6470 チェーンを駆動する。

    ``chain`` は :class:`L6470Chain` 互換 (``run_all`` / ``soft_stop_all`` を持つ)
    オブジェクト。テストではフェイクを注入できる。``configure_all`` は呼び出し側
    (スクリプト) で事前に済ませておくこと。
    """

    def __init__(
        self,
        chain,
        kinematics: KiwiKinematics | None = None,
        limits: RampLimits | None = None,
    ) -> None:
        self.chain = chain
        self.kin = kinematics or KiwiKinematics()
        self.limits = limits or RampLimits()
        self._target = (0.0, 0.0, 0.0)   # 目標 (vx, vy, omega)
        self._current = (0.0, 0.0, 0.0)  # ランプ後の現在指令値

    # -- 指令 ---------------------------------------------------------------
    def set_velocity(self, vx: float, vy: float, omega: float) -> None:
        """目標ボディ速度を設定する (即時には反映されずランプで追従)。"""
        self._target = (vx, vy, omega)

    def stop(self) -> None:
        """目標を 0 にする (ランプで減速。即時停止ではない)。"""
        self._target = (0.0, 0.0, 0.0)

    @property
    def current_velocity(self) -> tuple[float, float, float]:
        return self._current

    @property
    def target_velocity(self) -> tuple[float, float, float]:
        return self._target

    def at_target(self, tol: float = 1e-9) -> bool:
        return all(abs(c - t) <= tol for c, t in zip(self._current, self._target))

    # -- 制御ループから一定 dt で呼ぶ ---------------------------------------
    def update(self, dt: float) -> tuple[float, float, float]:
        """ランプを dt 進め、現在指令値で 3 輪を駆動する。現在速度を返す。"""
        tvx, tvy, tw = self._target
        cvx, cvy, cw = self._current
        lin = self.limits.max_linear_accel_mps2 * dt
        ang = self.limits.max_angular_accel_radps2 * dt
        cvx = _rate_limit(cvx, tvx, lin)
        cvy = _rate_limit(cvy, tvy, lin)
        cw = _rate_limit(cw, tw, ang)
        self._current = (cvx, cvy, cw)
        self._command(cvx, cvy, cw)
        return self._current

    def _command(self, vx: float, vy: float, omega: float) -> None:
        # ボディ速度 -> 各輪の符号付き周速 [m/s] -> 向き + フルステップ/s (正)
        wheel_mps = self.kin.body_to_wheels(vx, vy, omega)
        dirs = [FWD if s >= 0 else REV for s in wheel_mps]
        step_hz = [abs(self.kin.wheel_speed_to_step_hz(s)) for s in wheel_mps]
        self.chain.run_all(dirs, step_hz)
