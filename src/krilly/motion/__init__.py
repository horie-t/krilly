"""運動制御: ボディ速度 (vx, vy, omega) -> 3輪、加減速ランプ。

M2 (issues #9, #11) と M4 (issue #17) で実装する。
"""

from .velocity_driver import RampLimits, VelocityDriver

__all__ = ["RampLimits", "VelocityDriver"]
