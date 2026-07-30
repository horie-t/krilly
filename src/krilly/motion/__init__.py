"""運動制御: ボディ速度 (vx, vy, omega) -> 3輪、加減速ランプ、位置連動プリミティブ。

M2 (issues #9, #11) と M4 (issue #17) で実装する。
"""

from .cell_motion import CellMotion, CellMotionConfig, Kind, Phase
from .velocity_driver import RampLimits, VelocityDriver

__all__ = [
    "CellMotion",
    "CellMotionConfig",
    "Kind",
    "Phase",
    "RampLimits",
    "VelocityDriver",
]
