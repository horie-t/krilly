"""カメラによる認識: 赤い壁の上端を検出する (2 レンジの HSV マスク -> 重心)。

M1 (issue #7) と M4 (issues #16, #17) で実装する。
"""

from .axis_yaw import (
    AxisYaw,
    AxisYawConfig,
    axis_yaw,
    calibrated_axis_yaw_config,
    median_axis_yaw,
    yaw_delta_rad,
)
from .red_wall import (
    RedDetectorConfig,
    RedRegion,
    annotate,
    detect_red_regions,
    red_mask,
)
from .wall_detect import (
    CALIBRATED_RED,
    Roi,
    WallDetector,
    WallDetectorConfig,
    body_walls_to_maze,
    calibrated_config,
    calibrated_rois,
    default_rois,
    roi_red_fraction,
    update_maze_walls,
)

__all__ = [
    "AxisYaw",
    "AxisYawConfig",
    "axis_yaw",
    "calibrated_axis_yaw_config",
    "median_axis_yaw",
    "yaw_delta_rad",
    "RedDetectorConfig",
    "RedRegion",
    "annotate",
    "detect_red_regions",
    "red_mask",
    "CALIBRATED_RED",
    "Roi",
    "WallDetector",
    "WallDetectorConfig",
    "body_walls_to_maze",
    "calibrated_config",
    "calibrated_rois",
    "default_rois",
    "roi_red_fraction",
    "update_maze_walls",
]
