"""カメラによる認識: 赤い壁の上端を検出する (2 レンジの HSV マスク -> 重心)。

M1 (issue #7) と M4 (issue #16) で実装する。
"""

from .red_wall import (
    RedDetectorConfig,
    RedRegion,
    annotate,
    detect_red_regions,
    red_mask,
)
from .wall_detect import (
    Roi,
    WallDetector,
    WallDetectorConfig,
    body_walls_to_maze,
    default_rois,
    roi_red_fraction,
    update_maze_walls,
)

__all__ = [
    "RedDetectorConfig",
    "RedRegion",
    "annotate",
    "detect_red_regions",
    "red_mask",
    "Roi",
    "WallDetector",
    "WallDetectorConfig",
    "body_walls_to_maze",
    "default_rois",
    "roi_red_fraction",
    "update_maze_walls",
]
