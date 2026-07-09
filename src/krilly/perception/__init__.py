"""Camera perception: detect red wall-tops (two-range HSV mask -> centroid).

Implemented in M1 (issue #7) and M4 (issue #16).
"""

from .red_wall import (
    RedDetectorConfig,
    RedRegion,
    annotate,
    detect_red_regions,
    red_mask,
)

__all__ = [
    "RedDetectorConfig",
    "RedRegion",
    "annotate",
    "detect_red_regions",
    "red_mask",
]
