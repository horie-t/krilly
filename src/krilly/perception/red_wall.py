"""Red wall-top detection (issue #7).

The downward camera looks at the maze walls, whose tops are painted **red**.
This module turns a BGR frame into a red mask and the centroids of the red
regions (wall tops) — pure OpenCV/NumPy so it is unit-testable without a camera.

Red wraps the HSV hue boundary, so we OR two hue ranges (low and high). Defaults
are a starting point; the competition rules warn walls fade and mix, so tune the
S/V floors on the real maze / lighting and lock the camera exposure & AWB.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RedDetectorConfig:
    """HSV thresholds (OpenCV ranges: H 0-179, S/V 0-255) and filtering."""

    h1_lo: int = 0
    h1_hi: int = 10
    h2_lo: int = 160
    h2_hi: int = 179
    s_min: int = 100
    v_min: int = 70
    min_area: float = 100.0   # ignore red blobs smaller than this (px^2)
    morph_kernel: int = 3     # 0 disables open/close denoising


@dataclass(frozen=True)
class RedRegion:
    """A detected red blob."""

    cx: float
    cy: float
    area: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


def red_mask(bgr: np.ndarray, config: RedDetectorConfig | None = None) -> np.ndarray:
    """Return a uint8 (0/255) mask of red pixels in a BGR image."""
    cfg = config or RedDetectorConfig()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo1 = np.array([cfg.h1_lo, cfg.s_min, cfg.v_min], dtype=np.uint8)
    hi1 = np.array([cfg.h1_hi, 255, 255], dtype=np.uint8)
    lo2 = np.array([cfg.h2_lo, cfg.s_min, cfg.v_min], dtype=np.uint8)
    hi2 = np.array([cfg.h2_hi, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))
    if cfg.morph_kernel > 0:
        k = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_red_regions(
    bgr: np.ndarray, config: RedDetectorConfig | None = None
) -> list[RedRegion]:
    """Detect red blobs and return them as centroids/bboxes, largest first."""
    cfg = config or RedDetectorConfig()
    mask = red_mask(bgr, cfg)
    # findContours returns (contours, hierarchy) on cv2 4.x, (img, ...) on 3.x
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    regions: list[RedRegion] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        regions.append(RedRegion(cx, cy, float(area), cv2.boundingRect(c)))
    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def annotate(bgr: np.ndarray, regions: list[RedRegion]) -> np.ndarray:
    """Draw bounding boxes and centroids for visual bring-up. Returns a copy."""
    out = bgr.copy()
    for r in regions:
        x, y, w, h = r.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (int(round(r.cx)), int(round(r.cy))), 4, (255, 0, 0), -1)
    return out
