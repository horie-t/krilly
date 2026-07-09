"""Unit tests for red wall-top detection (synthetic images, no camera)."""

import cv2
import numpy as np

from krilly.perception.red_wall import (
    RedDetectorConfig,
    annotate,
    detect_red_regions,
    red_mask,
)


def _blank(h=120, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _hsv_bgr(h, s, v):
    """A single BGR color from an HSV triple (OpenCV ranges)."""
    px = np.array([[[h, s, v]]], dtype=np.uint8)
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


def test_red_mask_flags_red_not_black():
    img = _blank()
    cv2.rectangle(img, (40, 30), (80, 70), (0, 0, 255), -1)  # BGR red
    mask = red_mask(img)
    assert mask[50, 60] == 255      # inside red
    assert mask[5, 5] == 0          # black background


def test_detect_single_red_region_centroid():
    img = _blank()
    cv2.rectangle(img, (40, 30), (80, 70), (0, 0, 255), -1)
    regions = detect_red_regions(img)
    assert len(regions) == 1
    r = regions[0]
    assert abs(r.cx - 60) <= 2      # rect center x
    assert abs(r.cy - 50) <= 2      # rect center y
    assert r.area > 1000


def test_detects_red_at_high_hue_end():
    # hue ~175 is still red and must be caught by the second range
    color = _hsv_bgr(175, 220, 220)
    img = _blank()
    cv2.rectangle(img, (30, 30), (90, 90), color, -1)
    assert len(detect_red_regions(img)) == 1


def test_blue_and_green_not_detected():
    img = _blank()
    cv2.rectangle(img, (10, 10), (50, 50), (255, 0, 0), -1)   # blue
    cv2.rectangle(img, (90, 60), (140, 100), (0, 255, 0), -1)  # green
    assert detect_red_regions(img) == []


def test_min_area_filters_small_specks():
    img = _blank()
    cv2.rectangle(img, (50, 50), (53, 53), (0, 0, 255), -1)   # ~3x3 red
    assert detect_red_regions(img, RedDetectorConfig(min_area=100)) == []


def test_two_regions_sorted_largest_first():
    img = _blank()
    cv2.rectangle(img, (10, 10), (30, 30), (0, 0, 255), -1)    # small
    cv2.rectangle(img, (80, 40), (140, 100), (0, 0, 255), -1)  # large
    regions = detect_red_regions(img)
    assert len(regions) == 2
    assert regions[0].area > regions[1].area


def test_annotate_returns_same_shape_copy():
    img = _blank()
    cv2.rectangle(img, (40, 30), (80, 70), (0, 0, 255), -1)
    regions = detect_red_regions(img)
    out = annotate(img, regions)
    assert out.shape == img.shape
    assert out is not img
    # a green box (0,255,0) was drawn somewhere
    assert (out[:, :, 1] == 255).sum() > 0
