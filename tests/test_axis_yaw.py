"""赤い壁エッジからの yaw 推定 (axis_yaw #17) のユニットテスト。

実機画像は使わず、既知の角度で赤帯を描いた合成画像で検証する。
"""

import math

import cv2
import numpy as np
import pytest

from krilly.perception.axis_yaw import (
    AxisYawConfig,
    axis_yaw,
    calibrated_axis_yaw_config,
    fold_deg,
    fold_rad,
    median_axis_yaw,
    yaw_delta_rad,
)
from krilly.perception.wall_detect import Roi

RED = (0, 0, 255)   # BGR


def frame_with_bands(angle_deg: float, size=(480, 640), offsets=(-120, 120)) -> np.ndarray:
    """画像中心を通る ``angle_deg`` 傾きの赤帯を、直交する 2 方向に描く。

    実機の「セルを囲む 4 枚の壁上面」を模した合成シーン。
    """
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = w / 2.0, h / 2.0
    for deg in (angle_deg, angle_deg + 90.0):
        a = math.radians(deg)
        ux, uy = math.cos(a), math.sin(a)          # 帯の長手方向
        nx, ny = -uy, ux                           # 法線方向 (オフセット用)
        for off in offsets:
            mx, my = cx + nx * off, cy + ny * off  # 帯の中心
            p1 = (int(mx - ux * 400), int(my - uy * 400))
            p2 = (int(mx + ux * 400), int(my + uy * 400))
            cv2.line(img, p1, p2, RED, thickness=14)
    return img


# --- 折り返し / 差分 -------------------------------------------------------
def test_fold_deg():
    assert fold_deg(0.0) == pytest.approx(0.0)
    assert fold_deg(30.0) == pytest.approx(30.0)
    assert fold_deg(90.0) == pytest.approx(0.0)
    assert fold_deg(100.0) == pytest.approx(10.0)
    assert fold_deg(46.0) == pytest.approx(-44.0)
    assert fold_deg(-91.0) == pytest.approx(-1.0)


def test_fold_rad_matches_fold_deg():
    for deg in (-100.0, -45.0, 0.0, 12.5, 89.0, 271.0):
        assert math.degrees(fold_rad(math.radians(deg))) == pytest.approx(fold_deg(deg), abs=1e-9)


def test_yaw_delta_is_folded_difference():
    a = axis_yaw(frame_with_bands(1.0), AxisYawConfig())
    b = axis_yaw(frame_with_bands(-2.0), AxisYawConfig())
    assert a is not None and b is not None
    # 1° -> -2° は 3° の CW 回転 (90° の倍数を除いた差分)
    assert math.degrees(yaw_delta_rad(a, b)) == pytest.approx(-3.0, abs=0.5)


# --- 軸角の推定 -----------------------------------------------------------
@pytest.mark.parametrize("angle", [-30.0, -10.0, -1.0, 0.0, 2.5, 12.0, 33.0])
def test_axis_yaw_recovers_drawn_angle(angle):
    result = axis_yaw(frame_with_bands(angle), AxisYawConfig())
    assert result is not None
    assert result.angle_deg == pytest.approx(angle, abs=0.5)
    assert result.segments > 0
    assert result.total_length_px > 150.0


def test_axis_yaw_is_periodic_every_90_deg():
    """迷路の軸は 90° 周期なので、+90° 回した画像は同じ軸角を返す。"""
    a = axis_yaw(frame_with_bands(8.0), AxisYawConfig())
    b = axis_yaw(frame_with_bands(98.0), AxisYawConfig())
    assert a is not None and b is not None
    assert a.angle_deg == pytest.approx(b.angle_deg, abs=0.3)


def test_axis_yaw_returns_none_without_red():
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    assert axis_yaw(black, AxisYawConfig()) is None


def test_axis_yaw_returns_none_when_evidence_is_too_short():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(img, (100, 100), (170, 100), RED, thickness=12)   # 70px の帯 1 本だけ
    assert axis_yaw(img, AxisYawConfig(min_total_length_px=500.0)) is None


# --- 除外領域 (機体の写り込み) --------------------------------------------
def test_exclude_region_is_ignored():
    """除外矩形の中に傾いた赤があっても、外の赤帯だけで角度が決まる。"""
    img = frame_with_bands(0.0)
    cv2.line(img, (250, 200), (420, 320), RED, thickness=20)   # 中央に斜めの偽赤
    without = axis_yaw(img, AxisYawConfig())
    with_mask = axis_yaw(img, AxisYawConfig(exclude=[Roi(200, 150, 260, 220)]))
    assert without is not None and with_mask is not None
    assert with_mask.angle_deg == pytest.approx(0.0, abs=0.5)     # 偽赤の影響なし
    assert abs(without.angle_deg) > abs(with_mask.angle_deg)      # 除外しないと引っ張られる


def test_calibrated_config_excludes_the_robot_body():
    cfg = calibrated_axis_yaw_config()
    assert cfg.exclude and cfg.exclude[0].w > 200   # 機体矩形が入っている
    assert cfg.red.s_min == 70                      # 影の壁も拾う緩めのしきい値


# --- 中央値 ---------------------------------------------------------------
def test_median_axis_yaw_picks_the_middle_frame():
    frames = [frame_with_bands(a) for a in (0.0, 2.0, 20.0)]
    result = median_axis_yaw(frames, AxisYawConfig())
    assert result is not None
    assert result.angle_deg == pytest.approx(2.0, abs=0.5)


def test_median_axis_yaw_returns_none_if_all_frames_fail():
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    assert median_axis_yaw([black, black], AxisYawConfig()) is None
