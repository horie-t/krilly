"""カメラの傾き測定 (issue #87) のユニットテスト。

壁上面は 3D では正確な格子なので、カメラが真下を向いていれば画像でも平行線になる。
傾いていれば消失点へ収束する。ここでは**既知の傾きで歪ませた合成格子**を作り、
その角度を測り返せることを確かめる。
"""

import math

import cv2
import numpy as np
import pytest

from krilly.perception.camera_tilt import (
    Line,
    find_bands,
    measure_tilt,
    vanishing_point,
)

W, H = 960, 720
FOCAL = 574.0          # px/mm 1.69 x 距離 340mm (実機の値)
PITCH = 305            # 1 セル 180mm の画素数


def grid_mask(pitch: int = PITCH, thickness: int = 24) -> np.ndarray:
    """真下から見た理想の格子 (帯の太さは実機と同じくらい)。"""
    m = np.zeros((H, W), np.uint8)
    for y in range(H // 2 % pitch, H, pitch):
        m[max(0, y - thickness // 2):y + thickness // 2, :] = 255
    for x in range(W // 2 % pitch, W, pitch):
        m[:, max(0, x - thickness // 2):x + thickness // 2] = 255
    return m


def tilt_homography(theta_deg: float, focal: float = FOCAL, about_x: bool = True):
    """カメラを ``theta_deg`` 傾けたときの、地面 -> 画像 の射影変換。

    真下向きの画像座標 (u0, v0) から傾いた画像座標 (u, v) への写像。中心は主点。

    ``about_x=True`` は**画像の X 軸**まわりの傾き。奥行きが画像の縦方向に沿って
    変わるので、**縦帯が収束する** = :attr:`TiltResult.pitch_deg` に出る
    (横帯は平行のまま)。``about_x=False`` はその逆。
    """
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    # 主点まわりで [[1,0,0],[0,c,-f s],[0,s/f,c]] (x 軸まわりの傾き)
    core = np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -focal * s],
                     [0.0, s / focal, c]])
    if not about_x:
        core = core[[1, 0, 2]][:, [1, 0, 2]]
    to_c = np.array([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]], float)
    fr_c = np.array([[1, 0, W / 2], [0, 1, H / 2], [0, 0, 1]], float)
    return fr_c @ core @ to_c


def tilted_mask(theta_deg: float, about_x: bool = True) -> np.ndarray:
    m = grid_mask()
    Hm = tilt_homography(theta_deg, about_x=about_x)
    return cv2.warpPerspective(m, Hm, (W, H), flags=cv2.INTER_NEAREST)


# --- 帯の抽出 ---------------------------------------------------------------
def test_finds_every_band_of_an_untilted_grid():
    m = grid_mask()
    hor = find_bands(m, True, min_length=100)
    ver = find_bands(m, False, min_length=100)
    assert len(hor) == len(range(H // 2 % PITCH, H, PITCH))
    assert len(ver) == len(range(W // 2 % PITCH, W, PITCH))


def test_merges_fragments_of_the_same_band():
    """自機に隠されて分断された帯を 1 本にまとめること。

    まとめないと 1 本の壁が複数の線に化け、消失点の当てはめが壊れる
    (実機のフレームで 4 本の壁が 6 本に分裂した)。
    """
    m = grid_mask()
    m[:, 380:560] = 0                      # 中央を自機で隠す
    hor = find_bands(m, True, min_length=100)
    assert len(hor) == len(range(H // 2 % PITCH, H, PITCH))
    assert all(ln.length > 600 for ln in hor), [ln.length for ln in hor]


# --- 消失点 -----------------------------------------------------------------
def test_parallel_lines_have_no_vanishing_point():
    lines = [Line(0.0, y, 1.0, 0.0, 900.0, 0.0) for y in (100.0, 400.0, 700.0)]
    assert vanishing_point(lines) is None


def test_converging_lines_meet_at_the_expected_point():
    # (1000, 300) で交わる 3 本
    target = (1000.0, 300.0)
    lines = []
    for y in (100.0, 300.0, 500.0):
        dx, dy = target[0] - 0.0, target[1] - y
        n = math.hypot(dx, dy)
        lines.append(Line(0.0, y, dx / n, dy / n, 900.0, 0.0))
    vp = vanishing_point(lines)
    assert vp == pytest.approx(target, abs=1e-6)


# --- 傾きの復元 -------------------------------------------------------------
def test_an_untilted_grid_reads_as_no_tilt():
    r = measure_tilt(grid_mask(), FOCAL, min_length=100)
    for angle in (r.roll_deg, r.pitch_deg):
        assert angle is None or abs(angle) < 0.5, r.describe()


@pytest.mark.parametrize("theta", [2.0, 4.0, 8.0])
def test_recovers_a_known_tilt(theta):
    """既知の傾きで歪ませた格子から、その角度を測り返せること。"""
    r = measure_tilt(tilted_mask(theta, about_x=True), FOCAL, min_length=100)
    assert r.pitch_deg == pytest.approx(theta, abs=0.4), r.describe()


@pytest.mark.parametrize("theta", [3.0, 6.0])
def test_recovers_a_tilt_about_the_other_axis(theta):
    r = measure_tilt(tilted_mask(theta, about_x=False), FOCAL, min_length=100)
    assert r.roll_deg == pytest.approx(theta, abs=0.4), r.describe()


def test_the_two_axes_are_independent():
    """片方の軸だけ傾けたとき、もう片方は 0 のままであること。

    軸の取り違えは符号だけ合って意味が違うので、これで固定しておく。
    """
    r = measure_tilt(tilted_mask(6.0, about_x=True), FOCAL, min_length=100)
    assert r.pitch_deg == pytest.approx(6.0, abs=0.4)      # 縦帯が収束
    assert r.roll_deg is None or abs(r.roll_deg) < 1.0, r.describe()


def test_tilt_scales_with_the_assumed_focal_length():
    """焦点距離 (= px/mm x 距離) の仮定が効くので、距離の誤差は角度の誤差になる。"""
    m = tilted_mask(4.0)
    near = measure_tilt(m, FOCAL * 0.9, min_length=100).pitch_deg
    far = measure_tilt(m, FOCAL * 1.1, min_length=100).pitch_deg
    assert near < far
    assert far / near == pytest.approx(1.1 / 0.9, rel=0.05)
