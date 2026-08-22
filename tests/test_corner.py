"""止まらない方向転換の幾何 (corner #80) のユニットテスト。

ここで固定するのは「丸めても通れる」という**安全性の根拠**。実機で確かめる前に、
数字が思い込みでないことをここで押さえる。
"""

import math

import pytest

from krilly.motion.corner import (
    MACHINE_RADIUS_M,
    blend_distance_m,
    blend_duration_s,
    corner_bulge_m,
    corner_path,
    corridor_clearance_m,
    min_post_clearance_m,
    peak_speed_ratio,
)
from krilly.motion.cell_motion import CellMotionConfig
from krilly.motion.velocity_driver import RampLimits

TUNED = CellMotionConfig()
TUNED_BLEND = blend_distance_m(TUNED.v_max, min(TUNED.decel_mps2,
                                                RampLimits().max_linear_accel_mps2))


def test_blend_distance_is_the_stopping_distance():
    """ブレンド距離 = 一定減速で止まれる距離。ここが**指令が連続になる**条件。

    切り替え時点の残量が d なら、古い軸の減速エンベロープ sqrt(2*a*d) は
    ちょうど v_max。値が違うと継ぎ目で速度が跳ぶ。
    """
    v, a = 0.24, 0.8
    d = blend_distance_m(v, a)
    assert d == pytest.approx(v * v / (2 * a))
    assert math.sqrt(2 * a * d) == pytest.approx(v)
    assert blend_duration_s(v, a) == pytest.approx(v / a)
    assert blend_distance_m(v, 0.0) == 0.0        # 0 除算しない


def test_corner_path_enters_and_leaves_on_the_centre_lines():
    d = 0.036
    path = corner_path(d)
    assert path[0] == pytest.approx((0.0, -d))     # 入りは +y の中心線上
    assert path[-1] == pytest.approx((d, 0.0))     # 出は +x の中心線上
    # 単調 (行きつ戻りつしない)
    assert all(b[0] >= a[0] and b[1] >= a[1] for a, b in zip(path, path[1:]))


def test_the_bulge_is_a_quarter_of_the_blend_distance():
    d = 0.036
    assert corner_bulge_m(d) == pytest.approx(d / 4)
    # 軌跡の中点が実際にそこを通る
    mid = corner_path(d, samples=3)[1]
    assert mid == pytest.approx((d / 4, -d / 4))


def test_rounding_the_corner_is_looser_than_the_straight_corridor():
    """**丸めても、通れるかを決めているのは相変わらず廊下の方**であること (#80)。

    膨らむ先は入りと出に挟まれた開いている象限で、そこには柱が 1 本あるだけ。
    廊下を柱の真横で通り抜けるときの方が狭く、そこは丸めても丸めなくても通る。
    """
    corridor = corridor_clearance_m()
    assert corridor == pytest.approx(0.084 - MACHINE_RADIUS_M)     # 21.4mm
    assert min_post_clearance_m(TUNED_BLEND) > corridor
    # 半セル分まで丸めても廊下より狭くならない
    for d in (0.01, 0.02, TUNED_BLEND, 0.06, 0.085):
        assert min_post_clearance_m(d) >= corridor, d


def test_the_tuned_blend_has_a_wide_margin():
    """採用中の設定 (v=0.24 / a=0.8) での実数値を固定する。"""
    assert TUNED_BLEND == pytest.approx(0.036, abs=0.001)
    assert corner_bulge_m(TUNED_BLEND) == pytest.approx(0.009, abs=0.001)
    assert min_post_clearance_m(TUNED_BLEND) == pytest.approx(0.034, abs=0.002)


def test_the_speed_dips_through_the_corner_so_the_wheels_see_no_new_peak():
    """コーナーで上がるのは**加速度**であって速度ではない。

    両軸が入れ替わる途中で速度ベクトルは 1/√2 まで落ちる。輪速のピークが増えない
    ので、MAX_SPEED 側の余裕はコーナーでむしろ広がる。
    """
    assert peak_speed_ratio() == pytest.approx(1 / math.sqrt(2))
    assert peak_speed_ratio() < 1.0
