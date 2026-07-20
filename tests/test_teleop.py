"""テレオペのキー→速度写像 apply_key のユニットテスト。"""

import pytest

from scripts.teleop import apply_key

# v_step, omega_step, v_max, omega_max
LIM = (0.1, 0.5, 0.3, 2.0)


def test_wasd_translation():
    assert apply_key(ord("w"), 0, 0, 0, *LIM) == (0.1, 0, 0)    # 前進
    assert apply_key(ord("s"), 0, 0, 0, *LIM) == (-0.1, 0, 0)   # 後退
    assert apply_key(ord("a"), 0, 0, 0, *LIM) == (0, 0.1, 0)    # 左
    assert apply_key(ord("d"), 0, 0, 0, *LIM) == (0, -0.1, 0)   # 右


def test_qe_rotation():
    assert apply_key(ord("q"), 0, 0, 0, *LIM) == (0, 0, 0.5)    # 左旋回(CCW)
    assert apply_key(ord("e"), 0, 0, 0, *LIM) == (0, 0, -0.5)   # 右旋回(CW)


def test_space_stops():
    assert apply_key(ord(" "), 0.2, -0.1, 1.0, *LIM) == (0, 0, 0)


def test_translation_clamped_to_v_max():
    vx = 0.0
    for _ in range(100):
        vx, _, _ = apply_key(ord("w"), vx, 0, 0, *LIM)
    assert vx == pytest.approx(0.3)   # v_max


def test_rotation_clamped_to_omega_max():
    omega = 0.0
    for _ in range(100):
        _, _, omega = apply_key(ord("q"), 0, 0, omega, *LIM)
    assert omega == pytest.approx(2.0)  # omega_max


def test_unknown_key_keeps_state():
    assert apply_key(ord("z"), 0.1, 0.2, 0.3, *LIM) == (0.1, 0.2, 0.3)
    assert apply_key(-1, 0.1, 0.2, 0.3, *LIM) == (0.1, 0.2, 0.3)
