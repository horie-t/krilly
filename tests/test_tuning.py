"""走行チューニング設定 (issue #21) の単体テスト。"""

from __future__ import annotations

import argparse
import math

import pytest

from krilly.hal.l6470 import NO_SPI, fault_flags
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.hal.l6470 import L6470Profile
from krilly.motion.cell_motion import CellMotionConfig
from krilly.motion.velocity_driver import RampLimits
from krilly.motion.tuning import (
    add_tuning_args,
    build_tuning,
    check_limits,
    describe_faults,
    peak_wheel_value,
)


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_tuning_args(p)
    return p.parse_args(argv)


def test_defaults_round_trip():
    """引数なしなら各設定クラスの既定 (= #21 の採用点) がそのまま出る。"""
    tuning = build_tuning(parse([]))
    assert tuning.motion == CellMotionConfig()
    assert tuning.limits == RampLimits()
    assert tuning.profile == L6470Profile()


def test_overrides_all_three_layers():
    tuning = build_tuning(parse([
        "--v", "0.24", "--omega", "2.0", "--accel", "1.0", "--angular-accel", "8.0",
        "--decel", "0.9", "--angular-decel", "7.0", "--kval", "0x60",
        "--kval-hold", "0x30", "--max-speed", "800", "--driver-accel", "2000",
    ]))
    assert tuning.motion.v_max == 0.24
    assert tuning.motion.decel_mps2 == 0.9
    assert tuning.limits.max_angular_accel_radps2 == 8.0
    assert tuning.profile.kval_run == tuning.profile.kval_acc == 0x60
    assert tuning.profile.kval_hold == 0x30
    assert tuning.profile.max_speed_steps_s == 800
    assert tuning.profile.acc_steps_s2 == tuning.profile.dec_steps_s2 == 2000


def test_other_gains_are_inherited():
    """速度以外のゲイン・許容値は base から引き継ぐ (勝手に既定へ戻さない)。"""
    base = CellMotionConfig(pos_tol_m=0.003, k_cross=5.0)
    tuning = build_tuning(parse(["--v", "0.2"]), motion=base)
    assert tuning.motion.pos_tol_m == 0.003
    assert tuning.motion.k_cross == 5.0
    assert tuning.motion.v_max == 0.2


def test_defaults_are_within_driver_limits():
    """現行の既定値は L6470 の MAX_SPEED / ACC に収まっている。"""
    assert check_limits(build_tuning(parse([]))) == []


def test_speed_over_max_speed_warns():
    warnings = check_limits(build_tuning(parse(["--v", "0.35"])))
    assert any("MAX_SPEED" in w for w in warnings)
    # --max-speed を上げれば消える
    assert check_limits(build_tuning(parse(["--v", "0.35", "--max-speed", "800"]))) == []


def test_ramp_over_driver_accel_warns():
    warnings = check_limits(build_tuning(parse(["--accel", "2.0", "--decel", "2.0"])))
    assert any("ACC/DEC" in w for w in warnings)


def test_decel_over_ramp_warns():
    """減速エンベロープはランプ上限に丸められるので、超えていたら知らせる。"""
    warnings = check_limits(build_tuning(parse(["--decel", "1.5"])))
    assert any("--decel" in w for w in warnings)
    assert check_limits(build_tuning(parse(
        ["--decel", "1.5", "--accel", "1.5", "--driver-accel", "3000"]))) == []


def test_peak_wheel_value_matches_kinematics():
    """符号の最悪組み合わせ = 各項の絶対値の和になっている。"""
    kin = KiwiKinematics()
    peak = peak_wheel_value(kin, 0.2, 0.1, 1.0)
    worst = max(
        abs(w)
        for sx in (1, -1) for sy in (1, -1) for sw in (1, -1)
        for w in kin.body_to_wheels(0.2 * sx, 0.1 * sy, 1.0 * sw)
    )
    assert peak == pytest.approx(worst)


def test_pure_forward_peak_is_the_sine_of_the_spoke_angle():
    """前進だけなら車輪速度のピークは vx*|sin(spoke)| (運動学の式そのもの)。"""
    kin = KiwiKinematics()
    expected = max(abs(math.sin(math.radians(a))) for a in kin.cfg.wheel_angles_deg)
    assert peak_wheel_value(kin, 1.0, 0.0, 0.0) == pytest.approx(expected)


def test_describe_faults():
    ok = 0x7E03            # フォールトビットが全て 1 (= 異常なし)
    assert describe_faults([ok, ok, ok]) is None
    step_loss = ok & ~(1 << 13)
    text = describe_faults([ok, step_loss, ok])
    assert text is not None and "M1" in text and "STEP_LOSS_A" in text
    # UVLO の電源投入ラッチなど、無視したいフラグは除ける
    uvlo = ok & ~(1 << 9)
    assert describe_faults([uvlo], ignore=("UVLO",)) is None


def test_fault_flags_detects_dead_spi():
    assert fault_flags(0x0000) == {NO_SPI}
    assert fault_flags(0xFFFF) == {NO_SPI}
    assert fault_flags(0x7E03) == set()
    # 0x7C03 は電源投入直後の値: UVLO だけがラッチされている
    assert fault_flags(0x7C03) == {"UVLO"}


def test_lateral_peak_is_checked_too():
    """横移動は前進より高い輪速を要求するので、そこで先に MAX_SPEED を超える (#76)。

    純 vy の係数は純 vx より大きく、飽和するのは W0 (駆動方向が真横の車輪)。
    前進だけ見ていると見落とし、横移動の距離だけが縮む。
    """
    kin = KiwiKinematics()
    m = CellMotionConfig()
    forward = peak_wheel_value(kin, m.v_max, m.v_cross_max, m.omega_hold_max)
    lateral = peak_wheel_value(kin, m.v_cross_max, m.v_max, m.omega_hold_max)
    assert lateral > forward                       # 横の方が高い

    # 前進は収まるが横は超える MAX_SPEED を選ぶと、横のケースだけが警告に出る
    between = (kin.wheel_speed_to_step_hz(forward) + kin.wheel_speed_to_step_hz(lateral)) / 2
    warnings = check_limits(build_tuning(parse(["--max-speed", str(between)])))
    assert any("MAX_SPEED" in w and "横移動" in w for w in warnings)
