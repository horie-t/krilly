"""CellMotion (1セル前進・90°ターンの位置連動 #17) のユニットテスト。

推定器は指令速度から積分する (オープンループ) ため、ここでは「指令どおりに
動く理想の車体」を相手に閉ループの収束・終端条件・誤差の非累積を確認する。
実機のスリップ相当は ``est`` に外乱を直接注入して模擬する。
"""

import math

import pytest

from krilly.config import MazeConfig, RobotConfig
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.motion.cell_motion import (
    CellMotion,
    CellMotionConfig,
    Kind,
    Phase,
    _clamp,
    _envelope,
)
from krilly.motion.velocity_driver import VelocityDriver

ROBOT = RobotConfig(
    wheel_diameter_m=0.048,
    wheel_count=3,
    center_to_wheel_m=0.05,
    steps_per_rev=200,
    microstep=16,
    wheel_angles_deg=[0.0, 120.0, 240.0],  # スポーク角
)

MAZE = MazeConfig(
    grid_size=16,
    cell_pitch_m=0.180,
    wall_thickness_m=0.012,
    wall_height_m=0.050,
    goal_min=(7, 7),
    goal_max=(8, 8),
)

DT = 0.02
MAX_TICKS = 2000  # 1 プリミティブに与える上限 tick 数 (無限ループ検出用)


class FakeChain:
    """run_all / soft_stop_all を記録するフェイク。"""

    def __init__(self):
        self.calls = []

    def run_all(self, directions, speeds):
        self.calls.append((list(directions), list(speeds)))

    def soft_stop_all(self):
        self.calls.append(("soft_stop",))


@pytest.fixture
def motion():
    kin = KiwiKinematics(config=ROBOT)
    driver = VelocityDriver(FakeChain(), kinematics=kin)
    est = DeadReckoning(kin)
    return CellMotion(driver, est, maze=MAZE)


def run_until_done(m: CellMotion, gyro_rate=None, max_ticks: int = MAX_TICKS) -> int:
    """完了までループを回して消費 tick 数を返す (発散したら fail)。"""
    for i in range(max_ticks):
        if m.update(DT, gyro_rate=gyro_rate):
            return i + 1
    pytest.fail(f"{max_ticks} tick で完了しなかった (phase={m.phase}, 残量={m.remaining})")


# --- ヘルパ関数 ------------------------------------------------------------
def test_clamp():
    assert _clamp(0.5, 0.2) == pytest.approx(0.2)
    assert _clamp(-0.5, 0.2) == pytest.approx(-0.2)
    assert _clamp(0.1, 0.2) == pytest.approx(0.1)


def test_envelope_shape():
    # 残量が大きければ v_max で飽和
    assert _envelope(1.0, 0.12, 0.4, DT) == pytest.approx(0.12)
    # 残量が小さいと sqrt(2*a*s) で減速 (0.01m -> sqrt(0.008)≈0.0894)
    assert _envelope(0.01, 0.12, 0.4, DT) == pytest.approx(math.sqrt(2 * 0.4 * 0.01))
    # 符号は残量の向きに従う (オーバーシュートは戻る方向へ)
    assert _envelope(-0.01, 0.12, 0.4, DT) < 0
    # 1 tick で残量を超えない (0.0001m は 0.0001/dt = 0.005 m/s で頭打ち)
    assert _envelope(0.0001, 0.12, 0.4, DT) == pytest.approx(0.0001 / DT)


def test_envelope_decel_is_clamped_to_driver_ramp(motion):
    # 既定 decel 0.4 <= driver ランプ 0.5 なのでそのまま
    assert motion._decel == pytest.approx(0.4)
    # driver のランプより大きい減速度を指定したらランプ側に丸められる
    kin = KiwiKinematics(config=ROBOT)
    driver = VelocityDriver(FakeChain(), kinematics=kin)
    m = CellMotion(
        driver,
        DeadReckoning(kin),
        config=CellMotionConfig(decel_mps2=10.0, angular_decel_radps2=100.0),
        maze=MAZE,
    )
    assert m._decel == pytest.approx(driver.limits.max_linear_accel_mps2)
    assert m._angular_decel == pytest.approx(driver.limits.max_angular_accel_radps2)


# --- 1セル前進 -------------------------------------------------------------
def test_forward_one_cell_reaches_cell_pitch(motion):
    motion.start_forward_cells(1)
    assert motion.kind is Kind.FORWARD
    assert not motion.done
    run_until_done(motion)
    x, y, phi = motion.est.pose
    assert x == pytest.approx(0.180, abs=motion.cfg.pos_tol_m)
    assert y == pytest.approx(0.0, abs=1e-3)
    assert phi == pytest.approx(0.0, abs=1e-3)
    assert motion.phase is Phase.DONE
    # 基準姿勢も 1 セル進んでいる
    assert motion.reference == pytest.approx((0.180, 0.0, 0.0))


def test_forward_stops_and_does_not_creep(motion):
    motion.start_forward_cells(1)
    run_until_done(motion)
    x_end = motion.est.x
    for _ in range(50):  # 完了後に update を続けても動かない
        motion.update(DT)
    assert motion.est.x == pytest.approx(x_end, abs=1e-9)
    assert motion.driver.current_velocity == pytest.approx((0.0, 0.0, 0.0))


def test_forward_ramps_up_before_decelerating(motion):
    """走行中に v_max を超えず、終盤は減速している (台形プロファイル)。"""
    motion.start_forward_cells(1)
    speeds = []
    while not motion.update(DT):
        speeds.append(motion.driver.current_velocity[0])
    assert max(speeds) == pytest.approx(motion.cfg.v_max, rel=1e-6)
    assert speeds[0] < motion.cfg.v_max          # 立ち上がりはランプ
    assert speeds[-1] < max(speeds)              # 終盤は減速している
    assert all(v <= motion.cfg.v_max + 1e-9 for v in speeds)


def test_forward_corrects_lateral_and_heading_disturbance(motion):
    """走行中に横ずれ・方位ずれを注入しても基準線へ戻る (位置連動)。"""
    motion.start_forward_cells(1)
    for _ in range(10):
        motion.update(DT)
    motion.est.y += 0.010                       # 10mm 右にずれた (推定上)
    motion.est.phi += math.radians(5.0)         # 5° 傾いた
    run_until_done(motion)
    along, cross, heading = motion.residual()
    assert abs(along) <= motion.cfg.pos_tol_m
    assert abs(cross) < 1e-3
    assert abs(heading) < math.radians(0.5)


def test_forward_multiple_cells_do_not_accumulate_error(motion):
    """毎セル外乱を入れても、基準は理想格子なので誤差が累積しない。"""
    for i in range(4):
        motion.start_forward_cells(1)
        for _ in range(5):
            motion.update(DT)
        motion.est.x += 0.004 * (1 if i % 2 == 0 else -1)   # 進み過ぎ/足りない
        motion.est.y += 0.003
        run_until_done(motion)
        assert motion.est.x == pytest.approx(0.180 * (i + 1), abs=motion.cfg.pos_tol_m)
        assert motion.est.y == pytest.approx(0.0, abs=2e-3)


def test_forward_retries_after_overshoot(motion):
    """整定中に行き過ぎたら低速でやり直して許容内に収める。"""
    motion.start_forward_cells(1)
    for _ in range(MAX_TICKS):        # 減速完了 (整定) まで進める
        motion.update(DT)
        if motion.phase is Phase.SETTLE:
            break
    motion.est.x += 0.010             # 10mm 行き過ぎ (残量が負になる)
    assert motion.remaining < 0
    run_until_done(motion)
    assert motion._retries == 1       # やり直しが 1 回入った
    assert motion.est.x == pytest.approx(0.180, abs=motion.cfg.pos_tol_m)


# --- 90°ターン -------------------------------------------------------------
def test_turn_left_90_reaches_target_heading(motion):
    motion.start_turn_left()
    assert motion.kind is Kind.TURN
    run_until_done(motion)
    x, y, phi = motion.est.pose
    assert phi == pytest.approx(math.pi / 2, abs=motion.cfg.angle_tol_rad)
    # その場旋回なので位置は保持される
    assert x == pytest.approx(0.0, abs=1e-3)
    assert y == pytest.approx(0.0, abs=1e-3)
    assert motion.reference[2] == pytest.approx(math.pi / 2)


def test_turn_right_90_is_clockwise(motion):
    motion.start_turn_right()
    omegas = []
    while not motion.update(DT):
        omegas.append(motion.driver.current_velocity[2])
    assert min(omegas) < 0                       # CW = 負の omega
    assert motion.est.phi == pytest.approx(-math.pi / 2, abs=motion.cfg.angle_tol_rad)


def test_turn_180_keeps_direction(motion):
    """180°旋回でも符号が反転しない (残角を積算で持っているため)。"""
    motion.start_turn_left(2)
    omegas = []
    while not motion.update(DT):
        omegas.append(motion.driver.current_velocity[2])
    assert all(w >= -1e-9 for w in omegas)       # 一貫して CCW
    assert abs(_wrap(motion.est.phi - math.pi)) <= motion.cfg.angle_tol_rad


def test_turn_corrects_position_drift(motion):
    """旋回中に位置がずれても (ホロノミックなので) 元の点へ戻す。"""
    motion.start_turn_left()
    for _ in range(10):
        motion.update(DT)
    motion.est.x += 0.008
    motion.est.y -= 0.006
    run_until_done(motion)
    assert motion.est.x == pytest.approx(0.0, abs=2e-3)
    assert motion.est.y == pytest.approx(0.0, abs=2e-3)


def test_turn_absorbs_initial_heading_error(motion):
    """開始時に方位誤差があっても、旋回後は基準方位に一致する。"""
    motion.est.phi = math.radians(-4.0)         # 基準 0° に対して -4° ずれている
    motion.start_turn_left()
    run_until_done(motion)
    assert motion.est.phi == pytest.approx(math.pi / 2, abs=motion.cfg.angle_tol_rad)


# --- 連結シーケンス --------------------------------------------------------
def test_forward_turn_forward_l_path(motion):
    """L字経路: 1セル前進 -> 左90° -> 1セル前進 で (0.18, 0.18, 90°)。"""
    motion.start_forward_cells(1)
    run_until_done(motion)
    motion.start_turn_left()
    run_until_done(motion)
    motion.start_forward_cells(1)
    run_until_done(motion)
    x, y, phi = motion.est.pose
    assert x == pytest.approx(0.180, abs=3e-3)
    assert y == pytest.approx(0.180, abs=3e-3)
    assert phi == pytest.approx(math.pi / 2, abs=math.radians(1.0))


def test_gyro_rate_is_used_for_heading(motion):
    """gyro_rate を渡すと方位はジャイロ側で積分される (#13 の融合方針)。

    ジャイロは「指令した回転 + バイアス残り」を観測する、というモデルで模擬する。
    方位保持の P 制御が効くので、定常偏差は drift/k_heading 程度に収まる。
    """
    drift = math.radians(2.0)   # 2 deg/s のバイアス残り
    motion.start_forward_cells(1)
    phi_moved = False
    for _ in range(MAX_TICKS):
        gz = motion.driver.current_velocity[2] + drift
        if motion.update(DT, gyro_rate=gz):
            break
        phi_moved = phi_moved or abs(motion.est.phi) > 1e-6
    else:
        pytest.fail("完了しなかった")
    assert phi_moved                                        # ジャイロ入力が効いている
    assert abs(motion.residual()[2]) < math.radians(1.0)    # 定常偏差 ≈ drift/k_heading
    assert abs(motion.residual()[1]) < 2e-3                 # 横ずれも抑えられている


def test_turn_compensates_rotational_slip_via_gyro(motion):
    """回転スリップ (実回転が指令の 80%) でも、ジャイロを見ているので 90° 回りきる。"""
    slip = 0.8
    motion.start_turn_left()
    ticks = 0
    for _ in range(MAX_TICKS):
        gz = motion.driver.current_velocity[2] * slip   # 実際の回転 = 指令 × slip
        ticks += 1
        if motion.update(DT, gyro_rate=gz):
            break
    else:
        pytest.fail("完了しなかった")
    assert motion.est.phi == pytest.approx(math.pi / 2, abs=motion.cfg.angle_tol_rad)
    assert ticks > 67   # スリップ分だけ時間が伸びる (理想は約 67 tick)


# --- 基準姿勢の操作・中断 --------------------------------------------------
def test_set_and_sync_reference(motion):
    motion.set_reference(x=0.09, y=0.09, phi=math.pi / 2)
    assert motion.reference == pytest.approx((0.09, 0.09, math.pi / 2))
    motion.est.reset(0.1, 0.1, 0.0)
    motion.sync_reference_to_estimate()
    assert motion.reference == pytest.approx((0.1, 0.1, 0.0))


def test_start_forward_uses_reference_heading(motion):
    """基準方位が北向き (90°) なら、前進は +Y 方向へ進む。"""
    motion.set_reference(phi=math.pi / 2)
    motion.est.reset(0.0, 0.0, math.pi / 2)
    motion.start_forward_cells(1)
    run_until_done(motion)
    assert motion.est.x == pytest.approx(0.0, abs=2e-3)
    assert motion.est.y == pytest.approx(0.180, abs=motion.cfg.pos_tol_m)


def test_abort_stops_and_marks_idle(motion):
    motion.start_forward_cells(1)
    for _ in range(10):
        motion.update(DT)
    motion.abort()
    assert motion.done and motion.phase is Phase.IDLE
    assert motion.driver.target_velocity == (0.0, 0.0, 0.0)
    for _ in range(100):
        motion.update(DT)
    assert motion.driver.current_velocity == pytest.approx((0.0, 0.0, 0.0))


def test_idle_update_keeps_wheels_stopped(motion):
    for _ in range(10):
        assert motion.update(DT) is True
    assert motion.driver.current_velocity == pytest.approx((0.0, 0.0, 0.0))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
