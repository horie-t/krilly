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
    # 既定の decel は driver ランプ以下なのでそのまま通る
    assert motion._decel == pytest.approx(motion.cfg.decel_mps2)
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


def test_floor_keeps_primary_axis_moving_but_not_past_tolerance(motion):
    """残量が許容外の間は下限速度を保証し、許容内では素通しする。"""
    motion.start_forward_cells(1)
    assert motion._with_floor(0.001, 0.010, 0.05) == pytest.approx(0.010)   # 下限まで持ち上げ
    assert motion._with_floor(-0.001, 0.010, -0.05) == pytest.approx(-0.010)  # 符号は残量側
    assert motion._with_floor(0.05, 0.010, 0.05) == pytest.approx(0.05)     # 下限以上は素通し
    tol = motion.cfg.pos_tol_m
    assert motion._with_floor(0.001, 0.010, tol / 2) == pytest.approx(0.001)  # 許容内は素通し


def test_floor_cannot_jump_over_the_tolerance_band(motion):
    """``下限 * dt <= 2 * 許容値`` — 下限速度でも 1 tick で許容帯を跳び越さない。"""
    cfg = motion.cfg
    assert cfg.min_v * DT <= 2 * cfg.pos_tol_m
    assert cfg.min_omega * DT <= 2 * cfg.angle_tol_rad


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
    # 注入した 10mm / 5° に対して十分戻っている。横ずれの上限は「保持の速度上限で
    # 1 セル走る間に詰められる量」で決まるので、v_max を上げると残りも増える。
    assert abs(cross) < 0.010 / 4
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
    # 終端は許容値で切るが、その後の惰行が乗るので不感帯の幅で見る
    assert motion.est.phi == pytest.approx(
        math.pi / 2, abs=motion.cfg.retry_angle_tol_rad)


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
    # 定常偏差 ≈ drift/k_heading。整定後の待ち (settle_dwell_s) の間は 0 指令なので
    # P 制御が効かず、バイアス残りがそのまま積み上がる分も足して見積もる。
    cfg = motion.cfg
    bound = drift / cfg.k_heading + drift * cfg.settle_dwell_s
    assert abs(motion.residual()[2]) < bound * 1.5
    assert abs(motion.residual()[1]) < 2e-3                 # 横ずれも抑えられている


def test_turn_compensates_rotational_slip_via_gyro(motion):
    """回転スリップ (実回転が指令の 80%) でも、ジャイロを見ているので 90° 回りきる。"""
    slip = 0.8
    motion.start_turn_left()
    ticks = 0
    commanded = 0.0
    for _ in range(MAX_TICKS):
        gz = motion.driver.current_velocity[2] * slip   # 実際の回転 = 指令 × slip
        commanded += motion.driver.current_velocity[2] * DT
        ticks += 1
        if motion.update(DT, gyro_rate=gz):
            break
    else:
        pytest.fail("完了しなかった")
    # 終端は許容値で切るが、その後の惰行が乗るので不感帯の幅で見る
    assert motion.est.phi == pytest.approx(
        math.pi / 2, abs=motion.cfg.retry_angle_tol_rad)
    # 補償の証拠: 実回転を 90° にするために、指令はその 1/slip 倍を出している
    assert commanded == pytest.approx(math.pi / 2 / slip, rel=0.1)


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


# --- 機械の不感帯 (ガタ) の中はやり直さないこと (#21) ------------------------
def _turn_settling_into_play(cfg: CellMotionConfig, offset_rad: float):
    """旋回させ、停止時に機体がガタの帯のどこかへ落ち着く様子を注入する。

    実機 (#21): 旋回を止めると機体はガタの帯 (1.6-2°) の中のどこかに落ち着いて
    **そこに留まる**。往復して 0 に戻る振動ではないので待っても消えず、車輪を
    動かしてやり直しても、車輪が動いた先でまた帯のどこかに落ちるだけで詰まらない。
    どの角速度でも残差 0.9-1.45°、やり直し 12/12 回、改善 0 回だった。
    """
    kin = KiwiKinematics(config=ROBOT)
    motion = CellMotion(VelocityDriver(FakeChain(), kinematics=kin),
                        DeadReckoning(kin), config=cfg, maze=MAZE)
    motion.start_turn_left()
    dropped = False
    ticks = 0
    for i in range(MAX_TICKS):
        gz = motion.driver.current_velocity[2]
        if motion.phase is not Phase.RUN and not dropped:
            gz -= offset_rad / DT      # 主軸が終わった瞬間に帯の端へ落ちる (1 tick 分)
            dropped = True
        ticks = i + 1
        if motion.update(DT, gyro_rate=gz):
            break
    else:
        pytest.fail("完了しなかった")
    return motion, ticks


def test_retry_deadband_skips_the_residual_the_play_makes_unfixable():
    """ガタの帯の内側の残差はやり直さない (追いかけても詰まらないため)。"""
    play = math.radians(1.2)          # 実機で観測された 0.9-1.45° の真ん中あたり
    cfg = CellMotionConfig(retry_angle_tol_rad=math.radians(1.6))
    motion, ticks = _turn_settling_into_play(cfg, play)
    assert motion.retries == 0
    # 許容値は超えているが不感帯の内側、という状態で完了している
    residual = abs(motion.residual()[2])
    assert cfg.angle_tol_rad < residual <= cfg.retry_angle_tol_rad

    # 不感帯を狭めると (従来の挙動) 追いかけに入り、時間を捨てたうえで詰まらない
    chasing = CellMotionConfig(retry_angle_tol_rad=cfg.angle_tol_rad)
    slow, slow_ticks = _turn_settling_into_play(chasing, play)
    assert slow.retries == 1
    assert slow_ticks > ticks


def test_retry_still_fires_for_a_real_overrun():
    """不感帯より大きく行き過ぎたら、従来どおりやり直す (安全側は殺さない)。"""
    cfg = CellMotionConfig()
    motion, _ticks = _turn_settling_into_play(cfg, math.radians(5.0))
    assert motion.retries == 1


# --- 横移動 (旋回せず平行移動する、#76) --------------------------------------
def _axis_motion():
    kin = KiwiKinematics(config=ROBOT)
    return CellMotion(VelocityDriver(FakeChain(), kinematics=kin),
                      DeadReckoning(kin), maze=MAZE)


@pytest.mark.parametrize(
    "k, expect",
    [
        (0, (0.180, 0.0)),      # 前 (基準方位 = +x)
        (1, (0.0, 0.180)),      # 左 (+y)
        (-1, (0.0, -0.180)),    # 右 (-y)
        (2, (-0.180, 0.0)),     # 後
    ],
)
def test_start_move_cells_advances_the_reference_along_the_axis(motion, k, expect):
    """基準点は「基準方位から k×90° 回した向き」へ進む。機体は回らない。"""
    motion.start_move_cells(1, k)
    x, y, phi = motion.reference
    assert (x, y) == pytest.approx(expect, abs=1e-12)
    assert phi == pytest.approx(0.0)          # 基準方位は変わらない


def test_start_move_uses_the_reference_heading_not_the_estimate():
    """主軸は基準方位を基準にする (推定方位がずれていても軸は動かない)。"""
    kin = KiwiKinematics(config=ROBOT)
    est = DeadReckoning(kin, phi=math.pi / 2)          # 北を向いて置いた
    m = CellMotion(VelocityDriver(FakeChain(), kinematics=kin), est, maze=MAZE)
    m.start_move_cells(1, +1)                          # 左 = 西へ
    x, y, _phi = m.reference
    assert (x, y) == pytest.approx((-0.180, 0.0), abs=1e-9)


@pytest.mark.parametrize("k", [1, -1, 2])
def test_lateral_move_converges_and_holds_heading(k):
    """横移動でも主軸が詰まり、方位は基準へ保持される。"""
    m = _axis_motion()
    m.start_move_cells(1, k)
    run_until_done(m)
    assert abs(m.remaining) <= m.cfg.pos_tol_m
    along, cross, heading = m.residual()
    assert abs(along) < 2e-3 and abs(cross) < 2e-3
    assert abs(heading) < math.radians(0.1)


def test_lateral_move_uses_the_distance_tolerance_not_the_angle_one():
    """横移動に角度許容が漏れていないこと (#76 の設計上の罠の回帰テスト)。

    許容値を「FORWARD なら距離、else 角度」と書くと、距離系の Kind を足したときに
    角度許容 (0.3° = 0.0052) が黙って適用される。pos_tol_m (0.0015) の 3.5 倍で、
    しかも単位が違うので値の大小では気づけない。
    """
    m = _axis_motion()
    m.start_move_cells(1, +1)
    assert m._tolerance() == m.cfg.pos_tol_m
    assert m._retry_tolerance() == m.cfg.retry_pos_tol_m
    run_until_done(m)
    assert abs(m.remaining) <= m.cfg.pos_tol_m        # 角度許容なら 0.0052 まで許してしまう


def test_lateral_move_corrects_cross_drift():
    """横移動中に主軸と直交する方向へずらしても戻る。"""
    m = _axis_motion()
    m.start_move_cells(1, +1)                          # 左へ = 主軸は +y、直交は x
    for _ in range(10):
        m.update(DT)
    m.est.x += 0.008                                   # 8mm 前へずれた
    run_until_done(m)
    along, cross, _ = m.residual()
    assert abs(along) < 2e-3 and abs(cross) < 2e-3


def test_square_of_lateral_and_forward_moves_returns_to_origin():
    """前 → 左 → 後 → 右 の 1 セル正方形で基準点も推定も原点に戻る。"""
    m = _axis_motion()
    for k in (0, 1, 2, -1):
        m.start_move_cells(1, k)
        run_until_done(m)
    assert m.reference[:2] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert m.est.pose[:2] == pytest.approx((0.0, 0.0), abs=3e-3)


def test_axis_projection_reduces_to_the_forward_one_at_k0(motion):
    """k=0 では主軸フレームと基準方位フレームが厳密に一致する。"""
    motion.start_move_cells(1, 0)
    for _ in range(20):
        motion.update(DT)
    assert motion._axis_remaining() == motion._along_remaining()
    assert motion._axis_cross() == motion._cross_error()


def test_turn_resets_the_axis_to_forward():
    """旋回したら主軸は前に戻る (横移動の指定が残ると次の前進が横へ飛ぶ)。"""
    m = _axis_motion()
    m.start_move_cells(1, +1)
    run_until_done(m)
    m.start_turn_left()
    assert m._axis_k == 0


def test_axis_from_quarter_turns_matches_the_maze_directions():
    """迷路方角 -> quarter_turns -> _axis_k の対応が正しいこと (#76)。

    走行スクリプトは ``start_move_cells(1, quarter_turns(facing, direction))`` で
    進むので、ここがずれると機体が違う方角へ飛ぶ。北を向いたまま各方角へ 1 セル
    動いて、基準点がそのセル中心へ行き、**方位が変わらない**ことを確認する。
    """
    from krilly.solver.maze import Direction
    from krilly.strategy.explorer import heading_rad, quarter_turns

    kin = KiwiKinematics(config=ROBOT)
    for d in Direction:
        est = DeadReckoning(kin, x=0.0, y=0.0, phi=heading_rad(Direction.N))
        m = CellMotion(VelocityDriver(FakeChain(), kinematics=kin), est, maze=MAZE)
        m.start_move_cells(1, quarter_turns(Direction.N, d))
        x, y, phi = m.reference
        want = tuple(v * MAZE.cell_pitch_m for v in d.delta)
        assert (x, y) == pytest.approx(want, abs=1e-9), d
        assert phi == pytest.approx(heading_rad(Direction.N))   # 機体は回らない
