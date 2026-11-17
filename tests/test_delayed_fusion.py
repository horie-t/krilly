"""遅れて届くカメラの測定の融合 (#186) のテスト。"""

from __future__ import annotations

import math

import pytest

from krilly.localization.delayed_fusion import (
    CameraMeasurement,
    DelayedFusion,
    FusionConfig,
    PoseHistory,
    fold_quarter,
)
from krilly.localization.estimator import DeadReckoning

MS = 1_000_000
PITCH = 0.18


def body_offset(x, y, phi):
    """真の位置の、最寄りのセル中央からのずれ (車体フレーム) = カメラが読む値。"""
    dx = x - round(x / PITCH) * PITCH
    dy = y - round(y / PITCH) * PITCH
    c, s = math.cos(phi), math.sin(phi)
    return dx * c + dy * s, -dx * s + dy * c


class Sim:
    """東 (φ=0) へ ``v`` で走る機体。推定は真の位置から ``bias`` だけずれている。

    カメラは撮影時刻 ``t`` の ``delay`` 前の**真の**位置を読む (= #184 の 40 ms)。
    """

    def __init__(self, v=0.30, bias=(0.0, 0.0, 0.0), delay_s=0.040, cfg=None):
        self.v = v
        self.true = [0.0, 0.0, 0.0]
        self.est = DeadReckoning(x=bias[0], y=bias[1], phi=bias[2])
        self.fusion = DelayedFusion(self.est, cfg or FusionConfig(delay_s=delay_s))
        self.delay_s = delay_s
        self.t_ns = 0
        self.true_log = []                  # (t_ns, true pose)

    def tick(self, dt=0.02):
        self.t_ns += int(dt * 1e9)
        self.true[0] += self.v * dt
        self.est.x += self.v * dt           # 推定も同じだけ進む (誤りは一定のまま)
        self.true_log.append((self.t_ns, tuple(self.true)))
        self.fusion.record(self.t_ns)

    def true_at(self, t_ns):
        best = min(self.true_log, key=lambda p: abs(p[0] - t_ns))
        return best[1]

    def measure(self, sensor_ts, yaw=None, along=True):
        x, y, phi = self.true_at(sensor_ts - self.delay_s * 1e9)
        f, l = body_offset(x, y, phi)
        return CameraMeasurement(int(sensor_ts), forward_m=f if along else None, left_m=l,
                                 axis_rad=yaw)


def test_history_interpolates_and_forgets():
    h = PoseHistory(0.1)
    for i in range(20):
        h.record(i * 10 * MS, (i * 0.01, 0.0, 0.0))
    assert h.at(185 * MS) == pytest.approx((0.185, 0.0, 0.0))
    assert h.at(50 * MS) is None             # 0.1 s より前は捨ててある
    h.shift(0.001, 0.002, 0.003)
    assert h.at(190 * MS) == pytest.approx((0.191, 0.002, 0.003))


def test_cross_track_bias_converges_with_clamped_steps():
    sim = Sim(bias=(0.0, -0.010, 0.0))      # 推定は 10 mm 右 (真は y=0)
    for _ in range(10):
        sim.tick()
    steps = []
    for k in range(60):
        sim.tick()
        res = sim.fusion.apply(sim.measure(sim.t_ns - 30 * MS))
        steps.append(abs(res.dy_m))
    assert sim.est.y == pytest.approx(0.0, abs=0.0005)
    assert max(steps) <= FusionConfig().max_step_m + 1e-12     # 1 回 2 mm まで


def test_along_track_is_skipped_while_moving_by_default():
    sim = Sim(bias=(0.005, 0.0, 0.0))
    for _ in range(10):
        sim.tick()
    res = sim.fusion.apply(sim.measure(sim.t_ns - 30 * MS))
    assert res.skipped.get("前後") == "進行方向"
    assert sim.est.x - sim.true[0] == pytest.approx(0.005)


def test_delay_compensation_removes_the_motion_bias():
    """進行方向も補正する設定で、遅れを補えば真の位置に収束し、補わなければ偏る。"""
    def run(cfg_delay):
        cfg = FusionConfig(delay_s=cfg_delay, along_track=True, gain=0.3,
                           max_offset_m=0.09, max_error_m=0.05)
        sim = Sim(v=0.30, bias=(0.0, 0.0, 0.0), cfg=cfg)
        for _ in range(10):
            sim.tick()
        for _ in range(25):
            sim.tick()
            sim.fusion.apply(sim.measure(sim.t_ns - 20 * MS))
        return sim.est.x - sim.true[0]

    good = run(0.040)
    assert good == pytest.approx(0.0, abs=0.001)
    bad = run(0.0)                           # 遅れを無視すると 0.3 m/s × 40 ms = 12 mm 遅れ側へ
    assert bad < -0.006


def test_two_measurements_in_flight_do_not_double_correct():
    cfg = FusionConfig(gain=1.0, max_step_m=1.0)
    sim = Sim(bias=(0.0, 0.008, 0.0), cfg=cfg)
    for _ in range(10):
        sim.tick()
    m1 = sim.measure(sim.t_ns - 60 * MS)     # 処理中だった古い絵
    m2 = sim.measure(sim.t_ns - 30 * MS)
    sim.fusion.apply(m2)
    assert sim.est.y == pytest.approx(0.0, abs=1e-6)
    res = sim.fusion.apply(m1)               # ずらした履歴と比べるので、もう直さない
    assert res.dy_m == pytest.approx(0.0, abs=1e-6)
    assert sim.est.y == pytest.approx(0.0, abs=1e-6)


def test_large_disagreement_is_rejected():
    sim = Sim(bias=(0.0, 0.030, 0.0))
    for _ in range(10):
        sim.tick()
    res = sim.fusion.apply(sim.measure(sim.t_ns - 30 * MS))
    assert res.skipped["左右"] == "食い違いが大きい"
    assert not res.applied
    assert sim.est.y == pytest.approx(0.030)


def test_far_from_cell_centre_is_not_used():
    cfg = FusionConfig(along_track=True)
    sim = Sim(v=0.0, bias=(0.0, 0.0, 0.0), cfg=cfg)
    sim.true[0] = sim.est.x = 0.070          # セル中央から 70 mm (帯の探索の外)
    for _ in range(10):
        sim.tick()
    res = sim.fusion.apply(CameraMeasurement(sim.t_ns - 30 * MS, forward_m=0.0, left_m=0.0))
    assert res.skipped["前後"] == "セル中央から遠い"
    assert "左右" not in res.skipped


def test_outside_history_and_rotation_are_rejected():
    sim = Sim()
    sim.tick()
    assert sim.fusion.apply(CameraMeasurement(-10**9, left_m=0.0)).skipped["all"] == "履歴の外"
    est = DeadReckoning()
    fusion = DelayedFusion(est)
    for i in range(20):                      # 2 rad/s でその場旋回
        est.phi = 2.0 * i * 0.02
        fusion.record(i * 20 * MS)
    res = fusion.apply(CameraMeasurement(300 * MS, left_m=0.0, axis_rad=0.0))
    assert res.skipped["all"] == "回転中"


def test_heading_correction_folds_and_uses_camera_offset():
    offset = math.radians(-0.8)
    cfg = FusionConfig(gain=1.0, max_step_rad=1.0, camera_yaw_offset_rad=offset)
    est = DeadReckoning(phi=math.pi / 2 + math.radians(0.5))    # 北向きで 0.5° CCW にずれた推定
    fusion = DelayedFusion(est, cfg)
    for i in range(10):
        fusion.record(i * 20 * MS)
    # 機体は真北。カメラは機体に対して -0.8° 回っているので軸角は -0.8° と読む
    res = fusion.apply(CameraMeasurement(150 * MS, axis_rad=offset))
    assert math.degrees(res.dphi_rad) == pytest.approx(-0.5, abs=1e-6)
    assert est.phi == pytest.approx(math.pi / 2, abs=1e-9)


def test_fold_quarter():
    assert math.degrees(fold_quarter(math.radians(91))) == pytest.approx(1.0)
    assert math.degrees(fold_quarter(math.radians(-46))) == pytest.approx(44.0)
