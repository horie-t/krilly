"""走行スクリプトへの融合の組み込み (#187) のテスト。"""

from __future__ import annotations

import argparse

import pytest

from krilly.app.motion_fusion import (
    MotionFusion,
    add_fusion_args,
    fusion_config_from_args,
    motion_fusion_from_args,
)
from krilly.localization.delayed_fusion import FusionConfig
from krilly.localization.estimator import DeadReckoning

MS = 1_000_000


class FakeLogger:
    """カメラのスレッドの代わり。start/stop の回数だけ数える。"""

    def __init__(self, camera, detector, probe, path, on_measure=None, yaw_every=0,
                 clock_ns=None):
        self.on_measure = on_measure
        self.probe = probe
        self.yaw_every = yaw_every
        self.frames = 0
        self.starts = self.stops = 0

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1


class Clock:
    def __init__(self):
        self.t = 1_000 * MS

    def __call__(self):
        return self.t


def make(est, **kw):
    clock = Clock()
    fuser = MotionFusion(None, None, est, FusionConfig(gain=1.0, max_step_m=1.0), clock_ns=clock,
                         logger_factory=FakeLogger, **kw)
    return fuser, clock


def row(t_ns, left_mm):
    return {"sensor_ts_ns": t_ns, "left_mm": left_mm, "forward_mm": None, "yaw_deg": None}


def test_begin_fuse_end_cycle():
    est = DeadReckoning(phi=0.0)                  # 東向きで止まっている
    fuser, clock = make(est)
    assert not fuser.active and fuser.before_update() is None
    fuser.begin("[1] 東へ")
    assert fuser.active and fuser.logger.starts == 1
    for _ in range(10):                           # 20ms ごとに履歴を積む
        clock.t += 20 * MS
        fuser.after_update(0.02)
    fuser._on_measure(row(clock.t - 30 * MS, +5.0))   # カメラ: 実は 5mm 左に居る
    fused = fuser.before_update()
    assert fused[3] == 1
    assert est.y == pytest.approx(0.005)
    s = fuser.end()
    assert not fuser.active and fuser.logger.stops == 1
    assert s.dy_m == pytest.approx(0.005) and s.applied == 1
    assert fuser.total.dy_m == pytest.approx(0.005)
    assert "融合 1 回" in s.describe()


def test_leftover_measurements_are_dropped_and_history_restarts():
    est = DeadReckoning(phi=0.0)
    fuser, clock = make(est)
    fuser.begin()
    clock.t += 200 * MS
    fuser.after_update(0.02)
    fuser.end()
    fuser._on_measure(row(clock.t, +5.0))         # 止まっている間に届いた古い測定
    est.y = 0.004                                 # 停止時の補正で推定が跳んだ
    fuser.begin()
    assert fuser.before_update() == (0.0, 0.0, 0.0, 0)   # 捨ててある
    assert len(fuser.fusion.history) == 1          # 履歴は作り直し


def test_yaw_is_off_by_default_and_args_round_trip():
    p = argparse.ArgumentParser()
    add_fusion_args(p)
    args = p.parse_args(["--fuse", "--fuse-gain", "0.3", "--fuse-delay", "50"])
    cfg = fusion_config_from_args(args, -0.8)
    assert cfg.gain == 0.3 and cfg.delay_s == pytest.approx(0.05)
    assert cfg.camera_yaw_offset_rad == pytest.approx(-0.8 * 3.141592653589793 / 180)
    assert args.fuse_yaw_every == 0
    assert motion_fusion_from_args(p.parse_args([]), None, None, DeadReckoning(), 0.0) is None
    est = DeadReckoning()
    fuser, _ = make(est, yaw_every=args.fuse_yaw_every)
    assert fuser.logger.probe.do_yaw is False and fuser.logger.probe.do_tilt is False


def test_stop_report_splits_cross_and_along_by_last_direction():
    from scripts.stop_report import group_args, parse

    lines = [
        "x INFO krilly.speed_run:     Nへ3セル 完了 2.30s 残差 ...",
        "x INFO krilly.speed_run:   停止 (0, 3): 軸角 +0.1° / 位置ずれ 実測 前後=+6.0mm 左右=-2.4mm (壁 1/2 枚) -> 補正",
        "x INFO krilly.speed_run:     Wへ1->Eへ2 完了 2.30s 残差 ...",
        "x INFO krilly.speed_run:   停止 (2, 3): 軸角 +0.1° / 位置ずれ 実測 前後=-1.2mm 左右=- (壁 1/0 枚) -> 補正",
        "x INFO krilly.speed_run:   停止 (2, 3): 軸角 +0.1° / 位置ずれ 実測 前後=+9.9mm 左右=+9.9mm (壁) -> 補正",
    ]
    stops = parse(lines)
    assert len(stops) == 2                      # 動作の後でない停止 (3 行目) は数えない
    assert stops[0] == {"dir": "N", "cell": (0, 3), "cross": -2.4, "along": 6.0}
    assert stops[1] == {"dir": "E", "cell": (2, 3), "cross": -1.2, "along": None}
    assert group_args(["--label", "a", "x.log", "y.log", "--label", "b", "z.log"]) == [
        ("a", ["x.log", "y.log"]), ("b", ["z.log"])]


def test_edge_budget_is_reset_when_the_leg_changes():
    est = DeadReckoning(phi=0.0)
    fuser, clock = make(est)

    class Motion:
        translating = True
        axis_k = 0
        reference = (0.36, 0.0, 0.0)

    m = Motion()
    fuser.begin()
    fuser.fusion.edge_total["x"] = 0.025
    fuser.fusion.edges_disabled = "上限"
    fuser.before_update(m)
    assert fuser.fusion.edges_disabled == "上限"          # 同じ脚のあいだはそのまま
    m.axis_k = 1                                          # 角を曲がった
    fuser.before_update(m)
    assert fuser.fusion.edges_disabled is None and fuser.fusion.edge_total["x"] == 0.0
