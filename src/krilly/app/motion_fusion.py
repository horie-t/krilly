"""走行スクリプトに、動作中のカメラの融合を組み込む部品 (issue #187)。

``speed_run`` / ``search_run`` は、動作 (``run_primitive``) の間は制御ループだけを
回し、止まってから ``camera.capture()`` で撮って壁と位置を見る。#186 の融合は
**走っている間**にカメラを撮り続けるので、カメラを取り合わないように:

- 動作の始め (:meth:`MotionFusion.begin`) にカメラのスレッドを起こし、
- 動作が終わったら (:meth:`MotionFusion.end`) スレッドを止めてから、
- 今までどおり止まった状態で撮る。

融合の履歴は**動作ごとに作り直す**。止まっている間に停止時の補正 (``apply_cell_offset``
/ ``apply_axis_heading``) が推定を跳ばすので、前の動作の履歴は跳ぶ前の推定のままで、
それと比べると同じずれを二度直してしまう。

止まるたびの補正はそのまま残す。そこでは進行方向の位置も直せる (走行中の融合は
既定で横ずれだけ、#186)。向きの融合も既定では切ってある (``--fuse-yaw-every 0``):
推定の向きが進む向きで +0.4° / −0.3° 違う原因 (方位保持の定常偏差?) が分かるまで、
向きは止まったときの補正に任せる (``docs/claude/in-motion-camera.md``)。
"""

from __future__ import annotations

import argparse
import math
import queue
import time
from dataclasses import dataclass

from krilly.localization.delayed_fusion import DelayedFusion, FusionConfig
from krilly.logging_config import get_logger

log = get_logger("krilly.motion_fusion")


def add_fusion_args(p: argparse.ArgumentParser) -> None:
    """走行中の融合の引数 (``cell_move_demo`` と同じ名前)。**既定はすべて今の挙動。**"""
    g = p.add_argument_group("走行中のカメラの融合 (#186/#187)")
    g.add_argument("--fuse", action="store_true",
                   help="動作の最中もカメラを撮り続け、横ずれを遅れを補って推定に融合する。"
                        "止まるたびの補正はそのまま")
    g.add_argument("--fuse-gain", type=float, default=None, metavar="割合",
                   help="1 回の測定で食い違いの何割を直すか (既定 0.2)")
    g.add_argument("--fuse-delay", type=float, default=None, metavar="ミリ秒",
                   help="絵が合う推定位置の、撮影時刻からの遅れ (既定 40、#184 の実測)")
    g.add_argument("--fuse-yaw-every", type=int, default=0, metavar="N",
                   help="走行中も向きを融合し、axis_yaw を N 枚に 1 枚測る (既定 0 = 向きは止まった"
                        "ときだけ。推定の向きが進む向きで違う件が未解決のため)")
    g.add_argument("--fuse-along", action="store_true",
                   help="走行中に進行方向の軸も補正する (既定は横ずれだけ)")
    g.add_argument("--frame-log", default=None, metavar="PREFIX",
                   help="--fuse の間のフレームを PREFIX_frames.csv に、制御の tick を "
                        "PREFIX_ticks.csv に記録する (scripts.frame_log_report で解析)")


def fusion_config_from_args(args, camera_yaw_offset_deg: float) -> FusionConfig:
    over = {"camera_yaw_offset_rad": math.radians(camera_yaw_offset_deg),
            "along_track": bool(getattr(args, "fuse_along", False))}
    if getattr(args, "fuse_gain", None) is not None:
        over["gain"] = args.fuse_gain
    if getattr(args, "fuse_delay", None) is not None:
        over["delay_s"] = args.fuse_delay / 1000.0
    return FusionConfig(**over)


@dataclass
class FusionSummary:
    """1 動作ぶんの融合の結果 (ログ用)。"""

    frames: int = 0
    applied: int = 0
    rejected: int = 0
    dx_m: float = 0.0          # 推定に足した量の合計 (世界座標)
    dy_m: float = 0.0
    dphi_rad: float = 0.0
    abs_m: float = 0.0         # 位置の補正の絶対値の合計

    def describe(self) -> str:
        return ("融合 %d 回 (フレーム %d、使わず %d) 補正 X%+.1f Y%+.1f mm%s" % (
            self.applied, self.frames, self.rejected, self.dx_m * 1000, self.dy_m * 1000,
            "" if not self.dphi_rad else " 方位%+.2f°" % math.degrees(self.dphi_rad)))


class MotionFusion:
    """動作の間だけカメラのスレッドを回し、届いた測定を推定に融合する。

    制御ループからは 1 tick ごとに :meth:`before_update` (届いた測定を融合) →
    ``motion.update`` → :meth:`after_update` (履歴と記録) の順に呼ぶ。推定に触る
    のは制御ループのスレッドだけ。``camera`` は ``capture_with_metadata`` を持つもの。
    """

    def __init__(self, camera, detector, est, config: FusionConfig, *, yaw_cfg=None,
                 yaw_every: int = 0, frame_log: str | None = None,
                 clock_ns=time.monotonic_ns, logger_factory=None) -> None:
        from krilly.perception.frame_log import FrameLogger, ProbeConfig, TickLog, focal_px

        self.est = est
        self.cfg = config
        self.clock_ns = clock_ns
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self.fusion: DelayedFusion | None = None
        self.current = FusionSummary()
        self.total = FusionSummary()
        probe = ProbeConfig(focal_px=focal_px(), yaw=yaw_cfg, do_yaw=yaw_every > 0,
                            do_tilt=False)
        factory = logger_factory or FrameLogger
        self.logger = factory(
            camera, detector, probe,
            f"{frame_log}_frames.csv" if frame_log else None,
            on_measure=self._on_measure, yaw_every=yaw_every, clock_ns=clock_ns)
        self.ticks = TickLog(f"{frame_log}_ticks.csv", clock_ns=clock_ns) if frame_log else None

    # -- カメラのスレッドから -------------------------------------------------
    def _on_measure(self, row: dict) -> None:
        from krilly.perception.frame_log import measurement_from_row

        m = measurement_from_row(row)
        if m is not None:
            self._queue.put(m)

    # -- 制御ループから ---------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.fusion is not None

    def begin(self, label: str = "") -> None:
        """動作の始めに呼ぶ: 履歴を作り直してカメラのスレッドを起こす。"""
        if self.active:
            self.end()
        while True:                       # 前の動作の取り残しを捨てる
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self.fusion = DelayedFusion(self.est, self.cfg)
        self.fusion.record(self.clock_ns())
        self.current = FusionSummary()
        if self.ticks is not None:
            self.ticks.label = label
        self.logger.start()

    def before_update(self) -> tuple[float, float, float, int] | None:
        """届いている測定をすべて融合し、この tick の補正量の合計を返す。"""
        if self.fusion is None:
            return None
        dx = dy = dphi = 0.0
        n = 0
        while True:
            try:
                m = self._queue.get_nowait()
            except queue.Empty:
                break
            res = self.fusion.apply(m)
            dx += res.dx_m
            dy += res.dy_m
            dphi += res.dphi_rad
            n += 1
            self.current.abs_m += abs(res.dx_m) + abs(res.dy_m)
        self.current.dx_m += dx
        self.current.dy_m += dy
        self.current.dphi_rad += dphi
        return (dx, dy, dphi, n)

    def after_update(self, dt: float, motion=None, rate: float | None = None,
                     fused: tuple[float, float, float, int] | None = None) -> None:
        if self.fusion is None:
            return
        self.fusion.record(self.clock_ns())
        if self.ticks is not None and motion is not None:
            self.ticks.record(dt, self.est.pose, motion.reference,
                              motion.driver.current_velocity, rate, fused)

    def end(self) -> FusionSummary:
        """動作の終わりに呼ぶ: スレッドを止める (以降は止まった状態でカメラを使える)。"""
        if self.fusion is None:
            return self.current
        self.logger.stop()
        s = self.current
        s.frames = self.logger.frames - self.total.frames
        s.applied, s.rejected = self.fusion.applied, self.fusion.rejected
        t = self.total
        t.frames += s.frames
        t.applied += s.applied
        t.rejected += s.rejected
        t.dx_m += s.dx_m
        t.dy_m += s.dy_m
        t.dphi_rad += s.dphi_rad
        t.abs_m += s.abs_m
        self.fusion = None
        return s

    def close(self) -> None:
        self.end()
        if self.ticks is not None:
            self.ticks.close()

    def __enter__(self) -> "MotionFusion":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def motion_fusion_from_args(args, camera, detector, est, camera_yaw_offset_deg: float,
                            yaw_cfg=None) -> MotionFusion | None:
    """``--fuse`` が付いていれば :class:`MotionFusion` を作る (無ければ None)。"""
    if not getattr(args, "fuse", False):
        return None
    cfg = fusion_config_from_args(args, camera_yaw_offset_deg)
    fuser = MotionFusion(camera, detector, est, cfg, yaw_cfg=yaw_cfg,
                         yaw_every=max(0, args.fuse_yaw_every),
                         frame_log=getattr(args, "frame_log", None))
    log.info("動作の最中もカメラを融合する (#187): gain %.2f / 遅れ %.0fms / 向き%s / 進行方向%s%s",
             cfg.gain, cfg.delay_s * 1000,
             "も (%d 枚に 1 枚)" % args.fuse_yaw_every if args.fuse_yaw_every > 0
             else "は止まったときだけ",
             "も補正" if cfg.along_track else "は補正しない",
             "、記録 %s_frames.csv / _ticks.csv" % args.frame_log if args.frame_log else "")
    return fuser
