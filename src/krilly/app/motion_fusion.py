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
import statistics
import time
from dataclasses import dataclass, field

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
    g.add_argument("--fuse-edges", action="store_true",
                   help="壁の切れ目の縁から、進行方向の位置も走行中に補正する (#189)")
    g.add_argument("--post-top", choices=("auto", "red", "white"), default="auto",
                   help="柱の上面の色 (#189)。auto = 縁ごとに判定。試走会で分かったら決め打ちする")
    g.add_argument("--fuse-edges-dry", action="store_true",
                   help="壁の切れ目の縁を探して補正量を試算するが、推定には入れない (#189 の確認用)。"
                        "動作の完了ログに「縁(試算)」として出る")
    g.add_argument("--ahead-check", action="store_true",
                   help="走りながら、今の脚が通るセルの境目に壁が見えたら止める (#188)")
    g.add_argument("--ahead-check-dry", action="store_true",
                   help="前方の境目の見張りを試算だけ (止めずに記録する、#188)")
    g.add_argument("--frame-log", default=None, metavar="PREFIX",
                   help="--fuse の間のフレームを PREFIX_frames.csv に、制御の tick を "
                        "PREFIX_ticks.csv に記録する (scripts.frame_log_report で解析)")
    g.add_argument("--frame-log-save", type=int, default=0, metavar="N",
                   help="--frame-log 中、N 枚に 1 枚を PREFIX_frames/ に JPEG で残す (処理時間を食う)")


def fusion_config_from_args(args, camera_yaw_offset_deg: float) -> FusionConfig:
    over = {"camera_yaw_offset_rad": math.radians(camera_yaw_offset_deg),
            "along_track": bool(getattr(args, "fuse_along", False)),
            "use_edges": bool(getattr(args, "fuse_edges", False)
                              or getattr(args, "fuse_edges_dry", False)),
            "edge_observe_only": bool(getattr(args, "fuse_edges_dry", False))}
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
    edge_would_dx_m: float = 0.0   # 縁の補正の試算 (--fuse-edges-dry)
    edge_would_dy_m: float = 0.0
    edge_note: str = ""            # 縁を止めた理由など
    #: 縁の食い違い [m] (フレームごと、軸の名前 -> 並び)。**試算の合計は目安にならない**
    #: — 試算では推定を直さないので同じ食い違いが毎フレーム残り、合計は上限に張り付く。
    edge_errors: dict = field(default_factory=dict)
    ahead_max: float = 0.0         # 今の脚の境目で見えた赤の割合の最大 (#188)
    ahead_checked: int = 0
    ahead_at: tuple | None = None   # 最大が出た (境目までの距離, 通り道からの横) [m]
    ahead_note: str = ""

    def describe(self) -> str:
        return ("融合 %d 回 (フレーム %d、使わず %d) 補正 X%+.1f Y%+.1f mm%s" % (
            self.applied, self.frames, self.rejected, self.dx_m * 1000, self.dy_m * 1000,
            "" if not self.dphi_rad else " 方位%+.2f°" % math.degrees(self.dphi_rad))) + (
            "".join(" / %s 食い違い 中央値 %+.1f mm (%d 枚)" % (
                name, statistics.median(v) * 1000, len(v))
                for name, v in sorted(self.edge_errors.items()) if v)) + (
            "" if not self.edge_note else " / ** %s **" % self.edge_note) + (
            "" if not self.ahead_checked else " / 前方 最大 %.2f (%d 回%s)" % (
                self.ahead_max, self.ahead_checked,
                "" if self.ahead_at is None else "、%.0f mm 先・横 %+.0f mm" % (
                    self.ahead_at[0] * 1000, self.ahead_at[1] * 1000))) + (
            "" if not self.ahead_note else " / ** %s **" % self.ahead_note)


class MotionFusion:
    """動作の間だけカメラのスレッドを回し、届いた測定を推定に融合する。

    制御ループからは 1 tick ごとに :meth:`before_update` (届いた測定を融合) →
    ``motion.update`` → :meth:`after_update` (履歴と記録) の順に呼ぶ。推定に触る
    のは制御ループのスレッドだけ。``camera`` は ``capture_with_metadata`` を持つもの。
    """

    def __init__(self, camera, detector, est, config: FusionConfig, *, yaw_cfg=None,
                 yaw_every: int = 0, frame_log: str | None = None, save_every: int = 0,
                 post_top: str = "auto", ahead=None,
                 clock_ns=time.monotonic_ns, logger_factory=None) -> None:
        from krilly.perception.frame_log import FrameLogger, ProbeConfig, TickLog, focal_px

        self.est = est
        self.cfg = config
        self.clock_ns = clock_ns
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self.fusion: DelayedFusion | None = None
        self._leg_axis: int | None = None     # 今の脚の主軸 (縁の上限を脚ごとに数え直す)
        self.current = FusionSummary()
        self.total = FusionSummary()
        probe = ProbeConfig(focal_px=focal_px(), yaw=yaw_cfg, do_yaw=yaw_every > 0,
                            do_tilt=False, do_edges=config.use_edges, post_top=post_top,
                            do_ahead=ahead is not None)
        #: 前方の境目の見張り (#188、``app.ahead_watch.AheadWatch``)
        self.ahead = ahead
        factory = logger_factory or FrameLogger
        self.logger = factory(
            camera, detector, probe,
            f"{frame_log}_frames.csv" if frame_log else None,
            on_measure=self._on_measure, yaw_every=yaw_every, clock_ns=clock_ns,
            **({"save_every": save_every, "save_dir": f"{frame_log}_frames"}
               if frame_log and save_every > 0 else {}))
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
        self._leg_axis = None
        if self.ahead is not None:
            self.ahead.reset()
        self.current = FusionSummary()
        if self.ticks is not None:
            self.ticks.label = label
        self.logger.start()

    @property
    def ahead_stop(self):
        """前方の境目に壁が見えて**止めるべき**なら :class:`AheadAlarm` (試算だけなら None)。"""
        if self.ahead is None or self.ahead.alarm is None or self.ahead.cfg.observe_only:
            return None
        return self.ahead.alarm

    def before_update(self, motion=None) -> tuple[float, float, float, int] | None:
        """届いている測定をすべて融合し、この tick の補正量の合計を返す。

        ``motion`` (``CellMotion``) を渡すと、前方の境目の見張り (#188) も回す。
        """
        if self.fusion is None:
            return None
        if motion is not None and motion.translating:
            # 縁の補正の合計の上限 (25 mm) は**脚ごと**に数え直す。経路全体を 1 動作で
            # 走ると (--chain-legs 4、約 3.4 m)、上限に届いて残りの区間で縁が効かなく
            # なった (2026-11-19 の long8)。
            if self._leg_axis is not None and motion.axis_k != self._leg_axis:
                self.fusion.reset_edge_budget()
            self._leg_axis = motion.axis_k
        dx = dy = dphi = 0.0
        n = 0
        while True:
            try:
                m = self._queue.get_nowait()
            except queue.Empty:
                break
            res = self.fusion.apply(m)
            if (self.ahead is not None and motion is not None and m.ahead is not None
                    and motion.translating):
                past = self.fusion.history.at(m.t_ns - self.cfg.delay_s * 1e9)
                if past is not None:
                    self.ahead.check(m.ahead, past, motion.reference, motion.axis_k,
                                     motion.reference[2])
            self.current.edge_would_dx_m += res.edge_would_dx_m
            self.current.edge_would_dy_m += res.edge_would_dy_m
            for name, err in res.errors.items():
                if name.startswith("縁"):
                    self.current.edge_errors.setdefault(name, []).append(err)
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
        if self.ahead is not None:
            s.ahead_max = self.ahead.max_fraction
            s.ahead_checked = self.ahead.checked
            s.ahead_at = self.ahead.max_at
            if self.ahead.alarm is not None:
                a = self.ahead.alarm
                s.ahead_note = ("前方の境目に壁 (%.0f mm 先、赤の割合 %.2f、%d フレーム)%s"
                                % (a.distance_m * 1000, a.fraction, a.frames,
                                   " [試算]" if self.ahead.cfg.observe_only else ""))
        if self.fusion.edges_disabled:
            s.edge_note = self.fusion.edges_disabled
            log.warning("  %s (#189)", self.fusion.edges_disabled)
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
    ahead = None
    if getattr(args, "ahead_check", False) or getattr(args, "ahead_check_dry", False):
        from krilly.app.ahead_watch import AheadWatch, AheadWatchConfig

        ahead = AheadWatch(AheadWatchConfig(observe_only=not args.ahead_check))
        log.info("前方の境目を見張る (#188)%s", "" if args.ahead_check else " [試算だけ、止めない]")
    fuser = MotionFusion(camera, detector, est, cfg, yaw_cfg=yaw_cfg, ahead=ahead,
                         yaw_every=max(0, args.fuse_yaw_every),
                         frame_log=getattr(args, "frame_log", None),
                         save_every=getattr(args, "frame_log_save", 0) or 0,
                         post_top=getattr(args, "post_top", "auto"))
    log.info("動作の最中もカメラを融合する (#187): gain %.2f / 遅れ %.0fms / 向き%s / 進行方向%s%s",
             cfg.gain, cfg.delay_s * 1000,
             "も (%d 枚に 1 枚)" % args.fuse_yaw_every if args.fuse_yaw_every > 0
             else "は止まったときだけ",
             "も補正" if cfg.along_track else
             ("は壁の切れ目で試算だけ (#189、推定には入れない)" if cfg.edge_observe_only
              else "は壁の切れ目で補正 (#189)" if cfg.use_edges else "は補正しない"),
             "、記録 %s_frames.csv / _ticks.csv" % args.frame_log if args.frame_log else "")
    return fuser
