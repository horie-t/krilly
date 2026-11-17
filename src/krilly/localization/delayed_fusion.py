"""遅れて届くカメラの測定を、走行中の推定に融合する (issue #186)。

走行中のカメラの測定は、届いた時点ですでに古い。#184 の実測 (黒い床、EV −2、
0.30 m/s、壁の帯だけ): ``SensorTimestamp`` から処理完了まで 29.5 ms、さらに絵は
撮影時刻より**約 40 ms 前**の推定位置と合う (露光の中心が撮影時刻より前にあること
と、推定が指令から積分されて実際の動きより先行すること)。合わせて約 70 ms、
0.30 m/s で 21 mm。届いた測定を「いま」の推定と比べると、その分だけ進行方向に
偏った補正を掛けてしまう。

そこで推定の**履歴**を持ち、測定は「撮影時刻 − ``delay_s``」の推定と比べる。その差
(= 補正量) を、いまの推定に足す。比べる相手が過去なので、補正量は「その時点での
推定の誤り」であり、推定が積分で運んできた並進は差に入らない。

補正の掛け方:

- **少しずつ寄せる** (``gain``) — 走行中の読みは 1 フレームごとに約 0.7 mm 振れる
  (#184、≈ 1 px)。30 fps で gain 0.2 なら約 5 フレームの平均になる。
- **1 回の補正に上限** (``max_step_*``) — 推定を一気に書き換えると ``CellMotion``
  の目標とのずれが跳ね、機体が揺さぶられる (2026-10-17 のコーナー)。
- **大きすぎる食い違いは捨てる** (``max_error_*``) — 位置を動かすものは強い証拠が
  要る (CLAUDE.md の安全規則)。弱い帯・飽和した帯はそもそも ``cell_offset`` が返さない。
- **補正したら履歴もずらす** — 処理中の次の測定 (補正より前に撮った絵) は、ずらした
  履歴と比べるので、同じ誤りを二度直さない。
- **セル中央から遠い軸は使わない** (``max_offset_m``) — 帯の探索は ±40 px (±24 mm)
  しか追えないので、推定が中央から大きく離れている軸の読みは別の壁を掴んでいる
  かもしれない。
- **進行方向の軸は既定で使わない** (``along_track``) — 進行方向と直角な帯は露光中に
  ブレて縁が崩れ (#184: 0.30 m/s で約 15 px)、走行中は端の壁が見える所でしか読めない。
  横ずれと向き (#187) を先に入れ、前後は壁の切れ目 (#189) で扱う。
- **回っている最中の絵は使わない** (``max_rotation_rad``) — その場旋回中の絵は回転で
  ブレ、帯の位置も軸角も信用できない。

推定の世界座標は、セル中心が ``pitch_m`` の格子点 (整数倍) にあること
(``cell_move_demo`` は開始セルの中心が原点、``speed_run`` は (x·pitch, y·pitch))。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from krilly.localization.estimator import DeadReckoning


def fold_quarter(a: float) -> float:
    """角度 [rad] を (-45°, 45°] に折り返す (迷路の軸は 90° 周期)。"""
    q = math.pi / 2.0
    return (a + q / 2.0) % q - q / 2.0


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


@dataclass(frozen=True)
class FusionConfig:
    """融合の設定。既定値は #184 の実測 (0.30 m/s、黒い床、EV −2) から。"""

    #: 絵が合う推定位置の、撮影時刻からの遅れ [s] (#184: 3 走行で 35〜45 ms)。
    delay_s: float = 0.040
    #: 推定の履歴を持つ長さ [s]。処理完了まで最大 37 ms + delay なので 0.5 s で十分。
    history_s: float = 0.5
    #: 1 回の測定で誤りの何割を直すか。
    gain: float = 0.2
    #: 1 回の補正の上限 [m] / [rad]。30 fps なら 2 mm で 60 mm/s、0.1° で 3°/s まで追える
    #: (#184 の横流れは 6 セルで最大 15 mm = 約 20 mm/s)。
    max_step_m: float = 0.002
    max_step_rad: float = math.radians(0.1)
    #: これより大きな食い違いは誤測定として捨てる [m] / [rad]。
    max_error_m: float = 0.015
    max_error_rad: float = math.radians(1.5)
    #: 推定がセル中央からこの距離より離れている軸では位置を補正しない [m]。
    max_offset_m: float = 0.040
    #: 進行方向の軸も補正するか (既定は横ずれだけ)。
    along_track: bool = False
    #: この速さ [m/s] 以上で動いている車体軸を「進行方向」とみなす。
    moving_mps: float = 0.03
    #: 撮影の直前 ``rotation_window_s`` の間にこれ以上回っていたら絵を使わない [rad]。
    max_rotation_rad: float = math.radians(0.5)
    rotation_window_s: float = 0.033
    #: セルのピッチ [m]。
    pitch_m: float = 0.18
    #: カメラが機体に対して回っている角度 [rad] (+ = CCW、``camera_yaw_offset_deg``)。
    camera_yaw_offset_rad: float = 0.0


@dataclass(frozen=True)
class CameraMeasurement:
    """1 フレームの測定。測れなかった値は None。"""

    t_ns: int                          # SensorTimestamp (time.monotonic_ns と同じ時計)
    forward_m: float | None = None     # セル中央に対する機体のずれ (車体 +x)
    left_m: float | None = None        # 同 (車体 +y)
    axis_rad: float | None = None      # axis_yaw の軸角 ((-45°, 45°]、カメラの向き)


@dataclass
class FusionResult:
    """1 回の :meth:`DelayedFusion.apply` で何をしたか (記録用)。"""

    dx_m: float = 0.0          # 推定に足した量 (世界座標)
    dy_m: float = 0.0
    dphi_rad: float = 0.0
    errors: dict[str, float] = field(default_factory=dict)   # 軸 -> 食い違い (補正前)
    skipped: dict[str, str] = field(default_factory=dict)    # 軸 -> 使わなかった理由

    @property
    def applied(self) -> bool:
        return bool(self.dx_m or self.dy_m or self.dphi_rad)


class PoseHistory:
    """推定姿勢の履歴。時刻で線形補間して引ける。"""

    def __init__(self, length_s: float) -> None:
        self.length_ns = int(length_s * 1e9)
        self._buf: deque[tuple[int, float, float, float]] = deque()

    def __len__(self) -> int:
        return len(self._buf)

    def record(self, t_ns: int, pose: tuple[float, float, float]) -> None:
        if self._buf and t_ns <= self._buf[-1][0]:
            return                       # 時刻が進んでいない記録は捨てる
        self._buf.append((t_ns, *pose))
        while self._buf and self._buf[0][0] < t_ns - self.length_ns:
            self._buf.popleft()

    def at(self, t_ns: float) -> tuple[float, float, float] | None:
        """時刻 ``t_ns`` の姿勢 (範囲外なら None)。"""
        buf = self._buf
        if len(buf) < 2 or t_ns < buf[0][0] or t_ns > buf[-1][0]:
            return None
        lo, hi = 0, len(buf) - 1
        while hi - lo > 1:               # 二分探索
            mid = (lo + hi) // 2
            if buf[mid][0] <= t_ns:
                lo = mid
            else:
                hi = mid
        t0, x0, y0, p0 = buf[lo]
        t1, x1, y1, p1 = buf[hi]
        w = 0.0 if t1 == t0 else (t_ns - t0) / (t1 - t0)
        return (x0 + w * (x1 - x0), y0 + w * (y1 - y0), p0 + w * (p1 - p0))

    def shift(self, dx: float, dy: float, dphi: float) -> None:
        """履歴全体をずらす (補正を掛けたとき。同じ誤りを二度直さないため)。"""
        self._buf = deque((t, x + dx, y + dy, p + dphi) for t, x, y, p in self._buf)


class DelayedFusion:
    """推定 ``est`` の履歴を持ち、遅れて届く :class:`CameraMeasurement` を融合する。

    使い方 (制御ループの 1 tick):

    1. 届いている測定を :meth:`apply` する
    2. ``CellMotion.update`` などで推定を進める
    3. :meth:`record` で、いまの時刻と推定を履歴に入れる
    """

    def __init__(self, est: DeadReckoning, config: FusionConfig | None = None) -> None:
        self.est = est
        self.cfg = config or FusionConfig()
        self.history = PoseHistory(self.cfg.history_s)
        self.applied = 0
        self.rejected = 0

    def record(self, t_ns: int) -> None:
        self.history.record(t_ns, self.est.pose)

    def apply(self, m: CameraMeasurement) -> FusionResult:
        cfg = self.cfg
        res = FusionResult()
        t = m.t_ns - cfg.delay_s * 1e9
        past = self.history.at(t)
        if past is None:
            res.skipped["all"] = "履歴の外"
            self.rejected += 1
            return res
        before = self.history.at(t - cfg.rotation_window_s * 1e9)
        if before is not None and abs(past[2] - before[2]) > cfg.max_rotation_rad:
            res.skipped["all"] = "回転中"
            self.rejected += 1
            return res
        x, y, phi = past
        c, s = math.cos(phi), math.sin(phi)
        # 撮影時点の車体フレームの速度 (進行方向の判定に使う)
        vel = (0.0, 0.0)
        earlier = self.history.at(t - 0.05e9)
        if earlier is not None:
            vx, vy = (x - earlier[0]) / 0.05, (y - earlier[1]) / 0.05
            vel = (vx * c + vy * s, -vx * s + vy * c)
        cx = round(x / cfg.pitch_m) * cfg.pitch_m
        cy = round(y / cfg.pitch_m) * cfg.pitch_m
        dx = dy = dphi = 0.0
        for name, meas, (ux, uy), v in (("前後", m.forward_m, (c, s), vel[0]),
                                        ("左右", m.left_m, (-s, c), vel[1])):
            if meas is None:
                continue
            if not cfg.along_track and abs(v) >= cfg.moving_mps:
                res.skipped[name] = "進行方向"
                continue
            current = (x - cx) * ux + (y - cy) * uy
            if abs(current) > cfg.max_offset_m:
                res.skipped[name] = "セル中央から遠い"
                continue
            error = meas - current
            res.errors[name] = error
            if abs(error) > cfg.max_error_m:
                res.skipped[name] = "食い違いが大きい"
                continue
            step = _clamp(cfg.gain * error, cfg.max_step_m)
            dx += step * ux
            dy += step * uy
        if m.axis_rad is not None:
            body_axis = m.axis_rad - cfg.camera_yaw_offset_rad
            error = fold_quarter(body_axis - phi)
            res.errors["方位"] = error
            if abs(error) > cfg.max_error_rad:
                res.skipped["方位"] = "食い違いが大きい"
            else:
                dphi = _clamp(cfg.gain * error, cfg.max_step_rad)
        if dx or dy or dphi:
            self.est.x += dx
            self.est.y += dy
            self.est.phi += dphi
            self.history.shift(dx, dy, dphi)
            self.applied += 1
        elif res.skipped:
            self.rejected += 1
        res.dx_m, res.dy_m, res.dphi_rad = dx, dy, dphi
        return res
