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
- **壁の切れ目の縁 (#189) は、進行方向でも使う** (``use_edges``)。縁は「セルの境目 ±
  柱の半幅」にあるので、帯に沿った方向の位置が分かる (``perception/wall_end.py``)。
  どの境目の縁かは撮影時点の推定から決める — 推定の誤りがセルの半分 (約 80 mm) より
  小さければ取り違えない。横ずれとは別のゲイン・上限を持つ: 縁は 1 本あたり約 2 mm
  の誤差 (#189 の実測) で、溜まった進行方向の誤り (7 セルで 13〜20 mm) を直すために
  食い違いの上限を広めにする。

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


def edge_offset_m(edge, current_m: float, pitch_m: float, inset_m: float) -> float:
    """壁の切れ目の縁 1 つから、その軸での機体の位置 (今のセルの中央から [m]) を出す。

    ``perception.wall_end.offset_from_edge`` と同じ計算 (層を逆向きに import しないため
    ここにも置く)。縁は「境目 ± inset」(切れ目の側へ) にあり、どの境目かは推定の位置
    ``current_m`` から決める。
    """
    side = -inset_m if edge.red_on_plus else inset_m
    if getattr(edge, "post", "red") == "white":
        side = -side                  # 白い柱: 赤は柱の手前で終わる
    edge_est = current_m + edge.offset_mm / 1000.0
    k = round((edge_est - side) / pitch_m - 0.5)
    return (k + 0.5) * pitch_m + side - edge.offset_mm / 1000.0


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
    #: 壁の切れ目の縁を使うか (#189)。
    use_edges: bool = False
    #: 縁の食い違いの何割を直すか / 1 回の上限 [m] / これより大きな食い違いは捨てる [m]。
    #: 縁は 1 フレームに写っている間 (0.30 m/s で約 1 s、30 枚) ずっと読めるので、
    #: 0.3 なら数枚で収束する。上限 3 mm/枚 = 90 mm/s。
    edge_gain: float = 0.3
    edge_max_step_m: float = 0.003
    edge_max_error_m: float = 0.030
    #: 同じ軸の縁どうしの食い違いがこれを超えたら、その軸の縁は使わない [m]
    #: (取り違えた縁や、テープ・継ぎ目を拾った偽の縁が混ざっている)。
    edge_max_spread_m: float = 0.008
    #: 縁の、セルの境目からの距離 [m] (柱の半幅、``wall_end.EDGE_INSET_MM``)。
    edge_inset_m: float = 0.006
    #: **縁は、前のフレームから推定の移動量どおりに動いたものだけを使う** [m]。機体に
    #: 付いて動くもの (影・テープ・ケーブル) は画面上でほとんど動かないので、ここで
    #: 落ちる。2026-11-19 の long8 の衝突では、偽の縁が機体の 120 mm の移動の間に
    #: 25〜30 mm しか動いていなかった。前のフレームとの移動がこれより小さい間は確かめ
    #: られないので使わない (止まっている間は縁の補正をしない)。
    edge_track_tol_m: float = 0.004
    edge_track_min_move_m: float = 0.002
    #: 1 つの融合 (= 1 動作、``MotionFusion.begin`` ごとに作り直す) の間に縁で直す量の
    #: 合計の上限 [m]。超えたらその動作では縁を使わない (暴走の歯止め)。7 セルの直線で
    #: 溜まる進行方向の誤りが 13〜20 mm (#187) なので 25 mm。
    edge_max_total_m: float = 0.025
    #: **縁が 1 本だけのフレーム**は、同じ動作で直前に使った縁の食い違い (から、その後に
    #: 掛けた補正を引いた値) とこの範囲で合うときだけ使う [m]。動作の最初の 1 本は、次の
    #: フレームで合うのを確かめてから使う。ずれて置かれた壁や抜けた柱の縁は 10〜20 mm
    #: 食い違う (2026-11-19 の long8、木の床: 東の列の西側と (5,4) の北の壁) — 同じフレームに
    #: 別の縁があれば ``edge_max_spread_m`` で落ちるが、1 本だけだとそれが効かない。
    edge_single_tol_m: float = 0.006
    #: 1 本だけの縁が直前と合わないことがこのフレーム数続いたら、期待値の方を捨てて
    #: 新しい縁で確かめ直す (直前に使った縁の方が間違っていた場合に、ずっと縁を使え
    #: なくならないため)。
    edge_single_reset_frames: int = 5
    #: 縁の補正を**試算だけ**して推定には入れない (``FusionResult.edge_would_*`` に残す)。
    #: 偽の縁が出ないことを、機体を動かさずに記録で確かめるため (2026-11-19 の衝突の後)。
    edge_observe_only: bool = False


@dataclass(frozen=True)
class CameraMeasurement:
    """1 フレームの測定。測れなかった値は None。"""

    t_ns: int                          # SensorTimestamp (time.monotonic_ns と同じ時計)
    forward_m: float | None = None     # セル中央に対する機体のずれ (車体 +x)
    left_m: float | None = None        # 同 (車体 +y)
    axis_rad: float | None = None      # axis_yaw の軸角 ((-45°, 45°]、カメラの向き)
    #: 壁の切れ目の縁 (#189、``perception.wall_end.BandEdge``: ``axis`` / ``offset_mm`` /
    #: ``red_on_plus`` を持つもの)。
    edges: tuple = ()
    #: 前方の境目の見張り用の、行・列ごとの赤の数 (#188、``perception.ahead.AheadProfiles``)。
    #: 融合には使わない — 撮影時刻と一緒に制御ループへ運ぶだけ。
    ahead: object = None


@dataclass
class FusionResult:
    """1 回の :meth:`DelayedFusion.apply` で何をしたか (記録用)。"""

    dx_m: float = 0.0          # 推定に足した量 (世界座標)
    dy_m: float = 0.0
    dphi_rad: float = 0.0
    errors: dict[str, float] = field(default_factory=dict)   # 軸 -> 食い違い (補正前)
    skipped: dict[str, str] = field(default_factory=dict)    # 軸 -> 使わなかった理由
    edge_would_dx_m: float = 0.0   # 縁の補正の試算 (``edge_observe_only`` のとき、世界座標)
    edge_would_dy_m: float = 0.0

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
        #: 縁の追跡用: 前のフレームの (撮影時点の推定姿勢, 縁の並び)
        self._prev_edges: tuple | None = None
        #: 軸ごとの縁による補正の合計 [m] と、止めたか
        self.edge_total = {"x": 0.0, "y": 0.0}
        #: 軸ごとの「次のフレームで期待する縁の食い違い」(1 本だけの縁を確かめるため)
        self._edge_expect: dict[str, float | None] = {"x": None, "y": None}
        self._edge_mismatch = {"x": 0, "y": 0}
        self.edges_disabled: str | None = None

    def _track_edges(self, edges, past) -> list:
        """前のフレームから、推定の移動量どおりに動いた縁だけを返す。

        縁の ``offset_mm`` は機体から見た位置なので、機体がその軸に Δs 進めば −Δs
        動くはず。同じ帯・同じ向き (``red_on_plus``) の縁で、この関係を満たすものが
        前のフレームにあれば本物とみなす。
        """
        cfg = self.cfg
        prev = self._prev_edges
        self._prev_edges = (past, tuple(edges))
        if prev is None or not edges:
            return []
        (px, py, pphi), pedges = prev
        x, y, phi = past
        c, s = math.cos(phi), math.sin(phi)
        moved = {"x": (x - px) * c + (y - py) * s, "y": -(x - px) * s + (y - py) * c}
        out = []
        for e in edges:
            ds = moved[e.axis]
            if abs(ds) < cfg.edge_track_min_move_m:
                continue                  # 動いていないと、機体に付いたものと区別できない
            for p in pedges:
                if p.band != e.band or p.red_on_plus != e.red_on_plus:
                    continue
                if abs((e.offset_mm - p.offset_mm) / 1000.0 + ds) <= cfg.edge_track_tol_m:
                    out.append(e)
                    break
        return out

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
        tracked = self._track_edges(m.edges, past) if cfg.use_edges else []
        if cfg.use_edges and m.edges and self.edges_disabled is None:
            for axis, (ux, uy), name in (("x", (c, s), "縁:前後"), ("y", (-s, c), "縁:左右")):
                current = (x - cx) * ux + (y - cy) * uy
                errs = []
                axis_edges = [e for e in m.edges if e.axis == axis]
                if axis_edges and not any(e.axis == axis for e in tracked):
                    res.skipped[name] = "前のフレームと動きが合わない (機体と一緒に動く?)"
                    continue
                for e in tracked:
                    if e.axis != axis:
                        continue
                    err = edge_offset_m(e, current, cfg.pitch_m, cfg.edge_inset_m) - current
                    if abs(err) <= cfg.edge_max_error_m:
                        errs.append(err)
                if not errs:
                    if any(e.axis == axis for e in m.edges):
                        res.skipped[name] = "食い違いが大きい"
                    continue
                if max(errs) - min(errs) > cfg.edge_max_spread_m:
                    res.skipped[name] = "縁どうしが食い違う"
                    continue
                error = sum(errs) / len(errs)
                expect = self._edge_expect[axis]
                if len(errs) == 1:
                    if expect is None:
                        # 確かめる相手がいない: 次のフレームで合えば使う
                        self._edge_expect[axis] = error
                        res.skipped[name] = "1 本だけの縁 (次のフレームで確かめる)"
                        continue
                    if abs(error - expect) > cfg.edge_single_tol_m:
                        self._edge_mismatch[axis] += 1
                        if self._edge_mismatch[axis] >= cfg.edge_single_reset_frames:
                            self._edge_expect[axis] = error      # 確かめ直し (次のフレームで)
                            self._edge_mismatch[axis] = 0
                        res.skipped[name] = "1 本だけの縁が直前と合わない"
                        continue
                self._edge_mismatch[axis] = 0
                res.errors[name] = error
                step = _clamp(cfg.edge_gain * error, cfg.edge_max_step_m)
                # 補正を掛けると次の食い違いはその分だけ小さくなるはず (試算では掛けない)
                self._edge_expect[axis] = error - (0.0 if cfg.edge_observe_only else step)
                if cfg.edge_observe_only:
                    # 試算では推定を直さないので食い違いが残り続け、合計の上限に意味が無い
                    res.edge_would_dx_m += step * ux
                    res.edge_would_dy_m += step * uy
                    continue
                if abs(self.edge_total[axis] + step) > cfg.edge_max_total_m:
                    self.edges_disabled = (
                        "縁の補正の合計が %.0f mm を超えるので、この動作では縁を使わない"
                        % (cfg.edge_max_total_m * 1000))
                    res.skipped[name] = self.edges_disabled
                    break
                self.edge_total[axis] += step
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
