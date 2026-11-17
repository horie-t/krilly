#!/usr/bin/env python3
"""走行中のカメラ記録 (``--frame-log``) を解析する (issue #184、off-Pi で動く)。

``cell_move_demo --frame-log PREFIX`` / ``frame_probe --out PREFIX`` が書いた
``PREFIX_frames.csv`` (と、あれば ``PREFIX_ticks.csv``) を読み、次を出す:

1. **処理**: 実効 fps、取りこぼし、各段の処理時間、撮影から受け取り/使えるまでの遅れ、露出
2. **制御ループ**: tick の間隔 (記録のスレッドが GIL で邪魔をしているか)
3. **場面ごとの読み** (静止 / 加速 / 等速 / 減速): カメラ − 推定 の残差のばらつき、傾き
4. **揺れの見分け**: 残差と傾きの比例 (てこ = 壁の上面からの高さなら機体全体の傾き)
5. **加速度との関係**: 傾き・残差が指令の加速度に比例するか
6. **止まった後の収束**: 読みが落ち着くまでの時間
7. **遅れの当てはめ**: 推定をずらして残差が最小になる時間差

推定 (オドメトリ + ジャイロ) は**指令した動き**なので、カメラ − 推定 の残差には
「実際の動き − 指令」と「カメラの揺れ」の両方が乗る。揺れは数 Hz の振動、実際の
ずれはゆっくり溜まるので、ばらつき (標準偏差) は主に揺れを見ている。

例:
    python -m scripts.frame_log_report survey/m7/d6_030
    python -m scripts.frame_log_report survey/m7/d6_030 --plot d6_030.png
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

import numpy as np

from krilly.perception.frame_log import (
    CAMERA_HEIGHT_MM,
    read_csv,
    tilt_to_shift_mm,
    wall_top_lever_mm,
)

CELL_M = 0.18

#: 場面分けのしきい値。指令速度がこれ未満なら静止、加速度がこれ以上なら加減速。
STILL_MPS = 0.005
ACCEL_MPS2 = 0.05

#: 止まった後に「落ち着いた」とみなす幅。
SETTLE_MM = 0.5


def fold_quarter_deg(a: float) -> float:
    """角度 [deg] を (-45, 45] に折り返す (迷路の軸は 90° 周期)。"""
    return (a + 45.0) % 90.0 - 45.0


def interp(ts: np.ndarray, values: np.ndarray, t: float) -> float | None:
    if len(ts) == 0 or t < ts[0] or t > ts[-1]:
        return None
    return float(np.interp(t, ts, values))


class Ticks:
    """tick の記録を時刻で引けるようにしたもの (配列で持つ)。"""

    def __init__(self, rows: list[dict]) -> None:
        rows = [r for r in rows if r.get("t_ns") is not None]
        self.t = np.array([r["t_ns"] for r in rows], dtype=float)
        self.rows = rows

        def col(k):
            return np.array([r.get(k) or 0.0 for r in rows], dtype=float)

        self.x, self.y, self.phi = col("x_m"), col("y_m"), col("phi_rad")
        self.vx, self.vy = col("cmd_vx"), col("cmd_vy")
        self.dt = col("dt_s")
        # 指令の加速度 (車体フレーム)。指令は tick ごとの階段なので差分で出る。
        if len(self.t) > 1:
            dts = np.diff(self.t) / 1e9
            dts[dts <= 0] = np.nan
            self.ax = np.concatenate([[0.0], np.diff(self.vx) / dts])
            self.ay = np.concatenate([[0.0], np.diff(self.vy) / dts])
            self.ax = np.nan_to_num(self.ax)
            self.ay = np.nan_to_num(self.ay)
        else:
            self.ax = self.ay = np.zeros(len(self.t))

    def __len__(self) -> int:
        return len(self.t)

    def at(self, t_ns: float) -> dict | None:
        """時刻 ``t_ns`` の推定姿勢・指令 (線形補間)。範囲外なら None。"""
        if len(self.t) < 2 or t_ns < self.t[0] or t_ns > self.t[-1]:
            return None
        i = int(np.searchsorted(self.t, t_ns))
        i = min(max(i, 1), len(self.t) - 1)
        out = {k: float(np.interp(t_ns, self.t, getattr(self, k)))
               for k in ("x", "y", "phi", "vx", "vy")}
        # 加速度は補間しない (階段の角が丸まる) — 直前の tick の値
        out["ax"], out["ay"] = float(self.ax[i]), float(self.ay[i])
        out["label"] = self.rows[i].get("label") or ""
        return out


def body_offset_from_estimate(x: float, y: float, phi: float) -> tuple[float, float]:
    """推定位置の、最寄りのセル中心からのずれを車体フレーム (前, 左) [mm] で返す。

    ``cell_move_demo`` の推定は開始セルの中心を (0, 0) とするので、セル中心は
    0.18m の格子。カメラの ``cell_offset`` と同じ量になる (理想なら一致する)。
    """
    dx = x - round(x / CELL_M) * CELL_M
    dy = y - round(y / CELL_M) * CELL_M
    c, s = math.cos(phi), math.sin(phi)
    return (dx * c + dy * s) * 1000.0, (-dx * s + dy * c) * 1000.0


def phase_of(state: dict) -> str:
    speed = math.hypot(state["vx"], state["vy"])
    if speed < STILL_MPS:
        return "静止"
    if speed > 0:
        along = (state["ax"] * state["vx"] + state["ay"] * state["vy"]) / speed
    else:
        along = 0.0
    if along > ACCEL_MPS2:
        return "加速"
    if along < -ACCEL_MPS2:
        return "減速"
    return "等速"


def enrich(frames: list[dict], ticks: Ticks | None, tau_ms: float = 0.0) -> list[dict]:
    """各フレームに撮影時刻の推定・指令・場面・残差を足す (``tau_ms`` だけ推定をずらす)。"""
    out = []
    for f in frames:
        g = dict(f)
        ts = f.get("sensor_ts_ns")
        g["state"] = None
        if ticks is not None and ts is not None:
            st = ticks.at(ts + tau_ms * 1e6)
            g["state"] = st
            if st is not None:
                g["phase"] = phase_of(st)
                ef, el = body_offset_from_estimate(st["x"], st["y"], st["phi"])
                g["res_forward_mm"] = (None if f.get("forward_mm") is None
                                       else f["forward_mm"] - ef)
                g["res_left_mm"] = None if f.get("left_mm") is None else f["left_mm"] - el
                g["res_yaw_deg"] = (None if f.get("yaw_deg") is None else
                                    fold_quarter_deg(f["yaw_deg"]
                                                     - fold_quarter_deg(math.degrees(st["phi"]))))
        out.append(g)
    return out


def stats(values: list[float]) -> str:
    v = [x for x in values if x is not None]
    if not v:
        return "—"
    if len(v) == 1:
        return f"{v[0]:+.2f} (1)"
    return f"{statistics.fmean(v):+.2f} ± {statistics.pstdev(v):.2f} ({len(v)})"


def fit_slope(xs: list[float], ys: list[float]) -> tuple[float, float, int] | None:
    """最小二乗の傾きと相関係数 (平均を除いて)。点が足りなければ None。"""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 5:
        return None
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    x = x - x.mean()
    y = y - y.mean()
    sxx = float((x * x).sum())
    syy = float((y * y).sum())
    if sxx <= 0 or syy <= 0:
        return None
    slope = float((x * y).sum() / sxx)
    r = float((x * y).sum() / math.sqrt(sxx * syy))
    return slope, r, len(pairs)


def pct(values, q):
    v = [x for x in values if x is not None]
    return None if not v else float(np.percentile(v, q))


def fmt_ms(v) -> str:
    return "—" if v is None else f"{v:.1f}"


def report_processing(frames: list[dict]) -> list[str]:
    lines = ["## 1. 処理"]
    ts = [f["sensor_ts_ns"] for f in frames if f.get("sensor_ts_ns") is not None]
    n = len(frames)
    if len(ts) >= 2:
        span = (ts[-1] - ts[0]) / 1e9
        gaps = np.diff(ts) / 1e6
        fd = [f["frame_duration_us"] / 1000.0 for f in frames if f.get("frame_duration_us")]
        nominal = statistics.median(fd) if fd else float(np.median(gaps))
        dropped = int(sum(max(0, round(g / nominal) - 1) for g in gaps))
        lines.append(f"- フレーム {n} 枚 / {span:.1f} s = **実効 {(len(ts) - 1) / span:.1f} fps** "
                     f"(センサーのフレーム間隔 {nominal:.1f} ms、取りこぼし {dropped} 枚)")
    else:
        lines.append(f"- フレーム {n} 枚 (撮影時刻が無いので fps と遅れは出せない)")
    for key, name in (("t_measure_ms", "壁の帯 + cell_offset"), ("t_yaw_ms", "axis_yaw"),
                      ("t_tilt_ms", "傾き (赤マスク + 消失点)")):
        v = [f.get(key) for f in frames]
        if any(x is not None for x in v):
            lines.append(f"- {name}: 中央値 {fmt_ms(pct(v, 50))} ms / 95% {fmt_ms(pct(v, 95))} "
                         f"/ 最大 {fmt_ms(pct(v, 100))} ms")
    total = [sum(f.get(k) or 0.0 for k in ("t_measure_ms", "t_yaw_ms", "t_tilt_ms"))
             for f in frames]
    lines.append(f"- 合計: 中央値 {fmt_ms(pct(total, 50))} ms / 95% {fmt_ms(pct(total, 95))} ms")
    recv = [(f["recv_ns"] - f["sensor_ts_ns"]) / 1e6 for f in frames
            if f.get("recv_ns") is not None and f.get("sensor_ts_ns") is not None]
    done = [(f["done_ns"] - f["sensor_ts_ns"]) / 1e6 for f in frames
            if f.get("done_ns") is not None and f.get("sensor_ts_ns") is not None]
    if recv:
        lines.append(f"- 撮影時刻 → 受け取り: 中央値 {fmt_ms(pct(recv, 50))} ms / "
                     f"95% {fmt_ms(pct(recv, 95))} ms")
        lines.append(f"- 撮影時刻 → **使える (処理完了)**: 中央値 {fmt_ms(pct(done, 50))} ms / "
                     f"95% {fmt_ms(pct(done, 95))} ms / 最大 {fmt_ms(pct(done, 100))} ms "
                     f"(0.30 m/s なら {pct(done, 50) * 0.30:.1f} mm 進む)")
    exp = [f.get("exposure_us") for f in frames if f.get("exposure_us")]
    gain = [f.get("gain") for f in frames if f.get("gain")]
    if exp:
        e = statistics.median(exp) / 1000.0
        lines.append(f"- 露出 {e:.1f} ms / ゲイン {statistics.median(gain):.2f} "
                     f"(露光中に 0.24 m/s で {e * 0.24:.1f} mm、0.30 m/s で {e * 0.30:.1f} mm 動く)")
    return lines


def report_ticks(ticks: Ticks) -> list[str]:
    lines = ["## 2. 制御ループ"]
    dt = [d for d in ticks.dt if d > 0]
    if not dt:
        return lines + ["- tick の記録なし"]
    nominal = statistics.median(dt)
    late = sum(1 for d in dt if d > 1.5 * nominal)
    lines.append(f"- tick {len(dt)} 回: 間隔 中央値 {nominal * 1000:.1f} ms / "
                 f"95% {pct(dt, 95) * 1000:.1f} / 最大 {max(dt) * 1000:.1f} ms、"
                 f"1.5 倍を超えた tick {late} 回 ({late / len(dt) * 100:.1f}%)")
    return lines


PHASES = ("静止", "加速", "等速", "減速")


def report_phases(rows: list[dict]) -> list[str]:
    lines = ["## 3. 場面ごとの読み (平均 ± 標準偏差 (枚数))",
             "残差 = カメラ − 推定。前後・左右は車体フレーム [mm]、軸角・傾きは [deg]。",
             "",
             "| 場面 | 残差 前後 | 残差 左右 | 残差 軸角 | ロール | ピッチ |",
             "|---|---|---|---|---|---|"]
    for ph in PHASES:
        sel = [r for r in rows if r.get("phase") == ph]
        if not sel:
            continue
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            ph, stats([r.get("res_forward_mm") for r in sel]),
            stats([r.get("res_left_mm") for r in sel]),
            stats([r.get("res_yaw_deg") for r in sel]),
            stats([r.get("roll_deg") for r in sel]),
            stats([r.get("pitch_deg") for r in sel])))
    return lines


def report_sway(rows: list[dict], camera_height_mm: float) -> list[str]:
    lever = wall_top_lever_mm(camera_height_mm)
    expect = tilt_to_shift_mm(1.0, camera_height_mm)
    lines = ["## 4. 揺れの見分け (残差と傾きの比例)",
             f"機体全体が傾くなら、残差は傾き 1° あたり ±{expect:.1f} mm "
             f"(てこ = 壁の上面からの高さ {lever:.0f} mm)。",
             "傾きと無関係に残差が振れるなら枠の横ずれ (天板の平行移動)。",
             "ロール (横帯の収束) は左右の読み、ピッチ (縦帯の収束) は前後の読みに効く。", ""]
    for name, res, tilt in (("左右 vs ロール", "res_left_mm", "roll_deg"),
                            ("前後 vs ピッチ", "res_forward_mm", "pitch_deg")):
        fit = fit_slope([r.get(tilt) for r in rows], [r.get(res) for r in rows])
        if fit is None:
            lines.append(f"- {name}: 点が足りない")
            continue
        slope, r, n = fit
        lines.append(f"- {name}: 傾き 1° あたり {slope:+.2f} mm (相関 {r:+.2f}, {n} 枚)、"
                     f"機体全体の傾きなら ±{expect:.1f}")
    yaw = [r.get("res_yaw_deg") for r in rows if r.get("phase") in ("等速", "加速", "減速")]
    if yaw and any(y is not None for y in yaw):
        lines.append(f"- 走行中の軸角の残差: {stats(yaw)} (天板のねじれならここだけが振れる)")
    return lines


def report_accel(rows: list[dict]) -> list[str]:
    lines = ["## 5. 加速度との関係 (走行中のフレーム)",
             "加減速で天板が傾く/ずれるなら、指令の加速度に比例する (偏りなので平均では消えない)。",
             "**往復 (F6,B6 など) では加速度の符号と場所が一致する** (端で加速・減速するので)。"
             "3b で場所ごとの静止の読みが違えば、ここの比例は場所の違いかもしれない。", ""]
    moving = [r for r in rows if r.get("state") is not None and r.get("phase") != "静止"]
    for name, acc, key in (("ピッチ vs 前後の加速度", "ax", "pitch_deg"),
                           ("ロール vs 左右の加速度", "ay", "roll_deg"),
                           ("残差 前後 vs 前後の加速度", "ax", "res_forward_mm"),
                           ("残差 左右 vs 左右の加速度", "ay", "res_left_mm")):
        fit = fit_slope([r["state"][acc] for r in moving], [r.get(key) for r in moving])
        unit = "°" if key.endswith("deg") else " mm"
        if fit is None:
            lines.append(f"- {name}: 点が足りない (またはその向きに加速していない)")
        else:
            slope, r, n = fit
            lines.append(f"- {name}: 1 m/s² あたり {slope:+.2f}{unit} (相関 {r:+.2f}, {n} 枚)")
    return lines


def still_segments(rows: list[dict]) -> list[list[dict]]:
    """静止のフレームを、間に走行を挟まない区間ごとに分ける。"""
    segs: list[list[dict]] = []
    prev = None
    for r in rows:
        if r.get("phase") is None:
            continue
        if r["phase"] == "静止":
            if prev != "静止":
                segs.append([])
            segs[-1].append(r)
        prev = r["phase"]
    return segs


def settle_times(rows: list[dict]) -> list[tuple[str, float | None, int]]:
    """止まった後の区間ごとに、位置の読みが後半の中央値 ±幅 に入って出なくなるまでの時間 [s]。

    軸角は使わない — 静止でも 1 枚ごとに ±0.2° ほど振れ (2026-11-17 実測)、
    幅 0.1° では雑音で「落ち着かない」と出てしまう。
    """
    out = []
    for k, seg in enumerate(still_segments(rows)):
        if k == 0 or len(seg) < 5:
            continue                  # 最初の区間は動く前 (止まった後ではない)
        t0 = seg[0]["sensor_ts_ns"]
        tail = seg[len(seg) // 2:]
        refs = {}
        for key in ("left_mm", "forward_mm"):
            v = [r.get(key) for r in tail if r.get(key) is not None]
            refs[key] = statistics.median(v) if v else None
        last_bad = None
        for r in seg:
            for key in ("left_mm", "forward_mm"):
                v = r.get(key)
                if refs[key] is not None and v is not None and abs(v - refs[key]) > SETTLE_MM:
                    last_bad = r["sensor_ts_ns"]
        settle = 0.0 if last_bad is None else (last_bad - t0) / 1e9
        label = seg[0]["state"]["label"] if seg[0].get("state") else ""
        out.append((label, settle, len(seg)))
    return out


def med(values) -> str:
    v = [x for x in values if x is not None]
    return "—" if not v else f"{statistics.median(v):+.2f}"


def report_still(rows: list[dict]) -> list[str]:
    """静止の区間ごとの読み (中央値)。**場所ごとに傾きの読みが違う**かを見る。"""
    lines = ["## 3b. 静止の区間ごと (中央値)",
             "傾きの読みが場所で違うなら、床の傾きか帯の組み合わせの違い。"
             "往復の走行では加速と減速の場所が決まっているので、5. の比例はこれと区別できない。", "",
             "| 区間 | 推定 x, y [mm] | 枚数 | 左右 | 前後 | 軸角 | ロール | ピッチ |",
             "|---|---|---|---|---|---|---|---|"]
    for k, seg in enumerate(still_segments(rows)):
        xs = [r["state"]["x"] for r in seg if r.get("state")]
        ys = [r["state"]["y"] for r in seg if r.get("state")]
        lines.append("| %d | %+.0f, %+.0f | %d | %s | %s | %s | %s | %s |" % (
            k, statistics.median(xs) * 1000 if xs else 0, statistics.median(ys) * 1000 if ys else 0,
            len(seg), med(r.get("left_mm") for r in seg), med(r.get("forward_mm") for r in seg),
            med(r.get("yaw_deg") for r in seg), med(r.get("roll_deg") for r in seg),
            med(r.get("pitch_deg") for r in seg)))
    return lines


def jitter(values: list[float], half: int = 2) -> list[float]:
    """前後 ``half`` 枚の中央値からの外れ (ゆっくりした本当のずれを除いた、速い振れ)。"""
    v = np.array(values, dtype=float)
    return [float(v[i] - np.median(v[i - half:i + half + 1])) for i in range(half, len(v) - half)]


def report_jitter(rows: list[dict]) -> list[str]:
    lines = ["## 3c. 等速中の速い振れ (前後 2 枚との中央値からの外れ)",
             "ゆっくり溜まる本当のずれ (横流れ) を除いた、フレームごとの振れ。"
             "揺れの上限の目安 (1px ≈ 0.6mm なので、それ以下は量子化と区別できない)。", ""]
    cruise = [r for r in rows if r.get("phase") == "等速"]
    for key, name in (("left_mm", "左右"), ("forward_mm", "前後"), ("yaw_deg", "軸角")):
        v = [r.get(key) for r in cruise]
        runs, cur = [], []
        for x in v:                       # 測れなかった所で切る
            if x is None:
                if cur:
                    runs.append(cur)
                cur = []
            else:
                cur.append(x)
        if cur:
            runs.append(cur)
        j = [x for run in runs if len(run) >= 5 for x in jitter(run)]
        if len(j) < 10:
            lines.append(f"- {name}: 測れた枚数が足りない")
            continue
        unit = "°" if key == "yaw_deg" else " mm"
        lines.append(f"- {name}: 標準偏差 {np.std(j):.2f}{unit} / 95% {np.percentile(np.abs(j), 95):.2f}{unit} "
                     f"({len(j)} 枚)")
    return lines


def report_settle(rows: list[dict]) -> list[str]:
    lines = ["## 6. 止まった後の収束",
             f"指令が 0 になってから、位置の読み (左右・前後 ±{SETTLE_MM} mm) が"
             "その区間の後半の中央値に収まって出なくなるまで。区間が短いと上限で切れる。", ""]
    st = settle_times(rows)
    if not st:
        return lines + ["- 止まった後の区間が無い"]
    for label, t, n in st:
        lines.append(f"- {label or '(名前なし)'}: {t:.2f} s ({n} 枚)")
    return lines


def tau_sweep(frames: list[dict], ticks: Ticks, lo: float = -150, hi: float = 150,
              step: float = 5) -> dict[str, tuple[float, float, float]]:
    """推定を τ [ms] ずらして、走行中の残差の標準偏差が最小になる τ を探す。

    返り値: 量 -> (最良の τ, そのときの標準偏差, τ=0 の標準偏差)。走行中の信号が
    変化しないと谷ができないので、平らなら意味が無い (結果にそれも出る)。
    """
    out = {}
    taus = np.arange(lo, hi + step / 2, step)
    for key in ("res_left_mm", "res_forward_mm", "res_yaw_deg"):
        best = None
        at_zero = None
        for tau in taus:
            rows = enrich(frames, ticks, float(tau))
            v = [r.get(key) for r in rows if r.get("phase") not in (None, "静止")]
            v = [x for x in v if x is not None]
            if len(v) < 10:
                break
            sd = statistics.pstdev(v)
            if abs(tau) < 1e-9:
                at_zero = sd
            if best is None or sd < best[1]:
                best = (float(tau), sd)
        if best is not None and at_zero is not None:
            out[key] = (best[0], best[1], at_zero)
    return out


def report_tau(frames: list[dict], ticks: Ticks) -> list[str]:
    lines = ["## 7. 遅れの当てはめ (推定を τ ずらして残差のばらつきが最小になる所)",
             "+τ = カメラの撮影時刻より後の推定と合う (= 撮影時刻の記録が実際より早い)。"
             "τ=0 と最良の差が小さければ、走行中の信号が平らで決まらない。", ""]
    res = tau_sweep(frames, ticks)
    if not res:
        return lines + ["- 走行中のフレームが足りない"]
    names = {"res_left_mm": "左右 [mm]", "res_forward_mm": "前後 [mm]", "res_yaw_deg": "軸角 [deg]"}
    for key, (tau, sd, sd0) in res.items():
        lines.append(f"- {names[key]}: 最良 τ = {tau:+.0f} ms (標準偏差 {sd:.3f}、τ=0 で {sd0:.3f})")
    return lines


def cross_track_mm(r: dict) -> float | None:
    """走行中のフレームの横ずれ [mm] (進行方向と直角な車体軸のカメラの読み)。"""
    st = r.get("state")
    if st is None:
        return None
    if abs(st["vx"]) >= abs(st["vy"]):
        return r.get("left_mm")
    return r.get("forward_mm")


def report_fusion(rows: list[dict], ticks: Ticks) -> list[str]:
    """#186: 融合した補正量と、走行中のカメラの横ずれ (融合が効けば 0 付近に保たれる)。"""
    lines = ["## 8. 走行中の横ずれと融合 (#186)",
             "カメラの横ずれ = セル中央からの、進行方向と直角なずれ (真の位置に近い)。"
             "融合が効いていれば走行中も 0 付近に保たれ、効いていなければ横流れがそのまま出る。", ""]
    cross = [cross_track_mm(r) for r in rows if r.get("phase") in ("加速", "等速", "減速")]
    cross = [c for c in cross if c is not None]
    if cross:
        lines.append(f"- 走行中のカメラの横ずれ: 平均 {statistics.fmean(cross):+.2f} mm、"
                     f"最大 |{max(abs(c) for c in cross):.2f}| mm、"
                     f"標準偏差 {statistics.pstdev(cross):.2f} mm ({len(cross)} 枚)")
    rows_t = ticks.rows
    n = [r.get("fuse_n") for r in rows_t if r.get("fuse_n") is not None]
    if not n:
        return lines + ["- 融合なし (--fuse を付けていない走行)"]
    dx = [r.get("fuse_dx_mm") or 0.0 for r in rows_t]
    dy = [r.get("fuse_dy_mm") or 0.0 for r in rows_t]
    dp = [r.get("fuse_dphi_deg") or 0.0 for r in rows_t]
    applied = sum(1 for a, b, c in zip(dx, dy, dp) if a or b or c)
    lines.append(f"- 測定 {int(sum(n))} 件を融合に回し、補正を掛けた tick {applied} 回")
    lines.append(f"- 補正の合計: x {sum(dx):+.2f} mm / y {sum(dy):+.2f} mm / "
                 f"方位 {sum(dp):+.3f}° (絶対値の合計 x {sum(map(abs, dx)):.2f} / "
                 f"y {sum(map(abs, dy)):.2f} mm / {sum(map(abs, dp)):.3f}°)")
    return lines


def plot(rows: list[dict], path: str) -> None:
    """時系列を PNG に描く (matplotlib を入れずに cv2 で。cv2 は日本語を描けないので見出しは英語)。"""
    import cv2

    pts = [r for r in rows if r.get("sensor_ts_ns") is not None]
    if len(pts) < 2:
        return
    t0 = pts[0]["sensor_ts_ns"]
    t = np.array([(r["sensor_ts_ns"] - t0) / 1e9 for r in pts])
    panels = [
        ("command: speed [m/s] / accel/10 [m/s2]", [
            ("speed", (60, 60, 60), lambda r: None if r.get("state") is None
             else math.hypot(r["state"]["vx"], r["state"]["vy"])),
            ("acc/10", (40, 120, 220), lambda r: None if r.get("state") is None
             else math.hypot(r["state"]["ax"], r["state"]["ay"]) / 10.0)]),
        ("residual camera - estimate [mm]", [("left", (200, 80, 0), lambda r: r.get("res_left_mm")),
                     ("forward", (0, 150, 0), lambda r: r.get("res_forward_mm"))]),
        ("camera cell_offset [mm]", [("left", (200, 80, 0), lambda r: r.get("left_mm")),
                          ("forward", (0, 150, 0), lambda r: r.get("forward_mm"))]),
        ("yaw residual / roll / pitch [deg]", [
            ("yaw", (160, 0, 160), lambda r: r.get("res_yaw_deg")),
            ("roll", (200, 80, 0), lambda r: r.get("roll_deg")),
            ("pitch", (0, 150, 0), lambda r: r.get("pitch_deg"))]),
    ]
    W, H, M = 1400, 230, 60
    img = np.full((H * len(panels) + 30, W, 3), 255, np.uint8)
    tmax = max(t[-1], 1e-6)
    for k, (title, series) in enumerate(panels):
        top = k * H + 20
        vals = [v for _n, _c, fn in series for v in (fn(r) for r in pts) if v is not None]
        cv2.rectangle(img, (M, top), (W - 20, top + H - 40), (200, 200, 200), 1)
        cv2.putText(img, title,
                    (M, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-6:
            lo, hi = lo - 1, hi + 1
        for label, val in ((f"{hi:.2f}", hi), (f"{lo:.2f}", lo)):
            y = int(top + (H - 40) * (1 - (val - lo) / (hi - lo)))
            cv2.putText(img, label, (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        for j, (name, color, fn) in enumerate(series):
            prev = None
            for ti, r in zip(t, pts):
                v = fn(r)
                if v is None:
                    prev = None
                    continue
                p = (int(M + (W - 20 - M) * ti / tmax),
                     int(top + (H - 40) * (1 - (v - lo) / (hi - lo))))
                if prev is not None:
                    cv2.line(img, prev, p, color, 1)
                cv2.circle(img, p, 1, color, -1)
                prev = p
            cv2.putText(img, name, (W - 160 + 0, top + 14 + 14 * j),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(img, f"t [s] 0 .. {tmax:.1f}", (M, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(path, img)


def analyze(prefix: str, camera_height_mm: float = CAMERA_HEIGHT_MM,
            plot_path: str | None = None) -> str:
    frames = read_csv(f"{prefix}_frames.csv")
    tick_path = f"{prefix}_ticks.csv"
    ticks = Ticks(read_csv(tick_path)) if os.path.exists(tick_path) else None
    out = [f"# {prefix}", ""]
    out += report_processing(frames) + [""]
    if ticks is not None and len(ticks):
        rows = enrich(frames, ticks)
        out += report_ticks(ticks) + [""]
        out += report_phases(rows) + [""]
        out += report_still(rows) + [""]
        out += report_jitter(rows) + [""]
        out += report_sway(rows, camera_height_mm) + [""]
        out += report_accel(rows) + [""]
        out += report_settle(rows) + [""]
        out += report_tau(frames, ticks) + [""]
        out += report_fusion(rows, ticks) + [""]
    else:
        rows = frames
        out += ["(tick の記録が無い = 静止のみ。読みのばらつきだけ出す)", ""]
        for key in ("left_mm", "forward_mm", "yaw_deg", "roll_deg", "pitch_deg"):
            out.append(f"- {key}: {stats([f.get(key) for f in frames])}")
    if plot_path:
        plot(rows, plot_path)
        out.append(f"(時系列を {plot_path} に描いた)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="走行中のカメラ記録の解析 (#184)")
    p.add_argument("prefix", nargs="+", help="--frame-log / --out に渡した PREFIX")
    p.add_argument("--camera-height", type=float, default=CAMERA_HEIGHT_MM, metavar="mm",
                   help="カメラの床からの高さ (傾き → ずれ の換算に使う)")
    p.add_argument("--plot", default=None, metavar="PNG",
                   help="時系列を PNG に描く (PREFIX が複数なら PREFIX ごとに _N を付ける)")
    args = p.parse_args(argv)
    for i, prefix in enumerate(args.prefix):
        path = args.plot
        if path and len(args.prefix) > 1:
            root, ext = os.path.splitext(path)
            path = f"{root}_{i}{ext}"
        print(analyze(prefix, args.camera_height, path))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
