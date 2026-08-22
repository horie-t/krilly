"""カメラの傾きを壁上面の格子から測る (issue #87).

**壁の上面は 3D では正確な格子**なので、カメラが真下を向いていれば画像でも
平行線の束になる。傾いていれば透視投影で**消失点へ収束する**。この収束は
スケールに依らないので、迷路のセルが正確な正方形でなくても影響を受けない
(自立式の柱に変えてから前後と左右の壁間隔は数 mm 違いうる、#65)。

測り方:

1. 赤マスクを横長・縦長の構造要素で開いて、横帯と縦帯を分ける
2. 各帯に主成分で直線を当てる
3. 帯の束が交わる点 (消失点) を最小二乗で求める
4. 主点から消失点までの距離 ``d`` と焦点距離 ``f`` から **傾き = atan(f / d)**

   真下向きなら平行 -> ``d`` は無限大 -> 傾き 0。真横向きなら ``d`` = 0 -> 90°。

画像の**横帯は機体の左右軸に沿って伸びる**ので、その収束は機体前後軸まわりの
傾き (ロール = 左右に傾く) を表す。縦帯の収束は左右軸まわり (ピッチ = 前後に傾く)。

感度: 1 セルピッチ 305px 離れた 2 本の平行線は、傾き 1° あたり約 0.53° 収束する。
帯の長さが 900px あれば 0.1° (= 端で 1.6px) は見えるので、**傾き 0.2° 程度まで測れる**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Line:
    """画像内の直線 (点 ``p`` を通り方向 ``d`` の単位ベクトル)。"""

    px: float
    py: float
    dx: float
    dy: float
    length: float          # 元になった帯の長さ [px]
    residual: float        # 当てはめの残差 (直交方向の RMS) [px]

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.dy, self.dx))


@dataclass(frozen=True)
class TiltResult:
    """格子から求めたカメラの傾き。"""

    roll_deg: float | None      # 機体前後軸まわり (左右に傾く)。横帯の収束から
    pitch_deg: float | None     # 機体左右軸まわり (前後に傾く)。縦帯の収束から
    horizontal: list[Line]
    vertical: list[Line]
    vp_horizontal: tuple[float, float] | None
    vp_vertical: tuple[float, float] | None
    spread_h_deg: float | None  # 横帯どうしの角度のばらつき (収束の生の証拠)
    spread_v_deg: float | None
    #: 透視モデルの当てはまり [deg]。各帯を消失点へ通すのに要る回転の RMS。
    #: **これが傾きの角度に対して小さくないと、測っているのは傾きではない**
    #: (自立式の壁が平行でない、レンズの歪曲が残っている、など)。
    residual_h_deg: float | None
    residual_v_deg: float | None

    @property
    def trustworthy(self) -> bool:
        """透視モデルが信号に対して十分よく当てはまっているか (残差 < 信号の 1/3)。"""
        for angle, resid in ((self.roll_deg, self.residual_h_deg),
                             (self.pitch_deg, self.residual_v_deg)):
            if angle is None or resid is None:
                return False
            if resid > abs(angle) / 3.0:
                return False
        return True

    def describe(self) -> str:
        def fmt(v, unit="°"):
            return "測定不能" if v is None else f"{v:+.2f}{unit}"
        def q(angle, resid):
            if angle is None or resid is None:
                return ""
            mark = "" if resid <= abs(angle) / 3.0 else "  <- 残差が大きい。要注意"
            return f" (当てはめ残差 {resid:.2f}°){mark}"
        return (
            f"横帯 {len(self.horizontal)} 本 (角度の幅 {fmt(self.spread_h_deg)}) "
            f"-> ロール {fmt(self.roll_deg)}{q(self.roll_deg, self.residual_h_deg)}\n"
            f"縦帯 {len(self.vertical)} 本 (角度の幅 {fmt(self.spread_v_deg)}) "
            f"-> ピッチ {fmt(self.pitch_deg)}{q(self.pitch_deg, self.residual_v_deg)}"
        )


def _fit_line(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float, float, float, float]:
    """点群に主成分で直線を当て、(px, py, dx, dy, 残差) を返す。"""
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    c = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    resid = float(np.sqrt(((pts - c) @ n) ** 2).mean() ** 0.5) if len(pts) else 0.0
    resid = float(np.sqrt((((pts - c) @ n) ** 2).mean()))
    return c[0], c[1], d[0], d[1], resid


def _components(mask: np.ndarray, horizontal: bool, min_length: int,
                min_area: int) -> list[np.ndarray]:
    """長い構造要素で開いてから連結成分に分け、各成分の点群を返す。"""
    k = (min_length, 1) if horizontal else (1, min_length)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k)
    opened = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        long_side, short_side = (w, h) if horizontal else (h, w)
        if long_side < min_length or long_side < 3 * short_side:
            continue
        ys, xs = np.nonzero(lab == i)
        out.append(np.stack([xs, ys], axis=1).astype(np.float64))
    return out


def find_bands(mask: np.ndarray, horizontal: bool, min_length: int = 120,
               min_area: int = 400, merge_px: float = 40.0) -> list[Line]:
    """赤マスクから横帯 (または縦帯) を抜き出し、それぞれに直線を当てる。

    **同じ壁の帯は自機に隠されて分断される**ので、共線の破片をまとめてから当てる。
    まとめないと 1 本の壁が複数の「線」に化け、消失点の当てはめが壊れる
    (実測では 4 本の壁が 6 本に分裂した)。

    まとめる基準は画像中心での切片: 破片どうしはほぼ同じ切片になり、隣の壁とは
    1 セルピッチ (約 305px) 離れるので、``merge_px`` はその間に取れば足りる。
    """
    h, w = mask.shape[:2]
    ref = w / 2.0 if horizontal else h / 2.0
    comps = _components(mask, horizontal, min_length, min_area)
    entries = []
    for pts in comps:
        px, py, dx, dy, _ = _fit_line(pts[:, 1], pts[:, 0])
        if horizontal:
            inter = py + (dy / dx) * (ref - px) if abs(dx) > 1e-9 else py
        else:
            inter = px + (dx / dy) * (ref - py) if abs(dy) > 1e-9 else px
        entries.append((inter, pts))
    entries.sort(key=lambda e: e[0])

    out: list[Line] = []
    group: list[np.ndarray] = []
    last = None
    for inter, pts in entries + [(float("inf"), None)]:
        if last is not None and inter - last > merge_px and group:
            merged = np.concatenate(group)
            px, py, dx, dy, resid = _fit_line(merged[:, 1], merged[:, 0])
            span = (merged[:, 0].max() - merged[:, 0].min() if horizontal
                    else merged[:, 1].max() - merged[:, 1].min())
            out.append(Line(px, py, dx, dy, float(span), resid))
            group = []
        if pts is not None:
            group.append(pts)
            last = inter
    return out


def vanishing_point(lines: list[Line], weighted: bool = True) -> tuple[float, float] | None:
    """直線の束が最もよく交わる点 (最小二乗)。平行なら None。

    **帯の長さで重み付けする。** 直線の向きの不確かさは長さに反比例するので、
    短い帯ほど消失点の位置を粗くしか決められない。重み付けしないと、フレーム端の
    短い帯 (実機で 245px、他は 600-900px) が端にあるぶん強い梃子になって推定を
    振り回す — 実測で消失点の勾配が 16% 変わり、傾きの再現性が 0.07° から 1.1° に
    悪化した。
    """
    if len(lines) < 2:
        return None
    A, b = [], []
    for ln in lines:
        nx, ny = -ln.dy, ln.dx          # 法線
        w = ln.length if weighted else 1.0
        A.append([nx * w, ny * w])
        b.append(w * (nx * ln.px + ny * ln.py))
    A = np.array(A); b = np.array(b)
    # 条件数が悪い (= ほぼ平行) ときは消失点が無限遠とみなす
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] / s[0] < 1e-6:
        return None
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(p[0]), float(p[1])


def _tilt_from_vp(vp: tuple[float, float] | None, principal: tuple[float, float],
                  focal_px: float, axis: int) -> float | None:
    """消失点から傾きを出す。**符号は光軸が倒れている向き**。

    消失点は光軸が倒れている側にできる (実測で確認: ピッチの消失点が画像の上に
    あるとき、光軸の足はセル中心より前に落ちていた)。符号が無いと、機体を 180°
    回して測り直したときに「カメラ側の傾き」と「床側の傾き」を区別できない。
    """
    if vp is None:
        return None
    d = math.hypot(vp[0] - principal[0], vp[1] - principal[1])
    if d <= 0:
        return 90.0
    sign = math.copysign(1.0, vp[axis] - principal[axis])
    return sign * math.degrees(math.atan(focal_px / d))


def _fit_residual_deg(lines: list[Line], vp: tuple[float, float] | None) -> float | None:
    """各帯を消失点へ通すのに要る回転角の RMS [deg]。

    透視なら 4 本は 1 点で交わるはずなので、これが小さいほどモデルが当たっている。
    **傾きの角度と同じくらい大きいなら、それは傾きではなく壁の据え付け誤差**。
    """
    if vp is None or len(lines) < 3:
        return None
    out = []
    for ln in lines:
        nx, ny = -ln.dy, ln.dx
        d = abs(nx * (vp[0] - ln.px) + ny * (vp[1] - ln.py))
        D = math.hypot(vp[0] - ln.px, vp[1] - ln.py)
        if D > 0:
            out.append(math.degrees(math.atan2(d, D)))
    return float(np.sqrt(np.mean(np.square(out)))) if out else None


def _spread(lines: list[Line]) -> float | None:
    if len(lines) < 2:
        return None
    a = np.array([ln.angle_deg % 180.0 for ln in lines])
    a = np.where(a > 90.0, a - 180.0, a)     # -90..90 に畳んで折り返しを避ける
    return float(a.max() - a.min())


def measure_tilt(mask: np.ndarray, focal_px: float,
                 principal: tuple[float, float] | None = None,
                 min_length: int = 120) -> TiltResult:
    """赤マスクの格子からカメラの傾きを測る。

    ``focal_px`` は焦点距離 [px] = (px/mm) x (カメラから壁上面までの距離 [mm])。
    ``principal`` を省くと画像中心を主点とみなす。
    """
    h, w = mask.shape[:2]
    pp = principal if principal is not None else (w / 2.0, h / 2.0)
    hor = find_bands(mask, True, min_length)
    ver = find_bands(mask, False, min_length)
    vp_h, vp_v = vanishing_point(hor), vanishing_point(ver)
    return TiltResult(
        roll_deg=_tilt_from_vp(vp_h, pp, focal_px, axis=0),
        pitch_deg=_tilt_from_vp(vp_v, pp, focal_px, axis=1),
        horizontal=hor, vertical=ver, vp_horizontal=vp_h, vp_vertical=vp_v,
        spread_h_deg=_spread(hor), spread_v_deg=_spread(ver),
        residual_h_deg=_fit_residual_deg(hor, vp_h),
        residual_v_deg=_fit_residual_deg(ver, vp_v),
    )
