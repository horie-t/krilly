"""走りながら、これから通るセルの境目に壁が写っていないかを確かめる (issue #188)。

止まらずに走ると、動き出す前の進路チェック (``path_is_clear``) だけでは足りない —
長い直線の途中で推定がずれていたり、地図が間違っていたりすると、次の境目の壁に
そのまま突っ込む。そこで**毎フレーム**、進む向きの境目の位置に壁の帯が写って
いないかを見る。

**計画した経路は開いている境目しか通らない**ので、経路上の境目に壁が見えたら、
地図と照らし合わせるまでもなく異常 (推定のずれか地図の誤り)。行き先のセルの向こう
側の境目は正当に壁がありうるので、呼び出し側が対象から外す。

画面の中央には**機体自身**が写っている (``axis_yaw.CALIBRATED_ROBOT_RECT``)。境目の
帯は「セルの内側」(左右の帯の間、または前後の帯の間) で測り、機体とテープ (緑) の
画素は数えない。前進なら機体から約 70 mm より遠い境目しか見えない — 0.30 m/s で止まる
のに 75 mm、遅れの分 (約 70 ms) が 21 mm なので、境目が約 200 mm 先に見え始めてから
約 110 mm になるまでの間 (約 10 フレーム) に見つけることになる。

ここは純計算: :func:`ahead_profiles` が 1 枚から「行ごと・列ごとの赤の画素と見えて
いる画素の数」を作り、:func:`boundary_fraction` が、機体からの距離を与えられた境目の
位置の赤の割合を返す。どの境目を見るか (経路上か、どれだけ先か) は呼び出し側が決める。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from krilly.perception.axis_yaw import CALIBRATED_ROBOT_RECT
from krilly.perception.red_wall import RedDetectorConfig, red_mask
from krilly.perception.wall_detect import (
    BACK,
    CALIBRATED_BANDS,
    CALIBRATED_GEOMETRY,
    CALIBRATED_RED,
    FRONT,
    LEFT,
    RIGHT,
    CameraGeometry,
)


@dataclass(frozen=True)
class AheadConfig:
    red: RedDetectorConfig = CALIBRATED_RED
    #: セルの内側として使う範囲を、左右 (前後) の帯の内側の縁からさらに内へ寄せる量 [px]
    #: (横の壁の帯や柱を数えないため)。
    inset_px: int = 8
    #: 境目の位置の前後に探す幅 [px]。推定の進行方向の誤り (#187: 7 セルで 13〜20 mm) と
    #: 帯の幅 (12 mm ≈ 20 px) を見込む。
    window_px: int = 30
    #: 見えている画素がこの割合に満たない行 (列) は判定に使わない (機体に隠れている)。
    min_visible: float = 0.25


@dataclass
class AheadProfiles:
    """1 枚から作る、帯に沿った赤の画素と見えている画素の数。

    ``rows_*`` は画像の行ごと (前後に進むときの境目は横向きの線 = 行)、``cols_*`` は列ごと
    (左右に進むときの境目は縦向きの線 = 列)。
    """

    rows_red: np.ndarray
    rows_vis: np.ndarray
    cols_red: np.ndarray
    cols_vis: np.ndarray


def _red_and_visible(bgr: np.ndarray, x0: int, y0: int,
                     cfg: AheadConfig) -> tuple[np.ndarray, np.ndarray]:
    """切り出した ``bgr`` (元の画像の (x0, y0) から) の赤のマスクと、数えてよい画素
    (機体とテープを除く)。全画面で計算すると 1 枚 9.7 ms かかったので、使う帯だけ。"""
    h, w = bgr.shape[:2]
    red = red_mask(bgr, cfg.red) > 0
    vis = np.ones((h, w), dtype=bool)
    r = CALIBRATED_ROBOT_RECT
    ry0, ry1 = max(0, r.y - y0), max(0, min(h, r.y + r.h - y0))
    rx0, rx1 = max(0, r.x - x0), max(0, min(w, r.x + r.w - x0))
    vis[ry0:ry1, rx0:rx1] = False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    vis &= ~((hsv[..., 0] > 60) & (hsv[..., 0] < 100) & (hsv[..., 1] > 60))
    return red, vis


def ahead_profiles(bgr: np.ndarray, shifts_px: dict[str, int] | None = None,
                   config: AheadConfig | None = None) -> AheadProfiles:
    """1 枚から、セルの内側の行ごと・列ごとの赤の画素と見えている画素の数を作る。

    ``shifts_px`` は :meth:`WallDetector.measure` の帯のずれ (横ずれで帯が動いた量)。
    """
    cfg = config or AheadConfig()
    sh = shifts_px or {}
    h, w = bgr.shape[:2]
    # セルの内側: 左右の帯の内側の縁の間 (列)、前後の帯の内側の縁の間 (行)
    c0 = max(0, CALIBRATED_BANDS[LEFT][1] + sh.get(LEFT, 0) + cfg.inset_px)
    c1 = min(w, CALIBRATED_BANDS[RIGHT][0] + sh.get(RIGHT, 0) - cfg.inset_px)
    r0 = max(0, CALIBRATED_BANDS[FRONT][1] + sh.get(FRONT, 0) + cfg.inset_px)
    r1 = min(h, CALIBRATED_BANDS[BACK][0] + sh.get(BACK, 0) - cfg.inset_px)
    red_c, vis_c = _red_and_visible(bgr[:, c0:c1], c0, 0, cfg)     # 前後に進むとき (行)
    red_r, vis_r = _red_and_visible(bgr[r0:r1, :], 0, r0, cfg)     # 左右に進むとき (列)
    return AheadProfiles(
        rows_red=(red_c & vis_c).sum(axis=1).astype(np.float32),
        rows_vis=vis_c.sum(axis=1).astype(np.float32),
        cols_red=(red_r & vis_r).sum(axis=0).astype(np.float32),
        cols_vis=vis_r.sum(axis=0).astype(np.float32),
    )


#: 主軸の向き (``CellMotion`` の ``axis_k``: 0=前 / +1=左 / -1=右 / 2=後) -> (行か列か, 符号)
#: 画像の上が車体の前 (+x)、左が車体の左 (+y)。
_AXIS = {0: ("rows", -1), 2: ("rows", +1), 1: ("cols", -1), -1: ("cols", +1), 3: ("cols", +1)}


def boundary_fraction(prof: AheadProfiles, axis_k: int, distance_m: float,
                      config: AheadConfig | None = None,
                      geometry: CameraGeometry | None = None) -> float | None:
    """主軸 ``axis_k`` の向きに ``distance_m`` 先の境目の位置の、赤の割合 (0〜1)。

    境目の位置の ±``window_px`` の行 (列) のうち、見えている画素が十分ある所の
    「赤の画素 / 見えている画素」の最大。画面の外か、機体に隠れていて見えなければ None。
    """
    cfg = config or AheadConfig()
    g = geometry or CALIBRATED_GEOMETRY
    kind, sign = _AXIS[((axis_k + 1) % 4) - 1]     # 0 / +1 / 2 / -1 にそろえる
    if kind == "rows":
        center, ppm, red, vis = g.cell_center[1], g.px_per_mm_y, prof.rows_red, prof.rows_vis
    else:
        center, ppm, red, vis = g.cell_center[0], g.px_per_mm_x, prof.cols_red, prof.cols_vis
    pos = center + sign * distance_m * 1000.0 * ppm
    lo, hi = int(round(pos)) - cfg.window_px, int(round(pos)) + cfg.window_px + 1
    if hi <= 0 or lo >= len(red):
        return None
    lo, hi = max(0, lo), min(len(red), hi)
    full = float(vis.max()) if len(vis) else 0.0
    if full <= 0:
        return None
    ok = vis[lo:hi] >= cfg.min_visible * full
    if not ok.any():
        return None
    return float((red[lo:hi][ok] / vis[lo:hi][ok]).max())
