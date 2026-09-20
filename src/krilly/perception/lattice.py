"""格子そのものから機体のセル内位置を測る (#85 の接触事故)。

:func:`~krilly.perception.cell_pose.cell_offset` は**自セルの 4 つの ROI** に写る帯を
使う。ROI の探索は ±40px (±23.5mm) なので、それより大きくずれると帯を見失い、
**幽霊の壁まで生える**。実機で壁に食い込んで止まったとき、読みは ``{N, E}`` になり
(真は ``{E}`` だけ)、リカバリは「迷子」と判定して走行を終わらせた。

ここでは**フレームに写っている赤を全部**使う。壁も柱も 180mm 格子の線の上にしか
無いので、赤の質量を格子の周期で位相平均すれば、ROI がどこにあろうと関係なく機体の
ずれが出る。出せるのは **mod 180mm** だが、機体は必ずセルの中に居るので
``(-90, +90]`` に折り返せば一意に決まる。

**位相平均は「どれが壁でどれが柱か」を一切問わない。** 壁は長いので重みが大きく、柱は
小さいが位置は正しい — どちらも同じ線の上なので、区別せず足してよい。この性質のおかげで、
機体自身が壁を隠していても (食い込んだ壁は機体の陰になる) 残りの赤で位相が決まる。

**弱点は迷路の外の赤** (#89 の赤い定規袋)。格子の位相に乗らない赤は位相をばらけさせる
ので、:attr:`LatticeOffset.confidence` が下がる形で現れる — 消えるわけではない。
だからこれは**リカバリ専用**であって、正常運転の位置補正を置き換えるものではない。
正常運転では ROI が帯を確実に掴んでいるので、そちらの方が素性が良い。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from krilly.perception.red_wall import RedDetectorConfig, red_mask
from krilly.perception.wall_detect import (
    CALIBRATED_GEOMETRY,
    CALIBRATED_RED,
    CameraGeometry,
)

#: 格子の間隔 [mm]。セルのピッチそのもの。
PITCH_MM = 180.0

#: 位相がどれだけ揃っていれば信じるか (0 = ばらばら、1 = 全部が同じ位相)。
#:
#: **実測**: 機体が壁に食い込んで止まったフレームで 0.78 / 0.86、正常に停止した
#: フレームで 0.90 前後。迷路の外に赤い物を置くとここが落ちる。0.35 は「半分近くが
#: 格子に乗っていない」に相当し、そこまで崩れたら測れていないと見なす。
MIN_CONFIDENCE = 0.35

#: 「棒」と見なす長さ [px]。これより短い赤は柱か断片として扱う。
#:
#: 壁は 180mm = 306px、柱は 12mm = 20px なので、その間ならどこでもよい。60px (35mm)
#: は柱を確実に外し、画角で切れた壁の断片は拾える位置。
BAR_PX = 60


@dataclass(frozen=True)
class LatticeOffset:
    """格子から測った、セル中央に対する機体のずれ (車体フレーム、単位 m)。

    符号は :class:`~krilly.perception.cell_pose.CellOffset` と同じ
    (+forward = 前、+left = 左)。信頼できない軸は None。
    """

    forward_m: float | None
    left_m: float | None
    forward_confidence: float
    left_confidence: float

    @property
    def measured(self) -> bool:
        """**どちらかの軸**が測れたか (:class:`CellOffset` と同じ意味)。

        `apply_cell_offset` は軸ごとに None を見るので、片方だけでも渡す価値がある。
        """
        return self.forward_m is not None or self.left_m is not None

    def describe(self) -> str:
        def one(v: float | None, c: float) -> str:
            return "測れず" if v is None else f"{v * 1e3:+.1f}mm(位相 {c:.2f})"
        return f"前後={one(self.forward_m, self.forward_confidence)} " \
               f"左右={one(self.left_m, self.left_confidence)}"


def _fold(value: float, pitch: float) -> float:
    """``value`` を ``(-pitch/2, +pitch/2]`` に折り返す。"""
    folded = math.fmod(value, pitch)
    if folded > pitch / 2:
        folded -= pitch
    elif folded <= -pitch / 2:
        folded += pitch
    return folded


def _phase(weights: np.ndarray, positions_mm: np.ndarray,
           pitch: float) -> tuple[float, float]:
    """周期 ``pitch`` の位相平均と、その揃い具合を返す。

    ``exp(2πi x / pitch)`` の重み付き平均を取るだけ。**閾値もピーク検出も要らない**
    ので、帯が短くても (柱だけでも) 効く。戻り値の 2 番目は ``|z| / Σw`` で、
    全部の赤が同じ位相に乗れば 1、一様にばらければ 0 になる。
    """
    total = float(weights.sum())
    if total <= 0.0:
        return (0.0, 0.0)
    angles = 2.0 * math.pi * positions_mm / pitch
    z = complex(float((weights * np.cos(angles)).sum()),
                float((weights * np.sin(angles)).sum()))
    return (math.atan2(z.imag, z.real) * pitch / (2.0 * math.pi), abs(z) / total)


def _split_by_direction(mask: np.ndarray, bar_px: int
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """マスクを (縦に長い, 横に長い, どちらでもない) に分ける。

    3 つ目は柱と、画角で切れた壁の断片。**柱は縦横どちらの格子線の上にも乗る**ので、
    捨てずに両方のプロファイルへ足す — 壁が少ないセルでは柱しか手掛かりが無い。
    """
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((bar_px, 1), np.uint8))
    horz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, bar_px), np.uint8))
    bars = cv2.bitwise_or(vert, horz)
    return vert, horz, cv2.bitwise_and(mask, cv2.bitwise_not(bars))


def lattice_offset(
    bgr: np.ndarray,
    yaw_rad: float = 0.0,
    *,
    red: RedDetectorConfig | None = None,
    geometry: CameraGeometry | None = None,
    pitch_mm: float = PITCH_MM,
    min_confidence: float = MIN_CONFIDENCE,
    bar_px: int = BAR_PX,
) -> LatticeOffset:
    """赤の格子から機体のセル内位置を測る。

    ``yaw_rad`` には :func:`~krilly.perception.axis_yaw.axis_yaw` の測定値を渡す
    (+ = 機体 CCW)。**格子を画像の軸に揃えてから位相を取る**ため: 5° 傾いていると
    1 本の壁が画像の端から端で 60px 動き、周期 306px の位相が潰れる。
    """
    cfg = red or CALIBRATED_RED
    g = geometry or CALIBRATED_GEOMETRY
    mask = red_mask(bgr, cfg)
    h, w = mask.shape
    if yaw_rad:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), math.degrees(yaw_rad), 1.0)
        mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST)

    # **縦と横を分けてから位相を取る。** 分けないと、直交する壁の赤がもう一方の軸では
    # ちょうど 1 周期ぶんに広がり (壁の長さ = ピッチ)、位相を偏らせないかわりに
    # 分母だけ増やして `confidence` を潰す。実機のフレームで前後軸が「測れず」に
    # なったのがこれ。柱はどちらの線の上にも乗るので、両方に足す。
    vert, horz, posts = _split_by_direction(mask, bar_px)
    cols = ((vert + posts) > 0).sum(axis=0).astype(np.float64)
    rows = ((horz + posts) > 0).sum(axis=1).astype(np.float64)
    x_mm = (np.arange(w, dtype=np.float64) - w / 2.0) / g.px_per_mm_x
    y_mm = (np.arange(h, dtype=np.float64) - h / 2.0) / g.px_per_mm_y
    phase_x, conf_x = _phase(cols, x_mm, pitch_mm)
    phase_y, conf_y = _phase(rows, y_mm, pitch_mm)

    # 機体がセル中央なら格子線は ±90mm、つまり位相は pitch/2。そこからのずれが
    # 機体のずれ (符号は上のコメントのとおり)。
    left = _fold(phase_x - pitch_mm / 2.0, pitch_mm) / 1000.0
    forward = _fold(phase_y - pitch_mm / 2.0, pitch_mm) / 1000.0
    return LatticeOffset(
        forward_m=forward if conf_y >= min_confidence else None,
        left_m=left if conf_x >= min_confidence else None,
        forward_confidence=conf_y,
        left_confidence=conf_x,
    )


def offset_to_shift_px(offset: LatticeOffset,
                       geometry: CameraGeometry | None = None) -> tuple[int, int]:
    """:class:`LatticeOffset` を、ROI を動かす画素数に直す。

    機体が左 (+y) へずれると世界の特徴は画像の +x へ動く (`cell_pose` と同じ約束)
    ので、ROI も同じ向きへ動かせば帯の上に戻る。測れなかった軸は 0 にする
    (動かさない = いまの挙動のまま) — 片方だけでも直せば読みは良くなる。
    """
    g = geometry or CALIBRATED_GEOMETRY
    dx = 0 if offset.left_m is None else offset.left_m * 1000.0 * g.px_per_mm_x
    dy = 0 if offset.forward_m is None else offset.forward_m * 1000.0 * g.px_per_mm_y
    return (int(round(dx)), int(round(dy)))
