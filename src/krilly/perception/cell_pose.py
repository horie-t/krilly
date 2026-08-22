"""赤い壁の帯の位置から、セル中央に対する機体のずれを測る (issue #54)。

並進はオープンループのオドメトリ (指令速度の積分) なので、誤差は走行距離に比例して
累積し、推定側からは見えない。実機では**その場回転の滑りで最大 25mm** 溜まっていた
(#56 の全セル調査)。ROI 判定はセル中央にいる前提なので、放置すると壁判定から崩れる。

本モジュールは、壁上面の赤帯が**校正位置からどれだけずれて写っているか**を
セル内の位置ずれとして読み取る。壁は必ずセル中央から 90mm の位置にあるので、
帯のずれ = 機体のずれ (符号は反転) である。

    帯のずれ [px] / (px per mm) = 機体のずれ [mm]

- 縦帯 (LEFT/RIGHT) の列方向のずれ → 機体の**左右**方向のずれ
- 横帯 (FRONT/BACK) の行方向のずれ → 機体の**前後**方向のずれ
- 壁が無い辺は帯が無いので測れない。**測れた軸だけ**補正する
- 対向する 2 枚の壁が見えていれば平均して雑音を落とす (並進では両方が同じ向きに
  ずれる。回転なら逆向きにずれるので、平均すると回転成分は落ちる)

px/mm は既知の幾何から出る: 対向する帯の間隔がセルピッチ 180mm に対応する
(:data:`CALIBRATED_BANDS` の実測位置から算出)。行方向と列方向でわずかに違うのは
レンズの歪みと取付高さのため、別々に持つ。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from krilly.perception.wall_detect import (
    BACK,
    CALIBRATED_GEOMETRY,
    FRONT,
    LEFT,
    RIGHT,
    CameraGeometry,
    WallDetector,
)

CELL_PITCH_MM = 180.0

# 対向する帯の間隔 = 1 セルピッチ。列方向 (左右) と行方向 (前後) で別に持つ。
# 実体は :class:`CameraGeometry` にあり、ここはその既定値の別名。**画角を変えたら
# geometry ごと差し替えること** (出力解像度も同じ倍率にすれば px/mm は変わらないので、
# #88 の全画素モードへの移行ではこの値は据え置きになる)。
PX_PER_MM_X = CALIBRATED_GEOMETRY.px_per_mm_x
PX_PER_MM_Y = CALIBRATED_GEOMETRY.px_per_mm_y


@dataclass(frozen=True)
class CellOffset:
    """セル中央に対する機体のずれ (車体フレーム、単位 m)。

    測れなかった軸は None。``walls_x`` / ``walls_y`` は各軸で根拠にした壁の枚数。
    """

    forward_m: float | None   # +x (前) 方向のずれ
    left_m: float | None      # +y (左) 方向のずれ
    walls_x: int              # 左右方向の根拠 (LEFT/RIGHT のうち壁だった枚数)
    walls_y: int              # 前後方向の根拠 (FRONT/BACK のうち壁だった枚数)
    saturated: tuple[str, ...] = ()   # 帯探索がフレーム端で頭打ちになり捨てた辺

    @property
    def measured(self) -> bool:
        return self.forward_m is not None or self.left_m is not None


# 位置測定に使う帯の最低赤割合。壁の有無判定 (threshold=0.08) より厳しくする。
# 実測 (#65) では実壁の帯は >=0.30、ケーブル等の偽帯は <=0.15 なので 0.25 で分離
# できる。位置補正は機体を物理的に動かすため、弱い証拠で動いてはいけない
# (偽帯の -24mm を信じて壁へ突進した)。壁判定の方は見落とし=衝突なので低いまま。
OFFSET_MIN_FRACTION = 0.25


def cell_offset(
    bgr: np.ndarray,
    detector: WallDetector,
    min_fraction: float = OFFSET_MIN_FRACTION,
    geometry: CameraGeometry | None = None,
    measured: dict[str, tuple[float, int, bool]] | None = None,
) -> CellOffset:
    """フレームから、セル中央に対する機体のずれを推定する。

    ``detector`` は :class:`WallDetector` (帯探索が有効なもの)。**確実に壁と言える
    帯 (赤割合 >= min_fraction)** のずれだけを使う。

    帯探索がフレーム端で頭打ちになった辺 (飽和) も捨てる。頭打ちの値は「小さめの
    もっともらしいずれ」に化けるので、そのまま使うと**実際より中央寄りに居ると
    誤解した補正**を掛けてしまう (#21: BACK は前方向へ +14.7mm までしか測れない)。

    ``measured`` に :meth:`WallDetector.measure` の結果を渡せば測り直さない。壁判定と
    同じフレームから位置も出すときに使う (全画素モードでは赤マスクだけで数十 ms かかる)。
    """
    if measured is None:
        measured = detector.measure(bgr)

    def usable(e: str) -> bool:
        fraction, _offset, saturated = measured[e]
        return (not saturated
                and fraction >= max(detector.cfg.threshold_for(e), min_fraction))

    dx_px = [float(measured[e][1]) for e in (LEFT, RIGHT) if usable(e)]
    dy_px = [float(measured[e][1]) for e in (FRONT, BACK) if usable(e)]
    saturated = tuple(e for e in (FRONT, BACK, LEFT, RIGHT) if measured[e][2])
    # 帯が画像上でずれる向きと機体がずれる向きは同じ符号になる:
    # 機体が左 (+y) へずれると、世界の特徴は機体の右へ動く = 画像の右 (+x) へ動く。
    g = geometry or CALIBRATED_GEOMETRY
    left_m = (sum(dx_px) / len(dx_px)) / g.px_per_mm_x / 1000.0 if dx_px else None
    forward_m = (sum(dy_px) / len(dy_px)) / g.px_per_mm_y / 1000.0 if dy_px else None
    return CellOffset(forward_m, left_m, len(dx_px), len(dy_px), saturated)


def body_offset_to_world(
    forward_m: float | None, left_m: float | None, phi: float
) -> tuple[float, float]:
    """車体フレームのずれを世界フレーム (東, 北) へ回す。None は 0 として扱う。"""
    f = forward_m or 0.0
    l = left_m or 0.0
    c, s = math.cos(phi), math.sin(phi)
    return (f * c - l * s, f * s + l * c)
