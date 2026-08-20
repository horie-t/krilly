"""赤い壁上部の検出 (issue #7)。

下向きのカメラは迷路の壁を捉える。壁の上部は **赤** に塗られている。
このモジュールは BGR フレームを赤のマスクと、赤い領域 (壁上部) の重心へと変換する。
純粋に OpenCV/NumPy のみで実装しているため、カメラなしでもユニットテストできる。

赤は HSV の hue 境界をまたぐため、2 つの hue 範囲 (低域と高域) を OR で結合する。
デフォルト値はあくまで出発点である。競技規則では壁が退色したり混ざったりすると
注意されているので、実際の迷路 / 照明に合わせて S/V の下限を調整し、
カメラの露出と AWB をロックすること。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RedDetectorConfig:
    """HSV しきい値 (OpenCV の範囲: H 0-179, S/V 0-255) とフィルタリング設定。"""

    h1_lo: int = 0
    h1_hi: int = 10
    h2_lo: int = 160
    h2_hi: int = 179
    s_min: int = 100
    v_min: int = 70
    min_area: float = 100.0   # これより小さい赤の blob は無視する (px^2)
    morph_kernel: int = 3     # 0 で open/close によるノイズ除去を無効化


@dataclass(frozen=True)
class RedRegion:
    """検出された 1 つの赤い blob。"""

    cx: float
    cy: float
    area: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


def red_mask_parts(
    bgr: np.ndarray, config: RedDetectorConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """赤マスクを**色相の 2 帯に分けたまま**返す ``(h1, h2)`` (どちらも 0/255)。

    OpenCV の H は 0-179 の一周なので、赤は下端 (``h1_lo..h1_hi``、オレンジ寄り) と
    上端 (``h2_lo..h2_hi``、マゼンタ寄り) の 2 帯に割れる。:func:`red_mask` はこれを
    OR して 1 枚にするが、**どちらの帯で拾ったかは調べものの決め手になる**ので分けて
    取れるようにしてある。

    実例 (#87): リボンケーブルに黒テープを貼った後も 10% ほど赤判定が残った。単色で
    重ねても原因が分からなかったが、2 帯を別の色で塗ると一目で分かった — 壁は
    すべて h2 (H 169) なのに、ケーブル上の画素は h1 (H 7) だった。隣のテープが H 15
    だったことから、**テープ自身が ``h1_hi`` の境界をまたいでいる**と特定できた
    (ラベルの貼り残しでも壁の映り込みでもない)。

    形態学的なノイズ除去は :func:`red_mask` と同じものを**各帯に**掛ける。そのため
    ``h1 | h2`` は :func:`red_mask` と厳密には一致しない (2 帯の境目で片方だけでは
    小さすぎて消える塊が、合わせれば残ることがある。実測で 1% 前後)。**判定には
    :func:`red_mask` を使い、この関数は調べもの専用にすること。**
    """
    cfg = config or RedDetectorConfig()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo1 = np.array([cfg.h1_lo, cfg.s_min, cfg.v_min], dtype=np.uint8)
    hi1 = np.array([cfg.h1_hi, 255, 255], dtype=np.uint8)
    lo2 = np.array([cfg.h2_lo, cfg.s_min, cfg.v_min], dtype=np.uint8)
    hi2 = np.array([cfg.h2_hi, 255, 255], dtype=np.uint8)
    parts = [cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2)]
    if cfg.morph_kernel > 0:
        k = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
        parts = [cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k),
                                  cv2.MORPH_CLOSE, k) for m in parts]
    return parts[0], parts[1]


def red_mask(bgr: np.ndarray, config: RedDetectorConfig | None = None) -> np.ndarray:
    """BGR 画像中の赤ピクセルを表す uint8 (0/255) のマスクを返す。"""
    cfg = config or RedDetectorConfig()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo1 = np.array([cfg.h1_lo, cfg.s_min, cfg.v_min], dtype=np.uint8)
    hi1 = np.array([cfg.h1_hi, 255, 255], dtype=np.uint8)
    lo2 = np.array([cfg.h2_lo, cfg.s_min, cfg.v_min], dtype=np.uint8)
    hi2 = np.array([cfg.h2_hi, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))
    if cfg.morph_kernel > 0:
        k = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_red_regions(
    bgr: np.ndarray, config: RedDetectorConfig | None = None
) -> list[RedRegion]:
    """赤い blob を検出し、重心 / bbox として面積の大きい順に返す。"""
    cfg = config or RedDetectorConfig()
    mask = red_mask(bgr, cfg)
    # findContours は cv2 4.x では (contours, hierarchy) を、3.x では (img, ...) を返す
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    regions: list[RedRegion] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        regions.append(RedRegion(cx, cy, float(area), cv2.boundingRect(c)))
    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def annotate(bgr: np.ndarray, regions: list[RedRegion]) -> np.ndarray:
    """動作確認用にバウンディングボックスと重心を描画する。コピーを返す。"""
    out = bgr.copy()
    for r in regions:
        x, y, w, h = r.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (int(round(r.cx)), int(round(r.cy))), 4, (255, 0, 0), -1)
    return out
