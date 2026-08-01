"""赤い壁上面の直線エッジから迷路軸に対する yaw を測る (issue #17 の検証用)。

下向きカメラは車体固定なので、**画像に写る迷路の軸の傾き = 車体の yaw** (符号は
「機体 CCW = 角度 +」、実機で確認済み)。壁上面の赤帯は長い直線なので、その
エッジの向きを平均すれば、ジャイロに依存しない方位の実測値が得られる。

迷路の軸は 90° 周期なので、測れるのは **(-45°, 45°] に折り返した軸角**である。
90° の倍数の旋回では折り返し後の差分がそのまま「行き過ぎ/足りない」量になるため、
1セル前進・90°ターン (#17) の検証にはこれで足りる。

処理の流れ:
1. ``red_wall.red_mask`` で赤マスクを作り、機体自身 (基板・配線・リボンケーブル)
   の固定領域を除外する。
2. マスクの Canny エッジを取り、**画像の縁とマスク境界に沿う人工エッジを削る**。
   これを省くと、画像端で切れた赤帯のエッジが画像軸に張り付き、推定角が 0° 側へ
   引っ張られる (実機で最大 6° の誤差を出した)。
3. HoughLinesP で線分を取り、長さで重み付けした **4θ 領域の円周平均** で
   90° 周期の平均角を求める。

純粋に OpenCV/NumPy のみなので、合成画像でカメラなしにテストできる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from krilly.perception.red_wall import RedDetectorConfig, red_mask
from krilly.perception.wall_detect import CALIBRATED_RED, Roi

# 実機 (640x480, カメラ中央・高さ約39cm) で機体自身が写る固定領域。
# Pi 基板・配線・オレンジのリボンケーブルを含む (ケーブルは赤として拾われる)。
CALIBRATED_ROBOT_RECT = Roi(190, 135, 285, 345)


@dataclass(frozen=True)
class AxisYawConfig:
    """軸角推定のパラメータ。"""

    red: RedDetectorConfig = CALIBRATED_RED
    exclude: list[Roi] = field(default_factory=list)   # 機体自身などの固定領域
    margin_px: int = 4              # 人工エッジとして削る縁の幅
    min_length_px: int = 60         # 採用する線分の最小長
    max_gap_px: int = 6             # HoughLinesP の maxLineGap
    hough_threshold: int = 40       # HoughLinesP の投票しきい値
    min_total_length_px: float = 150.0   # 総線分長がこれ未満なら証拠不足


def calibrated_axis_yaw_config() -> AxisYawConfig:
    """実機校正済みの設定 (赤しきい値を緩め、機体の写り込みを除外)。"""
    return AxisYawConfig(exclude=[CALIBRATED_ROBOT_RECT])


@dataclass(frozen=True)
class AxisYaw:
    """軸角の推定結果。"""

    angle_rad: float          # (-45°, 45°] に折り返した軸角 (+ = 機体 CCW)
    segments: int             # 使った線分の本数
    total_length_px: float    # 線分の総長 (信頼度の目安)

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle_rad)


def fold_rad(a: float) -> float:
    """角度を (-45°, 45°] 相当 (-π/4, π/4] に折り返す (軸は 90° 周期)。"""
    quarter = math.pi / 2.0
    return (a + quarter / 2.0) % quarter - quarter / 2.0


def fold_deg(a: float) -> float:
    """角度[deg]を (-45, 45] に折り返す。"""
    return (a + 45.0) % 90.0 - 45.0


def yaw_delta_rad(before: AxisYaw, after: AxisYaw) -> float:
    """2 つの観測の間に機体が回った角度から 90° の倍数を除いた差分[rad] (+ = CCW)。

    指令が 90° の倍数 (90°ターン / 1セル前進) なら理想の差分は 0 なので、この値が
    そのまま **理想からの行き過ぎ量** になる (+ = CCW 側へ行き過ぎ、- = 足りない)。
    """
    return fold_rad(after.angle_rad - before.angle_rad)


def _artificial_edge_free(mask: np.ndarray, cfg: AxisYawConfig) -> np.ndarray:
    """マスクの Canny エッジから、画像の縁・除外矩形の境界に沿う成分を削る。"""
    edges = cv2.Canny(mask, 50, 150)
    m = cfg.margin_px
    if m > 0:
        edges[:m, :] = 0
        edges[-m:, :] = 0
        edges[:, :m] = 0
        edges[:, -m:] = 0
    for r in cfg.exclude:
        cv2.rectangle(
            edges,
            (r.x - m, r.y - m),
            (r.x + r.w + m, r.y + r.h + m),
            0,
            thickness=2 * m + 1,
        )
    return edges


def axis_yaw(bgr: np.ndarray, config: AxisYawConfig | None = None) -> AxisYaw | None:
    """フレームから軸角を推定する。証拠が足りなければ None。"""
    cfg = config or AxisYawConfig()
    mask = red_mask(bgr, cfg.red)
    for r in cfg.exclude:
        mask[r.y : r.y + r.h, r.x : r.x + r.w] = 0
    lines = cv2.HoughLinesP(
        _artificial_edge_free(mask, cfg),
        1,
        np.pi / 720,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.min_length_px,
        maxLineGap=cfg.max_gap_px,
    )
    if lines is None:
        return None
    sin_sum = cos_sum = total = 0.0
    count = 0
    for x1, y1, x2, y2 in lines[:, 0]:
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        folded = fold_rad(math.atan2(float(y2 - y1), float(x2 - x1)))
        # 4θ 領域で平均すると 90° 周期の角度を正しく平均できる
        sin_sum += length * math.sin(4.0 * folded)
        cos_sum += length * math.cos(4.0 * folded)
        total += length
        count += 1
    if total < cfg.min_total_length_px:
        return None
    return AxisYaw(math.atan2(sin_sum, cos_sum) / 4.0, count, total)


def median_axis_yaw(
    frames: Iterable[np.ndarray], config: AxisYawConfig | None = None
) -> AxisYaw | None:
    """複数フレームの推定の**中央値**を返す (外れフレームに強くする)。

    返り値の ``segments`` / ``total_length_px`` は中央値を与えたフレームのもの。
    """
    results = [r for r in (axis_yaw(f, config) for f in frames) if r is not None]
    if not results:
        return None
    results.sort(key=lambda r: r.angle_rad)
    return results[len(results) // 2]


def annotate(bgr: np.ndarray, config: AxisYawConfig | None = None) -> np.ndarray:
    """デバッグ用: 赤マスク・除外矩形・採用した線分を重ねた画像を返す。"""
    cfg = config or AxisYawConfig()
    vis = bgr.copy()
    mask = red_mask(bgr, cfg.red)
    for r in cfg.exclude:
        mask[r.y : r.y + r.h, r.x : r.x + r.w] = 0
    vis[mask > 0] = (0, 255, 255)
    for r in cfg.exclude:
        cv2.rectangle(vis, (r.x, r.y), (r.x + r.w, r.y + r.h), (255, 0, 0), 2)
    lines = cv2.HoughLinesP(
        _artificial_edge_free(mask, cfg),
        1,
        np.pi / 720,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.min_length_px,
        maxLineGap=cfg.max_gap_px,
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return vis
