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
3. HoughLinesP で線分を取り、**迷路の軸から外れた線分を捨ててから**、長さで重み付けした
   **4θ 領域の円周平均** で 90° 周期の平均角を求める (#152)。捨てる基準は「最も多くの線分
   (長さの合計) が集まる角度」から ``inlier_deg`` 以上離れていること。2026-10-17 の会場で、
   閉じたマスに斜めに立てかけた余りの壁 (35-43° の線分、総長の 1/3) が平均を +5.76° /
   +4.39° へ引っ張り、+4.39° の方は補正のガードをすり抜けて機体を 4.4° 傾けた。

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

# 機体自身が写る固定領域 (基板・配線・リボンケーブル。ケーブルは赤として拾われうる)。
# 元は 640x480 (2026-08) で測った Roi(190, 135, 285, 345)。#88 で 960x720 の全画素モードへ
# 移ったとき (px/mm 据え置き・画像中心が (320, 240) -> (480, 360)) に動かし忘れ、機体ではなく
# 左上の迷路 (左の壁と前の壁の左半分) を消していた。中心を合わせて (+160, +120) ずらす。
# 会場の 3 セッション (498 枚) で読みの RMS は 0.60-1.12° -> 0.60-1.08° と変わらず、害は
# 出ていなかったが、機体を除外できていないので戻す (#152)。
CALIBRATED_ROBOT_RECT = Roi(350, 255, 285, 345)


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
    # 迷路の軸として採用する線分の幅 [deg]。最も線分が集まる角度からこれ以上離れた線分は
    # 迷路の外の物 (倒れた壁・斜めに置いた物) として捨てる (#152)。端の画素で透視と
    # レンズの歪みのため実機の壁でも ±3.4° まで振れるので 5°。0 で捨てない (従来の挙動)
    inlier_deg: float = 5.0


def calibrated_axis_yaw_config() -> AxisYawConfig:
    """実機校正済みの設定 (赤しきい値を緩め、機体の写り込みを除外)。"""
    return AxisYawConfig(exclude=[CALIBRATED_ROBOT_RECT])


@dataclass(frozen=True)
class AxisYaw:
    """軸角の推定結果。"""

    angle_rad: float          # (-45°, 45°] に折り返した軸角 (+ = 機体 CCW)
    segments: int             # 使った線分の本数
    total_length_px: float    # 線分の総長 (信頼度の目安)
    rejected_length_px: float = 0.0   # 迷路の軸から外れて捨てた線分の総長 (#152)

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
    segments = [(fold_rad(math.atan2(float(y2 - y1), float(x2 - x1))),
                 math.hypot(float(x2 - x1), float(y2 - y1)))
                for x1, y1, x2, y2 in lines[:, 0]]
    inliers = maze_axis_inliers(segments, math.radians(cfg.inlier_deg))
    rejected = sum(length for _a, length in segments) - sum(length for _a, length in inliers)
    sin_sum = cos_sum = total = 0.0
    for folded, length in inliers:
        # 4θ 領域で平均すると 90° 周期の角度を正しく平均できる
        sin_sum += length * math.sin(4.0 * folded)
        cos_sum += length * math.cos(4.0 * folded)
        total += length
    if total < cfg.min_total_length_px:
        return None
    return AxisYaw(math.atan2(sin_sum, cos_sum) / 4.0, len(inliers), total, rejected)


def maze_axis_inliers(segments: list[tuple[float, float]], tol_rad: float
                      ) -> list[tuple[float, float]]:
    """``(折り返した角度[rad], 長さ)`` の列から、迷路の軸に乗っている線分だけを返す。

    「その角度の ±tol に入る線分の長さの合計」が最大になる線分の角度を軸とみなし
    (平均ではなく**最頻**なので、外れた線分に引っ張られない)、そこから ±tol の線分を残す。
    ``tol_rad`` が 0 以下なら全部返す。角度は 90° 周期なので差も折り返して比べる。

    **迷路の外の物の線分が迷路の線分より多ければ、そちらを軸と取り違える。** 会場の画像
    では迷路が総長の 2/3 だったので効いたが、限界はある。それを止めるのは推定方位との差で
    見る補正のガード (:data:`krilly.localization.grid.MAX_HEADING_CORRECTION_RUNNING_RAD`)。
    """
    if tol_rad <= 0.0 or not segments:
        return list(segments)

    def near(a: float, b: float) -> bool:
        return abs(fold_rad(a - b)) <= tol_rad

    axis = max(segments,
               key=lambda s: sum(length for a, length in segments if near(a, s[0])))[0]
    return [s for s in segments if near(s[0], axis)]


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
