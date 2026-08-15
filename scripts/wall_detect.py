#!/usr/bin/env python3
"""セル壁判定の可視化・調整スクリプト (issue #16).

各辺の ROI と、その中の赤割合・壁有無を画像に重ねて保存する。保存済み画像
(--image) に対してオフラインで ROI 位置や閾値を追い込める。実機ではライブ取得。

``--measure N`` は**位置測定 (cell_pose) の再現性**を見るモード (#21)。機体を止めた
まま N フレーム撮り、辺ごとの赤割合と帯のずれ、そこから出る前後・左右のずれを並べる。
静止しているのだから値は動いてはいけない。動くならその辺の帯探索が別のもの
(ケーブルなど) を掴んでいる。定規で機体を既知量ずらして前後で撮れば、絶対量の
確認にもなる。

例:
    # 保存画像に対して ROI と判定を可視化
    python -m scripts.wall_detect --image ~/Documents/krilly/red_detect_20260730.png --out /tmp/walls.png
    # ライブ (カメラ)
    python -m scripts.wall_detect --out walls.png --thickness 70 --span 0.5
    # 位置測定の再現性 (実機・静止のまま 10 フレーム)
    python -m scripts.wall_detect --measure 10

ROI (front/back/left/right) を自機・ケーブル・支柱の外へ、壁が写る辺に合わせて
--thickness / --span / --threshold で調整する。カメラ取付の回転に応じて画像の
どの辺が機体前後左右かを確認すること。
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from krilly.logging_config import get_logger, setup_logging
from krilly.perception.cell_pose import (
    OFFSET_MIN_FRACTION,
    PX_PER_MM_X,
    PX_PER_MM_Y,
    CellOffset,
    cell_offset,
)
from krilly.perception.red_wall import RedDetectorConfig, red_mask
from krilly.perception.wall_detect import (
    BACK,
    CALIBRATED_RED,
    FRONT,
    LEFT,
    RIGHT,
    WallDetector,
    WallDetectorConfig,
    calibrated_config,
    calibrated_rois,
)

log = get_logger("krilly.wall_detect")


def measure_repeatability(count: int, interval: float, save_prefix: str | None) -> None:
    """静止したまま N フレーム撮り、位置測定のばらつきを表にする (#21)。

    帯探索は ROI を ±40px スライドして赤割合が最大の位置を採るので、本物の帯の
    近くに別の赤 (カメラのリボンケーブルなど) があると、フレームによって掴む対象が
    入れ替わり、測定値が跳ぶ。静止中のばらつきがそのまま「位置補正がどれだけ嘘を
    つきうるか」になる。
    """
    import time

    from krilly.hal.camera import Camera

    det = WallDetector(calibrated_config())
    px_per_mm = {FRONT: PX_PER_MM_Y, BACK: PX_PER_MM_Y,
                 LEFT: PX_PER_MM_X, RIGHT: PX_PER_MM_X}
    edges = (FRONT, BACK, LEFT, RIGHT)
    rows: list[tuple[dict[str, tuple[float, float]], CellOffset]] = []
    with Camera() as cam:
        for i in range(count):
            if i:
                time.sleep(interval)
            frame = cam.capture()
            measured = det.measure(frame)
            off = cell_offset(frame, det)
            rows.append((
                {e: (measured[e][0], measured[e][1] / px_per_mm[e]) for e in edges}, off))
            if save_prefix:
                cv2.imwrite(f"{save_prefix}_{i + 1:02d}.png", frame)
            log.info(
                "#%02d %s | 前後=%s 左右=%s", i + 1,
                " ".join("%s %.2f/%+.1fmm" % (e, rows[-1][0][e][0], rows[-1][0][e][1])
                         for e in edges),
                "--" if off.forward_m is None else "%+.1fmm" % (off.forward_m * 1e3),
                "--" if off.left_m is None else "%+.1fmm" % (off.left_m * 1e3),
            )

    def spread(values: list[float]) -> str:
        if not values:
            return "測定なし"
        lo, hi = min(values), max(values)
        mean = sum(values) / len(values)
        return "平均 %+.1fmm 幅 %.1fmm (%.1f〜%.1f)" % (mean, hi - lo, lo, hi)

    log.info("--- 静止中のばらつき (動いていないので幅は 0 であるべき) ---")
    for e in edges:
        used = [row[0][e][1] for row in rows
                if row[0][e][0] >= max(det.cfg.threshold_for(e), OFFSET_MIN_FRACTION)]
        log.info("%-6s 位置測定に使えたフレーム %d/%d  %s",
                 e, len(used), len(rows), spread(used))
    log.info("%-6s %s", "前後", spread([r[1].forward_m * 1e3 for r in rows
                                        if r[1].forward_m is not None]))
    log.info("%-6s %s", "左右", spread([r[1].left_m * 1e3 for r in rows
                                        if r[1].left_m is not None]))


def main() -> None:
    p = argparse.ArgumentParser(description="セル壁判定の可視化")
    p.add_argument("--measure", type=int, default=0, metavar="N",
                   help="静止したまま N フレーム撮り、位置測定の再現性を表示する")
    p.add_argument("--interval", type=float, default=0.3, help="--measure のフレーム間隔 [s]")
    p.add_argument("--save-prefix", default=None, help="--measure のフレーム保存先プレフィクス")
    p.add_argument("--image", default=None, help="入力画像 (未指定ならカメラ取得)")
    p.add_argument("--out", default="walls.png", help="注釈画像の保存先")
    p.add_argument("--threshold", type=float, default=0.15, help="壁ありとみなす赤割合")
    p.add_argument("--s-min", type=int, default=CALIBRATED_RED.s_min, help="赤HSVのS下限")
    p.add_argument("--v-min", type=int, default=CALIBRATED_RED.v_min, help="赤HSVのV下限")
    args = p.parse_args()

    setup_logging()
    if args.measure:
        measure_repeatability(args.measure, args.interval, args.save_prefix)
        return
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            log.error("画像を読み込めません: %s", args.image)
            return
    else:
        from krilly.hal.camera import Camera

        with Camera() as cam:
            frame = cam.capture()

    red = RedDetectorConfig(s_min=args.s_min, v_min=args.v_min)
    rois = calibrated_rois()  # 実機校正済みの各辺 ROI
    det = WallDetector(WallDetectorConfig(rois=rois, threshold=args.threshold, red=red))

    fractions = det.red_fractions(frame)
    walls = det.detect(frame)

    # 可視化: 検出した赤を薄く重ね、ROI 矩形とラベルを描く
    out = frame.copy()
    mask = red_mask(frame, red)
    out[mask > 0] = (0.4 * out[mask > 0] + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
    for d, roi in rois.items():
        present = walls[d]
        color = (0, 255, 0) if present else (0, 200, 255)  # 緑=壁あり / 黄=なし
        cv2.rectangle(out, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), color, 2)
        cv2.putText(out, f"{d} {fractions[d]:.2f} {'WALL' if present else '-'}",
                    (roi.x + 2, max(roi.y + 16, 16)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(args.out, out)
    for d in rois:
        log.info("%-6s red=%.3f -> %s", d, fractions[d], "壁あり" if walls[d] else "なし")
    log.info("注釈画像を保存: %s", args.out)


if __name__ == "__main__":
    main()
