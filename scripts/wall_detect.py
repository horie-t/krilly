#!/usr/bin/env python3
"""セル壁判定の可視化・調整スクリプト (issue #16).

各辺の ROI と、その中の赤割合・壁有無を画像に重ねて保存する。保存済み画像
(--image) に対してオフラインで ROI 位置や閾値を追い込める。実機ではライブ取得。

例:
    # 保存画像に対して ROI と判定を可視化
    python -m scripts.wall_detect --image ~/Documents/krilly/red_detect_20260730.png --out /tmp/walls.png
    # ライブ (カメラ)
    python -m scripts.wall_detect --out walls.png --thickness 70 --span 0.5

ROI (front/back/left/right) を自機・ケーブル・支柱の外へ、壁が写る辺に合わせて
--thickness / --span / --threshold で調整する。カメラ取付の回転に応じて画像の
どの辺が機体前後左右かを確認すること。
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from krilly.logging_config import get_logger, setup_logging
from krilly.perception.red_wall import RedDetectorConfig, red_mask
from krilly.perception.wall_detect import (
    WallDetector,
    WallDetectorConfig,
    default_rois,
)

log = get_logger("krilly.wall_detect")


def main() -> None:
    p = argparse.ArgumentParser(description="セル壁判定の可視化")
    p.add_argument("--image", default=None, help="入力画像 (未指定ならカメラ取得)")
    p.add_argument("--out", default="walls.png", help="注釈画像の保存先")
    p.add_argument("--thickness", type=int, default=70, help="ROI の辺方向の厚み[px]")
    p.add_argument("--span", type=float, default=0.5, help="ROI の辺に沿う割合")
    p.add_argument("--threshold", type=float, default=0.15, help="壁ありとみなす赤割合")
    p.add_argument("--s-min", type=int, default=RedDetectorConfig.s_min)
    p.add_argument("--v-min", type=int, default=RedDetectorConfig.v_min)
    args = p.parse_args()

    setup_logging()
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            log.error("画像を読み込めません: %s", args.image)
            return
    else:
        from krilly.hal.camera import Camera

        with Camera() as cam:
            frame = cam.capture()

    h, w = frame.shape[:2]
    red = RedDetectorConfig(s_min=args.s_min, v_min=args.v_min)
    rois = default_rois(w, h, thickness=args.thickness, span=args.span)
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
