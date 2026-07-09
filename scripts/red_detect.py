#!/usr/bin/env python3
"""カメラ取得＋赤壁検出の可視化スクリプト (issue #7).

下向きカメラのフレームから赤い壁上面を検出し、検出結果をログ表示しつつ
注釈付き画像を保存する（ヘッドレス前提。表示環境があれば --preview）。

例:
    # 赤検出を続け、注釈画像を red_detect.png に随時保存
    python -m scripts.red_detect --out red_detect.png --duration 20

    # 色が反転している場合(R/B入替)
    python -m scripts.red_detect --swap-rb

    # 閾値調整
    python -m scripts.red_detect --s-min 120 --v-min 80 --min-area 200

配線/事前: Camera Module V3 を接続し `rpicam-hello --list-cameras` で認識確認。
"""

from __future__ import annotations

import argparse
import time

import cv2

from krilly.hal.camera import Camera
from krilly.logging_config import get_logger, setup_logging
from krilly.perception.red_wall import RedDetectorConfig, annotate, detect_red_regions

log = get_logger("krilly.red_detect")


def main() -> None:
    p = argparse.ArgumentParser(description="カメラ赤壁検出の可視化")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--out", default="red_detect.png", help="注釈画像の保存先")
    p.add_argument("--hz", type=float, default=10.0, help="処理レート")
    p.add_argument("--duration", type=float, default=None, help="秒数 (既定: 無限)")
    p.add_argument("--swap-rb", action="store_true", help="R/B を入れ替える")
    p.add_argument("--preview", action="store_true", help="cv2.imshow で表示(要ディスプレイ)")
    p.add_argument("--s-min", type=int, default=RedDetectorConfig.s_min)
    p.add_argument("--v-min", type=int, default=RedDetectorConfig.v_min)
    p.add_argument("--min-area", type=float, default=RedDetectorConfig.min_area)
    args = p.parse_args()

    setup_logging()
    cfg = RedDetectorConfig(s_min=args.s_min, v_min=args.v_min, min_area=args.min_area)
    period = 1.0 / args.hz if args.hz > 0 else 0.0

    log.info("Ctrl+C で終了 (自動終了は --duration)")
    with Camera(width=args.width, height=args.height) as cam:
        start = time.monotonic()
        try:
            while args.duration is None or (time.monotonic() - start) < args.duration:
                frame = cam.capture()
                if args.swap_rb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                regions = detect_red_regions(frame, cfg)
                if regions:
                    top = regions[0]
                    log.info("赤領域 %d 個 | 最大: 中心=(%.0f,%.0f) 面積=%.0f",
                             len(regions), top.cx, top.cy, top.area)
                else:
                    log.info("赤領域なし")
                annotated = annotate(frame, regions)
                cv2.imwrite(args.out, annotated)
                if args.preview:
                    cv2.imshow("red_detect", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                time.sleep(period)
        except KeyboardInterrupt:
            log.info("終了します")
        finally:
            if args.preview:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
