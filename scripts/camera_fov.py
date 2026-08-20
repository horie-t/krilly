#!/usr/bin/env python3
"""カメラの画角を測り、全画素モードと比べる (issue #87).

**picamera2 は要求サイズから勝手にセンサーモードを選ぶ。** 640x480 を頼むと IMX708 の
1536x864 モードが選ばれるが、このモードは ``crop_limits`` が (768, 432, 3072, 1728) で
**それ自体が中央 67% の切り出し**。さらに 16:9 のセンサーから 4:3 を切り出すので横が
50% になり、結局**センサー面積の 33% しか使っていない**。

全画素モード (2304x1296、``crop_limits`` が 4608x2592) を明示すると縦横とも 1.5 倍の
画角になる。**出力も 1.5 倍にすれば分解能は据え置き**で画角だけ広がる
(640x480 -> 960x720 で px/mm は 1.70 のまま)。画像処理は 3.4ms -> 6.0ms しか増えず、
1 セルの停止 0.44s に対して無視できる。

前提: **4 辺すべてに壁のあるセルの中央に機体を置く**。帯の位置から px/mm を出すので、
対向する帯の間隔が 1 ピッチ = 180mm であることを使う。

例:
    python -m scripts.camera_fov                      # 現状と全画素を比べる
    python -m scripts.camera_fov --size 960x720       # 全画素側の出力を指定
    python -m scripts.camera_fov --out-dir fov        # フレームも保存する
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from krilly.config import load_maze_config
from krilly.hal.camera import Camera
from krilly.logging_config import get_logger, setup_logging
from krilly.perception.red_wall import red_mask
from krilly.perception.wall_detect import (
    BACK,
    CALIBRATED_RED,
    FRONT,
    LEFT,
    RIGHT,
    band_positions,
)

log = get_logger(__name__)


def parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def measure(cam: Camera, pitch_mm: float, label: str, out_dir: Path | None):
    """1 枚撮って帯の位置から px/mm と画角を出す。"""
    frame = cam.capture()
    h, w = frame.shape[:2]
    bands = band_positions(red_mask(frame, CALIBRATED_RED))
    log.info("[%s] %dx%d  帯 %s", label, w, h,
             {k: v for k, v in sorted(bands.items())})
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{label}.png"), frame)

    if not {FRONT, BACK} <= bands.keys() or not {LEFT, RIGHT} <= bands.keys():
        log.warning("[%s] 4 辺すべての帯が見えない -> px/mm を出せない "
                    "(4 辺に壁のあるセルの中央に置くこと)", label)
        return None

    def centre(edge):
        lo, hi = bands[edge]
        return (lo + hi) / 2

    # 対向する帯の間隔がちょうど 1 ピッチ
    px_y = (centre(BACK) - centre(FRONT)) / pitch_mm
    px_x = (centre(RIGHT) - centre(LEFT)) / pitch_mm
    cy, cx = (centre(FRONT) + centre(BACK)) / 2, (centre(LEFT) + centre(RIGHT)) / 2
    log.info("[%s] px/mm  X %.3f  Y %.3f", label, px_x, px_y)
    log.info("[%s] 画角 %.0f x %.0f mm = %.2f x %.2f セル",
             label, w / px_x, h / px_y, w / px_x / pitch_mm, h / px_y / pitch_mm)
    log.info("[%s] セル中心の画素 (%.0f, %.0f) / 光軸 (%.0f, %.0f) "
             "-> 取付のずれ 前 %+.0fmm 右 %+.0fmm",
             label, cx, cy, w / 2, h / 2, (cy - h / 2) / px_y, (w / 2 - cx) / px_x)
    log.info("[%s] セル中心から 前 %.0f / 後 %.0f / 左 %.0f / 右 %.0f mm "
             "(壁は %.0fmm 先、隣セルの奥の壁は %.0fmm)",
             label, cy / px_y, (h - 1 - cy) / px_y, cx / px_x, (w - 1 - cx) / px_x,
             pitch_mm / 2, pitch_mm * 1.5)
    return px_x, px_y, cx, cy, w, h


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="960x720",
                   help="全画素モード側の出力 (既定 960x720 = 画角 1.5 倍で px/mm 据え置き)")
    p.add_argument("--current-size", default="640x480", help="比較する現行の出力")
    p.add_argument("--out-dir", default=None, help="フレームの保存先")
    p.add_argument("--full-only", action="store_true", help="全画素モードだけ測る")
    args = p.parse_args()
    setup_logging()

    pitch_mm = load_maze_config().cell_pitch_m * 1000.0
    out_dir = Path(args.out_dir) if args.out_dir else None
    results = {}

    if not args.full_only:
        w, h = parse_size(args.current_size)
        with Camera(width=w, height=h) as cam:
            results["現行"] = measure(cam, pitch_mm, "current", out_dir)

    w, h = parse_size(args.size)
    with Camera(width=w, height=h, full_fov=True) as cam:
        results["全画素"] = measure(cam, pitch_mm, "full_fov", out_dir)

    a, b = results.get("現行"), results.get("全画素")
    if a and b:
        log.info("---")
        # 画角は「地面の範囲 = 出力画素数 / px/mm」で比べる。px/mm だけ見ると、
        # 出力も一緒に大きくしたときに 1.00 倍と出てしまう (画角は広がっているのに)。
        log.info("画角の比: X %.2f 倍 / Y %.2f 倍 (期待 1.50)",
                 (b[4] / b[0]) / (a[4] / a[0]), (b[5] / b[1]) / (a[5] / a[1]))
        log.info("分解能の比: %.2f 倍 (1.00 なら壁判定のしきい値はそのまま使える見込み)",
                 b[0] / a[0])


if __name__ == "__main__":
    main()
