#!/usr/bin/env python3
"""1台ずつ L6470 + モーターの動作確認をするブリングアップスクリプト (issue #5).

例:
    # SPI0 CE0 のドライバを 400 step/s で正転 3秒
    python -m scripts.motor_spin --device 0 --speed 400 --duration 3

    # CE1 のドライバを逆転、相対 3200 マイクロステップだけ動かす
    python -m scripts.motor_spin --device 1 --dir rev --move 3200

各ドライバを順に CE0 / CE1 ... に差し替えて 1台ずつ確認する想定。
故障基板の切り分けに使う。
"""

from __future__ import annotations

import argparse
import time

from krilly.hal.l6470 import FWD, REV, L6470, L6470Profile
from krilly.logging_config import get_logger, setup_logging

log = get_logger("krilly.motor_spin")


def _decode_status(status: int) -> str:
    # よく見るフラグだけ簡易デコード (L6470 STATUS, アクティブLowに注意)
    flags = []
    if not (status & (1 << 9)):
        flags.append("OCD(過電流)")
    if not (status & (1 << 10)):
        flags.append("TH_WRN(熱警告)")
    if not (status & (1 << 11)):
        flags.append("TH_SD(熱遮断)")
    if not (status & (1 << 7)):
        flags.append("UVLO(低電圧)")
    if status & (1 << 0):
        flags.append("HiZ(出力停止)")
    return ", ".join(flags) if flags else "正常"


def main() -> None:
    p = argparse.ArgumentParser(description="L6470 単体動作確認")
    p.add_argument("--bus", type=int, default=0, help="SPI バス (既定 0)")
    p.add_argument("--device", type=int, default=0, help="SPI デバイス/CE (既定 0)")
    p.add_argument("--speed", type=float, default=400.0, help="回転速度 steps/s")
    p.add_argument("--dir", choices=["fwd", "rev"], default="fwd", help="回転方向")
    p.add_argument("--duration", type=float, default=3.0, help="Run の継続秒数")
    p.add_argument("--move", type=int, default=None,
                   help="指定すると Run ではなく相対マイクロステップ Move を実行")
    args = p.parse_args()

    setup_logging()
    direction = FWD if args.dir == "fwd" else REV

    with L6470(bus=args.bus, device=args.device) as drv:
        status = drv.configure(L6470Profile(max_speed_steps_s=max(args.speed, 400.0)))
        log.info("初期化後 STATUS=0x%04X (%s)", status, _decode_status(status))

        if args.move is not None:
            log.info("Move: dir=%s steps=%d", args.dir, args.move)
            drv.move(direction, args.move)
            # Move 完了待ち (BUSY フラグが立つまでポーリングするのが本来だが簡易に待つ)
            time.sleep(args.duration)
        else:
            log.info("Run: dir=%s speed=%.0f step/s を %.1f 秒", args.dir, args.speed, args.duration)
            drv.run(direction, args.speed)
            time.sleep(args.duration)
            drv.soft_stop()

        time.sleep(0.3)
        log.info("終了 STATUS=0x%04X (%s)", drv.get_status(), _decode_status(drv.get_status()))


if __name__ == "__main__":
    main()
