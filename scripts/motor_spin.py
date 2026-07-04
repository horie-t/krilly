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


def _decode_status(status: int, first_read: bool = False) -> str:
    # L6470 STATUS のフォールト系ビットを簡易デコード。
    # UVLO/TH_WRN/TH_SD/OCD/STEP_LOSS はアクティブLow (0 = 発生)。
    # first_read=True: リセット後の初回 GetStatus。UVLO の扱いを注記する。
    if status in (0x0000, 0xFFFF):
        return ("★SPI通信不可の可能性 (応答が全ビット %s)。正常なら 0x7C03 付近。"
                "配線(MISO/MOSI/SCK/CS/GND)・電源(VDD/VS)・SPI有効化・CE番号を確認"
                % ("0" if status == 0x0000 else "1"))
    flags = []
    if not (status >> 9) & 1:
        # UVLO はアクティブLow。電源投入時に必ずラッチされ、最初の GetStatus で
        # クリアされる。よって初回読み出しでの UVLO は正常（実際の低電圧ではない）。
        flags.append("UVLO(初回読み出しなら電源投入時の正常フラグ)"
                     if first_read else "UVLO(低電圧)")
    if not (status >> 10) & 1:
        flags.append("TH_WRN(熱警告)")
    if not (status >> 11) & 1:
        flags.append("TH_SD(熱遮断)")
    if not (status >> 12) & 1:
        flags.append("OCD(過電流)")
    if not (status >> 13) & 1:
        flags.append("STEP_LOSS_A")
    if not (status >> 14) & 1:
        flags.append("STEP_LOSS_B")
    if status & 1:
        flags.append("HiZ(出力停止)")
    return ", ".join(flags) if flags else "フォールトなし"


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
        log.info("初期化後 STATUS=0x%04X (%s)", status, _decode_status(status, first_read=True))

        if status in (0x0000, 0xFFFF):
            log.error("SPI 応答が異常のためモーター指令を中止します。"
                      "配線・電源・SPI有効化・CE番号を確認してください。")
            return

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
        end_status = drv.get_status()
        log.info("終了 STATUS=0x%04X (%s)", end_status, _decode_status(end_status))


if __name__ == "__main__":
    main()
