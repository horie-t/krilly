#!/usr/bin/env python3
"""L6470 デイジーチェーン (3台同時) の動作確認スクリプト (issue #25).

単体駆動(#5, motor_spin.py)で各基板の良品を確認した後、3台を1本の CS で
デイジーチェーン接続して同時制御できるかを確認する。

例:
    # 3台を 400 step/s で正転 3秒 (全台同方向)
    python -m scripts.motor_chain --devices 3 --speed 400 --duration 3

    # 台ごとに方向を変える (dev0 正転, dev1 逆転, dev2 正転)
    python -m scripts.motor_chain --devices 3 --dirs fwd,rev,fwd

配線ノート:
- Pi.MOSI -> dev0.SDI, dev0.SDO -> dev1.SDI, ..., dev(n-1).SDO -> Pi.MISO
- CS/SCLK は全台共有。index0 が MOSI に最も近い。逆順配線なら --miso-index0。
"""

from __future__ import annotations

import argparse
import time

from krilly.hal.l6470 import FWD, REV, L6470Profile, decode_status
from krilly.hal.l6470_chain import L6470Chain
from krilly.logging_config import get_logger, setup_logging

log = get_logger("krilly.motor_chain")


def _parse_dirs(s: str | None, n: int, default_dir: str) -> list[int]:
    tokens = s.split(",") if s else [default_dir] * n
    if len(tokens) != n:
        raise SystemExit(f"--dirs は {n} 個指定してください (例: fwd,rev,fwd)")
    return [FWD if t.strip() == "fwd" else REV for t in tokens]


def main() -> None:
    p = argparse.ArgumentParser(description="L6470 デイジーチェーン動作確認")
    p.add_argument("--devices", type=int, default=3, help="連結台数 (既定 3)")
    p.add_argument("--bus", type=int, default=0, help="SPI バス (既定 0)")
    p.add_argument("--device", type=int, default=0, help="SPI デバイス/CE (既定 0)")
    p.add_argument("--speed", type=float, default=400.0, help="回転速度 steps/s (全台共通)")
    p.add_argument("--dir", choices=["fwd", "rev"], default="fwd", help="既定回転方向")
    p.add_argument("--dirs", default=None, help="台ごとの方向 例: fwd,rev,fwd")
    p.add_argument("--duration", type=float, default=3.0, help="Run 継続秒数")
    p.add_argument("--miso-index0", action="store_true",
                   help="index0 が MISO 側 (逆順配線) の場合に指定")
    args = p.parse_args()

    setup_logging()
    n = args.devices
    dirs = _parse_dirs(args.dirs, n, args.dir)
    speeds = [args.speed] * n

    chain = L6470Chain(
        num_devices=n, bus=args.bus, device=args.device,
        mosi_is_index0=not args.miso_index0,
    )
    with chain:
        statuses = chain.configure_all(L6470Profile(max_speed_steps_s=max(args.speed, 400.0)))
        for i, st in enumerate(statuses):
            log.info("dev%d 初期化後 STATUS=0x%04X (%s)", i, st, decode_status(st, first_read=True))

        bad = [i for i, st in enumerate(statuses) if st in (0x0000, 0xFFFF)]
        if bad:
            log.error("dev%s が応答しません。チェーン配線(SDO->次段SDI)・CS/SCLK共有・"
                      "電源(VDD/VS)・台数(--devices)を確認してください。中止します。", bad)
            return

        log.info("run_all: dirs=%s speed=%.0f step/s を %.1f 秒",
                 ["fwd" if d else "rev" for d in dirs], args.speed, args.duration)
        chain.run_all(dirs, speeds)
        time.sleep(args.duration)
        chain.soft_stop_all()

        time.sleep(0.3)
        for i, st in enumerate(chain.get_status_all()):
            log.info("dev%d 終了 STATUS=0x%04X (%s)", i, st, decode_status(st))


if __name__ == "__main__":
    main()
