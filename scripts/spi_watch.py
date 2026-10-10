#!/usr/bin/env python3
"""SPI の接触を見張る (試走会 2026-09-26)。

L6470 の STATUS を 20Hz で読み続け、**応答が無い** (3 台とも 0x0000 / 0xFFFF) 回と
**WRONG_CMD** (化けたコマンドを受けた) 回を数える。実行中に SPI の線とコネクタを指で
揺らし、どこで途切れるかを探す。

試走会では、走行中に 3 台とも ``NO_SPI`` になって中断した。VS (モータ電源) が落ちた
なら L6470 はリセットされ、次の STATUS に UVLO が残る — それが**残っておらず**、M2 に
WRONG_CMD が立っていたので、電源ではなく SPI の線の接触を疑い、この見張りで再現した
(揺らすと 30 秒で 12 回途切れ、挿し直した後は 598 回とも正常)。**SPI が途切れている間、
L6470 は最後の Run 指令のまま回り続ける**ので、走行前に確かめる価値がある。

例:
    python -m scripts.spi_watch --seconds 30           # 機体は動かない (HiZ のまま)
    python -m scripts.spi_watch --seconds 30 --spin    # 車輪を回しながら (**機体を浮かせる**)

Ctrl-C で止まる (``emergency_stop`` がモータを解放する)。
"""

from __future__ import annotations

import argparse
import time

#: 応答が無いときに読める値 (MISO が張り付いている)。
DEAD = (0x0000, 0xFFFF)
#: STATUS の WRONG_CMD ビット (active-high、読むと消える)。
WRONG_CMD = 0x0100


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="L6470 の SPI 応答を見張る")
    p.add_argument("--seconds", type=float, default=30.0, help="見張る時間 [s]")
    p.add_argument("--spin", action="store_true",
                   help="車輪を 100 step/s で回しながら読む (**機体を浮かせてから**)")
    p.add_argument("--bus", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    return p


def main() -> int:
    args = build_parser().parse_args()

    from krilly.hal.l6470_chain import L6470Chain
    from krilly.motion.emergency_stop import emergency_stop

    with L6470Chain(bus=args.bus, device=args.device) as chain, emergency_stop(chain):
        chain.configure_all()
        chain.get_status_all()                  # 電源投入時のフラグを読み捨てる
        if args.spin:
            chain.run_all([1, 1, 1], [100.0, 100.0, 100.0])
        n = dead = wrong = 0
        end = time.monotonic() + args.seconds
        last_report = 0.0
        while time.monotonic() < end:
            st = chain.get_status_all()
            n += 1
            if all(s in DEAD for s in st):
                dead += 1
                print(f"{time.strftime('%H:%M:%S')} 応答なし {[hex(s) for s in st]}", flush=True)
            elif any(s & WRONG_CMD for s in st):
                wrong += 1
                print(f"{time.strftime('%H:%M:%S')} WRONG_CMD {[hex(s) for s in st]}", flush=True)
            if time.monotonic() - last_report > 5:
                print(f"  ... {n} 回読んで 応答なし {dead} / WRONG_CMD {wrong}", flush=True)
                last_report = time.monotonic()
            time.sleep(0.05)
        chain.soft_stop_all()
        chain.hard_hiz_all()
    print(f"結果: {n} 回読んで 応答なし {dead} 回 / WRONG_CMD {wrong} 回")
    return 0 if dead == 0 and wrong == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
