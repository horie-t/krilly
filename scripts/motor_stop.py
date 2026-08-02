#!/usr/bin/env python3
"""緊急停止: L6470 の出力を解放してモーターを止める。

走行スクリプトが強制終了 (kill -9 など) されて後始末が走らなかった場合、L6470 は
最後の Run 指令のまま**回り続ける**。そのときにこれを実行すると止まる。

    python -m scripts.motor_stop

やっていること: ``soft_stop_all`` (減速停止) -> ``hard_hiz_all`` (ブリッジ開放)。
実行しても止まらない場合はソフト側では制御できない状態なので、**モーター電源 (VS)
を落とすこと**。

なお通常の停止は **Ctrl-C** (SIGINT)。走行スクリプトは
:func:`krilly.motion.emergency_stop.emergency_stop` を使っているので、Ctrl-C でも
``timeout`` の SIGTERM でも解放される。**ESC キーは効かない** (キー入力は読んでいない)。
"""

from __future__ import annotations

import argparse

from krilly.hal.l6470_chain import L6470Chain
from krilly.logging_config import get_logger, setup_logging
from krilly.motion.emergency_stop import release

log = get_logger("krilly.motor_stop")


def main() -> None:
    p = argparse.ArgumentParser(description="L6470 の出力を解放して緊急停止する")
    p.add_argument("--devices", type=int, default=3, help="連結台数")
    p.add_argument("--bus", type=int, default=0, help="SPI バス")
    p.add_argument("--device", type=int, default=0, help="SPI デバイス/CE")
    args = p.parse_args()

    setup_logging()
    chain = L6470Chain(num_devices=args.devices, bus=args.bus, device=args.device)
    release(chain)
    log.info("soft_stop + hard_hiz を送信した (出力を解放)。"
             "止まらない場合はモーター電源 (VS) を落とすこと。")


if __name__ == "__main__":
    main()
