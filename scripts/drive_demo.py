#!/usr/bin/env python3
"""加減速ランプ付き速度指令の動作確認スクリプト (issue #9).

3輪を協調ランプさせ、直進 → 横移動 → その場回転 のシーケンスを実行する。
各区間は目標速度までランプアップし、保持後に 0 までランプダウンする。

例:
    python -m scripts.drive_demo --v 0.1 --omega 1.0 --hold 2

配線/事前: L6470 3台をデイジーチェーン接続 (docs/coordinate-frames.md の
M0->M1->M2 順)、VS 投入。座標系は +x前 / +y左 / +omega=CCW。
"""

from __future__ import annotations

import argparse
import time

from krilly.hal.l6470 import L6470Profile
from krilly.hal.l6470_chain import L6470Chain
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.logging_config import get_logger, setup_logging
from krilly.motion.velocity_driver import VelocityDriver

log = get_logger("krilly.drive_demo")


def main() -> None:
    p = argparse.ArgumentParser(description="協調ランプ速度指令のデモ")
    p.add_argument("--devices", type=int, default=3, help="連結台数")
    p.add_argument("--v", type=float, default=0.1, help="直進/横移動の速度 [m/s]")
    p.add_argument("--omega", type=float, default=1.0, help="旋回角速度 [rad/s]")
    p.add_argument("--hold", type=float, default=2.0, help="各区間の保持秒数")
    p.add_argument("--dt", type=float, default=0.02, help="制御周期 [s]")
    args = p.parse_args()

    setup_logging()
    kin = KiwiKinematics()
    period = args.dt

    def ramp_to(driver: VelocityDriver, vx: float, vy: float, omega: float, hold_s: float) -> None:
        """目標へランプ→hold_s 保持。update は純計算なのでここで dt スリープする。"""
        driver.set_velocity(vx, vy, omega)
        deadline = time.monotonic() + hold_s
        while time.monotonic() < deadline:
            driver.update(period)
            time.sleep(period)

    with L6470Chain(num_devices=args.devices) as chain:
        statuses = chain.configure_all(L6470Profile())
        log.info("configure_all STATUS = %s", [f"0x{s:04X}" for s in statuses])
        driver = VelocityDriver(chain, kin)

        log.info("直進 (+x) vx=%.2f", args.v)
        ramp_to(driver, args.v, 0.0, 0.0, args.hold)
        log.info("横移動 (+y 左) vy=%.2f", args.v)
        ramp_to(driver, 0.0, args.v, 0.0, args.hold)
        log.info("その場回転 (+omega 左) omega=%.2f", args.omega)
        ramp_to(driver, 0.0, 0.0, args.omega, args.hold)

        log.info("停止までランプダウン")
        driver.stop()
        ramp_to(driver, 0.0, 0.0, 0.0, 1.0)  # 0 目標のまま減速しきる
        # with 終了で hard_hiz により出力を解放


if __name__ == "__main__":
    main()
