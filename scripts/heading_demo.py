#!/usr/bin/env python3
"""ジャイロ融合の動作確認デモ (issue #13).

その場回転を走らせ、「車輪オドメトリのみ」の姿勢φと「ジャイロ融合」の姿勢φを
並べて表示する。BNO055 の融合 heading の変化量も参考表示する。回転はスリップの
影響が出やすいので、ジャイロ融合φの方が実際の回転に近くなることを確認する。

例:
    python -m scripts.heading_demo --omega 1.0 --duration 4
    python -m scripts.heading_demo --gyro-sign -1   # 回転方向とジャイロ符号が逆なら

配線: L6470×3 デイジーチェーン + BNO055 (I2C 0x28)。座標系 +omega=CCW。
"""

from __future__ import annotations

import argparse
import math
import time

from krilly.hal.imu import Bno055Imu
from krilly.hal.l6470 import L6470Profile
from krilly.hal.l6470_chain import L6470Chain
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.logging_config import get_logger, setup_logging
from krilly.motion.velocity_driver import VelocityDriver

log = get_logger("krilly.heading_demo")


def main() -> None:
    p = argparse.ArgumentParser(description="ジャイロ融合デモ")
    p.add_argument("--devices", type=int, default=3)
    p.add_argument("--omega", type=float, default=1.0, help="旋回角速度 [rad/s]")
    p.add_argument("--duration", type=float, default=4.0, help="旋回秒数")
    p.add_argument("--dt", type=float, default=0.02, help="制御周期 [s]")
    p.add_argument("--gyro-sign", type=float, default=1.0, help="ジャイロz符号 (+1/-1)")
    args = p.parse_args()

    setup_logging()
    kin = KiwiKinematics()

    with L6470Chain(num_devices=args.devices) as chain, Bno055Imu() as imu:
        statuses = chain.configure_all(L6470Profile())
        if any(s in (0x0000, 0xFFFF) for s in statuses):
            log.error("SPI 応答異常。配線/電源を確認。中止。")
            return
        imu.begin()
        log.info("静止のままジャイロバイアス計測中…")
        bias = imu.measure_gyro_bias()
        log.info("ジャイロバイアス z=%.3f deg/s", bias[2])

        driver = VelocityDriver(chain, kin)
        est_odom = DeadReckoning(kin)   # 車輪オドメトリのみ
        est_gyro = DeadReckoning(kin)   # ジャイロ融合
        heading0 = imu.heading_deg

        driver.set_velocity(0.0, 0.0, args.omega)
        last = time.monotonic()
        deadline = last + args.duration
        while time.monotonic() < deadline:
            time.sleep(args.dt)
            now = time.monotonic()
            dt = now - last
            last = now
            cur = driver.update(dt)
            wheel_mps = kin.body_to_wheels(*cur)
            gz = math.radians(imu.gyro[2] - bias[2]) * args.gyro_sign  # rad/s
            est_odom.update_wheel_speeds(wheel_mps, dt)
            est_gyro.update_with_gyro_rate(wheel_mps, gz, dt)

        driver.stop()
        for _ in range(int(1.0 / args.dt)):
            time.sleep(args.dt)
            now = time.monotonic(); dt = now - last; last = now
            cur = driver.update(dt)
            wheel_mps = kin.body_to_wheels(*cur)
            gz = math.radians(imu.gyro[2] - bias[2]) * args.gyro_sign
            est_odom.update_wheel_speeds(wheel_mps, dt)
            est_gyro.update_with_gyro_rate(wheel_mps, gz, dt)

        heading_delta = imu.heading_deg - heading0
        log.info("オドメトリのみ φ = %+.1f°", math.degrees(est_odom.pose[2]))
        log.info("ジャイロ融合   φ = %+.1f°", math.degrees(est_gyro.pose[2]))
        log.info("BNO055 heading 変化(参考) = %+.1f°", heading_delta)


if __name__ == "__main__":
    main()
