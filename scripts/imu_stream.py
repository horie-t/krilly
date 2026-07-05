#!/usr/bin/env python3
"""BNO055 (I2C) の読み出し確認スクリプト (issue #6).

方位・ジャイロ・キャリブレーション状態をライブ表示する。

例:
    # I2C バス1・アドレス0x28 から方位/ジャイロを連続表示
    python -m scripts.imu_stream

    # 起動時に静止させてジャイロバイアスを計測してから表示
    python -m scripts.imu_stream --calibrate-gyro

配線 (AE-BNO055-BO, 出荷時 I2C/0x28, ジャンパ変更不要):
    1 VIN -> Pi 3.3V (pin1)      2/7 GND -> GND
    3 SDA -> Pi SDA (GPIO2/pin3) 4 SCL  -> Pi SCL (GPIO3/pin5)
事前に raspi-config で I2C を有効化。クロックストレッチ対策として
/boot/firmware/config.txt に dtparam=i2c_arm_baudrate=10000 を推奨。
"""

from __future__ import annotations

import argparse
import time

from krilly.hal.imu import DEFAULT_ADDRESS, Bno055Imu
from krilly.logging_config import get_logger, setup_logging

log = get_logger("krilly.imu_stream")


def main() -> None:
    p = argparse.ArgumentParser(description="BNO055 I2C 読み出し確認")
    p.add_argument("--address", type=lambda s: int(s, 0), default=DEFAULT_ADDRESS,
                   help="I2C アドレス (既定 0x28)")
    p.add_argument("--bus-id", type=int, default=1, help="I2C バス番号 (既定 1)")
    p.add_argument("--hz", type=float, default=10.0, help="表示レート")
    p.add_argument("--duration", type=float, default=None, help="秒数 (既定: 無限)")
    p.add_argument("--external-crystal", action="store_true",
                   help="外部32.768kHz水晶を使用 (AE-BNO055-BO は搭載)")
    p.add_argument("--calibrate-gyro", action="store_true",
                   help="起動時に静止させてジャイロバイアスを計測")
    args = p.parse_args()

    setup_logging()
    period = 1.0 / args.hz if args.hz > 0 else 0.0

    with Bno055Imu(address=args.address, bus_id=args.bus_id) as imu:
        imu.begin(use_external_crystal=args.external_crystal)
        log.info("BNO055 初期化完了 (NDOF, addr=0x%02X)", args.address)

        if args.calibrate_gyro:
            log.info("静止させてください… ジャイロバイアス計測中")
            bias = imu.measure_gyro_bias()
            log.info("ジャイロバイアス [deg/s] = (%.3f, %.3f, %.3f)", *bias)

        start = time.monotonic()
        while args.duration is None or (time.monotonic() - start) < args.duration:
            heading, roll, pitch = imu.euler
            gx, gy, gz = imu.gyro
            sys_, gyr, acc, mag = imu.calibration_status
            log.info(
                "heading=%6.1f roll=%6.1f pitch=%6.1f | gyro=(%7.2f,%7.2f,%7.2f) "
                "| calib sys=%d gyr=%d acc=%d mag=%d",
                heading, roll, pitch, gx, gy, gz, sys_, gyr, acc, mag,
            )
            time.sleep(period)


if __name__ == "__main__":
    main()
