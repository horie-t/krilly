#!/usr/bin/env python3
"""BNO055 (UART) の読み出し確認スクリプト (issue #6).

方位・ジャイロ・キャリブレーション状態をライブ表示する。

例:
    # /dev/serial0 から方位/ジャイロを連続表示
    python -m scripts.imu_stream

    # 起動時に静止させてジャイロバイアスを計測してから表示
    python -m scripts.imu_stream --calibrate-gyro

配線: BNO055 PS1=High(UART選択), TX/RX を Pi の /dev/serial0 に接続、GND 共通。
raspi-config でシリアルログインを無効・シリアルHWを有効にしておくこと。

キャリブレーション: NDOF 融合はセンサ側で自走校正する。calibration_status が
(sys, gyro, accel, mag) それぞれ 3 に近づくほど良好。mag は 8 の字を描くと上がる。
"""

from __future__ import annotations

import argparse
import time

from krilly.hal.imu import Bno055Imu
from krilly.logging_config import get_logger, setup_logging

log = get_logger("krilly.imu_stream")


def main() -> None:
    p = argparse.ArgumentParser(description="BNO055 UART 読み出し確認")
    p.add_argument("--port", default="/dev/serial0", help="シリアルポート")
    p.add_argument("--baud", type=int, default=115200, help="ボーレート")
    p.add_argument("--hz", type=float, default=10.0, help="表示レート")
    p.add_argument("--duration", type=float, default=None, help="秒数 (既定: 無限)")
    p.add_argument("--external-crystal", action="store_true",
                   help="外部32.768kHz水晶を使用 (基板に水晶がある場合のみ)")
    p.add_argument("--calibrate-gyro", action="store_true",
                   help="起動時に静止させてジャイロバイアスを計測")
    args = p.parse_args()

    setup_logging()
    period = 1.0 / args.hz if args.hz > 0 else 0.0

    with Bno055Imu(port=args.port, baudrate=args.baud) as imu:
        imu.begin(use_external_crystal=args.external_crystal)
        log.info("BNO055 初期化完了 (NDOF)")

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
