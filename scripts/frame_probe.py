#!/usr/bin/env python3
"""止まったままカメラを撮り続け、1 フレームずつ記録する (issue #184、実機)。

走らせずに測れること — 1 フレームの処理時間、撮影から使えるまでの遅れ、
静止時の読みのばらつき (揺れの基準)、**機体を手で押して離したときの振れ** — を
測る。記録の形式は ``cell_move_demo --frame-log`` と同じなので、解析も
``scripts.frame_log_report`` でそのままできる。

例:
    python -m scripts.frame_probe --seconds 10 --ev -2 --out survey/m7/still
    python -m scripts.frame_probe --seconds 10 --ev -2 --out survey/m7/still_wall --no-yaw --no-tilt
                                         # 壁の帯だけの処理時間 (各段の内訳を見る)
    python -m scripts.frame_probe --seconds 15 --ev -2 --energize --out survey/m7/push
                                         # 励磁したまま (車輪が回らない状態で) 機体を押して離す
    python -m scripts.frame_log_report survey/m7/still --plot still.png
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time

from krilly.hal.camera import add_camera_args, camera_kwargs
from krilly.logging_config import get_logger, setup_logging
from krilly.motion.tuning import add_tuning_args
from krilly.perception.axis_yaw import calibrated_axis_yaw_config
from krilly.perception.frame_log import (
    CAMERA_HEIGHT_MM,
    FrameLogger,
    ProbeConfig,
    focal_px,
)
from krilly.perception.wall_detect import WallDetector, calibrated_config

log = get_logger("krilly.frame_probe")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="静止でカメラを撮り続けて記録する (#184)")
    p.add_argument("--out", required=True, metavar="PREFIX",
                   help="PREFIX_frames.csv に書く (追記。走行ごとに PREFIX を変えること)")
    p.add_argument("--seconds", type=float, default=10.0, help="記録する秒数")
    p.add_argument("--no-yaw", action="store_true", help="axis_yaw を測らない (処理時間の内訳用)")
    p.add_argument("--no-tilt", action="store_true", help="傾きを測らない (処理時間の内訳用)")
    p.add_argument("--save-every", type=int, default=0, metavar="N",
                   help="N 枚に 1 枚を PREFIX_frames/ に JPEG で残す (処理時間を食う)")
    p.add_argument("--camera-height", type=float, default=CAMERA_HEIGHT_MM, metavar="mm",
                   help="カメラの床からの高さ (傾きの角度の換算に使う)")
    p.add_argument("--energize", action="store_true",
                   help="L6470 を励磁して車輪を止めたまま記録する (押して離す試験用)")
    p.add_argument("--devices", type=int, default=3, help="連結台数 (--energize)")
    p.add_argument("--bus", type=int, default=0, help="SPI バス (--energize)")
    p.add_argument("--device", type=int, default=0, help="SPI デバイス/CE (--energize)")
    add_tuning_args(p)
    add_camera_args(p)
    args = p.parse_args(argv)
    setup_logging()

    with contextlib.ExitStack() as stack:
        if args.energize:
            from krilly.hal.l6470_chain import L6470Chain
            from krilly.kinematics.kiwi import KiwiKinematics
            from krilly.motion.emergency_stop import emergency_stop
            from krilly.motion.tuning import build_tuning
            from krilly.motion.velocity_driver import VelocityDriver

            chain = stack.enter_context(L6470Chain(num_devices=args.devices, bus=args.bus,
                                                   device=args.device))
            stack.enter_context(emergency_stop(chain))
            chain.configure_all(build_tuning(args).profile)   # --kval は走行と揃える
            VelocityDriver(chain, KiwiKinematics()).energize()
            log.info("励磁した (車輪は回らない)。Ctrl-C で止めて解放する")

        from krilly.hal.camera import Camera   # 遅延 import (実機専用の依存)

        camera = stack.enter_context(Camera(**camera_kwargs(args)))
        probe = ProbeConfig(focal_px=focal_px(args.camera_height),
                            yaw=calibrated_axis_yaw_config(),
                            do_yaw=not args.no_yaw, do_tilt=not args.no_tilt)
        logger = FrameLogger(camera, WallDetector(calibrated_config()), probe,
                             f"{args.out}_frames.csv", save_every=args.save_every,
                             save_dir=f"{args.out}_frames")
        stack.enter_context(logger)
        log.info("%.0f 秒記録する: %s_frames.csv", args.seconds, args.out)
        try:
            time.sleep(args.seconds)
        except KeyboardInterrupt:
            log.info("中断")
        logger.stop()
        log.info("記録したフレーム %d 枚 (失敗 %d)。解析: python -m scripts.frame_log_report %s",
                 logger.frames, logger.errors, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
