#!/usr/bin/env python3
"""デッドレコニングの動作確認デモ (issue #12).

VelocityDriver で L字経路 (前進 → 左90°旋回 → 前進) を走らせ、指令した各輪速度から
DeadReckoning で推定姿勢 [X, Y, φ] を積算して表示する (オープンループ)。
走行後に実機の位置・向きと推定値を見比べて、デッドレコニングの妥当性を確認する。

例:
    python -m scripts.odometry_demo --v 0.1 --seg 1.0

座標系は docs/coordinate-frames.md (+x前 / +y左 / +omega=CCW)。
"""

from __future__ import annotations

import argparse
import math
import time

from krilly.hal.l6470 import L6470Profile
from krilly.hal.l6470_chain import L6470Chain
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.motion.velocity_driver import VelocityDriver
from krilly.logging_config import get_logger, setup_logging

log = get_logger("krilly.odometry_demo")


def main() -> None:
    p = argparse.ArgumentParser(description="デッドレコニング デモ")
    p.add_argument("--devices", type=int, default=3, help="連結台数")
    p.add_argument("--v", type=float, default=0.1, help="直進速度 [m/s]")
    p.add_argument("--omega", type=float, default=1.0, help="旋回角速度 [rad/s]")
    p.add_argument("--seg", type=float, default=0.5, help="各直進区間の目標距離 [m]")
    p.add_argument("--dt", type=float, default=0.02, help="制御周期 [s]")
    args = p.parse_args()

    setup_logging()
    kin = KiwiKinematics()

    with L6470Chain(num_devices=args.devices) as chain:
        statuses = chain.configure_all(L6470Profile())
        if any(s in (0x0000, 0xFFFF) for s in statuses):
            log.error("SPI 応答異常。配線/電源を確認。中止。")
            return
        driver = VelocityDriver(chain, kin)
        est = DeadReckoning(kin)

        def run_for(vx: float, vy: float, omega: float, duration: float) -> None:
            driver.set_velocity(vx, vy, omega)
            last = time.monotonic()
            deadline = last + duration
            while time.monotonic() < deadline:
                time.sleep(args.dt)
                now = time.monotonic()
                dt = now - last
                last = now
                cur = driver.update(dt)
                est.update_wheel_speeds(kin.body_to_wheels(*cur), dt)

        def ramp_down() -> None:
            driver.stop()
            last = time.monotonic()
            deadline = last + 1.0
            while time.monotonic() < deadline:
                time.sleep(args.dt)
                now = time.monotonic(); dt = now - last; last = now
                cur = driver.update(dt)
                est.update_wheel_speeds(kin.body_to_wheels(*cur), dt)

        def report(label: str) -> None:
            x, y, phi = est.pose
            log.info("%s: 推定 X=%.3f Y=%.3f φ=%.1f°", label, x, y, math.degrees(phi))

        seg_time = args.seg / args.v
        turn_time = (math.pi / 2) / args.omega  # 90°旋回

        log.info("L字経路を走行 (前進→左90°→前進)")
        run_for(args.v, 0.0, 0.0, seg_time)
        report("前進1後")
        run_for(0.0, 0.0, args.omega, turn_time)
        report("左90°後")
        run_for(args.v, 0.0, 0.0, seg_time)
        report("前進2後")
        ramp_down()
        report("停止後")
        log.info("期待値(理想): X≈%.2f Y≈%.2f φ≈90°", args.seg, args.seg)


if __name__ == "__main__":
    main()
