#!/usr/bin/env python3
"""1セル前進・90°ターン (位置連動) の実機確認スクリプト (issue #17).

:class:`CellMotion` で「1 セル前進」「90°ターン」を **自己位置推定の残量で
終端判定しながら** 実行する。各プリミティブごとに、理想格子上の基準姿勢と
推定姿勢・残差を表示するので、実機の停止位置とセル中心のズレを見比べて
ゲイン (CellMotionConfig) を詰められる。

例:
    python -m scripts.cell_move_demo                       # 既定 F,L,F,R (L字→復帰)
    python -m scripts.cell_move_demo --seq F,F,L,F         # 任意のシーケンス
    python -m scripts.cell_move_demo --seq U --no-imu      # 180°ターンをオドメトリのみで
    python -m scripts.cell_move_demo --v 0.08 --omega 1.0  # ゆっくり

シーケンスのトークン: F=1セル前進 / B=1セル後退 / L=左90° / R=右90° / U=180°
配線: L6470×3 デイジーチェーン + BNO055 (I2C 0x28)。座標系 +x前 / +y左 / +omega=CCW。
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time

from krilly.hal.imu import Bno055Imu
from krilly.hal.l6470 import L6470Profile
from krilly.hal.l6470_chain import L6470Chain
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.logging_config import get_logger, setup_logging
from krilly.motion.cell_motion import CellMotion, CellMotionConfig
from krilly.motion.velocity_driver import VelocityDriver

log = get_logger("krilly.cell_move_demo")

TOKENS = {
    "F": ("1セル前進", lambda m: m.start_forward_cells(1)),
    "B": ("1セル後退", lambda m: m.start_forward_cells(-1)),
    "L": ("左90°", lambda m: m.start_turn_left()),
    "R": ("右90°", lambda m: m.start_turn_right()),
    "U": ("180°", lambda m: m.start_turn_left(2)),
}


def parse_seq(text: str) -> list[str]:
    """"F,L,F" / "FLF" いずれの書き方も受け付けてトークン列にする。"""
    tokens = [t.strip().upper() for t in text.replace(",", "") if not t.isspace()]
    bad = [t for t in tokens if t not in TOKENS]
    if bad:
        raise SystemExit(f"未知のトークン {bad} (使えるのは {'/'.join(TOKENS)})")
    return tokens


def main() -> None:
    p = argparse.ArgumentParser(description="1セル前進・90°ターンの位置連動デモ")
    p.add_argument("--devices", type=int, default=3, help="連結台数")
    p.add_argument("--bus", type=int, default=0, help="SPI バス (既定 0)")
    p.add_argument("--device", type=int, default=0, help="SPI デバイス/CE (既定 0)")
    p.add_argument("--seq", default="F,L,F,R", help="動作シーケンス (F/B/L/R/U)")
    p.add_argument("--v", type=float, default=0.12, help="前進の最大速度 [m/s]")
    p.add_argument("--omega", type=float, default=1.5, help="旋回の最大角速度 [rad/s]")
    p.add_argument("--dt", type=float, default=0.02, help="制御周期 [s]")
    p.add_argument("--pause", type=float, default=0.5, help="プリミティブ間の停止秒数")
    p.add_argument("--no-imu", action="store_true", help="ジャイロ融合せずオドメトリのみ")
    p.add_argument("--gyro-sign", type=float, default=1.0, help="ジャイロz符号 (+1/-1)")
    p.add_argument("--timeout", type=float, default=10.0, help="1プリミティブの上限秒数")
    args = p.parse_args()

    setup_logging()
    seq = parse_seq(args.seq)
    kin = KiwiKinematics()
    cfg = CellMotionConfig(v_max=args.v, omega_max=args.omega)

    with contextlib.ExitStack() as stack:
        chain = stack.enter_context(
            L6470Chain(num_devices=args.devices, bus=args.bus, device=args.device)
        )
        statuses = chain.configure_all(L6470Profile())
        if any(s in (0x0000, 0xFFFF) for s in statuses):
            log.error("SPI 応答異常 (STATUS=%s)。配線/電源を確認。中止。",
                      [f"0x{s:04X}" for s in statuses])
            return

        imu = None
        bias_z = 0.0
        if not args.no_imu:
            imu = stack.enter_context(Bno055Imu())
            imu.begin()
            log.info("静止のままジャイロバイアス計測中…")
            bias_z = imu.measure_gyro_bias()[2]
            log.info("ジャイロバイアス z=%.3f deg/s", bias_z)

        driver = VelocityDriver(chain, kin)
        est = DeadReckoning(kin)
        motion = CellMotion(driver, est, config=cfg)

        def gyro_rate() -> float | None:
            if imu is None:
                return None
            return math.radians(imu.gyro[2] - bias_z) * args.gyro_sign

        def run_primitive(label: str) -> None:
            """完了 (または timeout) までループを回す。update は純計算なので寝るのはここ。"""
            t0 = time.monotonic()
            last = t0
            while True:
                time.sleep(args.dt)
                now = time.monotonic()
                dt = now - last
                last = now
                if motion.update(dt, gyro_rate=gyro_rate()):
                    break
                if now - t0 > args.timeout:
                    log.warning("%s: %.1fs で終わらず打ち切り (残量 %.4f)",
                                label, args.timeout, motion.remaining)
                    motion.abort()
                    break
            x, y, phi = est.pose
            rx, ry, rphi = motion.reference
            along, cross, dphi = motion.residual()
            log.info(
                "%s 完了 %.2fs: 推定 X=%.4f Y=%.4f φ=%+.2f° / 基準 X=%.4f Y=%.4f φ=%+.2f°"
                " / 残差 前後=%+.4fm 左右=%+.4fm 方位=%+.2f°",
                label, time.monotonic() - t0, x, y, math.degrees(phi),
                rx, ry, math.degrees(rphi), along, cross, math.degrees(dphi),
            )

        def coast(seconds: float) -> None:
            """停止指令のまま update を回して減速しきる (推定も継続)。"""
            last = time.monotonic()
            deadline = last + seconds
            while time.monotonic() < deadline:
                time.sleep(args.dt)
                now = time.monotonic(); dt = now - last; last = now
                motion.update(dt, gyro_rate=gyro_rate())

        log.info("シーケンス %s を実行 (1セル=%.3fm)", "".join(seq), motion.cell_pitch_m)
        for i, token in enumerate(seq, 1):
            label, start = TOKENS[token]
            log.info("[%d/%d] %s 開始", i, len(seq), label)
            start(motion)
            run_primitive(f"[{i}/{len(seq)}] {label}")
            coast(args.pause)

        along, cross, dphi = motion.residual()
        log.info("完了。理想との総残差: 前後=%+.4fm 左右=%+.4fm 方位=%+.2f°",
                 along, cross, math.degrees(dphi))
        # with 終了で hard_hiz により出力を解放


if __name__ == "__main__":
    main()
