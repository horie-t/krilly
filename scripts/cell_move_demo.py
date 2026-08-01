#!/usr/bin/env python3
"""1セル前進・90°ターン (位置連動) の実機確認スクリプト (issue #17).

:class:`CellMotion` で「1 セル前進」「90°ターン」を **自己位置推定の残量で
終端判定しながら** 実行する。各プリミティブごとに、理想格子上の基準姿勢と
推定姿勢・残差を表示するので、実機の停止位置とセル中心のズレを見比べて
ゲイン (CellMotionConfig) を詰められる。

``--camera-yaw`` を付けると、動作の**前後でカメラのフレームから軸角を実測**し
(:mod:`krilly.perception.axis_yaw`)、ジャイロ由来の推定方位と突き合わせる。
シーケンスの総回転が 90° の倍数なら、カメラ側の差分がそのまま
**理想からの行き過ぎ量 (+ = CCW 側)** になるので、方位の真値として使える。

例:
    python -m scripts.cell_move_demo                       # 既定 F,L,F,R (L字)
    python -m scripts.cell_move_demo --seq F,F,L,F         # 任意のシーケンス
    python -m scripts.cell_move_demo --seq U --no-imu      # 180°ターンをオドメトリのみで
    python -m scripts.cell_move_demo --v 0.08 --omega 1.0  # ゆっくり
    python -m scripts.cell_move_demo --seq L --camera-yaw   # 90°ターンの方位をカメラで実測

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
from krilly.perception.axis_yaw import (
    AxisYaw,
    calibrated_axis_yaw_config,
    fold_deg,
    median_axis_yaw,
    yaw_delta_rad,
)

log = get_logger("krilly.cell_move_demo")

TOKENS = {
    "F": ("1セル前進", lambda m: m.start_forward_cells(1)),
    "B": ("1セル後退", lambda m: m.start_forward_cells(-1)),
    "L": ("左90°", lambda m: m.start_turn_left()),
    "R": ("右90°", lambda m: m.start_turn_right()),
    "U": ("180°", lambda m: m.start_turn_left(2)),
}


def wrapped_deg(rad: float) -> float:
    """表示用: 角度を (-180, 180] deg に正規化する。

    推定φは積算値なので 360° を超えて伸びる (2周すれば +720°)。基準φは
    (-180, 180] に正規化してあるため、並べて読むには表示側も揃える必要がある。
    """
    return math.degrees((rad + math.pi) % (2 * math.pi) - math.pi)


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
    p.add_argument("--gyro-scale", type=float, default=None,
                   help="ジャイロzスケール補正 (既定 robot.yaml の gyro_scale_z)")
    p.add_argument("--timeout", type=float, default=10.0, help="1プリミティブの上限秒数")
    p.add_argument("--camera-yaw", action="store_true",
                   help="動作前後の方位をカメラで実測してジャイロ推定と突き合わせる")
    p.add_argument("--yaw-samples", type=int, default=5, help="カメラ実測のフレーム数 (中央値)")
    p.add_argument("--save-frames", default=None,
                   help="カメラ実測フレームの保存先プレフィクス 例: /tmp/yaw")
    args = p.parse_args()

    setup_logging()
    seq = parse_seq(args.seq)
    kin = KiwiKinematics()
    cfg = CellMotionConfig(v_max=args.v, omega_max=args.omega)
    gyro_scale = args.gyro_scale if args.gyro_scale is not None else kin.cfg.gyro_scale_z

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
            log.info("ジャイロバイアス z=%.3f deg/s / スケール %.4f", bias_z, gyro_scale)

        camera = None
        if args.camera_yaw:
            from krilly.hal.camera import Camera   # 遅延 import (実機専用の依存)

            camera = stack.enter_context(Camera())
        yaw_cfg = calibrated_axis_yaw_config()

        driver = VelocityDriver(chain, kin)
        est = DeadReckoning(kin)
        motion = CellMotion(driver, est, config=cfg)

        def gyro_rate() -> float | None:
            """バイアス減算・符号・スケール補正を掛けた角速度 [rad/s]。"""
            if imu is None:
                return None
            return math.radians(imu.gyro[2] - bias_z) * args.gyro_sign * gyro_scale

        def measure_yaw(tag: str) -> AxisYaw | None:
            """カメラで軸角を実測する (複数フレームの中央値)。"""
            if camera is None:
                return None
            frames = [camera.capture() for _ in range(args.yaw_samples)]
            result = median_axis_yaw(frames, yaw_cfg)
            if args.save_frames:
                import cv2

                from krilly.perception.axis_yaw import annotate

                cv2.imwrite(f"{args.save_frames}_{tag}.png", frames[-1])
                cv2.imwrite(f"{args.save_frames}_{tag}_yaw.png", annotate(frames[-1], yaw_cfg))
            if result is None:
                log.warning("カメラ実測 (%s): 赤い壁エッジが足りず測定不能", tag)
            else:
                log.info("カメラ実測 (%s): 軸角 %+.3f° (線分 %d 本, 総長 %.0fpx)",
                         tag, result.angle_deg, result.segments, result.total_length_px)
            return result

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
                label, time.monotonic() - t0, x, y, wrapped_deg(phi),
                rx, ry, wrapped_deg(rphi), along, cross, math.degrees(dphi),
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
        yaw_before = measure_yaw("before")
        phi_before = est.phi
        for i, token in enumerate(seq, 1):
            label, start = TOKENS[token]
            log.info("[%d/%d] %s 開始", i, len(seq), label)
            start(motion)
            run_primitive(f"[{i}/{len(seq)}] {label}")
            coast(args.pause)
        yaw_after = measure_yaw("after")

        along, cross, dphi = motion.residual()
        log.info("完了。理想との総残差: 前後=%+.4fm 左右=%+.4fm 方位=%+.2f°",
                 along, cross, math.degrees(dphi))
        if yaw_before is not None and yaw_after is not None:
            # 軸は 90° 周期なので、ジャイロ側の総回転も同じ折り返しで比べる
            gyro_delta = fold_deg(math.degrees(est.phi - phi_before))
            cam_delta = math.degrees(yaw_delta_rad(yaw_before, yaw_after))
            log.info(
                "方位の実測 (90°の倍数を除いた差分): カメラ %+.3f° / ジャイロ推定 %+.3f°"
                " / 差 %+.3f°  (+ = CCW 側へ行き過ぎ)",
                cam_delta, gyro_delta, cam_delta - gyro_delta,
            )
        # with 終了で hard_hiz により出力を解放


if __name__ == "__main__":
    main()
