#!/usr/bin/env python3
"""flood-fill 探索ランの実機スクリプト (issue #18).

スタートセルからゴールまで、**1 セル進むごとにカメラで壁を判定 → flood-fill を
塗り直す**探索ランを実行する。未探索セルは「壁なし=通れる」と楽観視されるので、
ゴールへ向かう過程で必要なセルだけを自然に開拓する。

1 ステップ:
    カメラ1枚 -> WallDetector で機体相対の壁 -> Explorer.observe で迷路へ反映
    -> flood-fill -> 次の方角 -> CellMotion で 90°ターン + 1セル前進

例:
    python -m scripts.search_run --size 3            # 3x3 の練習迷路 (ゴール=中央)
    python -m scripts.search_run                     # 16x16 (config/maze.yaml)
    python -m scripts.search_run --size 3 --verbose   # 毎ステップ ASCII 迷路を表示
    python -m scripts.search_run --dry-run --size 3   # 走らせず壁判定だけ確認

前提: **スタートセルの中心に、迷路の北を向いて置く** (迷路グリッドは 東=+x/北=+y、
スタート (0,0)、ゴールは中央 2x2 / 奇数サイズなら中央 1 セル)。
配線: L6470×3 デイジーチェーン + BNO055 (I2C 0x28) + Pi カメラ。
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time

from krilly.config import load_maze_config
from krilly.hal.imu import Bno055Imu
from krilly.hal.l6470_chain import L6470Chain
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.logging_config import get_logger, setup_logging
from krilly.motion.cell_motion import CellMotion
from krilly.localization.grid import apply_axis_heading, apply_cell_offset
from krilly.motion.emergency_stop import emergency_stop
from krilly.motion.tuning import add_tuning_args, build_tuning, check_limits, describe_faults
from krilly.motion.velocity_driver import VelocityDriver
from krilly.perception.axis_yaw import axis_yaw, calibrated_axis_yaw_config
from krilly.perception.cell_pose import cell_offset
from krilly.perception.wall_detect import (
    BODY_DIRS,
    FRONT,
    WallDetector,
    calibrated_config,
)
from krilly.solver.maze import Direction, Maze
from krilly.strategy.explorer import (
    Explorer,
    Step,
    Unreachable,
    cell_center,
    heading_rad,
)
from krilly.strategy.shortest_path import (
    DEFAULT_TURN_COST,
    describe_legs,
    path_cost,
    path_to_legs,
    shortest_path,
)

log = get_logger("krilly.search_run")

TURN_LABEL = {0: "直進", 1: "左90°", -1: "右90°", 2: "180°"}


def turn_label(turn: int) -> str:
    return TURN_LABEL.get(turn, f"{turn * 90}°")


def main() -> None:
    p = argparse.ArgumentParser(description="flood-fill 探索ラン")
    p.add_argument("--devices", type=int, default=3, help="連結台数")
    p.add_argument("--bus", type=int, default=0, help="SPI バス (既定 0)")
    p.add_argument("--device", type=int, default=0, help="SPI デバイス/CE (既定 0)")
    p.add_argument("--size", type=int, default=None,
                   help="迷路サイズ (既定 maze.yaml の grid_size=16)")
    add_tuning_args(p, omega=1.5)
    p.add_argument("--dt", type=float, default=0.02, help="制御周期 [s]")
    p.add_argument("--pause", type=float, default=0.4, help="動作の前後で止まる秒数")
    p.add_argument("--max-steps", type=int, default=200, help="打ち切りステップ数")
    p.add_argument("--timeout", type=float, default=10.0, help="1動作の上限秒数")
    p.add_argument("--no-imu", action="store_true", help="ジャイロ融合せずオドメトリのみ")
    p.add_argument("--gyro-sign", type=float, default=1.0, help="ジャイロz符号 (+1/-1)")
    p.add_argument("--gyro-scale", type=float, default=None,
                   help="ジャイロzスケール補正 (既定 robot.yaml の gyro_scale_z)")
    p.add_argument("--dry-run", action="store_true",
                   help="走行せず、その場の壁判定と次の1手だけを表示する")
    p.add_argument("--verbose", action="store_true", help="毎ステップ ASCII 迷路を表示")
    p.add_argument("--turn-cost", type=float, default=DEFAULT_TURN_COST,
                   help="最速経路の旋回コスト [セル換算] (既定 1.0 = 実測の 90°ターン≒1セル)")
    p.add_argument("--no-correct", action="store_true",
                   help="カメラによるセル内位置の絶対補正を無効にする (#54)")
    p.add_argument("--correct-weight", type=float, default=1.0,
                   help="位置補正の引き込み率 0..1 (既定 1.0 = 測定値をそのまま採用)")
    p.add_argument("--no-front-check", action="store_true",
                   help="前進直前の前方壁チェックを無効にする (既定は有効)")
    p.add_argument("--save-frames", default=None, help="判定フレームの保存先プレフィクス")
    args = p.parse_args()

    setup_logging()
    kin = KiwiKinematics()
    tuning = build_tuning(args)
    maze_cfg = load_maze_config()
    maze = Maze(args.size) if args.size else Maze.from_config(maze_cfg)
    maze.set_outer_walls()
    explorer = Explorer(maze)
    detector = WallDetector(calibrated_config())
    yaw_cfg = calibrated_axis_yaw_config()
    gyro_scale = args.gyro_scale if args.gyro_scale is not None else kin.cfg.gyro_scale_z

    log.info("迷路 %dx%d / ゴール %s / スタート %s 北向き",
             maze.size, maze.size, maze.goal_cells(), maze.start)
    log.info("チューニング: %s", tuning.describe())
    for warning in check_limits(tuning, kin):
        log.warning("%s", warning)

    with contextlib.ExitStack() as stack:
        from krilly.hal.camera import Camera   # 遅延 import (実機専用の依存)

        camera = stack.enter_context(Camera())
        chain = imu = None
        bias_z = 0.0
        if not args.dry_run:
            chain = stack.enter_context(
                L6470Chain(num_devices=args.devices, bus=args.bus, device=args.device)
            )
            stack.enter_context(
                emergency_stop(chain, on_stop=lambda sig: log.warning(
                    "シグナル %s を受信。モーターを解放した。", sig))
            )
            statuses = chain.configure_all(tuning.profile)
            if any(s in (0x0000, 0xFFFF) for s in statuses):
                log.error("SPI 応答異常 (STATUS=%s)。配線/電源を確認。中止。",
                          [f"0x{s:04X}" for s in statuses])
                return
            log.info("停止は Ctrl-C (ESC は効かない)。"
                     "強制終了して回り続けたら python -m scripts.motor_stop")
            if not args.no_imu:
                imu = stack.enter_context(Bno055Imu())
                imu.begin()
                log.info("静止のままジャイロバイアス計測中…")
                bias_z = imu.measure_gyro_bias()[2]
                log.info("ジャイロバイアス z=%.3f deg/s / スケール %.4f", bias_z, gyro_scale)

        # スタートセル中心・北向きを初期姿勢にする (基準姿勢も同じ = 理想格子の原点)
        est = DeadReckoning(kin, x=0.0, y=0.0, phi=heading_rad(Direction.N))
        motion = None
        if chain is not None:
            motion = CellMotion(
                VelocityDriver(chain, kin, limits=tuning.limits),
                est,
                config=tuning.motion,
                maze=maze_cfg,
            )

        if motion is not None:
            # 最初の駆動指令でロータが谷へスナップし車体が最大 0.5° 跳ねる。
            # カメラで壁・位置を測る前に済ませておく (VelocityDriver.energize 参照)。
            motion.driver.energize()
            time.sleep(0.3)

        def gyro_rate() -> float | None:
            if imu is None:
                return None
            return math.radians(imu.gyro[2] - bias_z) * args.gyro_sign * gyro_scale

        def run_primitive(label: str) -> None:
            """1 動作を完了 (または timeout) まで回す。update は純計算なので寝るのはここ。"""
            t0 = last = time.monotonic()
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
            faults = describe_faults(chain.get_status_all(), ignore=("UVLO",))
            if faults:
                log.warning("  %s: L6470 フォールト %s (トルク不足の疑い)", label, faults)
            along, cross, dphi = motion.residual()
            log.info("  %s 完了 %.2fs 残差 前後=%+.4fm 左右=%+.4fm 方位=%+.2f°",
                     label, time.monotonic() - t0, along, cross, math.degrees(dphi))

        def coast(seconds: float) -> None:
            """停止指令のまま update を回して減速しきる (推定も継続)。"""
            last = time.monotonic()
            deadline = last + seconds
            while time.monotonic() < deadline:
                time.sleep(args.dt)
                now = time.monotonic(); dt = now - last; last = now
                motion.update(dt, gyro_rate=gyro_rate())

        def observe_here() -> None:
            """カメラ1枚から壁を判定して迷路へ反映し、セル内位置も補正する (#54)。"""
            frame = camera.capture()
            measured = detector.measure(frame)
            walls_body = {d: measured[d][0] >= detector.cfg.threshold_for(d) for d in BODY_DIRS}
            walls_maze = explorer.observe(walls_body)
            if args.save_frames:
                import cv2

                cv2.imwrite(f"{args.save_frames}_{explorer.steps:03d}.png", frame)
            log.info("  壁 赤割合 %s -> 迷路 %s",
                     {d: f"{measured[d][0]:.2f}" for d in BODY_DIRS},
                     {d.name: p for d, p in sorted(walls_maze.items())})
            # 迷路軸に対する方位を測って絶対補正する。ジャイロの基準は「走行開始時の
            # 機体の向き」なので、放置すると走行のたびに迷路軸から傾いていく。
            if not args.no_correct:
                yaw = axis_yaw(frame, yaw_cfg)
                if yaw is not None:
                    before = est.phi
                    if apply_axis_heading(est, yaw.angle_rad, weight=args.correct_weight):
                        log.info("  方位補正: 軸から %+.2f° -> φ %+.2f°",
                                 yaw.angle_deg, math.degrees(est.phi - before))
                    else:
                        log.info("  方位補正: 軸から %+.2f° は大きすぎるので棄却",
                                 yaw.angle_deg)
            # 帯のずれからセル内の位置ずれを測り、推定位置を絶対補正する
            off = cell_offset(frame, detector)
            if off.saturated:
                # フレーム端で頭打ちになった辺は「中央寄り」の嘘を返す (#21)
                log.info("  位置補正: %s は帯がフレーム端で飽和したので不採用",
                         "/".join(off.saturated))
            if not off.measured:
                log.info("  位置補正: 壁が無く測定不能")
                return

            def mm(value):
                return "-" if value is None else f"{value * 1000:+.1f}mm"

            center = cell_center(explorer.cell, maze_cfg.cell_pitch_m)
            before = est.pose[:2]
            applied = (
                not args.no_correct
                and motion is not None
                and apply_cell_offset(est, center, off.forward_m, off.left_m,
                                      weight=args.correct_weight)
            )
            log.info("  位置ずれ 実測 前後=%s 左右=%s (壁 %d/%d 枚) -> %s",
                     mm(off.forward_m), mm(off.left_m), off.walls_y, off.walls_x,
                     f"補正 X{est.pose[0] - before[0]:+.4f} Y{est.pose[1] - before[1]:+.4f}"
                     if applied else "補正なし")

        def front_is_clear() -> bool:
            """前進の直前に、いま見えている前方の壁を確認する。

            計画は地図に基づくので、地図が正しければ前方は空いている。ここで壁が
            見えたら**自機の姿勢がずれている**サインなので、突っ込む前に止める。
            """
            if args.no_front_check:
                return True
            fraction = detector.red_fractions(camera.capture())[FRONT]
            if fraction >= detector.cfg.threshold:
                log.error("前進中止: 前方に壁が見える (赤割合 %.2f)。"
                          "姿勢がずれている可能性が高い。", fraction)
                return False
            return True

        def execute(step: Step) -> bool:
            """1 手を実行する。安全チェックで中止したら False。"""
            if step.turn:
                motion.start_turn_left(step.turn)
                run_primitive(turn_label(step.turn))
                coast(args.pause)
            if not front_is_clear():
                return False
            motion.start_forward_cells(1)
            run_primitive("1セル前進")
            coast(args.pause)
            return True

        # -- 探索ラン本体 ---------------------------------------------------
        for _ in range(args.max_steps):
            log.info("%s", explorer.progress())
            observe_here()
            if args.verbose:
                log.info("現在の迷路:\n%s", maze.to_ascii())
            try:
                step = explorer.plan()
            except Unreachable as e:
                log.error("探索中止: %s", e)
                break
            if step is None:
                log.info("ゴール到達! %d 手 / 訪問 %d セル",
                         explorer.steps, len(explorer.visited))
                break
            log.info("  次の1手: %s -> %s へ (%s)",
                     turn_label(step.turn), step.to_cell, step.direction.name)
            if args.dry_run:
                log.info("dry-run のため走行しない。終了。")
                break
            if not execute(step):
                log.error("安全チェックで中止した (セル %s 向き %s)。",
                          explorer.cell, explorer.facing.name)
                break
            explorer.advance(step)
            x, y = est.pose[0], est.pose[1]
            log.info("  推定 X=%.4f Y=%.4f φ=%+.1f° (セル %s の中心は %s)",
                     x, y, math.degrees(est.pose[2]), step.to_cell,
                     tuple(round(c * maze_cfg.cell_pitch_m, 4) for c in step.to_cell))
        else:
            log.warning("%d 手で終わらなかった (現在 %s)", args.max_steps, explorer.cell)

        log.info("判明した迷路:\n%s", maze.to_ascii())
        report_shortest_path(explorer, args.turn_cost)


def report_shortest_path(explorer: Explorer, turn_cost: float) -> None:
    """確定した壁情報から最速ランの最短経路を求めて表示する (#19)。

    通すのは**観測済みセルのみ** (未探索セルは壁が未確定なので最速では走れない)。
    """
    maze = explorer.maze
    path = shortest_path(
        maze,
        maze.start,
        start_facing=Direction.N,
        known=explorer.visited,
        turn_cost=turn_cost,
    )
    if not path:
        log.warning("最速経路なし: 観測済みセルだけではゴールへ繋がっていない "
                    "(訪問 %d セル)。探索を続ける必要がある。", len(explorer.visited))
        return
    legs = path_to_legs(path, Direction.N)
    turns = sum(abs(leg.turn) for leg in legs)
    log.info("最速経路: %d セル / 旋回 %d 回 / コスト %.1f (旋回コスト %.2f)",
             len(path) - 1, turns, path_cost(path, Direction.N, turn_cost), turn_cost)
    log.info("  %s", describe_legs(legs))
    log.info("  セル列: %s", " -> ".join(str(c) for c in path))


if __name__ == "__main__":
    main()
