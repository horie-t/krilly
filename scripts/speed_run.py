#!/usr/bin/env python3
"""競技の通しラン: 探索 → 復帰 → 最速 ×N (issue #20)。

:class:`krilly.app.run_manager.RunManager` の差配で、10 分・最大 5 走の予算内を
自動で回す。探索ランは :mod:`scripts.search_run` と同じ閉ループ (壁観測 +
flood-fill + 位置補正)。復帰と最速は確定した壁情報から :func:`shortest_path` の
Leg 列 (旋回 + 連続直進) を作り、**複数セルを 1 動作で**走る。

例:
    python -m scripts.speed_run --size 5 --omega 1.0
    python -m scripts.speed_run --size 5 --max-runs 2      # 探索 + 最速 1 本だけ
    python -m scripts.speed_run                            # 16x16 (config/maze.yaml)

前提: **スタートセルの中心に、迷路の北を向いて置く**。停止は Ctrl-C (ESC は効かない)。
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time

from krilly.app.run_manager import RunManager, RunPhase
from krilly.config import load_maze_config
from krilly.hal.imu import Bno055Imu
from krilly.hal.l6470 import L6470Profile
from krilly.hal.l6470_chain import L6470Chain
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.localization.grid import apply_axis_heading, apply_cell_offset
from krilly.logging_config import get_logger, setup_logging
from krilly.motion.cell_motion import CellMotion, CellMotionConfig
from krilly.motion.emergency_stop import emergency_stop
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
    Unreachable,
    cell_center,
    heading_rad,
    quarter_turns,
)
from krilly.strategy.shortest_path import Leg, describe_legs

log = get_logger("krilly.speed_run")

TURN_LABEL = {0: "直進", 1: "左90°", -1: "右90°", 2: "180°"}


def main() -> None:
    p = argparse.ArgumentParser(description="競技の通しラン (探索 -> 復帰 -> 最速 xN)")
    p.add_argument("--devices", type=int, default=3)
    p.add_argument("--bus", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--size", type=int, default=None,
                   help="迷路サイズ (既定 maze.yaml の grid_size=16)")
    p.add_argument("--v", type=float, default=0.12, help="前進の最大速度 [m/s]")
    p.add_argument("--omega", type=float, default=1.0, help="旋回の最大角速度 [rad/s]")
    p.add_argument("--dt", type=float, default=0.02, help="制御周期 [s]")
    p.add_argument("--pause", type=float, default=0.4, help="動作の前後で止まる秒数")
    p.add_argument("--time-limit", type=float, default=600.0, help="持ち時間 [s]")
    p.add_argument("--max-runs", type=int, default=5, help="最大走行回数")
    p.add_argument("--max-steps", type=int, default=400, help="探索の打ち切りステップ数")
    p.add_argument("--timeout", type=float, default=10.0, help="1動作の上限秒数 (直進はセル数で延長)")
    p.add_argument("--no-imu", action="store_true", help="ジャイロ融合なし")
    p.add_argument("--gyro-sign", type=float, default=1.0)
    p.add_argument("--gyro-scale", type=float, default=None)
    p.add_argument("--no-correct", action="store_true", help="カメラの位置補正を無効化")
    p.add_argument("--no-front-check", action="store_true", help="前進前の前方確認を無効化")
    p.add_argument("--save-frames", default=None, help="判定フレームの保存先プレフィクス")
    args = p.parse_args()

    setup_logging()
    kin = KiwiKinematics()
    maze_cfg = load_maze_config()
    maze = Maze(args.size) if args.size else Maze.from_config(maze_cfg)
    maze.set_outer_walls()
    explorer = Explorer(maze)
    manager = RunManager(explorer, time_limit_s=args.time_limit, max_runs=args.max_runs)
    detector = WallDetector(calibrated_config())
    yaw_cfg = calibrated_axis_yaw_config()
    gyro_scale = args.gyro_scale if args.gyro_scale is not None else kin.cfg.gyro_scale_z
    log.info("迷路 %dx%d / ゴール %s / 持ち時間 %.0fs / 最大 %d 走",
             maze.size, maze.size, maze.goal_cells(), args.time_limit, args.max_runs)

    with contextlib.ExitStack() as stack:
        import cv2

        from krilly.hal.camera import Camera

        camera = stack.enter_context(Camera())
        chain = stack.enter_context(
            L6470Chain(num_devices=args.devices, bus=args.bus, device=args.device)
        )
        stack.enter_context(
            emergency_stop(chain, on_stop=lambda sig: log.warning(
                "シグナル %s を受信。モーターを解放した。", sig))
        )
        statuses = chain.configure_all(L6470Profile())
        if any(s in (0x0000, 0xFFFF) for s in statuses):
            log.error("SPI 応答異常 (STATUS=%s)。中止。", [f"0x{s:04X}" for s in statuses])
            return
        log.info("停止は Ctrl-C (ESC は効かない)。"
                 "強制終了して回り続けたら python -m scripts.motor_stop")
        imu = None
        bias_z = 0.0
        if not args.no_imu:
            imu = stack.enter_context(Bno055Imu())
            imu.begin()
            log.info("静止のままジャイロバイアス計測中…")
            bias_z = imu.measure_gyro_bias()[2]
            log.info("ジャイロバイアス z=%.3f deg/s / スケール %.4f", bias_z, gyro_scale)

        est = DeadReckoning(kin, x=0.0, y=0.0, phi=heading_rad(Direction.N))
        motion = CellMotion(
            VelocityDriver(chain, kin), est,
            config=CellMotionConfig(v_max=args.v, omega_max=args.omega), maze=maze_cfg,
        )
        frame_no = [0]

        def gyro_rate() -> float | None:
            if imu is None:
                return None
            return math.radians(imu.gyro[2] - bias_z) * args.gyro_sign * gyro_scale

        def run_primitive(label: str, timeout: float) -> None:
            t0 = last = time.monotonic()
            while True:
                time.sleep(args.dt)
                now = time.monotonic(); dt = now - last; last = now
                if motion.update(dt, gyro_rate=gyro_rate()):
                    break
                if now - t0 > timeout:
                    log.warning("%s: %.1fs で打ち切り (残量 %.4f)", label, timeout, motion.remaining)
                    motion.abort()
                    break
            deadline = time.monotonic() + args.pause
            while time.monotonic() < deadline:
                time.sleep(args.dt)
                now = time.monotonic(); dt = now - last; last = now
                motion.update(dt, gyro_rate=gyro_rate())

        def capture():
            frame = camera.capture()
            if args.save_frames:
                frame_no[0] += 1
                cv2.imwrite(f"{args.save_frames}_{frame_no[0]:03d}.png", frame)
            return frame

        def correct_at(cell: tuple[int, int], facing: Direction, frame=None) -> None:
            """停止中に方位と位置を絶対補正する (強い帯のみ、#54/#65)。"""
            if args.no_correct:
                return
            frame = capture() if frame is None else frame
            # 方位: ジャイロの基準は走行開始時の向きなので、迷路軸へ引き戻す
            yaw = axis_yaw(frame, yaw_cfg)
            if yaw is not None:
                apply_axis_heading(est, yaw.angle_rad)
            off = cell_offset(frame, detector)
            if not off.measured:
                return
            apply_cell_offset(est, cell_center(cell, maze_cfg.cell_pitch_m),
                              off.forward_m, off.left_m, phi=heading_rad(facing))

        def front_is_clear() -> bool:
            if args.no_front_check:
                return True
            fraction = detector.measure(capture())[FRONT][0]
            if fraction >= detector.cfg.threshold_for(FRONT):
                log.error("前進中止: 前方に壁が見える (赤割合 %.2f)。姿勢ずれの可能性。", fraction)
                return False
            return True

        def turn_to(facing: Direction, target: Direction) -> Direction:
            turn = quarter_turns(facing, target)
            if turn:
                motion.start_turn_left(turn)
                run_primitive(TURN_LABEL.get(turn, str(turn)), args.timeout)
            return target

        # -- 探索ラン (search_run と同じ閉ループ) ------------------------------
        def search_to_goal() -> tuple[tuple[int, int], Direction] | None:
            for _ in range(args.max_steps):
                frame = capture()
                measured = detector.measure(frame)
                walls_body = {d: measured[d][0] >= detector.cfg.threshold_for(d)
                              for d in BODY_DIRS}
                explorer.observe(walls_body)
                correct_at(explorer.cell, explorer.facing, frame=frame)
                try:
                    step = explorer.plan()
                except Unreachable as e:
                    log.error("探索中止: %s", e)
                    return None
                if step is None:
                    log.info("ゴール到達! %d 手 / 訪問 %d セル",
                             explorer.steps, len(explorer.visited))
                    return (explorer.cell, explorer.facing)
                turn_to(explorer.facing, step.direction)
                if not front_is_clear():
                    return None
                motion.start_forward_cells(1)
                run_primitive("1セル前進", args.timeout)
                explorer.advance(step)
            log.error("探索が %d 手で終わらなかった", args.max_steps)
            return None

        # -- Leg 列の実行 (復帰・最速: 複数セルを 1 動作で) ---------------------
        def execute_legs(
            legs: list[Leg], cell: tuple[int, int], facing: Direction, label: str
        ) -> tuple[tuple[int, int], Direction] | None:
            log.info("%s: %s", label, describe_legs(legs))
            for leg in legs:
                facing = turn_to(facing, Direction((facing - leg.turn) % 4))
                correct_at(cell, facing)
                if not front_is_clear():
                    return None
                motion.start_forward_cells(leg.cells)
                run_primitive(f"直進{leg.cells}", max(args.timeout, leg.cells * 3.0))
                for _ in range(leg.cells):
                    cell = maze.neighbor(*cell, facing)
            correct_at(cell, facing)
            return (cell, facing)

        # -- 状態機械を回す -----------------------------------------------------
        t_run = time.monotonic()
        manager.start_search(time.monotonic())
        log.info("[走行 1] 探索ラン開始")
        pose = search_to_goal()
        log.info("探索ラン %.1fs", time.monotonic() - t_run)
        if pose is None:
            manager.abort()
        while manager.phase is not RunPhase.FINISHED:
            cell, facing = pose
            home = manager.goal_reached(time.monotonic(), cell, facing)
            if home is None:
                break
            log.info("復帰 (%s)", manager.summary(time.monotonic()))
            pose = execute_legs(home, cell, facing, "復帰経路")
            if pose is None:
                manager.abort()
                break
            legs = manager.home_reached(time.monotonic(), pose[1])
            if legs is None:
                break
            log.info("[走行 %d] 最速ラン開始 (見積 %.1fs)",
                     manager.runs_used, manager.estimate_s(legs))
            t_run = time.monotonic()
            pose = execute_legs(legs, maze.start, pose[1], "最速経路")
            if pose is None:
                manager.abort()
                break
            log.info("[走行 %d] 最速ラン %.1fs", manager.runs_used, time.monotonic() - t_run)

        log.info("終了: %s", manager.summary(time.monotonic()))
        log.info("判明した迷路:\n%s", maze.to_ascii())


if __name__ == "__main__":
    main()
