#!/usr/bin/env python3
"""競技の通しラン: 探索 → 復帰 → 最速 ×N (issue #20)。

:class:`krilly.app.run_manager.RunManager` の差配で、7 分・最大 5 走の予算内を
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
from krilly.hal.l6470_chain import L6470Chain
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.localization.grid import apply_axis_heading, apply_cell_offset
from krilly.logging_config import get_logger, setup_logging
from krilly.motion.cell_motion import CellMotion
from krilly.motion.emergency_stop import emergency_stop
from krilly.motion.tuning import add_tuning_args, build_tuning, check_limits, describe_faults
from krilly.motion.velocity_driver import VelocityDriver
from krilly.perception.axis_yaw import axis_yaw, calibrated_axis_yaw_config
from krilly.perception.cell_pose import cell_offset
from krilly.perception.wall_detect import (
    BODY_DIRS,
    WallDetector,
    calibrated_config,
    path_block_threshold,
    path_check_slots,
)
from krilly.solver.maze import Direction, Maze
from krilly.strategy.explorer import (
    Explorer,
    Unreachable,
    cell_center,
    heading_rad,
    quarter_turns,
)
from krilly.strategy.shortest_path import (
    DEFAULT_COST,
    LEGACY_COST,
    Leg,
    describe_legs,
)

log = get_logger("krilly.speed_run")

TURN_LABEL = {0: "直進", 1: "左90°", -1: "右90°", 2: "180°"}


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数の定義。**``main`` と分けてある**のはテストから触るため
    (``tests/test_run_scripts.py`` が「main が読む名前が実在するか」を突き合わせる)。"""
    p = argparse.ArgumentParser(description="競技の通しラン (探索 -> 復帰 -> 最速 xN)")
    p.add_argument("--devices", type=int, default=3)
    p.add_argument("--bus", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--size", type=int, default=None,
                   help="迷路サイズ (既定 maze.yaml の grid_size=16)")
    add_tuning_args(p)
    p.add_argument("--dt", type=float, default=0.02, help="制御周期 [s]")
    p.add_argument("--pause", type=float, default=0.4, help="動作の前後で止まる秒数")
    p.add_argument("--time-limit", type=float, default=420.0,
                   help="持ち時間 [s] (既定 420 = クラシック規定の 7 分。大会により 5 分)")
    p.add_argument("--max-runs", type=int, default=5, help="最大走行回数")
    p.add_argument("--max-steps", type=int, default=400, help="探索の打ち切りステップ数")
    p.add_argument("--timeout", type=float, default=10.0, help="1動作の上限秒数 (直進はセル数で延長)")
    p.add_argument("--no-imu", action="store_true", help="ジャイロ融合なし")
    p.add_argument("--gyro-sign", type=float, default=1.0)
    p.add_argument("--gyro-scale", type=float, default=None)
    p.add_argument("--no-correct", action="store_true", help="カメラの位置補正を無効化")
    p.add_argument("--turn-in-place", action="store_true",
                   help="機体を旋回させて走る従来モード (既定は旋回レス走行 #76)")
    p.add_argument("--max-heading-residual", type=float, default=2.5,
                   help="平行移動で許す方位残差 [deg]。超えたら接触とみなして中止する")
    p.add_argument("--no-front-check", action="store_true", help="前進前の前方確認を無効化")
    p.add_argument("--chain-legs", type=int, default=1,
                   help="最速・復帰で止まらずに繋ぐ区間の本数の上限 (#80)。"
                        "1 = 区間ごとに停止 (従来)。繋ぐと位置補正の間隔も伸びる")
    p.add_argument("--no-neighbors", action="store_true",
                   help="左右の隣セルを読まない (#89 を切る。1 セルずつ止まって進む)")
    p.add_argument("--pass-cells", type=int, default=None,
                   help="探索で止まらずに通過してよいセル数の上限 (既定 2、隣を読まないなら 1)")
    p.add_argument("--save-frames", default=None, help="判定フレームの保存先プレフィクス")
    return p


def main() -> None:
    args = build_parser().parse_args()

    setup_logging()
    kin = KiwiKinematics()
    tuning = build_tuning(args)
    maze_cfg = load_maze_config()
    maze = Maze(args.size) if args.size else Maze.from_config(maze_cfg)
    maze.set_outer_walls()
    explorer = Explorer(maze, holonomic=not args.turn_in_place)
    manager = RunManager(explorer, time_limit_s=args.time_limit, max_runs=args.max_runs,
                         holonomic=not args.turn_in_place,
                         cost=LEGACY_COST if args.turn_in_place else DEFAULT_COST)
    neighbors = not args.no_neighbors
    detector = WallDetector(calibrated_config(neighbors=neighbors))
    pass_cells = args.pass_cells if args.pass_cells else (2 if neighbors else 1)
    yaw_cfg = calibrated_axis_yaw_config()
    gyro_scale = args.gyro_scale if args.gyro_scale is not None else kin.cfg.gyro_scale_z
    log.info("迷路 %dx%d / ゴール %s / 持ち時間 %.0fs / 最大 %d 走",
             maze.size, maze.size, maze.goal_cells(), args.time_limit, args.max_runs)
    log.info("探索の観測: 自セルの 4 壁%s / 1 動作で最大 %d セル",
             " + 左右の隣セル (#89)" if neighbors else "", pass_cells)
    log.info("最速・復帰: 1 動作で最大 %d 区間%s",
             args.chain_legs,
             " (止まらずに曲がる #80)" if args.chain_legs > 1 else " (区間ごとに停止)")
    log.info("チューニング: %s", tuning.describe())
    for warning in check_limits(tuning, kin):
        log.warning("%s", warning)

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
        statuses = chain.configure_all(tuning.profile)
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
            VelocityDriver(chain, kin, limits=tuning.limits), est,
            config=tuning.motion, maze=maze_cfg,
        )
        frame_no = [0]

        # 最初の駆動指令でロータが谷へスナップし車体が最大 0.5° 跳ねる。
        # カメラで壁・位置を測る前に済ませておく (VelocityDriver.energize 参照)。
        motion.driver.energize()
        time.sleep(0.3)

        def gyro_rate() -> float | None:
            if imu is None:
                return None
            return math.radians(imu.gyro[2] - bias_z) * args.gyro_sign * gyro_scale

        timings: dict[str, list[float]] = {}

        def run_primitive(label: str, timeout: float, kind: str = "") -> None:
            """1 動作を回し、所要時間を記録する (RunManager の見積もり定数の実測用)。"""
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
            elapsed = time.monotonic() - t0
            if kind:
                timings.setdefault(kind, []).append(elapsed)
            faults = describe_faults(chain.get_status_all(), ignore=("UVLO",))
            if faults:
                log.warning("    %s: L6470 フォールト %s (トルク不足の疑い)", label, faults)
            along, cross, dphi = motion.residual()
            log.info("    %s 完了 %.2fs 残差 前後=%+.4fm 左右=%+.4fm 方位=%+.2f°%s",
                     label, elapsed, along, cross, math.degrees(dphi),
                     "" if motion.retries == 0 else " / やり直し %d 回 (判定時の残量 %+.4f)"
                     % (motion.retries, motion.retry_remaining))
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

        def correct_at(cell: tuple[int, int], facing: Direction, frame=None,
                       measured=None) -> None:
            """停止中に方位と位置を絶対補正する (強い帯のみ、#54/#65)。"""
            if args.no_correct:
                return
            frame = capture() if frame is None else frame
            # 方位: ジャイロの基準は走行開始時の向きなので、迷路軸へ引き戻す
            yaw = axis_yaw(frame, yaw_cfg)
            if yaw is not None:
                apply_axis_heading(est, yaw.angle_rad)
            off = cell_offset(frame, detector, measured=measured)
            if off.saturated:
                log.info("  位置補正: %s は帯がフレーム端で飽和したので不採用",
                         "/".join(off.saturated))
            if not off.measured:
                return
            apply_cell_offset(est, cell_center(cell, maze_cfg.cell_pitch_m),
                              off.forward_m, off.left_m, phi=heading_rad(facing))

        def path_is_clear(direction: Direction, facing: Direction,
                          cells: int = 1) -> bool:
            """進む直前に、進行方角の壁が見えていないか確認する。

            見る辺は進行方角から決める (ホロノミック走行では進行方向と機体の向きが
            一致しない)。2 セル以上進むときは通過するセルの出口も見る — ただし
            左右方向だけで、前後方向の 2 セル目は画角の外 (:func:`path_check_slots`)。
            しきい値は ``path_block_threshold`` (辺別の壁しきい値と進路チェックの
            下限の大きい方) を使う。
            """
            if args.no_front_check:
                return True
            measured = detector.measure(capture())
            for i, slot in enumerate(path_check_slots(direction, facing, cells,
                                                      neighbors)):
                fraction = measured[slot][0]
                threshold = path_block_threshold(detector.cfg, slot)
                if fraction >= threshold:
                    log.error("進行中止: %s 方向 %d セル目 (%s) に壁が見える "
                              "(赤割合 %.2f >= %.2f)。姿勢ずれの可能性。",
                              direction.name, i + 1, slot, fraction, threshold)
                    return False
            return True

        def move_was_clean(label: str, timeout: float | None = None,
                           kind: str = "") -> bool:
            """移動を回し、**指令していない回転**が出ていないかを見る。

            平行移動では回転を一切指令しないので、測れた回転は異常の証拠になる
            (実機 #76: 壁に接触した移動が -3.01° を残し、その後カメラの軸角測定が
            壊れて走行が崩壊した)。旋回する走り方では旋回を指令するので判定しない。
            """
            run_primitive(label, timeout if timeout is not None else args.timeout,
                          kind=kind)
            if args.turn_in_place:
                return True
            residual = abs(math.degrees(motion.residual()[2]))
            if residual > args.max_heading_residual:
                log.error("進行中止: 回転を指令していないのに %.2f° 回った (上限 %.2f°)。"
                          "壁との接触かスリップの可能性が高い。",
                          residual, args.max_heading_residual)
                return False
            return True

        def turn_to(facing: Direction, target: Direction) -> Direction:
            turn = quarter_turns(facing, target)
            if turn:
                motion.start_turn_left(turn)
                run_primitive(TURN_LABEL.get(turn, str(turn)), args.timeout,
                              kind=f"turn{abs(turn)}")
            return target

        # -- 探索ラン (search_run と同じ閉ループ) ------------------------------
        def search_to_goal() -> tuple[tuple[int, int], Direction] | None:
            stops = 0
            for _ in range(args.max_steps):
                frame = capture()
                measured = detector.measure(frame)
                walls_body = {d: measured[d][0] >= detector.cfg.threshold_for(d)
                              for d in BODY_DIRS}
                explorer.observe(walls_body,
                                 detector.neighbor_walls(measured) if neighbors
                                 else None)
                correct_at(explorer.cell, explorer.facing, frame=frame,
                           measured=measured)
                try:
                    steps = explorer.plan_leg(pass_cells)
                except Unreachable as e:
                    log.error("探索中止: %s", e)
                    return None
                if not steps:
                    log.info("ゴール到達! %d 手 / %d 停止 / 訪問 %d セル / 壁が確定 %d セル",
                             explorer.steps, stops, len(explorer.visited),
                             len(explorer.known))
                    return (explorer.cell, explorer.facing)
                direction, cells = steps[0].direction, len(steps)
                axis = quarter_turns(explorer.facing, direction)
                facing = explorer.facing
                if args.turn_in_place:
                    facing = turn_to(explorer.facing, direction)   # 旋回後の向きで確認する
                    axis = 0
                if not path_is_clear(direction, facing, cells):
                    return None
                motion.start_move_cells(cells, axis)
                if not move_was_clean(f"{direction.name}へ{cells}セル",
                                      timeout=max(args.timeout, cells * 3.0)):
                    return None
                stops += 1
                for step in steps:
                    explorer.advance(step)
            log.error("探索が %d 手で終わらなかった", args.max_steps)
            return None

        # -- Leg 列の実行 (復帰・最速: 複数セルを 1 動作で) ---------------------
        def leg_chunks(legs: list[Leg]) -> list[list[Leg]]:
            """止まらずに走る区間のまとまりに切る (#80)。

            旋回する走り方では繋がない (区間の間に旋回が入るので必ず止まる)。
            まとまりの**先頭でしか位置補正もカメラの進路確認もできない**ので、
            長くするほど誤差の蓄積に賭けることになる。
            """
            size = 1 if args.turn_in_place else max(1, args.chain_legs)
            return [legs[i:i + size] for i in range(0, len(legs), size)]

        def execute_legs(
            legs: list[Leg], cell: tuple[int, int], facing: Direction, label: str
        ) -> tuple[tuple[int, int], Direction] | None:
            log.info("%s: %s", label, describe_legs(legs))
            for chunk in leg_chunks(legs):
                head = chunk[0]
                if args.turn_in_place:
                    facing = turn_to(facing, head.direction)
                correct_at(cell, facing)
                # 進路確認は**いま居るセルから見える範囲だけ**。まとまりの 2 本目以降は
                # 画角の外なので、観測済みの地図に賭けることになる (#89 と同じ扱い)。
                if not path_is_clear(head.direction, facing, head.cells):
                    return None
                motion.start_path_cells([
                    (leg.cells, 0 if args.turn_in_place
                     else quarter_turns(facing, leg.direction))
                    for leg in chunk
                ])
                cells = sum(leg.cells for leg in chunk)
                # 実測の集計は**進行軸で分ける**。旋回レスでは南北が機体の前後軸、
                # 東西が左右軸になり、所要時間が違いうる (時間定数の校正に要る)。
                # 繋いだ動作は軸が混ざるので別の名前で数える。
                if len(chunk) > 1:
                    kind = f"chain{len(chunk)}"
                else:
                    axis_name = ("ns" if head.direction in (Direction.N, Direction.S)
                                 else "ew")
                    kind = f"{axis_name}{head.cells}"
                if not move_was_clean(
                        "->".join(f"{leg.direction.name}へ{leg.cells}" for leg in chunk),
                        timeout=max(args.timeout, cells * 3.0), kind=kind):
                    return None
                if len(chunk) > 1:
                    log.info("  止まらずに曲がった回数 %d", motion.corners)
                for leg in chunk:
                    for _ in range(leg.cells):
                        cell = maze.neighbor(*cell, leg.direction)
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
            # 競技規定 3-4: 始点に戻って自動的に再スタートする場合、始点で 2 秒以上
            # 停止しなければならない。持ち時間から確実に引かれるが、削ってはいけない。
            log.info("[走行 %d] 始点で %.1fs 停止 (規定 3-4)",
                     manager.runs_used, manager.restart_dwell_s)
            time.sleep(manager.restart_dwell_s)
            log.info("[走行 %d] 最速ラン開始 (見積 %.1fs)",
                     manager.runs_used, manager.estimate_s(legs))
            t_run = time.monotonic()
            pose = execute_legs(legs, maze.start, pose[1], "最速経路")
            if pose is None:
                manager.abort()
                break
            log.info("[走行 %d] 最速ラン %.1fs", manager.runs_used, time.monotonic() - t_run)

        log.info("終了: %s", manager.summary(time.monotonic()))
        if timings:
            log.info("動作あたりの実測時間 (RunManager の見積もり定数の校正用):")
            for kind in sorted(timings):
                vals = timings[kind]
                per = ""
                for prefix in ("cells", "ns", "ew"):
                    if kind.startswith(prefix) and kind[len(prefix):].isdigit():
                        n = int(kind[len(prefix):])
                        per = f" / 1セルあたり {sum(vals) / len(vals) / n:.2f}s"
                log.info("  %-8s n=%2d 平均 %.2fs (最小 %.2f 最大 %.2f)%s",
                         kind, len(vals), sum(vals) / len(vals), min(vals), max(vals), per)
            # 時間定数の当てはめ: 同じ軸の 1 セルと n セルから「1 セルあたり」と
            # 「区間 1 本あたりの固定費」を分ける (連続直進はランプの固定費を償却する)
            for axis, name in (("ns", "南北 (前後軸)"), ("ew", "東西 (左右軸)")):
                points = sorted((int(k[len(axis):]), sum(v) / len(v))
                                for k, v in timings.items()
                                if k.startswith(axis) and k[len(axis):].isdigit())
                if len(points) >= 2:
                    (n1, t1), (n2, t2) = points[0], points[-1]
                    per_cell = (t2 - t1) / (n2 - n1)
                    fixed = t1 - per_cell * n1
                    log.info("  -> %s: 1セルあたり %.2fs + 区間あたり %.2fs "
                             "(%dセルと%dセルから)", name, per_cell, fixed, n1, n2)
        log.info("判明した迷路:\n%s", maze.to_ascii())


if __name__ == "__main__":
    main()
