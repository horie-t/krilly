#!/usr/bin/env python3
"""壁判定の校正データを実機で収集する (issue #56).

**既知の形状**を与えて全セルを巡回し、各セル・各向きでフレームを撮る。経路計画に
真の地図を使うので壁を横切らず、**壁判定の結果に依存せずに走れる** (壁判定を直す
ためのデータ収集なので、これが必須)。同時に各辺の正解ラベル (その姿勢で見えるはず
の壁) を書き出すので、あとはオフラインでしきい値を詰められる。

部屋の照明はセルごとに当たり方が変わり、白飛びの出方も違う。**全セル分のラベル付き
データ**が無いと、1 枚で合わせたしきい値が別のセルで破綻する (#56 がまさにそれ)。

出力:
    <prefix>_c<x><y>_<facing>.png   各姿勢のフレーム
    <prefix>_labels.csv             file,x,y,facing,edge,wall,fraction

例:
    python -m scripts.wall_survey --maze maze3.txt --out-dir survey
    python -m scripts.wall_survey --maze maze3.txt --dry-run     # 巡回順路だけ表示
    python -m scripts.wall_survey --maze maze3.txt --facings 1   # 到着した向きのみ撮る

前提: **スタートセルの中心に、迷路の北を向いて置く**。--maze には to_ascii 形式
(セル幅は任意) のテキストファイルを渡す。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import time
from pathlib import Path

from krilly.config import load_maze_config
from krilly.hal.imu import Bno055Imu
from krilly.hal.l6470 import L6470Profile
from krilly.hal.l6470_chain import L6470Chain
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.logging_config import get_logger, setup_logging
from krilly.motion.cell_motion import CellMotion, CellMotionConfig
from krilly.localization.grid import apply_cell_offset
from krilly.motion.emergency_stop import emergency_stop
from krilly.motion.velocity_driver import VelocityDriver
from krilly.perception.cell_pose import cell_offset
from krilly.perception.wall_detect import (
    BODY_DIRS,
    WallDetector,
    calibrated_config,
    maze_walls_to_body,
)
from krilly.solver.maze import Direction, Maze
from krilly.strategy.explorer import Step, cell_center, heading_rad, quarter_turns
from krilly.strategy.flood_fill import UNREACHABLE, flood_fill, next_direction

log = get_logger("krilly.wall_survey")

TURN_LABEL = {0: "直進", 1: "左90°", -1: "右90°", 2: "180°"}


def wrapped_deg(rad: float) -> float:
    """表示用: 角度を (-180, 180] deg に正規化する (推定φは積算値)。"""
    return math.degrees((rad + math.pi) % (2 * math.pi) - math.pi)


def plan_tour(
    maze: Maze, start: tuple[int, int] = (0, 0), facing: Direction = Direction.N
) -> list[Step]:
    """既知の地図で全セルを巡回する順路を Step 列で返す (壁は横切らない)。

    毎回「最も近い未訪問セル」を目標にして flood-fill の勾配を降りる。到達できない
    セルは飛ばす (壁で隔離されている場合)。
    """
    cell, face = start, facing
    visited = {cell}
    all_cells = {(x, y) for x in range(maze.size) for y in range(maze.size)}
    steps: list[Step] = []
    while True:
        remaining = all_cells - visited
        if not remaining:
            return steps
        # 現在地からの距離で最も近い未訪問セルを選ぶ (同距離なら座標順で決定的に)
        here = flood_fill(maze, goals=[cell])
        reachable = [c for c in remaining if here[c[0]][c[1]] < UNREACHABLE]
        if not reachable:
            log.warning("到達できないセルを飛ばす: %s", sorted(remaining))
            return steps
        target = min(reachable, key=lambda c: (here[c[0]][c[1]], c))
        dist = flood_fill(maze, goals=[target])
        while cell != target:
            d = next_direction(maze, cell, face, dist)
            if d is None:                      # 起こらないはずだが保険
                log.warning("%s から %s へ進めない", cell, target)
                return steps
            nxt = maze.neighbor(*cell, d)
            steps.append(Step(quarter_turns(face, d), d, nxt))
            cell, face = nxt, d
            visited.add(cell)


def main() -> None:
    p = argparse.ArgumentParser(description="壁判定の校正データ収集 (既知形状を巡回)")
    p.add_argument("--maze", required=True, help="既知形状の ASCII テキストファイル")
    p.add_argument("--out-dir", default="survey", help="出力ディレクトリ")
    p.add_argument("--prefix", default="cell", help="ファイル名のプレフィクス")
    p.add_argument("--facings", type=int, default=4, choices=(1, 4),
                   help="各セルで撮る向きの数 (4=その場で1周)")
    p.add_argument("--devices", type=int, default=3)
    p.add_argument("--bus", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--v", type=float, default=0.12, help="前進の最大速度 [m/s]")
    p.add_argument("--omega", type=float, default=1.5, help="旋回の最大角速度 [rad/s]")
    p.add_argument("--dt", type=float, default=0.02, help="制御周期 [s]")
    p.add_argument("--pause", type=float, default=0.4, help="動作後に止まる秒数")
    p.add_argument("--timeout", type=float, default=10.0, help="1動作の上限秒数")
    p.add_argument("--no-imu", action="store_true", help="ジャイロ融合なし")
    p.add_argument("--gyro-sign", type=float, default=1.0)
    p.add_argument("--gyro-scale", type=float, default=None)
    p.add_argument("--dry-run", action="store_true", help="走行せず順路だけ表示")
    p.add_argument("--no-correct", action="store_true",
                   help="カメラによるセル内位置の絶対補正を無効にする (#54 の前後比較用)")
    args = p.parse_args()

    setup_logging()
    maze = Maze.from_ascii(Path(args.maze).read_text(encoding="utf-8"))
    log.info("既知形状 (%dx%d) を読み込み:\n%s", maze.size, maze.size, maze.to_ascii())
    tour = plan_tour(maze)
    log.info("巡回順路 %d 手: %s", len(tour),
             " ".join(f"{TURN_LABEL.get(s.turn, s.turn)}->{s.to_cell}" for s in tour))
    if args.dry_run:
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kin = KiwiKinematics()
    maze_cfg = load_maze_config()
    detector = WallDetector(calibrated_config())
    gyro_scale = args.gyro_scale if args.gyro_scale is not None else kin.cfg.gyro_scale_z
    rows: list[dict] = []

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

        def gyro_rate() -> float | None:
            if imu is None:
                return None
            return math.radians(imu.gyro[2] - bias_z) * args.gyro_sign * gyro_scale

        def run_primitive(label: str) -> None:
            t0 = last = time.monotonic()
            while True:
                time.sleep(args.dt)
                now = time.monotonic(); dt = now - last; last = now
                if motion.update(dt, gyro_rate=gyro_rate()):
                    break
                if now - t0 > args.timeout:
                    log.warning("%s: 打ち切り (残量 %.4f)", label, motion.remaining)
                    motion.abort()
                    break
            deadline = time.monotonic() + args.pause
            while time.monotonic() < deadline:      # 停止指令のまま整定させる
                time.sleep(args.dt)
                now = time.monotonic(); dt = now - last; last = now
                motion.update(dt, gyro_rate=gyro_rate())

        shot = [0]   # 通し番号 (同じセルを再訪してもファイル名が衝突しないように)

        def shoot(cell: tuple[int, int], facing: Direction) -> None:
            """1 枚撮って、正解ラベルと現行設定での赤割合を記録する。"""
            frame = camera.capture()
            shot[0] += 1
            name = f"{args.prefix}{shot[0]:02d}_c{cell[0]}{cell[1]}_{facing.name}.png"
            cv2.imwrite(str(out / name), frame)
            truth = maze_walls_to_body(
                {d: maze.has_wall(cell[0], cell[1], d) for d in Direction}, facing
            )
            fractions = detector.red_fractions(frame)
            # セル内の位置ずれを測って記録し (#54)、有効なら推定位置を補正する
            off = cell_offset(frame, detector)
            center = cell_center(cell, maze_cfg.cell_pitch_m)
            applied = (not args.no_correct) and apply_cell_offset(
                est, center, off.forward_m, off.left_m, phi=heading_rad(facing)
            )

            def _mm(value):
                return "-" if value is None else f"{value * 1000:+.1f}"

            log.info("    位置ずれ 前後=%smm 左右=%smm (壁 %d/%d 枚) %s",
                     _mm(off.forward_m), _mm(off.left_m), off.walls_y, off.walls_x,
                     "-> 補正した" if applied else "-> 補正なし")
            for edge in BODY_DIRS:
                rows.append({
                    "file": name, "x": cell[0], "y": cell[1], "facing": facing.name,
                    "edge": edge, "wall": int(truth[edge]),
                    "fraction": f"{fractions[edge]:.4f}",
                    "off_fwd_mm": "" if off.forward_m is None else f"{off.forward_m*1000:.1f}",
                    "off_left_mm": "" if off.left_m is None else f"{off.left_m*1000:.1f}",
                })
            log.info("  %s: 正解 %s / 赤割合 %s", name,
                     {e: int(truth[e]) for e in BODY_DIRS},
                     {e: f"{fractions[e]:.2f}" for e in BODY_DIRS})

        def shoot_all_facings(cell: tuple[int, int], facing: Direction) -> Direction:
            """その場で 4 向き撮る (facings=1 なら 1 枚)。戻り値は撮影終了時の向き。

            元の向きへ戻す旋回はしない。次の移動方向へは呼び出し側が実際の向きから
            直接回る (quarter_turns)。無駄な旋回は滑り = 位置ずれ = 壁接触の機会を
            増やすだけなので、旋回数は最小にする (#65: 自立壁はかすっただけで倒れる)。
            """
            shoot(cell, facing)
            if args.facings == 1:
                return facing
            for _ in range(3):
                motion.start_turn_left(1)
                run_primitive("左90°")
                facing = Direction((facing - 1) % 4)   # CCW = 反時計回り
                shoot(cell, facing)
            return facing

        def correct_before_move(cell: tuple[int, int], facing: Direction) -> None:
            """前進の直前に 1 枚撮って位置を補正する (#65)。

            旋回中の滑りはオドメトリに見えないまま前進に持ち込まれ、通路の
            クリアランス (片側 ~20mm) を食い潰して壁に接触する。移動前に絶対補正を
            入れておけば、CellMotion の横ずれ保持が正しい位置を基準に働く。
            """
            if args.no_correct:
                return
            off = cell_offset(camera.capture(), detector)
            if not off.measured:
                return
            applied = apply_cell_offset(
                est, cell_center(cell, maze_cfg.cell_pitch_m),
                off.forward_m, off.left_m, phi=heading_rad(facing),
            )
            def _mm(value):
                return "-" if value is None else f"{value * 1000:+.1f}"
            log.info("    移動前補正 前後=%smm 左右=%smm %s",
                     _mm(off.forward_m), _mm(off.left_m),
                     "-> 補正した" if applied else "-> 棄却")

        cell, facing = (0, 0), Direction.N
        log.info("[1/%d] セル %s", len(tour) + 1, cell)
        facing = shoot_all_facings(cell, facing)
        for i, step in enumerate(tour, 2):
            # 撮影終了時の実際の向きから、次の移動方向へ最短で回る
            turn = quarter_turns(facing, step.direction)
            if turn:
                motion.start_turn_left(turn)
                run_primitive(TURN_LABEL.get(turn, str(turn)))
                facing = step.direction
            correct_before_move(cell, facing)
            motion.start_forward_cells(1)
            run_primitive("1セル前進")
            cell = step.to_cell
            log.info("[%d/%d] セル %s (推定 X=%.4f Y=%.4f φ=%+.1f°)",
                     i, len(tour) + 1, cell, *est.pose[:2], wrapped_deg(est.pose[2]))
            facing = shoot_all_facings(cell, facing)

        csv_path = out / f"{args.prefix}_labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file", "x", "y", "facing", "edge", "wall", "fraction",
                            "off_fwd_mm", "off_left_mm"]
            )
            writer.writeheader()
            writer.writerows(rows)
        log.info("%d 枚 / %d ラベルを %s に保存", len(rows) // 4, len(rows), csv_path)
        log.info("最終 推定 X=%.4f Y=%.4f φ=%+.1f° (セル %s)",
                 *est.pose[:2], wrapped_deg(est.pose[2]), cell)


if __name__ == "__main__":
    main()
