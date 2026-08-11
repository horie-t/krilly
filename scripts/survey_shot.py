#!/usr/bin/env python3
"""手動 survey: 機体は人が置き、撮影・正解照合・記録だけを行う (issue #65)。

自律走行での校正データ収集 (``wall_survey``) は、カメラ補正が誤っていると
旋回中の位置保持が機体を壁へ運ぶ、という鶏と卵の問題を抱える (校正するための
走行が、校正できていないカメラに依存する)。本スクリプトは**モーターに一切
触らない**。機体の移動・微調整は teleop / cell_move_demo / 手で行い、姿勢が
決まったらここでセルと向きを入力して 1 枚記録する。

対話コマンド:
    0 0 N        セル (0,0)・北向きで撮影して記録
    r            直前と同じ姿勢でもう 1 枚 (置き直しの確認用)
    q            終了 (CSV は 1 枚ごとに追記済みなのでいつ止めてもよい)

出力は wall_survey と同じ形式 (フレーム PNG + labels.csv、追記式):
    file,x,y,facing,edge,wall,fraction,off_fwd_mm,off_left_mm

撮影のたびに、既知形状から出した正解と判定を並べて表示するので、不一致が
出たらその場で気づける (機体の置き直しか、しきい値の問題かをすぐ切り分ける)。

例:
    python -m scripts.survey_shot --maze maze5.txt --out-dir survey5m
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from krilly.logging_config import get_logger, setup_logging
from krilly.perception.cell_pose import cell_offset
from krilly.perception.wall_detect import (
    BODY_DIRS,
    WallDetector,
    calibrated_config,
    maze_walls_to_body,
)
from krilly.solver.maze import Direction, Maze

log = get_logger("krilly.survey_shot")

CSV_FIELDS = ["file", "x", "y", "facing", "edge", "wall", "fraction",
              "off_fwd_mm", "off_left_mm"]


def parse_pose(text: str, size: int) -> tuple[int, int, Direction] | None:
    """"0 0 N" / "0,0,n" をセル座標と向きに解釈する。不正なら None。"""
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        return None
    try:
        x, y = int(parts[0]), int(parts[1])
        facing = Direction[parts[2].upper()]
    except (ValueError, KeyError):
        return None
    if not (0 <= x < size and 0 <= y < size):
        return None
    return (x, y, facing)


def next_shot_number(out: Path, prefix: str) -> int:
    """既存ファイルと衝突しない通し番号 (追記セッションでも上書きしない)。"""
    numbers = [
        int(p.name[len(prefix):len(prefix) + 2])
        for p in out.glob(f"{prefix}[0-9][0-9]*.png")
        if p.name[len(prefix):len(prefix) + 2].isdigit()
    ]
    return max(numbers, default=0) + 1


def main() -> None:
    p = argparse.ArgumentParser(description="手動 survey (撮影・照合・記録のみ)")
    p.add_argument("--maze", required=True, help="既知形状の ASCII テキストファイル")
    p.add_argument("--out-dir", default="survey", help="出力ディレクトリ (追記)")
    p.add_argument("--prefix", default="shot", help="ファイル名のプレフィクス")
    args = p.parse_args()

    setup_logging()
    maze = Maze.from_ascii(Path(args.maze).read_text(encoding="utf-8"))
    log.info("既知形状 (%dx%d):\n%s", maze.size, maze.size, maze.to_ascii())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{args.prefix}_labels.csv"
    write_header = not csv_path.exists()
    detector = WallDetector(calibrated_config())

    import cv2

    from krilly.hal.camera import Camera

    with Camera() as camera, open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        shot = next_shot_number(out, args.prefix)
        last: tuple[int, int, Direction] | None = None
        log.info("入力例: '0 0 N' で撮影 / 'r' で直前の姿勢をもう一枚 / 'q' で終了")
        while True:
            try:
                text = input("セル x y 向き> ").strip()
            except EOFError:
                break
            if text.lower() in ("q", "quit", "exit"):
                break
            if text.lower() == "r":
                if last is None:
                    log.warning("直前の姿勢が無い。まず '0 0 N' の形式で入力すること。")
                    continue
                pose = last
            else:
                pose = parse_pose(text, maze.size)
                if pose is None:
                    log.warning("解釈できない: %r (例: '0 0 N', 'q' で終了)", text)
                    continue
            x, y, facing = pose
            last = pose

            frame = camera.capture()
            name = f"{args.prefix}{shot:02d}_c{x}{y}_{facing.name}.png"
            cv2.imwrite(str(out / name), frame)
            shot += 1

            truth = maze_walls_to_body(
                {d: maze.has_wall(x, y, d) for d in Direction}, facing
            )
            measured = detector.measure(frame)
            off = cell_offset(frame, detector)
            mismatches = []
            for edge in BODY_DIRS:
                fraction, band_off = measured[edge]
                verdict = fraction >= detector.cfg.threshold_for(edge)
                ok = verdict == truth[edge]
                if not ok:
                    mismatches.append(edge)
                log.info("  %-6s 赤割合 %.3f (帯ずれ %+3dpx) 判定 %-4s 正解 %-4s %s",
                         edge, fraction, band_off,
                         "壁" if verdict else "なし", "壁" if truth[edge] else "なし",
                         "OK" if ok else "<-- 不一致!")
                writer.writerow({
                    "file": name, "x": x, "y": y, "facing": facing.name,
                    "edge": edge, "wall": int(truth[edge]),
                    "fraction": f"{measured[edge][0]:.4f}",
                    "off_fwd_mm": "" if off.forward_m is None else f"{off.forward_m*1000:.1f}",
                    "off_left_mm": "" if off.left_m is None else f"{off.left_m*1000:.1f}",
                })
            f.flush()
            def _mm(value):
                return "-" if value is None else f"{value * 1000:+.1f}mm"
            log.info("  %s: 位置ずれ 前後=%s 左右=%s%s", name,
                     _mm(off.forward_m), _mm(off.left_m),
                     "  ** 不一致あり: 置き直して 'r' で再撮影を推奨 **" if mismatches else "")
        log.info("終了。記録先: %s", csv_path)


if __name__ == "__main__":
    main()
