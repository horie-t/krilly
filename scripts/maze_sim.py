#!/usr/bin/env python3
"""多数の迷路をまとめてシミュレートする (issue #77)。

**実機不要。** 探索 → 復帰 → 最速 xN の 10 分セッションを、真の迷路を相手に丸ごと
回して「完走したか / 地図が一致したか / 予算に収まったか」を出す。実機の役割を
「誤差の蓄積」と「カメラ判定」に絞るための土台 (#23)。

例:
    # 手持ちのファイルを回す (ASCII、セル幅は任意。S/G でスタート・ゴールを書ける)
    python -m scripts.maze_sim --maze mazes/*.txt

    # ランダム迷路を 50 面
    python -m scripts.maze_sim --generate 50 --size 16

    # 意地悪なパターン (全面開放 / 長い一本道 / 袋小路 / ゴール封鎖)
    python -m scripts.maze_sim --pattern all --size 8

    # ゴールを対角に置くストレステスト
    python -m scripts.maze_sim --generate 20 --size 16 --goal diagonal

    # 見積もりより 40% 遅かったら予算判断が保つか (time_margin=1.5 の検算)
    python -m scripts.maze_sim --generate 20 --size 16 --actual-scale 1.4

    # 組めるかだけ調べる (実機の準備。壁の枚数と健全性)
    python -m scripts.maze_sim --maze mazes/maze8.txt --check-only --wall-budget 70

問題が 1 つでもあれば終了コード 1 を返すので、そのまま CI に載せられる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from krilly.logging_config import get_logger, setup_logging
from krilly.sim import (
    check_maze,
    comb_maze,
    diagonal_goal,
    open_maze,
    random_maze,
    seal_goal,
    serpentine_maze,
    simulate_session,
)
from krilly.solver.maze import Maze
from krilly.strategy.shortest_path import DEFAULT_COST, LEGACY_COST

log = get_logger(__name__)

# 名前 -> (説明, 作り方, 完走するのが正解か)。
# "sealed" は**到達不能が正解**なので、完走したらそちらが異常。
PATTERNS = {
    "open": ("全面開放", lambda n, seed: open_maze(n), True),
    "serpentine": ("長い一本道", lambda n, seed: serpentine_maze(n), True),
    "comb": ("袋小路だらけ", lambda n, seed: comb_maze(n), True),
    "sealed": ("ゴール封鎖", lambda n, seed: seal_goal(random_maze(n, seed=seed)), False),
}


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("迷路の指定 (併用可)")
    src.add_argument("--maze", nargs="*", default=[], help="ASCII 迷路ファイル")
    src.add_argument("--generate", type=int, default=0, help="ランダム迷路の面数")
    src.add_argument("--pattern", default="", help=f"意地悪なパターン (all / "
                     f"{','.join(PATTERNS)})")
    src.add_argument("--size", type=int, default=16, help="生成する迷路の 1 辺 (既定 16)")
    src.add_argument("--seed", type=int, default=0, help="生成の乱数種")
    src.add_argument("--loop-ratio", type=float, default=0.10,
                     help="ランダム迷路のループの多さ (既定 0.10。0 で完全迷路、\n                           大きいほど疎で易しい)")

    cfg = p.add_argument_group("走らせ方")
    cfg.add_argument("--goal", choices=("center", "diagonal"), default="center",
                     help="ゴールの位置 (既定 center = 本番設定の中央 2x2)")
    cfg.add_argument("--turn-in-place", action="store_true",
                     help="旋回する走り方で見積もる (既定は旋回レス #76)")
    cfg.add_argument("--time-limit", type=float, default=600.0, help="持ち時間 [s]")
    cfg.add_argument("--max-runs", type=int, default=5, help="最大走行回数")
    cfg.add_argument("--actual-scale", type=float, default=1.0,
                     help="実際は見積もりの何倍かかるか (予算判断の余裕を試す)")
    cfg.add_argument("--search-overhead", type=float, default=0.0,
                     help="探索 1 手あたりの追加時間 [s] (壁判定・位置補正の分)")

    out = p.add_argument_group("出力")
    out.add_argument("--check-only", action="store_true", help="健全性チェックだけ行う")
    out.add_argument("--wall-budget", type=int, default=None,
                     help="手持ちの壁の枚数 (足りなければ注意を出す)")
    out.add_argument("--ascii", action="store_true", help="迷路を ASCII で表示する")
    out.add_argument("--verbose", action="store_true", help="各走行の内訳も出す")
    return p


def collect(args) -> list[tuple[str, Maze, bool]]:
    """指定された迷路をすべて集める。第 3 要素は「完走するのが正解か」。"""
    mazes: list[tuple[str, Maze, bool]] = []
    for path in args.maze:
        text = Path(path).read_text(encoding="utf-8")
        mazes.append((Path(path).stem, Maze.from_ascii(text), True))
    names = list(PATTERNS) if args.pattern == "all" else [
        n for n in args.pattern.split(",") if n
    ]
    for name in names:
        if name not in PATTERNS:
            raise SystemExit(f"不明なパターン: {name} (使えるのは {','.join(PATTERNS)})")
        label, make, expect_ok = PATTERNS[name]
        mazes.append((f"{name}({label})", make(args.size, args.seed), expect_ok))
    for i in range(args.generate):
        mazes.append((f"random{i}",
                      random_maze(args.size, seed=args.seed + i, loop_ratio=args.loop_ratio),
                      True))
    if args.goal == "diagonal":
        for _, m, _ok in mazes:
            diagonal_goal(m)
    return mazes


def main() -> int:
    args = build_args().parse_args()
    setup_logging()
    mazes = collect(args)
    if not mazes:
        raise SystemExit("迷路が指定されていない (--maze / --generate / --pattern)")

    cost = LEGACY_COST if args.turn_in_place else DEFAULT_COST
    problems = 0
    for name, maze, expect_ok in mazes:
        if args.ascii:
            log.info("%s:\n%s", name, maze.to_ascii())
        report = check_maze(maze, wall_budget=args.wall_budget)
        log.info("%-24s %s", name, report.describe().replace("\n", "\n" + " " * 25))
        if args.check_only:
            if report.ok is not expect_ok:
                problems += 1
            continue
        if not report.ok:
            # 健全性チェックで落ちた迷路は走らせない (走らせるまでもなく結論が出ている)
            problems += 0 if not expect_ok else 1
            continue

        result = simulate_session(
            maze, holonomic=not args.turn_in_place, cost=cost,
            time_limit_s=args.time_limit, max_runs=args.max_runs,
            search_step_overhead_s=args.search_overhead, actual_scale=args.actual_scale,
        )
        good = result.ok is expect_ok
        speed = f"{result.best_speed_s:.1f}s" if result.best_speed_s else "-"
        log.info("%-24s %s 探索 %.1fs / 最速 %s / 経過 %.1fs / 走行 %d 回%s",
                 "", "OK " if good else "NG ",
                 result.search.duration_s if result.search else float("nan"),
                 speed, result.elapsed_s, result.runs_used,
                 "" if good else "  <- 要調査")
        if args.verbose or not good:
            log.info("%s", "  " + result.describe().replace("\n", "\n  "))
        if not good:
            problems += 1

    log.info("---")
    log.info("%d 面中 %d 面に問題", len(mazes), problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
