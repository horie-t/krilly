#!/usr/bin/env python3
"""大会迷路から実機で組める盤面を切り出して選ぶ (issue #85 / #23)。

**実機なし。** #84 で書き起こした 16x16 に窓を当てて N×N を切り出し、組めるか
(壁と柱の枚数・外周が閉じているか・到達可能か・柱に壁が付いているか) を確かめ、
**実機でしか試せない要素**の多い順に並べる。

選ぶ基準は速さではない。速さはシミュレーションで測れる。実機でしか分からないのは
**位置補正が入らないセルが続いたときに誤差がどこまで育つか**なので、そこを優先する
(:attr:`~krilly.sim.excerpt.Difficulty.score`)。

例:
    # 8x8 の候補を上位 10 件
    python -m scripts.maze_excerpt --size 8 --maze mazes/contest/*.txt

    # 手持ちの壁 70 枚に収まるものだけ
    python -m scripts.maze_excerpt --size 8 --wall-budget 70 --maze mazes/contest/*.txt

    # 中断の発生率を測るための盤面を選ぶ (#85: 曲がりが多いほどサンプルが増える)
    python -m scripts.maze_excerpt --size 8 --maze mazes/contest/*.txt \
        --wall-budget 80 --sort exposure --min-cells 24 --out mazes/twisty8.txt

    # 選んだものを書き出す (既定は 1 位、--rank で順位を選ぶ)
    python -m scripts.maze_excerpt --size 7 --maze mazes/contest/*.txt --out mazes/excerpt7.txt
    python -m scripts.maze_excerpt --size 8 --maze mazes/contest/*.txt --rank 2 --out mazes/excerpt8_2015.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from krilly.logging_config import get_logger, setup_logging
from krilly.sim import check_maze, open_maze, sense, sense_neighbors
from krilly.sim.check import posts_without_wall, reachable_cells
from krilly.sim.excerpt import (
    Difficulty,
    excerpts,
    goal_variants,
    longest_blind_cross_run,
    longest_blind_run,
    pieces_needed,
)
from krilly.sim.check import wall_counts
from krilly.app.run_manager import RunManager
from krilly.strategy.shortest_path import chunk_legs
from krilly.solver.maze import Direction, Maze
from krilly.strategy.explorer import Explorer, Unreachable
from krilly.strategy.shortest_path import path_to_legs, shortest_path

log = get_logger(__name__)

#: 実機の既定と同じ設定 (#80 / #85)。連結動作の本数はここで決まる。
CHAIN_LEGS = 2
#: 露出を数えるときに想定するセッションの走行回数 (`speed_run --max-runs 25`)。
SESSION_RUNS = 25


def measure(maze: Maze, max_steps: int = 2000) -> Difficulty | None:
    """探索を回し、最速経路を引いて指標を作る。走り切れなければ None。"""
    learned = open_maze(maze.size)
    learned.set_goal(maze.goal_min, maze.goal_max)
    ex = Explorer(learned, cell=maze.start)
    for _ in range(max_steps):
        ex.observe(sense(maze, ex.cell, ex.facing),
                   sense_neighbors(maze, ex.cell, ex.facing))
        try:
            steps = ex.plan_leg(2)
        except Unreachable:
            return None
        if not steps:
            break
        for step in steps:
            ex.advance(step)
    else:
        return None
    path = shortest_path(maze, maze.start, start_facing=Direction.N, known=ex.known)
    if not path:
        return None
    legs = path_to_legs(path)
    counts = wall_counts(maze)
    # **実機と同じ設定**で復帰・最速の区間を引き、1 セッションぶんの連結動作を数える。
    # 区間長の上限 (#85) と束ねる本数で本数が変わるので、既定値をそのまま使う。
    mgr = RunManager(ex, chain_legs=CHAIN_LEGS)
    run_legs = [mgr.speed_legs(Direction.N), mgr.return_legs(ex.cell, ex.facing)]
    chained = sum(
        sum(1 for chunk in chunk_legs(one, CHAIN_LEGS) if len(chunk) > 1)
        for one in run_legs if one
    ) * (SESSION_RUNS - 1)          # 探索の 1 走を除く = 復帰 + 最速の回数
    return Difficulty(
        # 柱は**実際に立てる本数** (格子点 (N+1)^2 からゴール中央の 1 本を引いたもの)。
        # 格子点の数をそのまま出すと 8x8 が 81 本に見え、手持ちの照らし合わせが 1 本ずれる。
        walls=counts.total, posts=counts.posts,
        search_steps=ex.steps, path_cells=len(path) - 1, legs=len(legs),
        longest_leg=max(leg.cells for leg in legs),
        blind_x=longest_blind_run(maze, path, 0),
        blind_y=longest_blind_run(maze, path, 1),
        blind_cross=longest_blind_cross_run(maze, path),
        no_wall_cells=sum(
            1 for x in range(maze.size) for y in range(maze.size)
            if not any(maze.has_wall(x, y, d) for d in Direction)
        ),
        chained_motions=chained,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maze", nargs="+", required=True, help="元にする迷路 (ASCII)")
    p.add_argument("--size", type=int, default=8, help="切り出す大きさ (既定 8)")
    p.add_argument("--wall-budget", type=int, default=None, help="手持ちの壁の枚数")
    p.add_argument("--post-budget", type=int, default=None, help="手持ちの柱の本数")
    p.add_argument("--min-search", type=int, default=0,
                   help="探索の手数の下限。**短い解の迷路を除くのに要る** — "
                        "直交の補正なしだけで並べるとゴールのすぐ隣を数手動く迷路が"
                        "上位に来て、探索の検証にならない")
    p.add_argument("--min-cells", type=int, default=0, help="最速経路のセル数の下限")
    p.add_argument("--max-stranded", type=int, default=4,
                   help="到達できないセルの上限 (窓を切ると必ず数個は出る)")
    p.add_argument("--sort", choices=("difficulty", "exposure"), default="difficulty",
                   help="並べ替えの目的。difficulty = 実機でしか試せない要素が多い順 "
                        "(既定)、exposure = **1 セッションで採れるサンプルが多い順** "
                        "(#85: 中断の発生率を測るための盤面。曲がりが多く直進が短い方が"
                        "連結動作が増える)")
    p.add_argument("--top", type=int, default=10, help="表示する件数")
    p.add_argument("--out", default=None, help="--rank 位を書き出すファイル")
    p.add_argument("--rank", type=int, default=1, metavar="N",
                   help="--out で書き出す順位 (既定 1)。**1 位が床にあるとは限らない** — "
                        "指標が同点なら組みやすさや探索の長さで選ぶことがある")
    args = p.parse_args()
    setup_logging()

    walls_max, posts_max = pieces_needed(args.size)
    log.info("%dx%d を組むのに要るのは 壁 %d 枚 / 柱 %d 本 (どんなレイアウトでも)",
             args.size, args.size, walls_max, posts_max)
    if args.post_budget is not None and args.post_budget < posts_max:
        log.warning("柱が %d 本足りない (%d 本必要)", posts_max - args.post_budget, posts_max)

    found: list[tuple[Difficulty, str, Maze]] = []
    rejected = {"組めない": 0, "予算超過": 0, "走れない": 0}
    for path in args.maze:
        source = Maze.from_ascii(Path(path).read_text(encoding="utf-8"))
        for x0, y0, window in excerpts(source, args.size):
            # 切り出したままではゴールが競技の形にならない (元の迷路では普通のセル
            # だった場所をゴールと宣言するため)。入口の選び方ごとに候補を作る。
            for maze in goal_variants(window):
                # 組む迷路なので、健全性は「誤りが無い」だけでは足りない:
                #   壁の付かない柱 -> 公式規則違反。物理的にも自立しない
                #   到達できないセル -> 窓を切ると必ず出るが、多いと盤面が無駄になる
                if not check_maze(maze).ok or posts_without_wall(maze):
                    rejected["組めない"] += 1
                    continue
                stranded = maze.size ** 2 - len(reachable_cells(maze))
                if stranded > args.max_stranded:
                    rejected["孤立が多い"] = rejected.get("孤立が多い", 0) + 1
                    continue
                if (args.wall_budget is not None
                        and wall_counts(maze).total > args.wall_budget):
                    rejected["予算超過"] += 1
                    continue
                metrics = measure(maze)
                if metrics is None:
                    rejected["走れない"] += 1
                    continue
                if (metrics.search_steps < args.min_search
                        or metrics.path_cells < args.min_cells):
                    rejected["易しすぎ"] = rejected.get("易しすぎ", 0) + 1
                    continue
                found.append((metrics, f"{Path(path).stem} ({x0},{y0})", maze))

    log.info("候補 %d 件 (除外: %s)", len(found),
             " / ".join(f"{k} {v}" for k, v in rejected.items()))
    if not found:
        log.error("条件を満たす切り出しが無い。--size か --wall-budget を見直すこと。")
        return 1
    key = (lambda t: t[0].exposure) if args.sort == "exposure" else (lambda t: t[0].score)
    found.sort(key=key, reverse=True)
    for metrics, name, _maze in found[:args.top]:
        log.info("  %-28s %s", name, metrics.describe())
    if args.out:
        if not 1 <= args.rank <= len(found):
            log.error("--rank %d は候補 %d 件の範囲外", args.rank, len(found))
            return 1
        best, name, maze = found[args.rank - 1]
        Path(args.out).write_text(maze.to_ascii() + "\n", encoding="utf-8")
        log.info("%d 位 (%s) を %s に書き出した:\n%s",
                 args.rank, name, args.out, maze.to_ascii())
    return 0


if __name__ == "__main__":
    sys.exit(main())
