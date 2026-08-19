#!/usr/bin/env python3
"""大会迷路のスクリーンショットを ASCII 迷路に書き起こす (issue #84).

NTF の記録集は PDF で機械可読な形が無いので、迷路の図から**画像処理で**壁を読む。
16x16 は壁の置き場所が 544 箇所あり、目視で写すと必ず読み違えるため、目で読むのは
「機械が出した結果が元図と合っているか」の確認だけにする。

前提とする図の形 (記録集の図はどの年もこの形):

- 迷路の外周が**太い実線の正方形**。セルの境界は細い点線
- 壁は境界線上の太い実線
- ゴールの 2x2 セルに ``G`` / ``GOAL`` の文字、スタート (0,0) に矢印
- 図の外側に 0..15 の軸ラベル (読み取りには使わない)

読み方:

1. 「ほぼ全幅が黒い」行・列の両端 = 外枠。そこを 16 等分してセル境界を得る
2. 各境界のセル 1 つ分について、線の**中央 60%** の黒率を測る。壁は 1.0 近く、
   点線のグリッドは 0.5 未満に出るので閾値 0.6 で分ける
   (**分離の余裕は毎回表示する。** 詰まっていたらその図は信用しない)
3. セルの内側 (中央 50%) にインクがあるセル = 文字のあるセル。(0,0) はスタートの
   矢印なので除き、残りをゴール領域とする

出力は :meth:`krilly.solver.maze.Maze.to_ascii` 形式なので、そのまま
``scripts/maze_sim.py`` に渡せる。

例:
    python -m scripts.maze_from_image shots/*.png --out-dir mazes/contest
    python -m scripts.maze_from_image shots/2024_japan.png --ascii   # 確認用に表示
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from krilly.logging_config import get_logger, setup_logging
from krilly.sim import check_maze
from krilly.sim.check import posts_without_wall
from krilly.solver.maze import Direction, Maze

log = get_logger(__name__)

# しきい値の探索で使う減点の重み。31 面を総当たりして決めた。
#: 四方を囲まれたセル。**大会迷路に実在する**ので (2020 年の (11,6) を拡大して確認)、
#: 重くすると正しい読み取りを弾いてしまう。
BOXED_PENALTY = 1
#: 壁の接していない柱。壁を読み落とすと増える。ただし**大会迷路にも実在する**ので
#: (2013 年エキスパート予選には 27 本あり、公式解の歩数と一致することを確認した)
#: 強くは効かせない。
BARE_POST_PENALTY = 5
#: 太さの分布の「谷の広さ」への加点。谷が広いほど、そこが本当の境界である見込みが高い。
#: これが無いと、ごく狭い谷で切って壁を数枚余分に読む解が選ばれてしまう。
GAP_BONUS = 60


class ReadError(RuntimeError):
    """図の形が想定と違って読み取れない。"""


def _border(dark: np.ndarray, axis: int) -> tuple[int, int]:
    """外枠の位置 (「ほぼ全幅が黒い」行/列の最初と最後)。"""
    counts = dark.sum(axis=axis)
    hits = np.where(counts >= 0.85 * counts.max())[0]
    if len(hits) < 2:
        raise ReadError("外枠が見つからない")
    return int(hits.min()), int(hits.max())


def _fit_lattice(counts: np.ndarray, origin: int, span: int, size: int) -> np.ndarray:
    """グリッド線の実位置 (size+1 本) を求める。

    セル幅は小数 (19.44px など) なので、外枠から等分して計算した位置は 16 セル分で
    1-2px ずれ、細い線の厚みを測る窓が隣にずれてしまう。そこで**実際に線が見える位置**
    を検出し、そこへ等間隔の格子 (原点 + ピッチ) を最小二乗で当てはめる。

    当てはめる理由は、壁がほとんど無い境界線は暗さが足りず検出から漏れるため
    (実測で 17 本中 15-16 本しか見つからない図があった)。モデルにすれば漏れた線も
    埋まり、かつ全体としてはサブピクセル精度になる。
    """
    thr = 0.25 * counts.max()
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, v in enumerate(counts):
        if v >= thr:
            cur.append(i)
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    centres = np.array([sum(i * counts[i] for i in gp) / sum(counts[i] for i in gp)
                        for gp in groups])
    if len(centres) < 3:
        raise ReadError(f"グリッド線が {len(centres)} 本しか見つからない")
    pitch0 = span / size
    idx = np.round((centres - origin) / pitch0)
    # centre ≈ offset + idx * pitch の最小二乗
    A = np.stack([np.ones_like(idx), idx], axis=1)
    offset, pitch = np.linalg.lstsq(A, centres, rcond=None)[0]
    resid = np.abs(centres - (offset + idx * pitch)).max()
    if resid > 0.25 * pitch:
        raise ReadError(f"グリッドが等間隔でない (残差 {resid:.1f}px / ピッチ {pitch:.1f}px)")
    return offset + np.arange(size + 1) * pitch


def _thickness(dark: np.ndarray, size: int,
               xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """各壁位置の線の**平均太さ** [px] を測る (壁ほど大きい)。

    線に直交する向きの黒画素数を、線に沿って**平均**する。これは
    「太さ x 連続率」に等しく、図によって効く手がかりが違うのを 1 つの数で吸収する:

    - **黒さ**だけでは分けられない。記録集の図はセル境界も点線で描かれており、
      解像度が低いと点が繋がって実線と見分けが付かない (2021/2023 は全 544 箇所が
      「黒い」と出た)
    - **太さ**だけでも分けられない。2010-2011 年の図は点線も実線と同じ 2px 幅で、
      違うのは**途切れているかどうか**だけ (中央値で測ると壁を 23-31 箇所読み落とした)
    - 平均太さなら、2023 年は 1px 対 3px、2010 年は 1px (2px x 50% の点線) 対 2px で、
      どちらも分かれる

    ``xs`` / ``ys`` は :func:`_fit_lattice` が出したグリッド線の実位置
    (``ys`` は画像の行なので北ほど小さい)。
    """
    cw, ch = float(np.diff(xs).mean()), float(np.diff(ys).mean())
    half = max(1, int(round(min(ch, cw) * 0.15)))
    out = np.zeros((2, size + 1, size + 1))
    for x in range(size):
        c0, c1 = int(xs[x] + 0.25 * cw), int(xs[x + 1] - 0.25 * cw)
        for k in range(size + 1):
            r = int(round(ys[size - k]))          # k=0 が南 = 画像の下端
            sub = dark[max(0, r - half):r + half + 1, c0:c1]
            out[0][x][k] = sub.sum(axis=0).mean() if sub.size else 0.0
    for j in range(size + 1):
        c = int(round(xs[j]))
        for y in range(size):
            r0, r1 = int(ys[size - y - 1] + 0.25 * ch), int(ys[size - y] - 0.25 * ch)
            sub = dark[r0:r1, max(0, c - half):c + half + 1]
            out[1][j][y] = sub.sum(axis=1).mean() if sub.size else 0.0
    return out


def _build(thick: np.ndarray, size: int, threshold: float) -> Maze:
    """厚みのしきい値から迷路を組み立てる。"""
    maze = Maze(size)
    for x in range(size):
        for k in range(size + 1):
            if thick[0][x][k] >= threshold:
                if k < size:
                    maze.set_wall(x, k, Direction.S)
                else:
                    maze.set_wall(x, size - 1, Direction.N)
    for j in range(size + 1):
        for y in range(size):
            if thick[1][j][y] >= threshold:
                if j < size:
                    maze.set_wall(j, y, Direction.W)
                else:
                    maze.set_wall(size - 1, y, Direction.E)
    return maze


def _score(maze: Maze) -> tuple[bool, int]:
    """(そもそも大会迷路として成立するか, 読み取りの粗さ)。粗さは小さいほど良い。

    **実際の大会迷路は必ず外周が閉じていて、必ずスタートからゴールへ行ける。**
    この 2 つは例外が無いので採点ではなく**足切り**にする (これを入れないと、
    しきい値を下げて壁を過剰に読んだ解が「裸の柱が少ない」という理由で選ばれてしまう)。

    残りは減点で測る。囲まれたセルは実在するので軽く、壁の接していない柱は
    ゴール中央の 1 本以外は規則違反なので重く見る (:data:`BARE_POST_PENALTY`)。
    """
    from krilly.sim.check import posts_without_wall, reachable_cells, wall_counts
    n = maze.size
    reachable = reachable_cells(maze)
    counts = wall_counts(maze)
    # 全セル到達可能なら内壁は (N-1)^2 枚以下 (#77)。囲まれたセルの分だけ上限が上がるので
    # 余裕を持たせるが、大きく超えるのは点線を壁と読んでいる証拠。
    viable = (counts.outer == 4 * n
              and counts.inner <= counts.inner_max + 30
              and any(g in reachable for g in maze.goal_cells()))
    unreachable = n * n - len(reachable)
    boxed = sum(all(maze.has_wall(x, y, d) for d in Direction)
                for x in range(n) for y in range(n))
    bare = len(posts_without_wall(maze))
    return viable, unreachable + BOXED_PENALTY * boxed + BARE_POST_PENALTY * bare


def read_maze(path: Path, size: int = 16):
    """画像から迷路を読む。(迷路, しきい値, 二山の間隔, 元画像, xs, ys) を返す。"""
    im = cv2.imread(str(path))
    if im is None:
        raise ReadError(f"画像を読めない: {path}")
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    dark = (gray < 128).astype(np.uint8)

    top, bottom = _border(dark, 1)
    left, right = _border(dark, 0)
    if not 0.9 < (right - left) / (bottom - top) < 1.1:
        raise ReadError(f"外枠が正方形でない ({right - left}x{bottom - top})")
    inner = dark[top:bottom + 1, left:right + 1]
    xs = left + _fit_lattice(inner.sum(axis=0), 0, right - left, size)
    ys = top + _fit_lattice(inner.sum(axis=1), 0, bottom - top, size)

    thick = _thickness(dark, size, xs, ys)
    vals = np.concatenate([thick[0][:size, :size + 1].ravel(),
                           thick[1][:size + 1, :size].ravel()])
    # 値が連続なので、隣り合う観測値の中点をしきい値の候補にする (二山の谷を総当たり)
    levels = sorted(set(np.round(vals, 3)))
    if len(levels) < 2:
        raise ReadError("線の太さが 1 種類しかない — 図の形が想定と違う")

    # 隣り合う厚みの中点をしきい値の候補にし、最もまともな迷路になるものを採る。
    best = None
    for lo, hi in zip(levels, levels[1:]):
        t = (lo + hi) / 2
        maze = _build(thick, size, t)
        viable, score = _score(maze)
        if not viable:
            continue                      # 外周が欠ける / ゴールへ行けないしきい値は論外
        score -= GAP_BONUS * (hi - lo)
        if best is None or score < best[0]:
            best = (score, t, hi - lo, maze)
    if best is None:
        raise ReadError("成立する迷路になるしきい値が無い — 線の太さが分離できていない")
    _score_v, threshold, gap, maze = best
    maze.start = (0, 0)
    _read_goal(dark, maze, xs, ys, size)
    return maze, threshold, gap, im, xs, ys


def _read_goal(dark, maze: Maze, xs: np.ndarray, ys: np.ndarray, size: int) -> None:
    """セルの内側に文字があるセルをゴールとみなす (スタートの矢印は除く)。"""
    marked = []
    for x in range(size):
        for y in range(size):
            c0, c1 = int(xs[x]), int(xs[x + 1])
            r0, r1 = int(ys[size - y - 1]), int(ys[size - y])
            w, h = c1 - c0, r1 - r0
            sub = dark[r0 + h // 4:r1 - h // 4, c0 + w // 4:c1 - w // 4]
            if sub.size and sub.mean() > 0.08:
                marked.append((x, y))
    marked = [c for c in marked if c != (0, 0)]          # スタートの矢印
    if not marked:
        log.warning("ゴールの文字が見つからない -> 中央 2x2 のまま")
        return
    def bounds(cells):
        lo = (min(c[0] for c in cells), min(c[1] for c in cells))
        hi = (max(c[0] for c in cells), max(c[1] for c in cells))
        return lo, hi, (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1)

    lo, hi, span = bounds(marked)
    if span != len(marked):
        # ゴールの文字は必ず隣接して並ぶ。図の隅の注記などを孤立した印として落とす。
        cluster = [c for c in marked
                   if any((c[0] + dx, c[1] + dy) in marked
                          for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
        if not cluster:
            raise ReadError(f"ゴールの文字が並んでいない (印のセル {sorted(marked)})")
        lo, hi, span = bounds(cluster)
        if span != len(cluster):
            raise ReadError(f"ゴールが矩形でない (印のセル {sorted(cluster)})")
    if span < 4 and size % 2 == 0:
        # クラシック競技のゴールは**常に中央 2x2**。図が代表して 1 セルにしか
        # 記号を書いていないことがあるので補う (2013 年エキスパート予選は 'G' が
        # 1 個だけで、中央 2x2 として解くと公式解の 52 歩に一致した)。
        c = size // 2
        log.info("ゴールの記号が %d セルだけ -> 中央 2x2 に補正", span)
        maze.set_goal((c - 1, c - 1), (c, c))
        return
    maze.set_goal(lo, hi)


def overlay(im, maze: Maze, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """読み取った壁を元画像に重ねる (目視での突き合わせ用)。

    #84 の手順 2「元画像と目視で突き合わせる」を速く確実にするためのもの。
    ASCII を睨むより、元図の上に読み取り結果を描いた方が食い違いは一目で分かる。
    """
    out = im.copy()
    n = maze.size
    for x in range(n):
        for k in range(n + 1):
            cell_y = k if k < n else n - 1
            d = Direction.S if k < n else Direction.N
            if maze.has_wall(x, cell_y, d):
                r = int(round(ys[n - k]))
                cv2.line(out, (int(xs[x]), r), (int(xs[x + 1]), r), (0, 200, 0), 2)
    for j in range(n + 1):
        for y in range(n):
            cell_x = j if j < n else n - 1
            d = Direction.W if j < n else Direction.E
            if maze.has_wall(cell_x, y, d):
                c = int(round(xs[j]))
                cv2.line(out, (c, int(ys[n - y - 1])), (c, int(ys[n - y])), (0, 0, 220), 2)
    for x, y in maze.goal_cells():
        cv2.circle(out, (int((xs[x] + xs[x + 1]) / 2), int((ys[n - y - 1] + ys[n - y]) / 2)),
                   3, (255, 0, 0), -1)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", help="迷路のスクリーンショット")
    p.add_argument("--size", type=int, default=16, help="1 辺のセル数 (既定 16)")
    p.add_argument("--out-dir", default=None, help="ASCII の書き出し先")
    p.add_argument("--ascii", action="store_true", help="読み取った迷路を表示する")
    p.add_argument("--overlay-dir", default=None,
                   help="読み取り結果を元画像に重ねた png の書き出し先 (目視確認用)")
    args = p.parse_args()
    setup_logging()

    out = Path(args.out_dir) if args.out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for path in sorted(Path(s) for s in args.images):
        try:
            maze, threshold, gap, im, xs, ys = read_maze(path, args.size)
        except ReadError as exc:
            log.error("%-32s 読み取り失敗: %s", path.name, exc)
            bad += 1
            continue
        report = check_maze(maze)
        # **四方を囲まれたセルは大会迷路に実在する** (2020 年の (11,6) を拡大して確認)。
        # 到達できないセルもその帰結なので、どちらも読み取り誤りの証拠にはならない。
        # 一方**壁の接していない柱**は、ゴール中央の 1 本を除けば公式規則で許されない
        # ので、2 本以上あれば壁を読み落としている。これが一番鋭い判定になる。
        bare = len(posts_without_wall(maze))
        # **裸の柱は少ないのが普通だが、実在もする** (2013 年エキスパート予選は 27 本で、
        # 公式解の歩数と一致することを確認済み)。多いときは読み落としの疑いがあるので
        # 目視確認に回すが、誤りと断定はしない。
        ok = report.ok and bare <= 1
        log.info("%-32s %s 太さ境界 %.2fpx (谷幅 %.2f) 裸の柱 %d ゴール %s..%s",
                 path.name, "OK " if ok else "NG ", threshold, gap, bare,
                 maze.goal_min, maze.goal_max)
        for line in report.describe().splitlines()[1:]:
            log.info("%34s%s", "", line.strip())
        if not ok:
            bad += 1
        if args.ascii:
            log.info("%s:\n%s", path.stem, maze.to_ascii())
        if out:
            (out / f"{path.stem}.txt").write_text(maze.to_ascii() + "\n", encoding="utf-8")
        if args.overlay_dir:
            od = Path(args.overlay_dir)
            od.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(od / f"{path.stem}.png"), overlay(im, maze, xs, ys))

    log.info("---")
    log.info("%d 面中 %d 面が要確認", len(args.images), bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
