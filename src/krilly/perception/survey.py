"""校正データ (labels.csv) の記録と解析 (issue #78).

**目的は「会場で 1 回走らせるだけで、壁判定にどれだけ余裕が残っているかを数値で
出す」こと。** 照明が変われば赤マスクの通り方が変わり、赤割合は下がる。どこまで
下がったら壁を見落とすのかは、しきい値との**距離**を測らないと分からない。

これまでその数字 (CLAUDE.md の「113 frames / 452 labels、front 0.000→0.302 …」) は
その場限りの手作業集計で、コードとして残っていなかった。ここがその置き場所。

記録の形式は ``scripts/survey_shot.py`` の CSV と同じ 9 列:

    file,x,y,facing,edge,wall,fraction,off_fwd_mm,off_left_mm

``edge`` は**スロット名**なので、隣セルを読むスロット (``"left:front"``, #89) も
同じ表に入る。``x,y`` は撮影したときの**機体の**セルで、隣セルのスロットでは
ラベルの対象セルと違う — 対象は :func:`slot_wall` が解決する。

ラベル (``wall``) の貼り方が本モジュールの肝で、2 通りある:

- **走行後の地図から貼る** (既定): 探索が終われば地図は確定しているので、走行中に
  撮ったフレームへ後からラベルを貼れる。追加の走行はゼロで、判定はいつもどおり
  行われるから、``wall_survey`` の鶏と卵 (カメラの誤測定が位置補正へ入り、
  再センタリングが校正中の機体を壁へ運ぶ) も起きない
- **既知形状の迷路から貼る**: ``mazes/practice5.txt`` のように形が分かっている
  迷路なら、そちらを正解にできる

前者には**限界がある**。地図はカメラ自身の出力なので、ある壁を毎回見落としていれば
ラベルも「壁なし」になり、誤判定が 0 と出る。だから分離の余裕 (:attr:`EdgeStats.ratio`)
の方を主に読むこと — 「全部当たった」より「一番弱い壁がしきい値の何倍だったか」が
効く数字である (#23 実測: 古い壁は 5-6 倍、新しいロットの壁は 1.4 倍)。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from krilly.perception.wall_detect import BODY_DIRS, maze_direction_for
from krilly.solver.maze import Direction

#: survey_shot / search_run / speed_run が共通で書く列。
CSV_FIELDS = ["file", "x", "y", "facing", "edge", "wall", "fraction",
              "off_fwd_mm", "off_left_mm"]


@dataclass(frozen=True)
class SurveyRow:
    """labels.csv の 1 行 = 「1 フレームの 1 スロット」の測定と正解。"""

    file: str
    x: int
    y: int
    facing: str
    edge: str                     # スロット名 (自セルの辺 or "left:front" 等)
    wall: bool                    # 正解 (地図から貼ったラベル)
    fraction: float               # 測定した赤割合
    off_fwd_mm: float | None = None
    off_left_mm: float | None = None

    @property
    def cell(self) -> tuple[int, int]:
        """撮影したときの**機体の**セル (ラベル対象のセルではない)。"""
        return (self.x, self.y)

    def to_csv(self) -> dict[str, str]:
        def mm(value: float | None) -> str:
            return "" if value is None else f"{value:.1f}"

        return {
            "file": self.file, "x": str(self.x), "y": str(self.y),
            "facing": self.facing, "edge": self.edge, "wall": str(int(self.wall)),
            "fraction": f"{self.fraction:.4f}",
            "off_fwd_mm": mm(self.off_fwd_mm), "off_left_mm": mm(self.off_left_mm),
        }

    @classmethod
    def from_csv(cls, row: dict[str, str]) -> "SurveyRow":
        def mm(text: str | None) -> float | None:
            return float(text) if text not in (None, "") else None

        return cls(
            file=row["file"], x=int(row["x"]), y=int(row["y"]),
            facing=row["facing"], edge=row["edge"],
            wall=bool(int(row["wall"])), fraction=float(row["fraction"]),
            off_fwd_mm=mm(row.get("off_fwd_mm")), off_left_mm=mm(row.get("off_left_mm")),
        )


def read_rows(path: str | Path) -> list[SurveyRow]:
    """labels.csv を読む。"""
    with open(path, newline="", encoding="utf-8") as f:
        return [SurveyRow.from_csv(r) for r in csv.DictReader(f)]


def write_rows(path: str | Path, rows: Iterable[SurveyRow], append: bool = False) -> int:
    """labels.csv を書く (``append`` なら追記し、ヘッダは無いときだけ出す)。"""
    path = Path(path)
    header = not (append and path.exists() and path.stat().st_size > 0)
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if header:
            writer.writeheader()
        count = 0
        for row in rows:
            writer.writerow(row.to_csv())
            count += 1
    return count


def slot_wall(slot: str, cell: tuple[int, int], facing: Direction,
              maze) -> tuple[tuple[int, int], Direction] | None:
    """スロットが見ている壁 = (セル, 迷路方角)。迷路の外を見ていれば None。

    自セルのスロット (``"front"``) はそのセルの辺、隣セルのスロット
    (``"left:front"``) は**左隣のセル**の前方の辺。どちらも機体の向き ``facing``
    で迷路方角へ写す (:func:`krilly.perception.wall_detect.maze_direction_for`)。

    None を返すのは「隣セルが迷路の外」のとき。そこには壁も柱も無いので正解ラベルが
    存在しない — が、**カメラには何かが写りうる** (#89 実機: 迷路の東に赤い定規袋を
    置いていたら ``right:right`` が 0.97 と出た)。ラベルの付けようがない行なので
    落とすが、落ちた数は報告すること。
    """
    side, _, edge = slot.rpartition(":")
    target = cell
    if side:
        target = maze.neighbor(*cell, maze_direction_for(side, facing))
        if not maze.in_bounds(*target):
            return None
    return (target, maze_direction_for(edge, facing))


def label_rows(file: str, cell: tuple[int, int], facing: Direction,
               measured: dict[str, tuple[float, int, bool]], maze,
               off_fwd_mm: float | None = None,
               off_left_mm: float | None = None) -> list[SurveyRow]:
    """1 フレームの測定に、地図から正解ラベルを貼って行にする。

    ``measured`` は :meth:`krilly.perception.wall_detect.WallDetector.measure` の
    戻り値そのまま (スロット名 -> (赤割合, 帯のずれ, 飽和))。``maze`` は正解に使う
    迷路 — 走行後に確定した地図でも、既知形状の迷路でもよい。
    """
    rows = []
    for slot in sorted(measured, key=_slot_order):
        target = slot_wall(slot, cell, facing, maze)
        if target is None:
            continue                          # 迷路の外を見ているスロット
        (tx, ty), direction = target
        rows.append(SurveyRow(
            file=file, x=cell[0], y=cell[1], facing=facing.name, edge=slot,
            wall=maze.has_wall(tx, ty, direction), fraction=measured[slot][0],
            off_fwd_mm=off_fwd_mm, off_left_mm=off_left_mm,
        ))
    return rows


def _slot_order(slot: str) -> tuple[int, str]:
    """自セルの 4 辺を front/back/left/right の順に、隣セルはその後ろに並べる。"""
    if slot in BODY_DIRS:
        return (0, f"{BODY_DIRS.index(slot):02d}")
    return (1, slot)


@dataclass
class EdgeStats:
    """1 スロットぶんの、赤割合の分布と「しきい値までの距離」。"""

    edge: str
    threshold: float
    walls: list[SurveyRow]        # 正解が「壁あり」の行 (赤割合の昇順)
    clears: list[SurveyRow]       # 正解が「壁なし」の行 (赤割合の昇順)

    @property
    def misses(self) -> list[SurveyRow]:
        """見落とし (壁があるのに赤割合がしきい値未満)。**衝突に直結する。**"""
        return [r for r in self.walls if r.fraction < self.threshold]

    @property
    def false_walls(self) -> list[SurveyRow]:
        """誤検出 (壁が無いのにしきい値以上)。回り道で済むので見落としより安い。"""
        return [r for r in self.clears if r.fraction >= self.threshold]

    @property
    def weakest_wall(self) -> SurveyRow | None:
        return self.walls[0] if self.walls else None

    @property
    def strongest_clear(self) -> SurveyRow | None:
        return self.clears[-1] if self.clears else None

    @property
    def separation(self) -> float | None:
        """壁の最小 − 壁なしの最大。負なら 2 群が重なっていて、どんなしきい値でも
        分けられない。"""
        if not self.walls or not self.clears:
            return None
        return self.walls[0].fraction - self.clears[-1].fraction

    @property
    def ratio(self) -> float | None:
        """一番弱い壁がしきい値の何倍か。**照明ロバスト性はこの数字で読む。**

        1 に近いほど、少しの劣化で見落としに変わる (#23 実測: 古い壁 5-6 倍に対し、
        新しいロットの壁は 0.11/0.08 = 1.4 倍しかなかった)。
        """
        if not self.walls or self.threshold <= 0:
            return None
        return self.walls[0].fraction / self.threshold

    @property
    def headroom(self) -> float | None:
        """壁なし側の余裕 (しきい値 − 壁なしの最大)。負なら誤検出が出ている。"""
        return None if not self.clears else self.threshold - self.clears[-1].fraction

    def quantiles(self, wall: bool) -> tuple[float, float, float] | None:
        """(最小, 中央, 最大)。該当する行が無ければ None。"""
        rows = self.walls if wall else self.clears
        if not rows:
            return None
        values = [r.fraction for r in rows]
        return (values[0], values[len(values) // 2], values[-1])


def summarize(rows: Sequence[SurveyRow],
              threshold_for: Callable[[str], float]) -> dict[str, EdgeStats]:
    """スロットごとに赤割合を wall=0/1 で層別する。

    ``threshold_for`` は ``WallDetectorConfig.threshold_for`` をそのまま渡せる
    (辺別のしきい値がある以上、分離の評価も辺別にしなければ意味がない)。
    """
    by_edge: dict[str, list[SurveyRow]] = {}
    for row in rows:
        by_edge.setdefault(row.edge, []).append(row)
    out: dict[str, EdgeStats] = {}
    for edge in sorted(by_edge, key=_slot_order):
        group = by_edge[edge]
        out[edge] = EdgeStats(
            edge=edge,
            threshold=threshold_for(edge),
            walls=sorted((r for r in group if r.wall), key=lambda r: r.fraction),
            clears=sorted((r for r in group if not r.wall), key=lambda r: r.fraction),
        )
    return out


def format_report(stats: dict[str, EdgeStats], sources: Sequence[str] = ()) -> list[str]:
    """:func:`summarize` の結果を人が読む表にする (ログ 1 行ずつ)。"""
    lines: list[str] = []
    if sources:
        lines.append("データ: " + ", ".join(sources))
    total = sum(len(s.walls) + len(s.clears) for s in stats.values())
    frames = len({r.file for s in stats.values() for r in s.walls + s.clears})
    lines.append(f"フレーム {frames} 枚 / ラベル {total} 個")
    lines.append("スロット      しきい値 |     壁あり 最小/中央/最大 |"
                 "     壁なし 最小/中央/最大 |  分離  余裕倍率 見落とし 誤検出")

    def band(values: tuple[float, float, float] | None, count: int) -> str:
        cell = ("   --    --    -- " if values is None
                else "%5.3f %5.3f %5.3f" % values)
        return "%3d枚 %s" % (count, cell)

    for edge, s in stats.items():
        lines.append(
            "%-13s %6.3f  | %s | %s | %s %s %6d %6d" % (
                edge, s.threshold,
                band(s.quantiles(True), len(s.walls)),
                band(s.quantiles(False), len(s.clears)),
                "  --  " if s.separation is None else "%+6.3f" % s.separation,
                "   -- " if s.ratio is None else "%5.1f倍" % s.ratio,
                len(s.misses), len(s.false_walls),
            )
        )
    misses = [r for s in stats.values() for r in s.misses]
    false_walls = [r for s in stats.values() for r in s.false_walls]
    lines.append("見落とし %d 個 / 誤検出 %d 個 (見落としは衝突、誤検出は回り道)"
                 % (len(misses), len(false_walls)))
    for row in misses:
        lines.append("  見落とし: %s %s セル(%d,%d) 赤割合 %.3f" %
                     (row.file, row.edge, row.x, row.y, row.fraction))
    for row in false_walls:
        lines.append("  誤検出  : %s %s セル(%d,%d) 赤割合 %.3f" %
                     (row.file, row.edge, row.x, row.y, row.fraction))
    # **一番弱い壁が、次に run を終わらせる壁である。** どのフレームか言っておく。
    weak = sorted((s.weakest_wall for s in stats.values() if s.weakest_wall),
                  key=lambda r: r.fraction)[:3]
    for row in weak:
        lines.append("  最も弱い壁: %s %s セル(%d,%d) 赤割合 %.3f" %
                     (row.file, row.edge, row.x, row.y, row.fraction))
    return lines


def confusion(stats: dict[str, EdgeStats]) -> tuple[int, int, int, int]:
    """(壁を壁, 壁を見落とし, 開を壁と誤検出, 開を開) の 4 数。"""
    walls = sum(len(s.walls) for s in stats.values())
    clears = sum(len(s.clears) for s in stats.values())
    misses = sum(len(s.misses) for s in stats.values())
    false_walls = sum(len(s.false_walls) for s in stats.values())
    return (walls - misses, misses, false_walls, clears - false_walls)


def format_confusion(stats: dict[str, EdgeStats]) -> list[str]:
    """混同行列。**2 種類の誤りは等価ではない**ので、左右で意味を書いておく。"""
    hit, miss, false_wall, reject = confusion(stats)
    return [
        "混同行列 (行=正解 / 列=判定)      判定:壁  判定:なし",
        "  正解: 壁あり                  %7d  %7d  <- 見落とし = 衝突" % (hit, miss),
        "  正解: 壁なし                  %7d  %7d  <- 誤検出 = 回り道" % (false_wall, reject),
    ]


def compare(before: dict[str, EdgeStats], after: dict[str, EdgeStats],
            labels: tuple[str, str] = ("前", "後")) -> list[str]:
    """2 回の走行 (例: 照明を変えた前後) を並べ、余裕がどれだけ縮んだかを出す。

    見るのは**一番弱い壁**と**一番強い壁なし**の 2 つだけ。分布の真ん中がどう動こうと、
    run を終わらせるのは端の方だから。
    """
    lines = ["スロット      | %s 弱壁/強開/分離      | %s 弱壁/強開/分離      | 弱壁の変化"
             % (labels[0], labels[1])]

    def pair(s: EdgeStats | None) -> str:
        if s is None:
            return "  --    --      --   "
        weak = "%5.3f" % s.walls[0].fraction if s.walls else "  -- "
        clear = "%5.3f" % s.clears[-1].fraction if s.clears else "  -- "
        sep = "%+6.3f" % s.separation if s.separation is not None else "  --  "
        return f"{weak} {clear} {sep}"

    for edge in sorted(set(before) | set(after), key=_slot_order):
        a, b = before.get(edge), after.get(edge)
        delta = "  --  "
        if a is not None and b is not None and a.walls and b.walls:
            delta = "%+6.3f" % (b.walls[0].fraction - a.walls[0].fraction)
        lines.append("%-13s | %s | %s | %s" % (edge, pair(a), pair(b), delta))
    return lines


@dataclass(frozen=True)
class FrameRecord:
    """走行中に撮った 1 フレームの記録 (ラベルはまだ貼らない)。

    探索中は地図が未確定なので、正解を貼れるのは**走り終わってから**。走行中は
    「どのセルで、どの向きで、何が測れたか」だけを溜めておき、最後に
    :func:`label_run` でまとめてラベル付けする。
    """

    file: str
    cell: tuple[int, int]
    facing: Direction
    measured: dict[str, tuple[float, int, bool]]
    off_fwd_mm: float | None = None
    off_left_mm: float | None = None


def label_run(records: Sequence[FrameRecord], maze) -> tuple[list[SurveyRow], int]:
    """溜めたフレーム記録に地図から正解を貼る。戻り値は (行, 迷路の外で捨てた数)。"""
    rows: list[SurveyRow] = []
    skipped = 0
    for rec in records:
        labelled = label_rows(rec.file, rec.cell, rec.facing, rec.measured, maze,
                              rec.off_fwd_mm, rec.off_left_mm)
        skipped += len(rec.measured) - len(labelled)
        rows.extend(labelled)
    return (rows, skipped)
