"""中断した走行を、その場で姿勢を作り直して続行する (issue #113)。

``--max-heading-residual`` のガードが落ちたとき、機体は**必ず停止している**
(残差の判定は整定後に行う) のでカメラが使える。失われるものは 3 つのうち 1 つだけ:

- **方位** — :func:`~krilly.perception.axis_yaw.median_axis_yaw` で厳密に測れる。
  迷路走行中の機体の向きは常に 90° の倍数なので、mod 90° の折り返しで足りる。
- **セル内位置** — :func:`~krilly.perception.cell_pose.cell_offset` で ±2-3mm。
  ただし**壁のある軸だけ**。
- **どのセルに居るか** — **幾何では回復できない。** 赤帯の見え方は 180mm 周期なので、
  1 セルずれていても同じに見える。

最後のものだけが別の手段を要する。それが :func:`verify_cell` で、**壁のパターン**を
学習済みマップと照合する。全セルで一意である必要はない — 距離誤差で起こりうるのは
進行軸 ±1 セルのずれなので、**隣 2 つと区別できれば足りる**。

**ガードの向きが正常運転とは逆になる。** :data:`~krilly.localization.grid.MAX_HEADING_CORRECTION_RAD`
(5°) は「正常運転でこれを超える読みは誤測定」という前提に立っている。回復時はその前提が
反転する — 姿勢が狂っていることは既に分かっているので、**補正は広く許し、代わりに測定の
質で門を作る**。path check の 0.20 対 壁判定の 0.08 と同じ非対称性で、**同じ数字を別の
前提で使い回してはいけない**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from krilly.solver.maze import Direction, Maze

#: 回復時に受け入れる方位補正の上限 [rad]。正常運転の 5° より広い (上の docstring)。
RECOVERY_MAX_HEADING_RAD = math.radians(20.0)

#: これを超えて回っていたら回復を試みない [rad]。
#:
#: ``axis_yaw`` は **mod 90°** なので、45° を超えて回っていると**自信を持って間違った軸へ
#: スナップする** (50° を -40° と読む)。その姿勢で走れば壁へ真っ直ぐ向かう。45° に余裕を
#: 持たせて 30° で切る。そもそも 30° 回っていれば壁は倒れていて走行は既に終わっている。
RECOVERY_ABANDON_RAD = math.radians(30.0)

#: 「1 セルずれていない」と言い切れる進行方向の残差の上限 [m]。
#:
#: 地図が未完成で壁の照合ができないとき (探索中) の代わりの証拠。セルの間隔は 180mm
#: なので、残差がこの程度なら 1 セル飛ばしてはいない。実機の中断 3 件の進行方向残差は
#: 10-14mm だったので、**そもそも 1 セルずれは今のところ observed されていない** —
#: ±1 の探索は安全網であって、期待されるケースではない。
CELL_SLIP_MAX_M = 0.030

#: 軸角のフレーム間ばらつきの上限 [rad]。**粗い「当てはめが崩れた」検出器**。
#:
#: 最初 0.5° にしたのは #88 の「960x720 でのばらつきは 0.11-0.26°」から取ったためだが、
#: **あれは機体を 1 セルに据え置いて測った値**で、実走には合っていなかった。実機で
#: リカバリが 14 回動いたときのばらつきは **0.08-0.45°** — 停止位置がセルごとに変わり、
#: 壁が少ないセルでは線分が短く・少なくなってばらつく (位置が `測れず` になった回ほど
#: 大きい傾向)。上端 0.45° は 0.5° のすぐ下で余裕が無く、**15 回目が 1.03° で落ちて
#: セッションが終わった**。実走の上端の 3 倍を取って 1.5° にする。
#:
#: **そもそもこれは弱い指標である。** 本当に危険なのは「自信を持った誤測定」で、それは
#: ばらつきに出ない — `axis_yaw` の誤測定が **-13.27° の「補正」**を当てて走行を破壊した
#: 件と、#89 の**赤い定規袋**が +5.4〜7.4° を安定して出した件は、どちらもばらつきは
#: 小さかったはず。実際に守っているのは :data:`RECOVERY_MAX_HEADING_RAD` の 20° (被害の
#: 大きさを縛る)、**壁パターンの照合** (独立な証拠 — 壁 ROI は位置ベースなので軸角の
#: 誤りに汚染されない)、そして**次の移動の path check** の 3 つ。ここを締めても
#: 危険な失敗は止まらず、正常な回復を断るだけになる。
RECOVERY_MAX_YAW_SPREAD_RAD = math.radians(1.5)


@dataclass(frozen=True)
class CellVerdict:
    """セル照合の結果。"""

    cell: tuple[int, int] | None    # 確定したセル (None = 確定できなかった)
    shift: int                      # 進行軸に何セルずれていたか (0 = 思っていたとおり)
    reason: str                     # 人間向けの要約 (ログに出す)
    #: True なら「矛盾した」のではなく「**照合材料が足りなかった**」。
    #: 探索中は地図が未完成なので普通に起きる。呼び出し側は他の証拠で 1 セルずれを
    #: 排除できれば続行してよい (矛盾した場合と混ぜてはいけない)。
    unverifiable: bool = False

    @property
    def ok(self) -> bool:
        return self.cell is not None


def _comparable(maze: Maze, cell: tuple[int, int],
                observed: dict[Direction, bool],
                known_edges: dict[tuple[int, int], set[Direction]] | None
                ) -> list[Direction]:
    """``cell`` について、照合に使える辺 (地図が実際に知っている辺) を返す。

    **`Maze` に壁の三値は無く、「未知」を担うのは観測済み集合の方**なので、
    `has_wall` が False を返しても「壁が無い」とは限らない — まだ見ていないだけ
    かもしれない。`flood_fill` はそれを楽観的に「開いている」と読んでよいが
    (見に行くため)、照合でそれをやると**探索中は必ず不一致になる**。
    実際にそうなった (#113 の実機 1 本目: 探索 2 手目で「迷子」と判定して停止)。

    ``known_edges`` が None なら全辺を既知として扱う (地図が完成している最速・
    復帰ランはこれでよい)。
    """
    if not maze.in_bounds(*cell):
        return []
    if known_edges is None:
        return list(observed)
    edges = known_edges.get(cell, set())
    return [d for d in observed
            if d in edges or not maze.in_bounds(*maze.neighbor(*cell, d))]


def _matches(maze: Maze, cell: tuple[int, int], observed: dict[Direction, bool],
             known_edges, min_edges: int) -> bool:
    """``cell`` の壁が観測と一致するか。**照合に使える辺が足りなければ不一致扱い**。"""
    usable = _comparable(maze, cell, observed, known_edges)
    if len(usable) < min_edges:
        return False
    return all(maze.has_wall(*cell, d) == observed[d] for d in usable)


def verify_cell(maze: Maze, believed: tuple[int, int], travel: Direction,
                observed: dict[Direction, bool], span: int = 1,
                known_edges: dict[tuple[int, int], set[Direction]] | None = None,
                min_edges: int = 1) -> CellVerdict:
    """壁のパターンから、いま居るセルを確定する (#113)。

    ``believed`` が一致すればそれを採る (**事前確率が高い方を優先する**)。一致しない
    ときだけ進行軸 ``travel`` の ±``span`` を調べ、**ちょうど 1 つだけ**一致すれば
    そこへ訂正する。0 個なら迷子、2 個以上なら区別がつかないので、どちらも諦める。

    ``observed`` は :func:`~krilly.perception.wall_detect.body_walls_to_maze` の出力
    (迷路方角 -> 壁の有無)。

    ``known_edges`` は :attr:`~krilly.strategy.explorer.Explorer.observed`
    (セル -> 観測済みの辺)。**これを渡さないと探索中は必ず迷子になる** —
    :func:`_comparable` を見ること。``min_edges`` は照合に要る辺の本数で、
    これを満たせないセルは「照合できない」として候補から外す。

    照合材料が足りず ``believed`` を確認できなかった場合は :attr:`CellVerdict.cell`
    が None・:attr:`CellVerdict.unverifiable` が True で返る。**呼び出し側が
    「材料が無いだけ」と「矛盾した」を区別できる**ようにするためで、前者は他の
    証拠 (進行方向の残差など) で 1 セルずれを排除できれば続行してよい。
    """
    if not observed:
        return CellVerdict(None, 0, "壁の観測が無い", unverifiable=True)
    if _matches(maze, believed, observed, known_edges, min_edges):
        return CellVerdict(believed, 0, "思っていたセルと壁が一致")

    usable = _comparable(maze, believed, observed, known_edges)
    dx, dy = travel.delta
    hits = []
    for k in range(-span, span + 1):
        if k == 0:
            continue
        cand = (believed[0] + dx * k, believed[1] + dy * k)
        if _matches(maze, cand, observed, known_edges, min_edges):
            hits.append((k, cand))
    if len(hits) == 1:
        k, cand = hits[0]
        return CellVerdict(cand, k, f"進行軸に {k:+d} セルずれていた ({believed} -> {cand})")
    if len(usable) < min_edges:
        return CellVerdict(None, 0,
                           f"照合できない ({believed} の辺のうち地図が知っているのは "
                           f"{len(usable)} 本で {min_edges} 本に足りない)",
                           unverifiable=True)
    if not hits:
        return CellVerdict(None, 0, "思っていたセルも隣も壁が一致しない (迷子)")
    return CellVerdict(None, 0,
                       "複数のセルが壁と一致して区別がつかない "
                       + " / ".join(str(c) for _, c in hits))


def recoverable(residual_rad: float, yaw_spread_rad: float | None) -> str | None:
    """回復を試みてよいか。駄目なら理由を返す (よければ None)。

    ``residual_rad`` は中断の原因になった方位残差、``yaw_spread_rad`` は軸角測定の
    フレーム間ばらつき (測れなかったら None)。
    """
    if abs(residual_rad) > RECOVERY_ABANDON_RAD:
        return (f"{math.degrees(abs(residual_rad)):.1f}° 回っており "
                f"{math.degrees(RECOVERY_ABANDON_RAD):.0f}° を超える "
                f"(mod 90° の折り返しで軸を誤る)")
    if yaw_spread_rad is None:
        return "軸角が測れない (赤帯が足りない)"
    if yaw_spread_rad > RECOVERY_MAX_YAW_SPREAD_RAD:
        return (f"軸角のフレーム間ばらつきが {math.degrees(yaw_spread_rad):.2f}° で "
                f"{math.degrees(RECOVERY_MAX_YAW_SPREAD_RAD):.1f}° を超える")
    return None
