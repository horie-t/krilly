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

#: 軸角のフレーム間ばらつきの上限 [rad]。これを超える測定は信用しない。
#:
#: 960x720 でのフレーム間ばらつきは実測 0.11-0.26° (#88) なので、0.5° は 2 倍の余裕。
#: **測定が暴れているときに広いガードで補正するのが最悪の組み合わせ**なので、ここは
#: 回復モードでこそ厳しくする。
RECOVERY_MAX_YAW_SPREAD_RAD = math.radians(0.5)


@dataclass(frozen=True)
class CellVerdict:
    """セル照合の結果。"""

    cell: tuple[int, int] | None    # 確定したセル (None = 確定できなかった)
    shift: int                      # 進行軸に何セルずれていたか (0 = 思っていたとおり)
    reason: str                     # 人間向けの要約 (ログに出す)

    @property
    def ok(self) -> bool:
        return self.cell is not None


def _matches(maze: Maze, cell: tuple[int, int],
             observed: dict[Direction, bool]) -> bool:
    """``cell`` の 4 壁が観測と一致するか。範囲外は不一致扱い。"""
    if not maze.in_bounds(*cell):
        return False
    return all(maze.has_wall(*cell, d) == present for d, present in observed.items())


def verify_cell(maze: Maze, believed: tuple[int, int], travel: Direction,
                observed: dict[Direction, bool], span: int = 1) -> CellVerdict:
    """壁のパターンから、いま居るセルを確定する (#113)。

    ``believed`` が一致すればそれを採る (**事前確率が高い方を優先する**)。一致しない
    ときだけ進行軸 ``travel`` の ±``span`` を調べ、**ちょうど 1 つだけ**一致すれば
    そこへ訂正する。0 個なら迷子、2 個以上なら区別がつかないので、どちらも諦める。

    ``observed`` は :func:`~krilly.perception.wall_detect.body_walls_to_maze` の出力
    (迷路方角 -> 壁の有無)。4 辺そろっていなくてもよい (渡された辺だけ照合する)。
    """
    if not observed:
        return CellVerdict(None, 0, "壁の観測が無い")
    if _matches(maze, believed, observed):
        return CellVerdict(believed, 0, "思っていたセルと壁が一致")

    dx, dy = travel.delta
    hits = []
    for k in range(-span, span + 1):
        if k == 0:
            continue
        cand = (believed[0] + dx * k, believed[1] + dy * k)
        if _matches(maze, cand, observed):
            hits.append((k, cand))
    if len(hits) == 1:
        k, cand = hits[0]
        return CellVerdict(cand, k, f"進行軸に {k:+d} セルずれていた ({believed} -> {cand})")
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
