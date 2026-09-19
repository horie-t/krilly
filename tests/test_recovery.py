"""中断した走行の姿勢作り直し (issue #113)。"""

import math

import pytest

from krilly.localization.recovery import (
    RECOVERY_ABANDON_RAD,
    RECOVERY_MAX_HEADING_RAD,
    RECOVERY_MAX_YAW_SPREAD_RAD,
    recoverable,
    verify_cell,
)
from krilly.localization.grid import MAX_HEADING_CORRECTION_RAD
from krilly.solver.maze import Direction
from tests.test_explorer import open_maze


def walls_of(maze, cell):
    """``cell`` の 4 壁を観測の形 (迷路方角 -> 有無) で返す。"""
    return {d: maze.has_wall(*cell, d) for d in Direction}


# --- セルの照合 -------------------------------------------------------------
def test_believed_cell_wins_when_it_matches():
    """**事前確率の高い方を優先する。** 一致するなら訂正しない。"""
    m = open_maze(5)
    m.set_wall(2, 2, Direction.E)
    v = verify_cell(m, (2, 2), Direction.N, walls_of(m, (2, 2)))
    assert v.ok and v.cell == (2, 2) and v.shift == 0


def test_a_one_cell_distance_error_is_corrected_along_the_travel_axis():
    """距離が 1 セル狂ったケース — 壁のパターンで拾い直せる。"""
    m = open_maze(5)
    m.set_wall(2, 2, Direction.E)          # (2,2) だけが持つ特徴
    # 実際は (2,2) に居るが (2,1) だと思っている
    v = verify_cell(m, (2, 1), Direction.N, walls_of(m, (2, 2)))
    assert v.ok and v.cell == (2, 2) and v.shift == +1
    # 逆向きに進んでいたなら -1 側を見る
    v = verify_cell(m, (2, 3), Direction.S, walls_of(m, (2, 2)))
    assert v.ok and v.cell == (2, 2) and v.shift == +1


def test_it_refuses_when_nothing_matches():
    """どのセルとも合わなければ**迷子**。続行してはいけない。"""
    m = open_maze(5)
    observed = {d: True for d in Direction}      # 四方を壁で囲まれたセルは無い
    v = verify_cell(m, (2, 2), Direction.N, observed)
    assert not v.ok and "迷子" in v.reason


def test_it_refuses_when_two_cells_match():
    """**区別がつかないときも諦める。** 開いた迷路の中央列はどのセルも同じに見える。"""
    m = open_maze(5)
    # (2,1) (2,2) (2,3) はどれも壁なし。(2,2) のつもりで進行軸 ±1 を見ると 2 つ当たる
    v = verify_cell(m, (2, 2), Direction.N, {Direction.E: False, Direction.W: False})
    assert v.ok and v.shift == 0     # believed が一致するので、そこで止まる
    # believed が一致しないケースを作る: believed を壁ありのセルにする
    m.set_wall(2, 2, Direction.E)
    v = verify_cell(m, (2, 2), Direction.N, {Direction.E: False, Direction.W: False})
    assert not v.ok and "区別がつかない" in v.reason


def test_the_travel_axis_is_what_gets_searched():
    """ずれは**進行軸にしか起きない**ので、直交する軸は候補にしない。"""
    m = open_maze(5)
    m.set_wall(2, 2, Direction.E)
    # (2,2) に居るのに (1,2) (東西にずれた位置) だと思っている場合、
    # 南北に進んでいたなら見つけられない = 諦めるのが正しい
    v = verify_cell(m, (1, 2), Direction.N, walls_of(m, (2, 2)))
    assert not v.ok


def test_out_of_bounds_candidates_are_not_matched():
    m = open_maze(5)
    v = verify_cell(m, (2, 0), Direction.S, walls_of(m, (2, 0)))
    assert v.ok and v.shift == 0       # believed が一致すれば範囲外は見ない


def test_no_observation_means_no_recovery():
    assert not verify_cell(open_maze(5), (2, 2), Direction.N, {}).ok


# --- 回復してよいかの判定 ---------------------------------------------------
def test_a_small_residual_with_a_tight_measurement_is_recoverable():
    assert recoverable(math.radians(2.7), math.radians(0.15)) is None


def test_a_large_rotation_is_not_recoverable_because_of_the_90_degree_fold():
    """**mod 90° の折り返しが罠。** 50° 回っていると -40° と読んで逆の軸へ倒す。"""
    why = recoverable(math.radians(50.0), math.radians(0.1))
    assert why is not None and "折り返し" in why
    # 記録されている中断 3 件 (2.78 / 5.43 / 25.68°) は全部回復圏内
    for deg in (2.78, 5.43, 25.68):
        assert recoverable(math.radians(deg), math.radians(0.1)) is None


def test_a_noisy_yaw_measurement_is_not_recoverable():
    """**測定が暴れているときに広いガードで補正するのが最悪の組み合わせ。**"""
    assert recoverable(math.radians(3.0), math.radians(1.2)) is not None
    assert recoverable(math.radians(3.0), None) is not None


def test_the_recovery_guard_is_wider_than_the_normal_one_but_below_the_fold():
    """回復モードではガードの前提が反転する (#113 の docstring)。

    正常運転の 5° より広く、しかし折り返しの 45° には遠いこと。
    """
    assert MAX_HEADING_CORRECTION_RAD < RECOVERY_MAX_HEADING_RAD < RECOVERY_ABANDON_RAD
    assert RECOVERY_ABANDON_RAD < math.radians(45.0)
    # 測定の質のガードは実測のばらつき (0.11-0.26°, #88) より緩く、1° より厳しい
    assert math.radians(0.26) < RECOVERY_MAX_YAW_SPREAD_RAD < math.radians(1.0)


# --- 実機の 3 件を再生する ---------------------------------------------------
def test_the_three_recorded_aborts_would_all_have_been_recovered():
    """実機で記録された中断 3 件が、設計上どう扱われるか (#107 -> #113)。

    2026-09-09 の N→E (-25.68°)、09-19 の S→E (-2.78°) と S→W (+5.43°)。
    どれも折り返しの上限 30° 内なので、**壁のパターンが一致すれば続行できた**。
    25.68° は余裕が無いことも同時に示しておく。
    """
    m = open_maze(8)
    m.set_wall(3, 3, Direction.E)
    for deg in (2.78, 5.43, 25.68):
        assert recoverable(math.radians(deg), math.radians(0.15)) is None, deg
        v = verify_cell(m, (3, 3), Direction.E, walls_of(m, (3, 3)))
        assert v.ok and v.shift == 0
    # 25.68° は上限 30° まで 4.3° しかない
    assert math.radians(25.68) < RECOVERY_ABANDON_RAD
    assert RECOVERY_ABANDON_RAD - math.radians(25.68) < math.radians(5.0)


def test_a_featureless_neighbour_is_still_found_if_it_is_the_only_one():
    """**「壁が無い」ことも特徴になる。** 隣が壁を持っていれば区別がつく。

    (2,2) に北壁があると、その北隣 (2,3) は南壁を共有するので無特徴ではない。
    したがって「四方とも壁なし」の観測は (2,1) を一意に指す。
    """
    m = open_maze(5)
    m.set_wall(2, 2, Direction.N)
    v = verify_cell(m, (2, 2), Direction.N, walls_of(m, (2, 1)))
    assert v.ok and v.cell == (2, 1) and v.shift == -1


def test_it_refuses_when_both_neighbours_look_alike():
    """**進行軸の両隣が同じに見えると回復できない。** 設計上の限界。

    進行軸に直交する壁は両隣から共有されないので、進行軸が東西のとき
    (2,2) の北壁は (1,2) / (3,2) のどちらにも現れない — 両方が無特徴になる。
    こうなったら諦めてその場で止まるのが正しい (#23 の cross-blind と同じ形)。
    """
    m = open_maze(5)
    m.set_wall(2, 2, Direction.N)
    v = verify_cell(m, (2, 2), Direction.E, walls_of(m, (1, 2)))
    assert not v.ok and "区別がつかない" in v.reason
