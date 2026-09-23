"""走行スクリプト (search_run / speed_run) の引数定義のテスト。

**なぜ引数だけをテストするのか**: この 2 本は実機でしか動かないので、CI では
``main`` の中身を回せない。しかし引数の定義と ``main`` の読み出しが食い違うと、
実機の前まで行ってから ``AttributeError`` で落ちる — 実際に #89 で
``--neighbors`` を ``--no-neighbors`` へ変えたとき、定義側だけ置換に失敗して
そうなった。両者の突き合わせは実機なしでできるので、ここで固定する。
"""

import argparse
import ast
import inspect
import pathlib
import textwrap

import pytest

from scripts import search_run, speed_run

SCRIPTS = (search_run, speed_run)
IDS = [m.__name__.rsplit(".", 1)[-1] for m in SCRIPTS]


def _args_attributes(func) -> set[str]:
    """``func`` の中で ``args.<名前>`` として読まれている名前をすべて集める。"""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }


@pytest.mark.parametrize("module", SCRIPTS, ids=IDS)
def test_every_option_main_reads_is_actually_defined(module):
    """``main`` が読む ``args.X`` が、パーサの定義に実在すること。"""
    namespace = module.build_parser().parse_args([])
    missing = sorted(a for a in _args_attributes(module.main)
                     if not hasattr(namespace, a))
    assert not missing, f"{module.__name__}: 定義に無い引数を読んでいる: {missing}"


@pytest.mark.parametrize("module", SCRIPTS, ids=IDS)
def test_the_defaults_are_the_machine_as_it_runs_today(module):
    """既定値が「いま実機で走らせている設定」であること (#76 / #89)。"""
    args = module.build_parser().parse_args([])
    assert args.no_neighbors is False      # 左右の隣セルを読む
    assert args.pass_cells is None         # -> 2 セルまで止まらずに通過 (main で決まる)
    assert args.turn_in_place is False     # 旋回せず平行移動する
    assert args.no_front_check is False    # 進路チェックは有効
    assert args.no_correct is False        # カメラの絶対補正は有効


@pytest.mark.parametrize("module", SCRIPTS, ids=IDS)
def test_the_legacy_switches_still_parse(module):
    """以前の挙動に戻す道が残っていること (#76 の旋回、#89 の 1 セルずつ)。"""
    args = module.build_parser().parse_args(
        ["--no-neighbors", "--pass-cells", "1", "--turn-in-place"])
    assert args.no_neighbors and args.pass_cells == 1 and args.turn_in_place


@pytest.mark.parametrize("module", SCRIPTS, ids=IDS)
def test_build_parser_does_not_touch_hardware(module):
    """パーサを作るだけで実機依存を import しないこと (テストが動く前提)。"""
    assert isinstance(module.build_parser(), argparse.ArgumentParser)


# --- カメラの露出引数がスクリプト間で食い違わないこと (#78 / #100) -----------

CAMERA_SCRIPTS = ("search_run", "speed_run", "survey_shot", "wall_detect",
                  "cell_move_demo")


def _source(name: str) -> str:
    return (pathlib.Path(__file__).resolve().parents[1]
            / "scripts" / f"{name}.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", CAMERA_SCRIPTS)
def test_every_script_that_opens_the_camera_uses_the_shared_args(name):
    """``Camera(`` を作るスクリプトは必ず ``camera_kwargs`` を渡すこと。

    **露出の引数を各スクリプトが別々に持つと、片方だけ直して食い違う。**
    実際に ``--max-frame-duration`` が 4 本に重複していた。黒い床では ``--ev -2``
    が無いと後方の壁を 0.09 で読む (#100) ので、1 本でも漏れると走らせられない。
    """
    src = _source(name)
    assert "add_camera_args(p)" in src, f"{name}: add_camera_args を呼んでいない"
    for line in src.splitlines():
        if "Camera(" in line and "camera_kwargs" not in line and "picam2" not in line:
            # 引数なしの Camera() は露出オプションが効かない
            assert "Camera()" not in line, f"{name}: {line.strip()} が引数を渡していない"


# --- ゴールを隅へ動かす (#125: 黒い床が 3x3 分しかない) ----------------------

@pytest.mark.parametrize("module", SCRIPTS, ids=IDS)
def test_goal_defaults_to_the_centre_and_can_be_moved(module):
    from krilly.solver.maze import Maze

    assert module.build_parser().parse_args([]).goal is None
    args = module.build_parser().parse_args(["--size", "3", "--goal", "2,2"])
    maze = Maze(args.size)
    maze.set_goal_arg(args.goal)
    assert maze.goal_cells() == [(2, 2)]


def test_the_white_goal_board_makes_the_search_pass_beside_the_white_wall():
    """``mazes/white_goal3.txt``: 探索は (1,2) を通り、ゴールの西の壁 (白) の真横に来る。
    そこで白を読めなければ、楽観的な flood fill はその壁を抜けて東へ入ろうとする。"""
    from pathlib import Path

    from krilly.sim.sense import sense, sense_neighbors
    from krilly.solver.maze import Direction, Maze
    from krilly.strategy.explorer import Explorer

    truth = Maze.from_ascii(Path("mazes/white_goal3.txt").read_text())
    assert truth.goal_cells() == [(2, 2)]
    assert truth.has_wall(2, 2, Direction.W) and not truth.has_wall(2, 2, Direction.S)
    maze = Maze(3)
    maze.set_outer_walls()
    maze.set_goal_arg("2,2")
    ex = Explorer(maze, holonomic=True)
    cells = [ex.cell]
    while not ex.at_goal:
        ex.observe(sense(truth, ex.cell, ex.facing), sense_neighbors(truth, ex.cell, ex.facing))
        for step in ex.plan_leg(2):
            ex.advance(step)
        cells.append(ex.cell)
    assert (1, 2) in cells and cells[-1] == (2, 2)
