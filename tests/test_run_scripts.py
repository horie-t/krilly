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
