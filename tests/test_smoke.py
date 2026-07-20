"""M0 スキャフォールドのスモークテスト: パッケージの import と config 読み込み。"""

import math

import krilly
from krilly.config import load_maze_config, load_robot_config
from krilly.logging_config import get_logger, setup_logging


def test_version():
    assert krilly.__version__


def test_setup_logging_and_logger():
    setup_logging("DEBUG")
    log = get_logger("krilly.test")
    assert log.name == "krilly.test"


def test_robot_config_defaults():
    cfg = load_robot_config()
    assert cfg.wheel_count == 3
    assert cfg.steps_per_rev == 200
    assert math.isclose(cfg.wheel_circumference_m, math.pi * cfg.wheel_diameter_m)
    assert cfg.microsteps_per_rev == cfg.steps_per_rev * cfg.microstep
    # オドメトリ分解能 (μ=16 で約47µm)
    assert 0 < cfg.metres_per_microstep < 1e-3
    assert len(cfg.wheel_angles_deg) == cfg.wheel_count


def test_maze_config_defaults():
    cfg = load_maze_config()
    assert cfg.grid_size == 16
    assert math.isclose(cfg.cell_pitch_m, 0.180)
    assert math.isclose(cfg.passage_width_m, 0.168)
    assert cfg.goal_min == (7, 7)
    assert cfg.goal_max == (8, 8)
