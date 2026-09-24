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
    # ジャイロzのスケール補正 (#17 でカメラ実測と比較して校正、1.0 なら無補正)
    assert 0.9 < cfg.gyro_scale_z < 1.1


def test_maze_config_defaults():
    cfg = load_maze_config()
    assert cfg.grid_size == 16
    assert math.isclose(cfg.cell_pitch_m, 0.180)
    assert math.isclose(cfg.passage_width_m, 0.168)
    assert cfg.goal_min == (7, 7)
    assert cfg.goal_max == (8, 8)


# --- 当日の走行設定 (#79) -------------------------------------------------------

def test_run_config_round_trips_and_keeps_the_command_verbatim(tmp_path):
    from krilly.config import RunConfig, load_run_config, save_run_config

    cfg = RunConfig("speed_run", ["--size", "16", "--ev", "-2", "--white-tops"],
                    saved_at="2026-09-24T10:00:00", buzzer_gpio=12)
    path = save_run_config(cfg, tmp_path / "run.yaml")
    back = load_run_config(path)
    assert back == cfg
    assert back.arg_value("--ev") == "-2"
    assert back.command("py") == ["py", "-m", "scripts.speed_run",
                                  "--size", "16", "--ev", "-2", "--white-tops"]


def test_run_config_refuses_an_unknown_script(tmp_path):
    import pytest

    from krilly.config import load_run_config

    p = tmp_path / "run.yaml"
    p.write_text("script: rm_rf\nargs: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_run_config(p)
