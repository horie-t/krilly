"""車体・迷路の設定を YAML から読み込む。

寸法はコードを変更せずにチューニングできるよう YAML (``robot.yaml`` /
``maze.yaml``) に置いている。これらの dataclass は型付きで検証済みの
アクセス手段を提供する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} did not parse to a mapping")
    return data


@dataclass(frozen=True)
class RobotConfig:
    """車体の物理パラメータ (単位: SI — メートル、ラジアン)。"""

    wheel_diameter_m: float
    wheel_count: int
    center_to_wheel_m: float          # L: 中心から各輪接地点までの距離
    steps_per_rev: int                # フルステップ数 (1.8° -> 200)
    microstep: int                    # マイクロステップ分割数 (1/μ)
    wheel_angles_deg: list[float]     # 各輪の駆動方向角 [deg]

    @property
    def wheel_circumference_m(self) -> float:
        return math.pi * self.wheel_diameter_m

    @property
    def microsteps_per_rev(self) -> int:
        return self.steps_per_rev * self.microstep

    @property
    def metres_per_microstep(self) -> float:
        return self.wheel_circumference_m / self.microsteps_per_rev


@dataclass(frozen=True)
class MazeConfig:
    """クラシック競技のマイクロマウス迷路の寸法。"""

    grid_size: int                    # N (クラシックは 16)
    cell_pitch_m: float               # 0.180 m
    wall_thickness_m: float           # 0.012 m
    wall_height_m: float              # 0.050 m
    goal_min: tuple[int, int]         # 0始まりインデックスの角 (端点を含む)
    goal_max: tuple[int, int]         # 0始まりインデックスの角 (端点を含む)

    @property
    def passage_width_m(self) -> float:
        return self.cell_pitch_m - self.wall_thickness_m


def load_robot_config(path: str | Path | None = None) -> RobotConfig:
    data = _load_yaml(path or _CONFIG_DIR / "robot.yaml")
    return RobotConfig(
        wheel_diameter_m=float(data["wheel_diameter_m"]),
        wheel_count=int(data["wheel_count"]),
        center_to_wheel_m=float(data["center_to_wheel_m"]),
        steps_per_rev=int(data["steps_per_rev"]),
        microstep=int(data["microstep"]),
        wheel_angles_deg=[float(a) for a in data["wheel_angles_deg"]],
    )


def load_maze_config(path: str | Path | None = None) -> MazeConfig:
    data = _load_yaml(path or _CONFIG_DIR / "maze.yaml")
    return MazeConfig(
        grid_size=int(data["grid_size"]),
        cell_pitch_m=float(data["cell_pitch_m"]),
        wall_thickness_m=float(data["wall_thickness_m"]),
        wall_height_m=float(data["wall_height_m"]),
        goal_min=tuple(data["goal_min"]),  # type: ignore[arg-type]
        goal_max=tuple(data["goal_max"]),  # type: ignore[arg-type]
    )
