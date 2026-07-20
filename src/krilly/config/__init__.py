"""設定の読み込み (車体・迷路の寸法、チューニング定数)。"""

from .loader import RobotConfig, MazeConfig, load_robot_config, load_maze_config

__all__ = [
    "RobotConfig",
    "MazeConfig",
    "load_robot_config",
    "load_maze_config",
]
