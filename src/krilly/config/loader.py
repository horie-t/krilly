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
    gyro_scale_z: float = 1.0         # BNO055 gyro z のスケール補正 (#17 で校正)
    #: 車輪ごとの実効径 [m]。None なら全輪 ``wheel_diameter_m``。
    #: 前進はほぼ W1/W2 だけで駆動するので (W0 の vx 係数は +0.026)、前進で校正した
    #: ``wheel_diameter_m`` は実質 W1/W2 の値になる。横移動は逆に W0 が主役 (係数 +1.000)
    #: なので、径が輪ごとに違うと**横だけスケールがずれる** (#76)。
    wheel_diameters_m: list[float] | None = None

    @property
    def wheel_circumference_m(self) -> float:
        return math.pi * self.wheel_diameter_m

    def wheel_circumference(self, wheel: int | None = None) -> float:
        """車輪 ``wheel`` の実効周長 [m]。``None`` なら共通値。"""
        if wheel is None or self.wheel_diameters_m is None:
            return self.wheel_circumference_m
        return math.pi * self.wheel_diameters_m[wheel]

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
        gyro_scale_z=float(data.get("gyro_scale_z", 1.0)),
        wheel_diameters_m=(
            [float(d) for d in data["wheel_diameters_m"]]
            if data.get("wheel_diameters_m") else None
        ),
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


# --- 当日の走行設定 (#79) -----------------------------------------------------
#: ランチャが起動できるスクリプト。
RUN_SCRIPTS = ("speed_run", "search_run")
#: 当日の走行設定の置き場所。**git には入れない** (機体ごと・会場ごとの状態なので)。
RUN_CONFIG_PATH = _CONFIG_DIR / "run.yaml"


@dataclass(frozen=True)
class RunConfig:
    """ボタンで起動する走行の設定 (#79)。

    **引数を 1 つずつ項目にせず、試走で走らせたコマンドをそのまま持つ。** 項目ごとの
    スキーマは ``speed_run`` の引数と必ずずれるうえ、手で YAML を書くと試走で検証して
    いない組み合わせが当日走る。``speed_run --save-run-config`` が完走したときにだけ
    書く (``saved_at`` はそのときの時刻)。

    ピンとブザーの種類は配線の都合なので、ここで上書きできるようにしてある。
    """

    script: str
    args: list[str]
    saved_at: str = ""
    button_gpio: int | None = None
    buzzer_gpio: int | None = None
    buzzer_passive: bool = True

    def command(self, python: str) -> list[str]:
        """実行するコマンド全体 (``python -m scripts.<script> <args...>``)。"""
        return [python, "-m", f"scripts.{self.script}", *self.args]

    def arg_value(self, name: str) -> str | None:
        """``args`` の中の ``name`` の値 (``--ev -2`` / ``--ev=-2`` のどちらも読む)。"""
        for i, a in enumerate(self.args):
            if a == name and i + 1 < len(self.args):
                return self.args[i + 1]
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
        return None


def load_run_config(path: str | Path | None = None) -> RunConfig:
    data = _load_yaml(path or RUN_CONFIG_PATH)
    script = str(data["script"])
    if script not in RUN_SCRIPTS:
        raise ValueError(f"script は {RUN_SCRIPTS} のどれか (読んだ値: {script!r})")
    args = data.get("args") or []
    if not isinstance(args, list):
        raise ValueError("args はリストで書くこと")
    return RunConfig(
        script=script,
        args=[str(a) for a in args],
        saved_at=str(data.get("saved_at", "")),
        button_gpio=(None if data.get("button_gpio") is None
                     else int(data["button_gpio"])),
        buzzer_gpio=(None if data.get("buzzer_gpio") is None
                     else int(data["buzzer_gpio"])),
        buzzer_passive=bool(data.get("buzzer_passive", True)),
    )


def save_run_config(cfg: RunConfig, path: str | Path | None = None) -> Path:
    """``RunConfig`` を YAML に書く。ピンの上書きは既存のファイルから引き継ぐ。"""
    target = Path(path or RUN_CONFIG_PATH)
    data: dict[str, Any] = {"script": cfg.script, "args": list(cfg.args),
                            "saved_at": cfg.saved_at}
    for key in ("button_gpio", "buzzer_gpio"):
        if getattr(cfg, key) is not None:
            data[key] = getattr(cfg, key)
    if not cfg.buzzer_passive:
        data["buzzer_passive"] = False
    with open(target, "w", encoding="utf-8") as f:
        f.write("# ボタンで起動する走行の設定 (#79)。speed_run --save-run-config が完走時に書く。\n")
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return target
