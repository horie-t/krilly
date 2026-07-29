"""カメラ画像から現在セルの壁有無を判定する (issue #16).

下向きカメラ (車体中央・高さ約39cm・1セルが画角内) では、壁があると赤い壁上面が
フレームの**各辺付近**に赤帯として現れる。中央は自機 (Pi 基板・カメラケーブル) で
占有され誤検出源になるため、判定は**各辺の ROI (関心領域) 内の赤割合**で行い、
中央や支柱・ケーブルを ROI の外に置いて除外する。

処理の流れ:
1. red_wall.red_mask で赤マスクを作る。
2. 機体前後左右 (FRONT/BACK/LEFT/RIGHT) の ROI ごとに赤割合を求め、閾値で壁有無を判定。
3. ロボットの向き (迷路の N/E/S/W) で機体相対 -> 迷路方角に写像し、Maze へ反映。

ROI の位置・閾値は実迷路 (セル中央にロボットを置いた画像) で調整する。カメラの
取付回転 (画像の上=機体のどの向きか) は取付依存なので、ROI をその向きに合わせる。
画素->地面のメートル投影は本判定には不要 ("辺付近に赤があるか" のみ見る)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from krilly.perception.red_wall import RedDetectorConfig, red_mask
from krilly.solver.maze import Direction

# 機体相対の方向 (FRONT=+x 前, BACK=-x 後, LEFT=+y 左, RIGHT=-y 右)
FRONT = "front"
BACK = "back"
LEFT = "left"
RIGHT = "right"
BODY_DIRS = (FRONT, BACK, LEFT, RIGHT)


@dataclass(frozen=True)
class Roi:
    """画像中の矩形 ROI (画素)。"""

    x: int
    y: int
    w: int
    h: int

    def view(self, img: np.ndarray) -> np.ndarray:
        return img[self.y : self.y + self.h, self.x : self.x + self.w]


def roi_red_fraction(mask: np.ndarray, roi: Roi) -> float:
    """ROI 内で赤 (mask>0) の占める割合 (0..1) を返す。"""
    sub = roi.view(mask)
    if sub.size == 0:
        return 0.0
    return float(np.count_nonzero(sub)) / sub.size


def default_rois(width: int = 640, height: int = 480, thickness: int = 70,
                 span: float = 0.5) -> dict[str, Roi]:
    """各辺の中央付近に帯状 ROI を作る (中央・角を避ける)。取付に合わせて要調整。

    既定は 画像上=FRONT / 下=BACK / 左=LEFT / 右=RIGHT と仮定 (実機で確認)。
    """
    sw = int(width * span)
    sh = int(height * span)
    x0 = (width - sw) // 2
    y0 = (height - sh) // 2
    return {
        FRONT: Roi(x0, 0, sw, thickness),
        BACK: Roi(x0, height - thickness, sw, thickness),
        LEFT: Roi(0, y0, thickness, sh),
        RIGHT: Roi(width - thickness, y0, thickness, sh),
    }


@dataclass
class WallDetectorConfig:
    rois: dict[str, Roi]
    threshold: float = 0.15                                     # 壁ありとみなす赤割合
    red: RedDetectorConfig = field(default_factory=RedDetectorConfig)
    # 固定の自己遮蔽領域 (Pi 基板・カメラケーブル等) を赤マスクから除外する矩形。
    # ケーブルの赤誤検出を消すために実機で位置を合わせる。
    exclude: list[Roi] = field(default_factory=list)


class WallDetector:
    """ROI ごとの赤割合で機体相対の壁有無を判定する。"""

    def __init__(self, config: WallDetectorConfig) -> None:
        self.cfg = config

    def red_fractions(self, bgr: np.ndarray) -> dict[str, float]:
        mask = red_mask(bgr, self.cfg.red)
        for r in self.cfg.exclude:                              # 自己遮蔽領域を除外
            mask[r.y : r.y + r.h, r.x : r.x + r.w] = 0
        return {d: roi_red_fraction(mask, roi) for d, roi in self.cfg.rois.items()}

    def detect(self, bgr: np.ndarray) -> dict[str, bool]:
        """機体相対 (front/back/left/right) の壁有無。"""
        return {d: f >= self.cfg.threshold for d, f in self.red_fractions(bgr).items()}


def body_walls_to_maze(
    walls_body: dict[str, bool], facing: Direction
) -> dict[Direction, bool]:
    """機体相対の壁有無を、ロボットの向き ``facing`` で迷路方角へ写像する。

    facing=前方の迷路方角。LEFT=facing の反時計回り(-90°)、RIGHT=時計回り(+90°)。
    """
    return {
        facing: walls_body[FRONT],
        Direction((facing + 2) % 4): walls_body[BACK],
        Direction((facing - 1) % 4): walls_body[LEFT],
        Direction((facing + 1) % 4): walls_body[RIGHT],
    }


def update_maze_walls(maze, x: int, y: int, walls_maze: dict[Direction, bool]) -> None:
    """判定した迷路方角の壁有無をセル (x, y) に反映する (共有エッジで隣接にも反映)。"""
    for d, present in walls_maze.items():
        maze.set_wall(x, y, d, present)
