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


# --- 実機校正済みの設定 (#16, #56 で再校正) --------------------------------
# 640x480、カメラ中央・高さ約39cm・セル中央。画像の上=車体前方(+x)。壁は端でなく
# 内側の帯に写り、四隅の格子点(赤ポスト)は各辺の中央 ROI で避ける。
# 壁ありで赤割合 ~0.36-0.55、壁なし 0.00 (閾値 0.15 で分離)。
#
# #56: 右壁は「影で暗い」のではなく**白飛び**していた (旧環境: 黒い床パネル)。
# 視野の大半が黒い床なので露出が開き、明るい壁上面が淡いピンク (S=46-54) に飛ぶ。
# s_min=70 では落ちて壁を見落とし、実機が壁に衝突した。s_min=50 まで緩めて拾う。
# #65: 床が木のフローリングに変わり、露出が絞られて彩度は戻った (S=94-190) が、
# 今度は右壁上面の**色相**が帯の場所によって H=141-155 までマゼンタ側へずれ、
# h2_lo=160 の外に出て帯の上 2/3 を取りこぼした。h2_lo=140 まで広げる (右帯の
# 赤割合 0.155 -> 0.442 で飽和)。木の床は H=30-66 / S=10-23 なので漏れない。
# 青配線は H≈120 なので 140 との余裕も十分。
CALIBRATED_RED = RedDetectorConfig(h2_lo=140, s_min=50, v_min=40)

# 実測した赤帯の位置 [px]。**機体を定規でセル中央に置いた姿勢**から測ること
# (ここが cell_pose のゼロ点であり、px/mm の基準でもある)。ROI はこの帯を内側に
# 含むように置く (帯から外れた分だけ赤割合が下がる)。
#
# 筐体改修 (モータドライバを壁より高い位置へ、ホイール位置変更、カメラの高さと
# 傾きを修正) で再測定した値。同じ姿勢での比較:
#   前後の帯間隔 290.5px -> 305.5px、左右 285.5px -> 305.0px (カメラが約 5% 低い)
#   px/mm は X 1.586->1.694 / Y 1.614->1.697。改修前は X と Y が 1.8% 食い違って
#   いたが、傾き修正後は 0.2% 以内で一致する。
CALIBRATED_BANDS = {
    FRONT: (119, 142),   # 行 (y)  定規で前後中央にした (2,1) 北向きで実測
    BACK: (425, 447),    # 行 (y)  同上
    LEFT: (134, 155),    # 列 (x)  定規で左右中央にした (0,1) 北向きで実測
    RIGHT: (438, 461),   # 列 (x)  同上
}


@dataclass(frozen=True)
class RoiSpec:
    """帯に対する ROI の**形**だけを mm で持つ (位置は幾何から決まる)。

    寸法を画素ではなく mm で持つ理由は、**カメラの画角を変えても形が保たれる**こと。
    #88 で 640x480 (センサーの 33% しか使っていなかった) から 960x720 の全画素モードへ
    移ったとき、px/mm は 1.70 のまま画角だけ 1.5 倍になったので、**ROI の寸法は据え置きで
    位置だけが変わった**。mm で持っていれば、どちらの場合も同じ定義で書ける。
    """

    thickness_mm: float      # 帯に直交する向きの長さ (帯 12mm を包む)
    length_mm: float         # 帯に沿う向きの長さ
    along_offset_mm: float = 0.0   # 帯に沿う向きのずらし (自機・ケーブルを避ける)


@dataclass(frozen=True)
class CameraGeometry:
    """校正の実体。**実測できる量だけ**で表す。

    - ``cell_center``: **定規でセル中央に置いた機体**でセル中心が写る画素。
      ここが位置測定のゼロ点になる (#64: ROI の中心をここに合わせて 6mm の偏りを消した)
    - ``px_per_mm_*``: 対向する帯の間隔が 1 セルピッチ (180mm) であることから出る

    壁は必ずセル中心から半ピッチ (90mm) の位置にあるので、**帯の位置はこの 2 つから
    計算できる**。実測の :data:`CALIBRATED_BANDS` と誤差 0.000px で一致する
    (当然で、この 2 つは帯の実測から作られている)。
    """

    width: int
    height: int
    cell_center: tuple[float, float]
    px_per_mm_x: float
    px_per_mm_y: float

    @classmethod
    def from_bands(cls, bands: dict[str, tuple[int, int]], width: int, height: int,
                   pitch_mm: float = 180.0) -> "CameraGeometry":
        """実測した 4 辺の帯位置から幾何を起こす。"""
        c = {e: (lo + hi) / 2.0 for e, (lo, hi) in bands.items()}
        return cls(
            width=width, height=height,
            cell_center=((c[LEFT] + c[RIGHT]) / 2.0, (c[FRONT] + c[BACK]) / 2.0),
            px_per_mm_x=(c[RIGHT] - c[LEFT]) / pitch_mm,
            px_per_mm_y=(c[BACK] - c[FRONT]) / pitch_mm,
        )

    def scale(self, edge: str) -> float:
        """その辺の帯に**直交**する向きの px/mm。"""
        return self.px_per_mm_x if edge in (LEFT, RIGHT) else self.px_per_mm_y

    def band_center(self, edge: str, half_pitch_mm: float = 90.0) -> float:
        """その辺の帯の中心 (縦帯なら列 x、横帯なら行 y)。"""
        sign = -1.0 if edge in (FRONT, LEFT) else +1.0
        base = self.cell_center[0] if edge in (LEFT, RIGHT) else self.cell_center[1]
        return base + sign * half_pitch_mm * self.scale(edge)

    def roi(self, edge: str, spec: RoiSpec) -> Roi:
        """帯の上に ROI を置く。**ROI の中心が、その軸の位置測定のゼロ点**になる。"""
        vertical = edge in (LEFT, RIGHT)
        cross = self.band_center(edge)
        along_scale = self.px_per_mm_y if vertical else self.px_per_mm_x
        along_base = self.cell_center[1] if vertical else self.cell_center[0]
        along = along_base + spec.along_offset_mm * along_scale
        thick = spec.thickness_mm * self.scale(edge)
        length = spec.length_mm * along_scale
        if vertical:
            return Roi(int(round(cross - thick / 2)), int(round(along - length / 2)),
                       int(round(thick)), int(round(length)))
        return Roi(int(round(along - length / 2)), int(round(cross - thick / 2)),
                   int(round(length)), int(round(thick)))

    def rois(self, specs: dict[str, RoiSpec]) -> dict[str, Roi]:
        return {e: self.roi(e, spec) for e, spec in specs.items()}


#: 実機校正済みの幾何 (640x480)。:data:`CALIBRATED_BANDS` から起こす。
CALIBRATED_GEOMETRY = CameraGeometry.from_bands(CALIBRATED_BANDS, 640, 480)

#: 各辺の ROI の形。**画角を変えても据え置き**で、位置だけ幾何が決める。
#: 値は #16 / #56 / #65 で手で追い込んだ画素値を mm に直したもの。
CALIBRATED_ROI_SPECS = {
    FRONT: RoiSpec(thickness_mm=29.5, length_mm=106.2, along_offset_mm=+13.6),
    # BACK だけ短く、少しずらしてある。リボンケーブルが赤く写るのを避けるため (#65)。
    # ケーブルは柔軟なので完全には避けられず、辺別しきい値 (BACK=0.25) と併用する。
    BACK: RoiSpec(thickness_mm=22.4, length_mm=53.1, along_offset_mm=+5.9),
    LEFT: RoiSpec(thickness_mm=27.1, length_mm=126.7, along_offset_mm=-2.2),
    RIGHT: RoiSpec(thickness_mm=27.1, length_mm=126.7, along_offset_mm=-2.2),
}


#: 解像度ごとの校正済み幾何。**撮影サイズと ROI は必ず対で使う。**
#: 食い違うと ROI が帯から外れ、壁ありでも赤割合が出ずに機体が壁へ突っ込む
#: (#56 で実際に起きた失敗と同じ形)。:func:`geometry_for` が対応付けを保証する。
CALIBRATED_GEOMETRIES = {(640, 480): CALIBRATED_GEOMETRY}


def geometry_for(size: tuple[int, int]) -> CameraGeometry:
    """撮影サイズに対応する校正済み幾何。無ければ **エラーにする**。

    黙って既定へ落とすと ROI が帯から外れたまま走ってしまう。校正していない解像度で
    走らせるくらいなら止まった方が安い。
    """
    try:
        return CALIBRATED_GEOMETRIES[size]
    except KeyError:
        known = ", ".join(f"{w}x{h}" for w, h in sorted(CALIBRATED_GEOMETRIES))
        raise KeyError(
            f"{size[0]}x{size[1]} の校正済み幾何が無い (あるのは {known})。"
            "camera_fov --emit-bands で測って CALIBRATED_GEOMETRIES に足すこと"
        ) from None


def calibrated_rois(geometry: CameraGeometry | None = None) -> dict[str, Roi]:
    """実機で校正した各辺 ROI。既定は 640x480 の :data:`CALIBRATED_GEOMETRY`。

    #56 で RIGHT / BACK を実測した帯 (:data:`CALIBRATED_BANDS`) に合わせ直した。
    #16 の校正写真は手置きで撮ったもので、閉ループ (#17 で ±1mm) の停止位置とは
    十数 px ずれており、RIGHT は帯と 20px しか重なっていなかった (壁ありでも
    赤割合 0.08-0.15 しか出ず、しきい値 0.15 を割って見落としていた)。
    ずれを疑うときは ``red_mask`` の列/行プロファイルで帯の位置を測ればよい
    (:func:`band_positions`)。

    **各 ROI の中心が、その辺のオフセット測定のゼロ点**。定規でセル中央に置いた
    機体で測った帯の中心に一致させてある (#64: ここを合わせて 6mm の偏りを消した)。
    ``geometry`` を渡せば別の画角 (全画素モードなど) の ROI を同じ形で作れる。
    """
    return (geometry or CALIBRATED_GEOMETRY).rois(CALIBRATED_ROI_SPECS)


def calibrated_config(threshold: float = 0.08,
                      size: tuple[int, int] = (640, 480)) -> "WallDetectorConfig":
    """実機校正済みの WallDetectorConfig を返す (ROI + 赤しきい値 + 帯探索)。

    しきい値は #65 の 5x5 手動調査 (``scripts/survey_shot.py``、113 枚 = 452 ラベル、
    木の床・3D プリント柱) の辺別分布から決めた:

    - front: 開 max 0.000 / 壁 min 0.302
    - back : 開 max 0.121 / 壁 min 0.509 (開側の 0.09-0.12 はリボンケーブルの偽帯)
    - left : 開 max 0.036 / 壁 min 0.344
    - right: 開 max 0.016 / 壁 min 0.379

    BACK だけ 0.25 に上げる (ケーブルと実壁の完全分離)。他は 0.08 のまま。
    **壁の見落とし (衝突) の方が偽陽性 (回り道) より高くつく**ので、迷ったら
    下げる側に振ること。

    ``size`` は撮影サイズ。ROI はその幾何から作られ、``measure`` が実フレームと
    突き合わせる (食い違ったまま走ると壁を見落とす)。
    """
    geometry = geometry_for(size)
    return WallDetectorConfig(
        rois=calibrated_rois(geometry), threshold=threshold,
        thresholds={BACK: 0.25}, red=CALIBRATED_RED, frame_size=size,
    )


@dataclass
class WallDetectorConfig:
    rois: dict[str, Roi]
    threshold: float = 0.15                                     # 壁ありとみなす赤割合
    # 辺別のしきい値上書き (#65)。BACK はリボンケーブル (オレンジ、柔軟で写る位置が
    # 動くため固定除外できない) が偽帯 <=0.12 を作るが、実壁の帯は太く >=0.51 で
    # 写るので、しきい値を上げるだけで完全に分離できる。
    thresholds: dict[str, float] = field(default_factory=dict)
    red: RedDetectorConfig = field(default_factory=RedDetectorConfig)
    # 固定の自己遮蔽領域 (Pi 基板・カメラケーブル等) を赤マスクから除外する矩形。
    # ケーブルの赤誤検出を消すために実機で位置を合わせる。
    exclude: list[Roi] = field(default_factory=list)
    # ROI を帯に直交する方向へ ±search_px ずらして最大の赤割合を採る (#56)。
    # セル内の位置ずれ (実機で最大 25mm ≒ 40px) で帯が ROI から外れるのを吸収する。
    # 0 で固定 ROI (従来の挙動)。隣のセルの帯は 292px 先なので 40px 程度なら
    # 取り違えない。
    search_px: int = 40
    search_step: int = 2

    #: この ROI を作ったときの撮影サイズ。``measure`` が実フレームと突き合わせる。
    #: None なら検算しない (合成フレームのテスト用)。
    frame_size: tuple[int, int] | None = None

    def threshold_for(self, edge: str) -> float:
        """この辺の壁判定しきい値 (辺別の上書きが無ければ共通値)。"""
        return self.thresholds.get(edge, self.threshold)


def _band_profile(mask: np.ndarray, roi: Roi, vertical: bool) -> np.ndarray:
    """ROI の長手方向に平均した赤割合プロファイル (探索軸に沿った 1 次元)。

    ``vertical`` は帯が縦 (LEFT/RIGHT) かどうか。縦帯なら列方向、横帯なら行方向。
    """
    if vertical:
        strip = mask[roi.y : roi.y + roi.h, :]
        return (strip > 0).mean(axis=0)
    strip = mask[:, roi.x : roi.x + roi.w]
    return (strip > 0).mean(axis=1)


def best_roi_red_fraction(
    mask: np.ndarray, roi: Roi, vertical: bool, search_px: int, step: int = 2
) -> tuple[float, int, bool]:
    """ROI を探索軸方向にずらして最大の赤割合、そのオフセット[px]、飽和したかを返す。

    帯が ROI より細いと最大値は**平坦**になる (帯を含む位置はどれも同じ割合) ので、
    オフセットは平坦部の**中心**を返す。すると「ROI を帯に乗せるのに必要な移動量」
    = ほぼ「帯が校正位置からずれた量」になり、セル内の位置ずれの観測値としても
    使える (#54 の画素->mm 投影の足がかり)。

    第 3 要素の **飽和フラグ**は「探索がフレーム端で打ち切られ、その端が最良だった」
    ことを表す。ROI をずらした先が画像の外へ出る位置は評価できないので、帯がそこより
    遠くへ動いていても最後に評価できた位置が返る — つまり**ずれが頭打ちになり、
    それらしい小さい値に化ける** (#21 実測: BACK ROI は下端まで +25px しか動かせず、
    機体を 20mm 前進させたのに読みは 7mm しか動かなかった)。位置補正に使う側は
    このフラグが立った辺を捨てること。壁の有無判定には引き続き使ってよい。
    """
    prof = _band_profile(mask, roi, vertical)
    start, length = (roi.x, roi.w) if vertical else (roi.y, roi.h)
    limit = len(prof) - length
    if limit < 0:
        return (0.0, 0, False)
    offsets = [o for o in range(-search_px, search_px + 1, max(1, step))
               if 0 <= start + o <= limit]
    if not offsets:
        return (0.0, 0, False)
    best_value = -1.0
    best_offsets: list[int] = []
    for offset in offsets:
        s = start + offset
        value = float(prof[s : s + length].mean())
        if value > best_value + 1e-12:
            best_value, best_offsets = value, [offset]
        elif value > best_value - 1e-12:
            best_offsets.append(offset)
    chosen = best_offsets[len(best_offsets) // 2]
    # 端が最良で、かつその端が「フレームに切られた端」なら飽和 (もっと先を見たかった)
    saturated = ((chosen == offsets[-1] and offsets[-1] < search_px)
                 or (chosen == offsets[0] and offsets[0] > -search_px))
    return (best_value, chosen, saturated)


class WallDetector:
    """ROI ごとの赤割合で機体相対の壁有無を判定する。"""

    def __init__(self, config: WallDetectorConfig) -> None:
        self.cfg = config

    def _mask(self, bgr: np.ndarray) -> np.ndarray:
        mask = red_mask(bgr, self.cfg.red)
        for r in self.cfg.exclude:                              # 自己遮蔽領域を除外
            mask[r.y : r.y + r.h, r.x : r.x + r.w] = 0
        return mask

    def measure(self, bgr: np.ndarray) -> dict[str, tuple[float, int, bool]]:
        """各辺の (赤割合, 帯のずれ[px], 飽和したか)。``search_px=0`` なら固定 ROI。"""
        if self.cfg.frame_size is not None:
            h, w = bgr.shape[:2]
            if (w, h) != self.cfg.frame_size:
                raise ValueError(
                    f"フレーム {w}x{h} と ROI の校正サイズ "
                    f"{self.cfg.frame_size[0]}x{self.cfg.frame_size[1]} が違う。"
                    "ROI が帯から外れて壁を見落とす (#56)"
                )
        mask = self._mask(bgr)
        out = {}
        for d, roi in self.cfg.rois.items():
            vertical = d in (LEFT, RIGHT)
            if self.cfg.search_px <= 0:
                out[d] = (roi_red_fraction(mask, roi), 0, False)
            else:
                out[d] = best_roi_red_fraction(
                    mask, roi, vertical, self.cfg.search_px, self.cfg.search_step
                )
        return out

    def red_fractions(self, bgr: np.ndarray) -> dict[str, float]:
        return {d: v[0] for d, v in self.measure(bgr).items()}

    def band_offsets(self, bgr: np.ndarray) -> dict[str, int]:
        """各辺の帯が校正位置からずれていた量[px] (壁が無い辺の値は無意味)。"""
        return {d: v[1] for d, v in self.measure(bgr).items()}

    def detect(self, bgr: np.ndarray) -> dict[str, bool]:
        """機体相対 (front/back/left/right) の壁有無。"""
        return {
            d: f >= self.cfg.threshold_for(d)
            for d, f in self.red_fractions(bgr).items()
        }


def band_positions(mask: np.ndarray, min_fraction: float = 0.25) -> dict[str, tuple[int, int]]:
    """赤マスクから 4 辺の帯の位置を測る (:data:`CALIBRATED_BANDS` を取り直す手順)。

    行/列プロファイルの山を探すだけの素朴な実装。FRONT/BACK は行 (y)、LEFT/RIGHT は
    列 (x) の範囲を返し、見つからない辺は入らない。**カメラの取付・高さ・画角を変えたら
    必ずこれで測り直す** — ROI が帯から外れると壁ありでも赤割合が出ず、機体が壁に
    突っ込む (#56 がまさにそれ)。

    帯は画像の中央寄りに 4 本出るので、上下・左右それぞれで中央から外側へ探し、
    最初に見つかった山を採る (中央は自機で占有されているので山にならない)。
    """
    h, w = mask.shape[:2]
    rows = (mask > 0).mean(axis=1)
    cols = (mask > 0).mean(axis=0)

    def run_from(profile, start, step):
        """``start`` から ``step`` 方向へ進み、最初の山 (連続して閾値以上) の範囲。"""
        i, n = start, len(profile)
        while 0 <= i < n and profile[i] < min_fraction:
            i += step
        if not (0 <= i < n):
            return None
        j = i
        while 0 <= j + step < n and profile[j + step] >= min_fraction:
            j += step
        return (min(i, j), max(i, j))

    out: dict[str, tuple[int, int]] = {}
    for name, prof, start, step in (
        (FRONT, rows, h // 2, -1), (BACK, rows, h // 2, +1),
        (LEFT, cols, w // 2, -1), (RIGHT, cols, w // 2, +1),
    ):
        found = run_from(prof, start, step)
        if found is not None:
            out[name] = found
    return out


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


def maze_walls_to_body(
    walls_maze: dict[Direction, bool], facing: Direction
) -> dict[str, bool]:
    """:func:`body_walls_to_maze` の逆写像 (迷路方角 -> 機体相対)。

    既知の迷路から「その姿勢でカメラに見えるはずの壁」を作れるので、シミュレーション
    と**校正データのラベル付け** (既知形状を走って撮る、#56) に使う。
    """
    return {
        FRONT: walls_maze[facing],
        BACK: walls_maze[Direction((facing + 2) % 4)],
        LEFT: walls_maze[Direction((facing - 1) % 4)],
        RIGHT: walls_maze[Direction((facing + 1) % 4)],
    }


#: 進路チェック (地図が開いていると言う方角に壁が見えないか) に使う最低赤割合。
#: **壁の有無判定より高くする。** 問うている内容が違うため:
#:   壁検出   … 「壁はあるか」。前提なし。見落とし = 衝突なのでしきい値は低く (0.08)
#:   進路確認 … 「地図は開いていると言うが本当か」。**開いている公算が高い**という前提が
#:               あり、誤検出は走行を無駄に終わらせる
#: 実測の分離 (#65, 113 フレーム / 452 ラベル) は 壁なし <= 0.121 / 壁あり >= 0.302 なので、
#: その間を取る。実機 (#76) では right の 0.09 で最速ランが誤って中止された。
#: なお本当に危険な状況 (姿勢がずれて壁を見失う) では赤割合は 0.00 まで落ちるので、
#: この値を上げても検出力はほとんど落ちない。
PATH_BLOCK_MIN_FRACTION = 0.20


def path_block_threshold(cfg: "WallDetectorConfig", edge: str) -> float:
    """進路チェックのしきい値 (辺ごとの壁判定値と :data:`PATH_BLOCK_MIN_FRACTION` の大きい方)。"""
    return max(cfg.threshold_for(edge), PATH_BLOCK_MIN_FRACTION)


def body_edge_for(direction: Direction, facing: Direction) -> str:
    """迷路方角 ``direction`` の壁が、``facing`` を向いた機体のどの辺に写るかを返す。

    :func:`maze_walls_to_body` の「辺だけ」版。**進もうとしている方角の壁を見る**のに使う
    (ホロノミック走行では進行方向と機体の向きが一致しないので、「前方を確認する」では
    足りない)。``facing`` を北に固定すると N→FRONT / S→BACK / W→LEFT / E→RIGHT の
    定数写像になる。
    """
    return _BODY_EDGE_BY_QUARTER[(direction - facing) % 4]


# facing からの時計回りの 90° 単位のずれ -> 機体の辺
_BODY_EDGE_BY_QUARTER = {0: FRONT, 1: RIGHT, 2: BACK, 3: LEFT}


def update_maze_walls(maze, x: int, y: int, walls_maze: dict[Direction, bool]) -> None:
    """判定した迷路方角の壁有無をセル (x, y) に反映する (共有エッジで隣接にも反映)。"""
    for d, present in walls_maze.items():
        maze.set_wall(x, y, d, present)
