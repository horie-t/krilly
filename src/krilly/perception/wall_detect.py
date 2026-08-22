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
#: 解像度ごとに実測した帯の位置。``camera_fov --emit-bands`` が出す値をそのまま貼る。
#: **定規でセル中央に置いた姿勢で測ること** — ここが位置測定のゼロ点になる (#64)。
#:
#: **カメラの傾きを直したら必ず測り直す** (#87 -> #88)。傾きを 4.3° から 0.87° へ
#: 追い込んだだけで FRONT/BACK の帯が 41px 動き、それまでの ROI とは**重なりが 0px**に
#: なった。壁ありでも赤割合が出ず、機体が壁へ突っ込む状態だった。
CALIBRATED_BANDS_BY_SIZE = {
    # 640x480 (センサーの 33% しか使わない従来モード)。傾き調整後に再測定。
    (640, 480): {
        FRONT: (79, 101),
        BACK: (384, 406),
        LEFT: (150, 173),
        RIGHT: (458, 480),
    },
    # 960x720 全画素モード (#88)。画角 1.5 倍・px/mm は据え置き 1.70。
    (960, 720): {
        FRONT: (199, 220),
        BACK: (504, 526),
        LEFT: (310, 333),
        RIGHT: (618, 639),
    },
}

#: 実機で使う撮影サイズ。**`hal.camera.Camera` の既定と対で変えること**
#: (食い違うと ROI が帯から外れる。``WallDetector.measure`` が検算する)。
DEFAULT_FRAME_SIZE = (960, 720)

#: 既定サイズの帯 (後方互換のための別名)。
CALIBRATED_BANDS = CALIBRATED_BANDS_BY_SIZE[DEFAULT_FRAME_SIZE]


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

    def wall_center(self, edge: str, cell: tuple[int, int] = (0, 0),
                    half_pitch_mm: float = 90.0,
                    pitch_mm: float = 180.0) -> tuple[float, float]:
        """機体から ``cell`` セル離れたセルの ``edge`` 側の壁が写る位置 [px]。

        ``cell`` は**機体相対**の (前方セル数, 左方セル数)。戻り値は
        (帯に直交する座標, 帯に沿う座標) で、縦帯 (LEFT/RIGHT) なら (x, y)、
        横帯 (FRONT/BACK) なら (y, x)。

        画像と機体の対応は「画像の上 = 機体前方 (+x)、画像の左 = 機体左 (+y)」なので、
        機体座標 [mm] から画素へは**符号が反転**する (前へ行くほど画像の上 = y が小)。
        """
        forward_mm, left_mm = cell[0] * pitch_mm, cell[1] * pitch_mm
        sign = +1.0 if edge in (FRONT, LEFT) else -1.0
        if edge in (LEFT, RIGHT):
            cross = self.cell_center[0] - (left_mm + sign * half_pitch_mm) * self.px_per_mm_x
            along = self.cell_center[1] - forward_mm * self.px_per_mm_y
        else:
            cross = self.cell_center[1] - (forward_mm + sign * half_pitch_mm) * self.px_per_mm_y
            along = self.cell_center[0] - left_mm * self.px_per_mm_x
        return (cross, along)

    def band_center(self, edge: str, half_pitch_mm: float = 90.0) -> float:
        """その辺の帯の中心 (縦帯なら列 x、横帯なら行 y)。"""
        return self.wall_center(edge, half_pitch_mm=half_pitch_mm)[0]

    def roi(self, edge: str, spec: RoiSpec, cell: tuple[int, int] = (0, 0)) -> Roi:
        """帯の上に ROI を置く。**ROI の中心が、その軸の位置測定のゼロ点**になる。

        ``cell`` を与えると隣のセルの壁を見る ROI になる (#89)。自セル (既定) では
        従来どおり。
        """
        vertical = edge in (LEFT, RIGHT)
        cross, along_base = self.wall_center(edge, cell)
        along_scale = self.px_per_mm_y if vertical else self.px_per_mm_x
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

    def target_roi(self, target: "WallTarget", spec: RoiSpec) -> Roi:
        return self.roi(target.edge, spec, target.cell)

    def contains(self, roi: Roi) -> bool:
        """ROI がフレームに収まっているか (隣セルの ROI は端に来るので要確認)。"""
        return (roi.x >= 0 and roi.y >= 0
                and roi.x + roi.w <= self.width and roi.y + roi.h <= self.height)


@dataclass(frozen=True)
class WallTarget:
    """「どのセルのどの辺の壁を見るか」(機体相対) (#89)。

    ``cell`` は機体からの (前方セル数, 左方セル数)。自セルは (0, 0)。全画素モード
    (#88) では左右の隣セルが画角に入るので、その 4 壁まで読める。
    """

    edge: str
    cell: tuple[int, int] = (0, 0)

    @property
    def vertical(self) -> bool:
        """帯が縦 (LEFT/RIGHT) か。帯探索の向きを決める。"""
        return self.edge in (LEFT, RIGHT)


#: 隣セルとして読む側 (機体の左右)。前後の隣は画角が 66mm 足りず読めない (#89)。
NEIGHBOR_SIDES = (LEFT, RIGHT)

#: 隣セルの向き -> 機体相対のセル座標 (前方セル数, 左方セル数)。
_NEIGHBOR_CELL = {LEFT: (0, +1), RIGHT: (0, -1)}


def neighbor_slot(side: str, edge: str) -> str:
    """隣セルの壁を指すスロット名 (``"left:front"`` など)。"""
    return f"{side}:{edge}"


def neighbor_targets(sides: tuple[str, ...] = NEIGHBOR_SIDES) -> dict[str, WallTarget]:
    """隣セルの「自分では見えない 3 辺」の観測対象。

    共有辺 (左の隣セルの RIGHT = 自セルの LEFT) は自セルの測定をそのまま使うので
    ROI を作らない。1 枚の壁を 2 つの ROI で測ると、食い違ったときにどちらを信じるか
    という問題が増えるだけで、得るものが無い。
    """
    out: dict[str, WallTarget] = {}
    for side in sides:
        cell = _NEIGHBOR_CELL[side]
        for edge in (FRONT, BACK, side):
            out[neighbor_slot(side, edge)] = WallTarget(edge, cell)
    return out


#: 解像度ごとの校正済み幾何。実測した帯から起こす。
CALIBRATED_GEOMETRIES = {
    size: CameraGeometry.from_bands(bands, *size)
    for size, bands in CALIBRATED_BANDS_BY_SIZE.items()
}

#: 既定サイズの幾何 (後方互換のための別名)。
CALIBRATED_GEOMETRY = CALIBRATED_GEOMETRIES[DEFAULT_FRAME_SIZE]


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


#: 隣セルの壁を見る ROI の形 (#89)。**自セルより薄い**のが要点:
#:
#: - 遠い側の壁 (自セル中心から 270mm) は 960 幅のフレームの端 (x=14.5 / 935.5) に写る。
#:   自セルと同じ 27mm 厚 (46px) では枠から食み出すので 16mm (27px) に絞る。
#: - 隣セルの前後の壁は自セルの帯と同じ行に、180mm 横へずれて写る。ケーブルは画像の
#:   中央下なので、BACK でも自セルのように短くする必要はない (支柱を避ける 106mm)。
#:
#: 実機実測 (5x5、壁あり / 壁なし): 前後 0.46-0.48 / 0.00、遠い側 0.81-0.82 / 0.00。
#: **遠い側が一番強い**のは ROI を薄くしてあるから (12mm の帯が 16mm の ROI をほぼ
#: 埋める)。隣の BACK も 0.47 出るので、自セルの BACK を短くしている理由 (ケーブル) は
#: 180mm 横では効かないことが確認できている。
NEIGHBOR_ROI_SPECS = {
    FRONT: RoiSpec(thickness_mm=26.0, length_mm=106.2),
    BACK: RoiSpec(thickness_mm=26.0, length_mm=106.2),
    LEFT: RoiSpec(thickness_mm=16.0, length_mm=126.7),
    RIGHT: RoiSpec(thickness_mm=16.0, length_mm=126.7),
}

#: 隣セルを「壁なし」と言い切れる赤割合の上限 (#89)。
#:
#: 自セルの判定は 2 値 (しきい値以上なら壁、未満なら壁なし) でよいが、**隣セルは
#: 3 値**にする — 壁 / 壁なし / **未確定**。理由は、未確定を「壁なし」と書くと
#: そのセルが「4 壁が分かったセル」に化け、**止まらずに通過する対象になってしまう**
#: から。見えなかっただけの壁へ全速で突っ込むことになる。
#: 判定が割れる帯 (この値以上・壁しきい値未満) のときは何も書かず、そのセルは
#: 通過対象から外れて普通に停止して観測する。停止 1 回で済むので安い。
NEIGHBOR_CLEAR_MAX_FRACTION = 0.04

#: 遠い側の帯の中心が、フレーム端からこれだけ内側に無ければ「壁なし」と言わない [px]。
#:
#: 遠い側の壁はセル中心から 270mm、フレーム端まで 281mm しかないので、帯の中心は
#: x=14.5 (左) / 935.5 (右) に写る。機体がその側へ寄ると帯は枠外へ出ていき、
#: **完全に出ると赤割合が 0 に落ちて「壁なし」と区別がつかなくなる**。
#: 逆に、帯の中心さえフレーム内にあれば帯の半分以上が写るので、赤割合は 0.4 前後出て
#: 壁として検出できる (半分に切れても十分しきい値を超える)。つまり
#: **危ないのは「中心が枠外」のときだけ**で、そこを 3px の余裕付きで除外すれば足りる。
#: 左の遠い帯なら、機体が右へ約 7mm ずれるまでは判定してよいということ。
#:
#: 実機で確かめた (5x5、左の遠い側に壁を立てて機体を右へずらす): 左右のずれ実測
#: +4.7mm で赤割合 0.82、-8.2mm で 0.36 (まだ壁と判定できる)、**-15.2mm で 0.00**。
#: つまり読めなくなるのは -15mm 付近で、この 3px は -6.7mm で止める = 13px ぶん
#: 保守側。**その余裕は残すこと** — 浮いた壁は単独でも弱く写る (#88 の 0.24)。
NEIGHBOR_FAR_MIN_MARGIN_PX = 3.0

#: 帯のずれを「機体のずれ」として信じてよい最低赤割合。位置の測定なので壁判定 (0.08)
#: より厳しくする — 弱い証拠で位置を語ってはいけない (#64)。
#: :data:`krilly.perception.cell_pose.OFFSET_MIN_FRACTION` と同じ値・同じ根拠で、
#: ``tests/test_wall_detect.py`` が両者の一致を固定している。
BAND_SHIFT_MIN_FRACTION = 0.25


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


def calibrated_neighbor_rois(geometry: CameraGeometry | None = None,
                             sides: tuple[str, ...] = NEIGHBOR_SIDES) -> dict[str, Roi]:
    """左右の隣セルの壁を見る ROI (#89)。キーは ``"left:front"`` 形式のスロット名。

    位置は自セルと同じ幾何から出る (壁は 180mm 格子の上にしか無い) ので、
    **新しく校正するものは無い**。確かめることがあるとすれば、フレーム端に来る
    遠い側の帯が本当にそこに写るかで、それは ``wall_detect --neighbors`` で見る。
    """
    g = geometry or CALIBRATED_GEOMETRY
    return {name: g.target_roi(t, NEIGHBOR_ROI_SPECS[t.edge])
            for name, t in neighbor_targets(sides).items()}


def calibrated_config(threshold: float = 0.08,
                      size: tuple[int, int] = DEFAULT_FRAME_SIZE,
                      neighbors: bool = False) -> "WallDetectorConfig":
    """実機校正済みの WallDetectorConfig を返す (ROI + 赤しきい値 + 帯探索)。

    しきい値は #65 の 5x5 手動調査 (``scripts/survey_shot.py``、113 枚 = 452 ラベル、
    木の床・3D プリント柱) の辺別分布から決めた:

    - front: 開 max 0.000 / 壁 min 0.302
    - back : 開 max 0.121 / 壁 min 0.509 (開側の 0.09-0.12 はリボンケーブルの偽帯)
    - left : 開 max 0.036 / 壁 min 0.344
    - right: 開 max 0.016 / 壁 min 0.379

    #65 当時は BACK だけ 0.25 に上げていた (ケーブルの偽帯 0.09-0.12 と分けるため)。
    **これは #88 で 0.08 へ下げた。** 理由が 2 つ:

    1. ケーブルに全長テープを貼り、全画素モードへ移った後の実測で、**壁なしの BACK は
       0.00** になった (偽帯が消えた)。0.25 を保つ理由が無くなった
    2. **0.25 のままで実際に壁を突き破った。** 5x5 の探索ランで本物の後方の壁が
       0.24 と出て、検出 (0.25) も進路確認 (max(0.25, 0.20) = 0.25) も**同時に**
       すり抜けた。辺別しきい値を上げると進路確認まで鈍るので、二重の防護が
       一重になっていた

    **壁の見落とし (衝突) の方が偽陽性 (回り道) より高くつく**ので、迷ったら
    下げる側に振ること。

    ``size`` は撮影サイズ。ROI はその幾何から作られ、``measure`` が実フレームと
    突き合わせる (食い違ったまま走ると壁を見落とす)。

    ``neighbors=True`` で左右の隣セルを読む ROI も足す (#89)。既定で切ってあるのは、
    フレーム端に来る遠い側の帯が実機で本当に読めるかを確かめてから使うため。
    """
    geometry = geometry_for(size)
    rois = calibrated_rois(geometry)
    slots = {e: WallTarget(e) for e in rois}
    if neighbors:
        rois |= calibrated_neighbor_rois(geometry)
        slots |= neighbor_targets()
    return WallDetectorConfig(
        rois=rois, slots=slots, threshold=threshold,
        red=CALIBRATED_RED, frame_size=size,
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

    #: スロット名 -> 「どのセルのどの辺を見るか」(#89)。``rois`` に有って
    #: ここに無いキーは自セルのその辺として扱う (従来の 4 辺だけの設定と互換)。
    slots: dict[str, WallTarget] = field(default_factory=dict)
    #: 隣セルを「壁なし」と言い切れる赤割合の上限 (これ以上・壁しきい値未満は未確定)。
    neighbor_clear: float = NEIGHBOR_CLEAR_MAX_FRACTION

    def target(self, name: str) -> WallTarget:
        """スロット名が見ている対象 (既定は自セルの同名の辺)。"""
        return self.slots.get(name) or WallTarget(name)

    def threshold_for(self, name: str) -> float:
        """このスロットの壁判定しきい値 (辺別の上書きが無ければ共通値)。

        隣セルのスロット (``"left:front"``) はまずスロット名で、無ければ**辺**で引く。
        """
        if name in self.thresholds:
            return self.thresholds[name]
        return self.thresholds.get(self.target(name).edge, self.threshold)


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
        for name, roi in self.cfg.rois.items():
            vertical = self.cfg.target(name).vertical
            if self.cfg.search_px <= 0:
                out[name] = (roi_red_fraction(mask, roi), 0, False)
            else:
                out[name] = best_roi_red_fraction(
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
            if self.cfg.target(d).cell == (0, 0)
        }

    def lateral_shift_px(
        self, measured: dict[str, tuple[float, int, bool]]
    ) -> float | None:
        """自セルの左右の帯から、機体が左右にどれだけずれているか [px] (+ = 画像の右)。

        :func:`krilly.perception.cell_pose.cell_offset` の左右成分と同じもの
        (あちらは mm に直して推定へ入れる)。ここでは**遠い側の帯がまだ写っているか**を
        判断するためだけに使うので、px のまま扱う。測れなければ None。
        """
        offsets = [float(measured[e][1]) for e in (LEFT, RIGHT)
                   if e in measured and not measured[e][2]
                   and measured[e][0] >= max(self.cfg.threshold_for(e),
                                             BAND_SHIFT_MIN_FRACTION)]
        return sum(offsets) / len(offsets) if offsets else None

    def neighbor_walls(
        self, measured: dict[str, tuple[float, int, bool]],
        sides: tuple[str, ...] = NEIGHBOR_SIDES,
        shift_px: float | None = None,
    ) -> dict[str, dict[str, bool]]:
        """左右の隣セルの壁有無を機体相対で返す (#89)。**未確定の辺はキーごと落とす。**

        戻り値は ``{"left": {"front": True, "back": False, ...}, ...}`` で、
        :meth:`krilly.strategy.explorer.Explorer.observe` の ``neighbors`` にそのまま渡せる。

        判定は 3 値 (:data:`NEIGHBOR_CLEAR_MAX_FRACTION` 参照):

        - 赤割合 >= しきい値 → 壁
        - 赤割合 <= ``neighbor_clear`` かつ**帯探索が飽和していない** → 壁なし
        - それ以外 → 未確定 (キーを入れない)

        **飽和を「壁なし」にしないこと**が要点。遠い側の壁はフレーム端 (x=14.5) に
        写るので、機体がその側へ 10mm ずれるだけで帯が枠外へ出て赤割合が 0 に落ちる
        (:func:`best_roi_red_fraction` が飽和フラグを立てる)。これを「壁なし」と
        書くと、見えなかった壁の向こうへ止まらずに突っ込む。

        **飽和だけでは足りない**ので、遠い側の辺はさらに「帯の中心がまだフレーム内か」を
        見る (:data:`NEIGHBOR_FAR_MIN_MARGIN_PX`)。帯が枠外へ**完全に**出ると赤割合は
        0 に落ち、探索プロファイルは平坦になるので飽和フラグすら立たない — 見えない
        ことと壁が無いことが区別できなくなる。機体の左右のずれは自セルの帯から測れる
        (:meth:`lateral_shift_px`) ので、測れなければ「壁なし」とは言わない。

        共有辺 (左の隣セルの RIGHT = 自セルの LEFT) は自セルの測定から埋める。
        """
        if shift_px is None:
            shift_px = self.lateral_shift_px(measured)
        out: dict[str, dict[str, bool]] = {}
        for side in sides:
            walls: dict[str, bool] = {}
            shared = _BODY_EDGE_BY_QUARTER[(_QUARTER_BY_BODY_EDGE[side] + 2) % 4]
            for edge, name in [(shared, side)] + [
                (e, neighbor_slot(side, e)) for e in (FRONT, BACK, side)
            ]:
                if name not in measured:
                    continue
                fraction, _offset, saturated = measured[name]
                if fraction >= self.cfg.threshold_for(name):
                    walls[edge] = True
                elif (fraction <= self.cfg.neighbor_clear and not saturated
                      and (edge is not side or self._far_band_in_frame(name, shift_px))):
                    walls[edge] = False
            if walls:
                out[side] = walls
        return out

    def _far_band_in_frame(self, name: str, shift_px: float | None) -> bool:
        """遠い側の帯が、いまの機体のずれでもフレーム内に写っているか。"""
        width = (self.cfg.frame_size or (0, 0))[0]
        if shift_px is None or not width:
            return False
        roi = self.cfg.rois[name]
        center = roi.x + roi.w / 2.0 + shift_px
        return (NEIGHBOR_FAR_MIN_MARGIN_PX <= center
                <= width - NEIGHBOR_FAR_MIN_MARGIN_PX)

    def detect_neighbors(self, bgr: np.ndarray,
                         sides: tuple[str, ...] = NEIGHBOR_SIDES
                         ) -> dict[str, dict[str, bool]]:
        """1 フレームから左右の隣セルの壁を読む (:meth:`neighbor_walls` の簡易版)。"""
        return self.neighbor_walls(self.measure(bgr), sides)


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
    """進路チェックのしきい値 (辺ごとの壁判定値と :data:`PATH_BLOCK_MIN_FRACTION` の大きい方)。

    **辺別しきい値をこの値より上げてはいけない。** 上げると進路確認まで一緒に鈍り、
    「壁検出をすり抜けたものを進路確認が拾う」という二重の防護が一重になる。
    #88 の探索ランで実際に起きた: BACK を 0.25 にしていたため、本物の壁の 0.24 が
    検出も進路確認も 0.01 差ですり抜け、機体が壁を突き破った。
    ``tests/test_wall_detect.py`` がこの関係を固定している。
    """
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

# 機体の辺 -> facing からの時計回りの 90° 単位のずれ (上の逆写像)
_QUARTER_BY_BODY_EDGE = {edge: q for q, edge in _BODY_EDGE_BY_QUARTER.items()}


def maze_direction_for(edge: str, facing: Direction) -> Direction:
    """機体の辺 ``edge`` が向いている迷路方角を返す (:func:`body_edge_for` の逆)。

    隣のセルを読むとき (#89) に「LEFT 側の隣」が迷路のどのセルかを決めるのに使う。
    ``facing`` を北に固定すると FRONT→N / RIGHT→E / BACK→S / LEFT→W の定数写像になる。
    """
    return Direction((facing + _QUARTER_BY_BODY_EDGE[edge]) % 4)


def path_check_slots(direction: Direction, facing: Direction, cells: int = 1,
                     neighbors: bool = False) -> list[str]:
    """進む前に「本当に開いているか」を確認できるスロット名を、進む順に返す。

    1 セル目の出口は自セルの辺に写る (:func:`body_edge_for`)。2 セル目の出口は
    **左右方向のときだけ**、隣セルの遠い側の帯として写る (#89)。前後方向の 2 セル目の
    出口は 270mm 先で、フレームの外 (±212mm) なので確認できない。

    確認できない分をここで返さないのは、**進路チェックの意味が「地図の確認」ではなく
    「自機の姿勢の確認」**だから (#76)。見えないものは見えないと言い、通過するセルの
    壁が観測済みであること (:attr:`~krilly.strategy.explorer.Explorer.known`) の方で
    担保する。
    """
    edge = body_edge_for(direction, facing)
    slots = [edge]
    if cells >= 2 and neighbors and edge in NEIGHBOR_SIDES:
        slots.append(neighbor_slot(edge, edge))
    return slots


def update_maze_walls(maze, x: int, y: int, walls_maze: dict[Direction, bool]) -> None:
    """判定した迷路方角の壁有無をセル (x, y) に反映する (共有エッジで隣接にも反映)。"""
    for d, present in walls_maze.items():
        maze.set_wall(x, y, d, present)
