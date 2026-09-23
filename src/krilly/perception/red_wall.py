"""赤い壁上部の検出 (issue #7)。白・黄の壁上面 (始点・終点、規定 2-1) は #125。

下向きのカメラは迷路の壁を捉える。壁の上部は **赤** に塗られている。
このモジュールは BGR フレームを赤のマスクと、赤い領域 (壁上部) の重心へと変換する。
純粋に OpenCV/NumPy のみで実装しているため、カメラなしでもユニットテストできる。

赤は HSV の hue 境界をまたぐため、2 つの hue 範囲 (低域と高域) を OR で結合する。
デフォルト値はあくまで出発点である。競技規則では壁が退色したり混ざったりすると
注意されているので、実際の迷路 / 照明に合わせて S/V の下限を調整し、
カメラの露出と AWB をロックすること。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RedDetectorConfig:
    """HSV しきい値 (OpenCV の範囲: H 0-179, S/V 0-255) とフィルタリング設定。"""

    h1_lo: int = 0
    h1_hi: int = 10
    h2_lo: int = 160
    h2_hi: int = 179
    s_min: int = 100
    v_min: int = 70
    min_area: float = 100.0   # これより小さい赤の blob は無視する (px^2)
    morph_kernel: int = 3     # 0 で open/close によるノイズ除去を無効化


@dataclass(frozen=True)
class RedRegion:
    """検出された 1 つの赤い blob。"""

    cx: float
    cy: float
    area: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


def red_mask_parts(
    bgr: np.ndarray, config: RedDetectorConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """赤マスクを**色相の 2 帯に分けたまま**返す ``(h1, h2)`` (どちらも 0/255)。

    OpenCV の H は 0-179 の一周なので、赤は下端 (``h1_lo..h1_hi``、オレンジ寄り) と
    上端 (``h2_lo..h2_hi``、マゼンタ寄り) の 2 帯に割れる。:func:`red_mask` はこれを
    OR して 1 枚にするが、**どちらの帯で拾ったかは調べものの決め手になる**ので分けて
    取れるようにしてある。

    実例 (#87): リボンケーブルに黒テープを貼った後も 10% ほど赤判定が残った。単色で
    重ねても原因が分からなかったが、2 帯を別の色で塗ると一目で分かった — 壁は
    すべて h2 (H 169) なのに、ケーブル上の画素は h1 (H 7) だった。隣のテープが H 15
    だったことから、**テープ自身が ``h1_hi`` の境界をまたいでいる**と特定できた
    (ラベルの貼り残しでも壁の映り込みでもない)。

    形態学的なノイズ除去は :func:`red_mask` と同じものを**各帯に**掛ける。そのため
    ``h1 | h2`` は :func:`red_mask` と厳密には一致しない (2 帯の境目で片方だけでは
    小さすぎて消える塊が、合わせれば残ることがある。実測で 1% 前後)。**判定には
    :func:`red_mask` を使い、この関数は調べもの専用にすること。**
    """
    cfg = config or RedDetectorConfig()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo1 = np.array([cfg.h1_lo, cfg.s_min, cfg.v_min], dtype=np.uint8)
    hi1 = np.array([cfg.h1_hi, 255, 255], dtype=np.uint8)
    lo2 = np.array([cfg.h2_lo, cfg.s_min, cfg.v_min], dtype=np.uint8)
    hi2 = np.array([cfg.h2_hi, 255, 255], dtype=np.uint8)
    parts = [cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2)]
    if cfg.morph_kernel > 0:
        k = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
        parts = [cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k),
                                  cv2.MORPH_CLOSE, k) for m in parts]
    return parts[0], parts[1]


@dataclass(frozen=True)
class WhiteYellowConfig:
    """白・黄の壁上面を拾うマスクの設定 (#125)。

    競技規定 2-1 は「壁の上面は赤」の例外として、**始点の区画と終点領域の区画の壁の
    上面を赤・白・黄のいずれか**にしてよいとしている。赤しか見ないと、ゴールの境界の
    壁 7 枚が全部「開」と読まれ、白い壁に突っ込む。

    **白は明るさの絶対値でも彩度でも拾えない** — 黒い床は天井灯を映して白っぽく
    光り (V 73-108)、その彩度は起動ごとに S 46-57 から S 11-30 まで動いて白 (S 6-39) と
    重なった (実測)。分けられるのは**形**だけ: 壁の上面は幅 ~20px の細い帯、床の光沢は
    広い斑。そこで帯に直交する向きの 1 次元トップハット (``tophat_px`` 幅の opening を
    引く = 「周囲よりどれだけ明るいか」) を取り、広い光沢を消して細い帯だけを残す。
    ``tophat_px`` は帯の幅より十分広く、光沢の斑より狭いこと。

    ``white_s_max`` は光沢を分けるためのものでは**ない**。機体を覆うフェルト・テープ
    (S 180-255) の縁を落とすため。外すとフェルトの縁が BACK で 0.19-0.39 と読める。
    **白の彩度は AWB のロックで動き、壁ごとにも違う**ので、白の側に寄せて置かないこと。
    最初は 60 にしていて、3x3 の実走で白い壁が S 64-73 (青みがかった白) と写り、
    **0.05 と読んでその壁に突っ込んだ**。同じ白い壁が前のセッションでは S 6-39 だった。
    80 に上げた直後にも、別の白い壁 (始点の西) が **S 中央 77・上側 5% で 86** と写り、
    帯の 27% が落ちて 0.29 まで弱った。**白の実測の端に合わせて上げるのをやめ**、
    落としたい側 (テープ S 180-) との中間の 120 に置く。

    ``min_width_px`` は床板の継ぎ目を落とすため。継ぎ目は幅 3px の明るい線
    (H 35 / S 30 / V 100) で、トップハットにも白にも入る (実走で開いた辺が 0.052)。
    壁の上面は ~20px あるので、帯に直交する向きにこの幅で opening を掛ければ線だけ消える。

    黄は有彩色なので色相で素直に取れる (H 25-29)。**木の床はこの黄に丸ごと入る** —
    EV -2 で撮った黒い床の外側の木の床は全面が黄と判定された。だから黒い床専用。
    """

    tophat_px: int = 41          # 帯に直交する向きのトップハット幅 [px]
    contrast_min: int = 40       # 周囲より明るい量 (V のトップハット) の下限
    white_s_max: int = 120       # 白とみなす彩度の上限 (フェルト・テープ S 180- を落とす)
    min_width_px: int = 9        # 帯に直交する向きで、これより細い線 (継ぎ目) を落とす
    yellow_h_lo: int = 18
    yellow_h_hi: int = 40
    yellow_s_min: int = 80
    yellow_v_min: int = 80


def white_yellow_mask_parts(
    bgr: np.ndarray, config: WhiteYellowConfig | None = None, vertical: bool = False,
    hsv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """白・黄の壁上面のマスクを ``(white, yellow)`` で返す (どちらも 0/255、#125)。

    ``vertical`` は帯が縦 (LEFT/RIGHT) かどうか。トップハットは帯に**直交する**向きに
    掛ける (縦帯なら横方向) — 帯に沿った向きに掛けると帯そのものが「広い構造」として
    消える。``hsv`` を渡せば色空間変換を省く (1 フレームで縦横 2 回呼ぶため)。
    """
    cfg = config or WhiteYellowConfig()
    if hsv is None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kernel = (np.ones((1, cfg.tophat_px), np.uint8) if vertical
              else np.ones((cfg.tophat_px, 1), np.uint8))
    contrast = cv2.morphologyEx(np.ascontiguousarray(hsv[..., 2]), cv2.MORPH_TOPHAT, kernel)
    white = ((contrast >= cfg.contrast_min)
             & (hsv[..., 1] <= cfg.white_s_max)).astype(np.uint8) * 255
    yellow = cv2.inRange(
        hsv,
        np.array([cfg.yellow_h_lo, cfg.yellow_s_min, cfg.yellow_v_min], dtype=np.uint8),
        np.array([cfg.yellow_h_hi, 255, 255], dtype=np.uint8),
    )
    if cfg.min_width_px > 1:
        thin = (np.ones((1, cfg.min_width_px), np.uint8) if vertical
                else np.ones((cfg.min_width_px, 1), np.uint8))
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, thin)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, thin)
    return white, yellow


def white_yellow_mask(
    bgr: np.ndarray, config: WhiteYellowConfig | None = None, vertical: bool = False,
    hsv: np.ndarray | None = None,
) -> np.ndarray:
    """白または黄の壁上面のマスク (0/255)。:func:`white_yellow_mask_parts` の OR。"""
    white, yellow = white_yellow_mask_parts(bgr, config, vertical, hsv)
    return cv2.bitwise_or(white, yellow)


#: 色相を信じてよい最低の彩度・明度。**これ未満の画素の H は意味を持たない** —
#: OpenCV は黒 (S=0,V=0) の H を 0 と返すので、赤の下側の帯 (h1_lo=0) にそのまま
#: 入ってしまう。床や影を「色相は赤なのに彩度で落ちた」と数えると、s_min を下げれば
#: 拾えるように見えてしまうが、実際には拾うものが無い。
HUE_MEANINGFUL_S = 40
HUE_MEANINGFUL_V = 40


@dataclass(frozen=True)
class RedBreakdown:
    """ある領域の画素が赤マスクを**どの条件で落ちたか** (issue #78)。

    「帯が弱く写った」だけでは何を直せばよいか決まらない。色相が帯の外へ流れたのか、
    彩度が足りないのか、そもそも赤いものが写っていないのかで、打つ手がまるで違う
    (#65 は色相、#56 は彩度、#88 は遮蔽だった)。**しきい値を下げる前にこれを見ること。**
    """

    total: int                                   # 領域の画素数
    accepted: int                                # 赤と判定された画素
    colorless: int                               # 無彩色・暗すぎて色相が意味を持たない
    hue_out: int                                 # 有彩色だが色相が赤の帯の外
    lost_to_s: int                               # 色相は赤だが S が足りない
    lost_to_v: int                               # 色相は赤、S も足りるが V が足りない
    hsv_accepted: tuple[float, float, float] | None   # 赤と判定された画素の中央値
    hsv_lost: tuple[float, float, float] | None       # 有彩色なのに落ちた画素の中央値

    @property
    def fraction(self) -> float:
        return self.accepted / self.total if self.total else 0.0

    @property
    def reason(self) -> str:
        """一番効いている落ち方。**これが次に直す対象。**"""
        lost = {"色相が帯の外": self.hue_out, "彩度 (s_min)": self.lost_to_s,
                "明度 (v_min)": self.lost_to_v}
        worst = max(lost, key=lambda k: lost[k])
        if lost[worst] == 0:
            return ("赤い画素はすべて採用された (弱いのは遮蔽か、そもそも壁が"
                    "写っていない)")
        if self.accepted >= lost[worst]:
            return f"取りこぼしの主因は {worst} (採用の方が多い)"
        return f"**取りこぼしの主因は {worst}**"

    def describe(self) -> str:
        def hsv(values):
            return "  --           " if values is None else "H%3.0f S%3.0f V%3.0f" % values

        return ("採用 %5d/%5d (%.2f) | 無彩色 %5d 色相外 %5d S落ち %5d V落ち %5d | "
                "採用 %s | 落ち %s | %s" % (
                    self.accepted, self.total, self.fraction, self.colorless,
                    self.hue_out, self.lost_to_s, self.lost_to_v,
                    hsv(self.hsv_accepted), hsv(self.hsv_lost), self.reason))


def red_breakdown(bgr: np.ndarray,
                  config: RedDetectorConfig | None = None) -> RedBreakdown:
    """領域内の画素を「赤マスクのどの条件で落ちたか」で分類する (#78)。

    渡すのは ROI で切り出した小片 (``roi.view(frame)``)。**形態学的なノイズ除去は
    掛けない** — ここで知りたいのは画素そのものの色であって、塊としての生き残りでは
    ないから。したがって ``accepted`` は :func:`red_mask` の結果と厳密には一致しない。
    """
    cfg = config or RedDetectorConfig()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colored = (s >= HUE_MEANINGFUL_S) & (v >= HUE_MEANINGFUL_V)
    hue_ok = colored & (((h >= cfg.h1_lo) & (h <= cfg.h1_hi))
                        | ((h >= cfg.h2_lo) & (h <= cfg.h2_hi)))
    s_ok = s >= cfg.s_min
    accepted = hue_ok & s_ok & (v >= cfg.v_min)
    lost_s = hue_ok & ~s_ok
    lost_v = hue_ok & s_ok & (v < cfg.v_min)

    def median(mask):
        if not mask.any():
            return None
        px = hsv[mask]
        return (float(np.median(px[:, 0])), float(np.median(px[:, 1])),
                float(np.median(px[:, 2])))

    return RedBreakdown(
        total=int(h.size), accepted=int(accepted.sum()),
        colorless=int((~colored).sum()), hue_out=int((colored & ~hue_ok).sum()),
        lost_to_s=int(lost_s.sum()), lost_to_v=int(lost_v.sum()),
        hsv_accepted=median(accepted),
        hsv_lost=median((colored & ~hue_ok) | lost_s | lost_v),
    )


def red_mask(bgr: np.ndarray, config: RedDetectorConfig | None = None) -> np.ndarray:
    """BGR 画像中の赤ピクセルを表す uint8 (0/255) のマスクを返す。"""
    cfg = config or RedDetectorConfig()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo1 = np.array([cfg.h1_lo, cfg.s_min, cfg.v_min], dtype=np.uint8)
    hi1 = np.array([cfg.h1_hi, 255, 255], dtype=np.uint8)
    lo2 = np.array([cfg.h2_lo, cfg.s_min, cfg.v_min], dtype=np.uint8)
    hi2 = np.array([cfg.h2_hi, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))
    if cfg.morph_kernel > 0:
        k = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_red_regions(
    bgr: np.ndarray, config: RedDetectorConfig | None = None
) -> list[RedRegion]:
    """赤い blob を検出し、重心 / bbox として面積の大きい順に返す。"""
    cfg = config or RedDetectorConfig()
    mask = red_mask(bgr, cfg)
    # findContours は cv2 4.x では (contours, hierarchy) を、3.x では (img, ...) を返す
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    regions: list[RedRegion] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        regions.append(RedRegion(cx, cy, float(area), cv2.boundingRect(c)))
    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def annotate(bgr: np.ndarray, regions: list[RedRegion]) -> np.ndarray:
    """動作確認用にバウンディングボックスと重心を描画する。コピーを返す。"""
    out = bgr.copy()
    for r in regions:
        x, y, w, h = r.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (int(round(r.cx)), int(round(r.cy))), 4, (255, 0, 0), -1)
    return out
