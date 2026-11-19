"""壁の帯の切れ目 (壁の端) から、帯に沿った方向の機体の位置を読む (issue #189)。

走行中の融合 (#186/#187) は横ずれしか直せない: 進行方向と直角な帯は露光中にブレる
うえ、廊下の途中には直角な壁が無い。#187 の実機 (long8) では 7 セルの直線の後に
進行方向が 13〜20 mm ずれた。そこで、**進行方向と平行な帯 (横の壁) が途切れる位置**
を使う。赤外線センサのマウスでいう「壁切れ補正」をカメラでやる。

- 横の壁は、走る向きに沿って伸びる帯として写る。壁が無いところで帯の赤が途切れる。
- **切れ目の縁は、セルの境目 (柱の中心) から柱の半幅 ``EDGE_INSET_MM`` (6 mm) だけ切れ目の
  側**にある (柱の上面も赤い)。2026-11-18 の f6b6_030_img (黒い床、EV −2、0.30 m/s) で、
  同じフレームに写った「前が赤」と「後ろが赤」の縁から出した位置の食い違いが、柱の
  半幅を 6.1 mm と置くと 0 になった (22 フレーム、標準偏差 2.7 mm)。**最初に 2 値の
  マスクで縁を読んだときは 8.4 mm に見えた** — ブレで淡くなった縁がしきい値の分だけ
  内側に寄ったためで、柱の形ではなかった (:func:`redness`)。
- 帯に沿った方向のブレ (進行方向の露光中の移動) は縁を台形に崩すが、半値の位置は
  露光の中点の位置を指す (#184: 0.30 m/s で縁が約 15 px 崩れ、半値幅は変わらない)。
  だから縁は**半値で読む**。

座標: 縦の帯 (LEFT/RIGHT) は画像の y に沿って伸び、上が車体の前 (+x)。横の帯
(FRONT/BACK) は画像の x に沿って伸び、左が車体の左 (+y)。:func:`find_band_edges`
は「その帯に沿った車体の座標で、セル中央の写る位置から何 mm 先か」を返す。

機体に貼ったテープ (緑) が帯を横切るところは赤が欠けて切れ目に見えるので、
テープの色が帯の上に出ている行の近くの縁は使わない。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from krilly.perception.red_wall import (
    RedDetectorConfig,
    WhiteYellowConfig,
    red_mask,
    white_yellow_mask_parts,
)
from krilly.perception.wall_detect import (
    BACK,
    CALIBRATED_BANDS,
    CALIBRATED_GEOMETRY,
    CALIBRATED_RED,
    FRONT,
    LEFT,
    RIGHT,
    CameraGeometry,
)

#: 切れ目の縁の、セルの境目からの距離 [mm] = 柱 (12 mm 角) の半幅 (実測と一致、モジュールの説明)。
EDGE_INSET_MM = 6.0

#: 帯に沿った車体の軸。LEFT/RIGHT は前後 (x)、FRONT/BACK は左右 (y)。
BAND_AXIS = {LEFT: "x", RIGHT: "x", FRONT: "y", BACK: "y"}


@dataclass(frozen=True)
class BandEdge:
    """帯の切れ目の縁 1 つ。"""

    band: str            # どの帯 (LEFT/RIGHT/FRONT/BACK)
    axis: str            # 帯に沿った車体の軸 ("x" = 前後、"y" = 左右)
    offset_mm: float     # セル中央の写る位置から、その軸の + 向きに何 mm 先に縁があるか
    red_on_plus: bool    # 縁の + 側が赤 (壁)、- 側が切れ目なら True
    pos_px: float        # 画像上の縁の位置 (縦の帯なら y、横の帯なら x)
    #: 縁の先の柱の上面の色 ("red" / "white")。**赤の縁の位置が 12 mm 変わる**:
    #: 赤い柱なら赤は柱の向こう側の面 (境目 + 6 mm、切れ目の側) まで続き、白い柱なら
    #: 手前の面 (境目 − 6 mm、赤の側) で終わる (:func:`offset_from_edge`)。
    post: str = "red"


@dataclass(frozen=True)
class EdgeConfig:
    """縁の検出の設定。"""

    red: RedDetectorConfig = CALIBRATED_RED
    #: 帯の幅に足す余白 [px] (横ずれで帯が動いても列から外れないように)。
    margin_px: int = 10
    #: 縁の両側で、壁側は赤・切れ目側は暗いことを確かめる長さ [px]。縁から ``gap_px``
    #: 離して測る — 縁はブレで台形に崩れていて (0.30 m/s で約 15 px)、すぐ隣は中間の値になる。
    side_px: int = 15
    gap_px: int = 10
    #: 縁の両側の赤さ (R − G) の差がこれより小さければ使わない (半値が決まらない)。
    min_contrast: float = 6.0
    #: 切れ目の側で許す赤のマスクの割合 (帯の幅に対して)。**切れ目かどうかはマスクで決める**:
    #: 赤さ (R − max(G, B)) は明るさに比例するので、機体の影や LED の光で壁の上面が
    #: 半分の明るさになると「暗い = 切れ目」に見える。2026-11-19 の long8 (木の床、EV 0、
    #: ゲイン 4.1) で、つながった東の壁に機体と一緒に動く偽の切れ目が出て、融合が推定を
    #: 57.6 mm 引っ張り、機体は北の壁に当たった。マスク (色相と彩度) は影の中でも赤を拾う。
    max_gap_mask: float = 0.03
    #: 赤のマスクがこの割合以上の位置を「帯が写っている」とみなす (縁の候補はこの切り替わり)。
    min_plateau: float = 0.2
    #: テープ (緑) の行から何 px 以内の縁を捨てるか。
    occlusion_px: int = 20
    #: 画像の端から何 px 以内の縁を捨てるか。
    border_px: int = 8
    #: 柱の上面の色の扱い: "auto" = 縁ごとに判定 / "red" / "white" = 会場で決め打ち。
    #: 競技規則に柱の色の決まりは無く、交流会 (2026-10-17) は白、家の黒い床のショップの
    #: 柱と木の床の 3D 印刷の柱は赤、大会の動画では混在もありうる。
    post_top: str = "auto"
    #: 白い柱の判定: 縁から切れ目の側へ ``post_px`` の間で白のマスクの割合がこれ以上、
    #: その先 (``post_px`` + 10 px から 20 px) は白も赤も無い (床) こと。柱は 12 mm ≈ 20 px。
    post_px: int = 22
    white_post_ratio: float = 0.3
    #: 切れ目の側の白い区間がこれより長ければ、白い柱ではなく**白飛びか白い壁**とみなして
    #: 捨てる [px]。白飛びすると壁の上面の赤がマスクからも消えて切れ目に見える
    #: (2026-11-19 の衝突の (0,4) は部屋の光源に直角で、白飛びしやすい場所だった)。
    max_white_px: int = 34
    #: 白のマスクを作るとき、帯の周りに広げて切り出す幅 [px] (トップハットが 41 px 要る)。
    white_pad_px: int = 30


def _green_mask(bgr: np.ndarray) -> np.ndarray:
    """機体に貼ったテープの色 (緑)。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    return (h > 60) & (h < 100) & (s > 60)


def redness(bgr: np.ndarray) -> np.ndarray:
    """連続した「赤さ」``R − G`` (0 未満は 0)。

    **縁は 2 値のマスクではなくこれで読む。** ブレは明るさを線形に混ぜるので、半値の
    位置は保たれる。2 値のマスクは「しきい値を超えた所」で切れるので、ブレて淡く
    なった縁はしきい値の分だけ内側に寄り、切れ目が狭く見える (合成画像で 15 px の
    ブレのとき 3 mm ずつ)。

    **``R − max(G, B)`` にしないこと。** 昼の光 (窓から) で AWB をロックすると、壁の上面が
    赤ではなくマゼンタ寄りのピンクに写り、B が R と同じくらい高くなる。2026-11-19 の
    黒い床の昼の走行 (b_red) では ``R − max(G, B)`` の平らな部分が 8 しかなく、縁の
    3 分の 2 を取りこぼした (赤のマスクは色相 125-180 も赤とみなすので 0.56 と読めて
    いた)。``R − G`` なら赤でもマゼンタでも高く、白・灰・黒い床では 0 に近い。
    """
    g, r = bgr[..., 1].astype(np.int16), bgr[..., 2].astype(np.int16)
    return np.clip(r - g, 0, None).astype(np.float32)


def _profile(mask: np.ndarray, band: str, lo: int, hi: int) -> np.ndarray:
    """帯の列 (または行) で、帯に沿った位置ごとの割合を出す。"""
    if band in (LEFT, RIGHT):
        return mask[:, lo:hi].mean(axis=1)
    return mask[lo:hi, :].mean(axis=0)


def find_band_edges(
    bgr: np.ndarray,
    bands: tuple[str, ...] = (LEFT, RIGHT, FRONT, BACK),
    shifts_px: dict[str, int] | None = None,
    config: EdgeConfig | None = None,
    geometry: CameraGeometry | None = None,
) -> list[BandEdge]:
    """``bands`` の帯で、赤が途切れる縁を探す。

    ``shifts_px`` は :meth:`WallDetector.measure` が返す帯のずれ (横ずれで帯が動いた量)。
    無ければ校正位置の帯をそのまま見る (``margin_px`` の範囲なら追える)。

    **帯の部分だけ**を切り出して色を計算する (全画面だと 1 枚 9.4 ms、走行中に毎フレーム
    回すには重い)。
    """
    cfg = config or EdgeConfig()
    g = geometry or CALIBRATED_GEOMETRY
    out: list[BandEdge] = []
    for band in bands:
        vertical = band in (LEFT, RIGHT)
        lo, hi = CALIBRATED_BANDS[band]
        shift = (shifts_px or {}).get(band, 0)
        across = bgr.shape[1] if vertical else bgr.shape[0]
        lo = max(0, lo + shift - cfg.margin_px)
        hi = min(across, hi + shift + cfg.margin_px)
        if hi - lo < 4:
            continue
        # 帯の周りを広めに切り出し、帯の赤の中心を帯に沿って直線で当てはめ、その直線に
        # 沿って帯の幅だけ取り出す。**固定の位置で切り出すと、傾いて写った壁 (機体の向き +
        # カメラの取り付けの回り) が画面の端で切り出しから外れて「切れ目」に見える**
        # (2026-11-19 の long8: 画面の端から端まで続く北の壁の右の方に偽の縁)。
        wlo, whi = max(0, lo - cfg.white_pad_px), min(across, hi + cfg.white_pad_px)
        wide = bgr[:, wlo:whi] if vertical else bgr[wlo:whi, :]
        white_wide, _ = white_yellow_mask_parts(wide, WhiteYellowConfig(), vertical=vertical)
        layers = [redness(wide), (red_mask(wide, cfg.red) > 0).astype(np.float32),
                  _green_mask(wide).astype(np.float32), (white_wide > 0).astype(np.float32)]
        if not vertical:                  # 帯に沿う向きを 0 軸にそろえる
            layers = [a.T for a in layers]
        red_w, mask_w, green_w, white_w = layers
        half_w = (hi - lo) // 2
        center = _band_center_line(mask_w, default=(lo + hi) / 2.0 - wlo)
        cols = np.clip(np.round(center)[:, None].astype(int)
                       + np.arange(-half_w, half_w + 1)[None, :], 0, mask_w.shape[1] - 1)

        def straight(a: np.ndarray) -> np.ndarray:
            return np.take_along_axis(a, cols, axis=1).mean(axis=1)

        p = straight(red_w)
        mask_frac = straight(mask_w)        # 帯の幅に対する赤のマスクの割合
        present = mask_frac > cfg.min_plateau
        occ = straight(green_w) > 0.05
        white_frac = straight(white_w)
        length = len(p)
        if int(present.sum()) < cfg.side_px:
            continue
        occ_idx = np.flatnonzero(occ)
        s, gap = cfg.side_px, cfg.gap_px
        # 候補は**赤のマスクの切り替わり**。半値は縁の両側の**その場の**赤さから取る —
        # 帯の赤さは場所で 2 倍以上違う (遠近で見える壁の白い側面が混ざる、影、ブレ) ので、
        # 帯全体の中央値を基準にすると赤さの低い所の縁を落とす (2026-11-19 の b_red で 3 分の 2)。
        found = set()
        for i in np.flatnonzero(present[:-1] != present[1:]):
            if i < s + gap or i > length - s - gap - 2:
                continue
            lo_w = slice(i - gap - s, i - gap + 1)
            hi_w = slice(i + 1 + gap, i + 1 + gap + s)
            low_mask, high_mask = float(np.mean(mask_frac[lo_w])), float(np.mean(mask_frac[hi_w]))
            # 切れ目の側は「赤がまったく無い」ことをマスクで確かめる (影で暗いだけの壁を除く)
            red_high = (high_mask >= cfg.min_plateau
                        and float(np.max(mask_frac[lo_w])) <= cfg.max_gap_mask)
            red_low = (low_mask >= cfg.min_plateau
                       and float(np.max(mask_frac[hi_w])) <= cfg.max_gap_mask)
            if not (red_high or red_low):
                continue
            red_level = float(np.mean(p[hi_w] if red_high else p[lo_w]))
            dark_level = float(np.mean(p[lo_w] if red_high else p[hi_w]))
            if red_level - dark_level < cfg.min_contrast:
                continue
            pos = _half_crossing(p, i, (red_level + dark_level) / 2.0, gap)
            if pos is None:
                continue
            # マスクの切り替わりはブレた縁の暗い側の端に寄るので、半値の位置を中心に
            # 両側の窓を取り直してもう一度 (台形の途中を「赤の側」に数えないため)
            c = int(round(pos))
            if s + gap <= c <= length - s - gap - 2:
                lo2, hi2 = slice(c - gap - s, c - gap + 1), slice(c + 1 + gap, c + 1 + gap + s)
                r2 = float(np.mean(p[hi2] if red_high else p[lo2]))
                d2 = float(np.mean(p[lo2] if red_high else p[hi2]))
                pos2 = _half_crossing(p, c, (r2 + d2) / 2.0, gap)
                if pos2 is not None:
                    pos = pos2
            key = round(pos)
            if key in found:
                continue
            found.add(key)
            if pos < cfg.border_px or pos > length - 1 - cfg.border_px:
                continue
            if len(occ_idx) and np.min(np.abs(occ_idx - pos)) <= cfg.occlusion_px:
                continue
            post = _classify_post(white_frac, mask_frac, pos, gap_up=red_low, cfg=cfg)
            if post is None:
                continue
            # 画像の座標を車体の軸へ: 縦の帯は上 (y 小) が前、横の帯は左 (x 小) が左
            if vertical:
                offset = (g.cell_center[1] - pos) / g.px_per_mm_y
            else:
                offset = (g.cell_center[0] - pos) / g.px_per_mm_x
            out.append(BandEdge(band, BAND_AXIS[band], float(offset),
                                bool(red_low),        # 座標の小さい側 = 車体の + 側
                                float(pos), post))
    return out


def _band_center_line(mask: np.ndarray, default: float, min_rows: int = 30,
                      min_px: int = 3) -> np.ndarray:
    """帯に沿った各位置での帯の中心 (幅の方向の画素)。

    **傾きだけ**を赤のマスクの重心の直線から取り、中心の位置は ``default`` (校正位置 +
    横ずれ) のまま画面の中央で合わせる。重心は遠近で見える壁の白い側面や交差する
    短い壁に引っ張られるので、位置まで当てはめると縁の精度が落ちた (黒い床の b_mixed で
    「前が赤 − 後ろが赤」の標準偏差が 2.9 → 5.3 mm)。

    ``mask`` は (帯に沿う向き, 幅の向き)。赤が ``min_px`` 画素以上ある位置が ``min_rows``
    に満たなければ傾き 0。外れ値を 1 回除いて当て直す。
    """
    n, w = mask.shape
    counts = mask.sum(axis=1)
    rows = np.flatnonzero(counts >= min_px)
    k = 0.0
    if len(rows) >= min_rows:
        cen = (mask[rows] * np.arange(w)[None, :]).sum(axis=1) / counts[rows]
        k, b = np.polyfit(rows, cen, 1)
        keep = np.abs(cen - (k * rows + b)) <= 4.0
        if keep.sum() >= min_rows:
            k, b = np.polyfit(rows[keep], cen[keep], 1)
        k = float(np.clip(k, -0.05, 0.05))      # ±2.9° より大きな傾きは誤った当てはめ
    return (default + k * (np.arange(n) - (n - 1) / 2.0)).astype(np.float32)


def _half_crossing(p: np.ndarray, center: int, half: float, reach: int) -> float | None:
    """``center`` の近く (±``reach``) で ``p`` が ``half`` を横切る位置 (1 画素以下まで)。"""
    best = None
    for j in range(max(0, center - reach), min(len(p) - 1, center + reach + 1)):
        a_, b_ = float(p[j]), float(p[j + 1])
        if (a_ - half) * (b_ - half) <= 0 and a_ != b_:
            cand = j + (half - a_) / (b_ - a_)
            if best is None or abs(cand - center) < abs(best - center):
                best = cand
    return best


def _classify_post(white_frac: np.ndarray, mask_frac: np.ndarray, pos: float,
                   gap_up: bool, cfg: EdgeConfig) -> str | None:
    """縁の先 (切れ目の側) を見て、柱の上面の色を決める。使えない縁なら None。

    ``gap_up`` は切れ目が座標の大きい側にあるか (= 赤が小さい側)。

    - 白が ``post_px`` の間だけあって、その先は床 → 白い柱
    - 白がほとんど無く、その先も床 → 赤い柱 (赤の中に柱も含まれている)
    - 白が ``max_white_px`` より長く続く → 白飛びか白い壁 → 使わない
    """
    n = len(white_frac)
    step = 1 if gap_up else -1
    start = int(round(pos)) + step * 2

    def window(a: int, b: int) -> np.ndarray:
        idx = [start + step * k for k in range(a, b)]
        idx = [i for i in idx if 0 <= i < n]
        return np.array(idx, dtype=int)

    near = window(0, cfg.post_px)
    far = window(cfg.post_px + 10, cfg.post_px + 30)
    long_ = window(0, cfg.max_white_px)
    if len(near) < cfg.post_px // 2 or len(far) < 5:
        return None
    near_white = float(white_frac[near].mean())
    # 白い区間がどこまで続くか (白飛び・白い壁なら長い)
    run = 0
    for i in long_:
        if white_frac[i] >= cfg.white_post_ratio:
            run += 1
        elif run:
            break
    if run >= cfg.max_white_px - 2:
        return None
    far_floor = (float(white_frac[far].max()) < cfg.white_post_ratio
                 and float(mask_frac[far].max()) <= cfg.max_gap_mask)
    if not far_floor:
        return None
    post = "white" if near_white >= cfg.white_post_ratio else "red"
    if cfg.post_top != "auto" and post != cfg.post_top:
        return None                  # 会場で決め打ちした色と違う縁は使わない
    return post


def encode_edges(edges: list[BandEdge]) -> str:
    """CSV に 1 欄で残すための文字列 (``band:offset_mm:red_on_plus:pos_px`` を ``|`` で)。"""
    return "|".join(f"{e.band}:{e.offset_mm:.2f}:{int(e.red_on_plus)}:{e.pos_px:.1f}"
                    f"{':w' if e.post == 'white' else ''}" for e in edges)


def decode_edges(text) -> list[BandEdge]:
    """:func:`encode_edges` の逆。空・None なら空のリスト。"""
    if not text or not isinstance(text, str):
        return []
    out = []
    for item in text.split("|"):
        parts = item.split(":")
        band, off, plus, pos = parts[:4]
        post = "white" if len(parts) > 4 and parts[4] == "w" else "red"
        out.append(BandEdge(band, BAND_AXIS[band], float(off), plus == "1", float(pos), post))
    return out


def offset_from_edge(edge: BandEdge, estimated_offset_mm: float, pitch_mm: float = 180.0,
                     post_half_mm: float = EDGE_INSET_MM) -> float:
    """縁 1 つから、その軸での機体の位置 (最寄りのセル中央からのずれ [mm]) を出す。

    縁は「セルの境目 ± ``post_half_mm``」にある (切れ目の側へ、既定 ``EDGE_INSET_MM``)。どの境目かは、推定の位置
    ``estimated_offset_mm`` (同じ軸、セル中央から) から決める — 推定の誤りが
    ``pitch/2 − post_half_mm`` (約 80 mm) より小さければ取り違えない。

    返す値の座標は ``estimated_offset_mm`` と同じ (セル中央が 0)。境目が隣のセルの
    ものなら、そのセルの中央からではなく**今のセルの中央から**のずれで返す。
    """
    # 縁の切れ目の側: + 側が赤なら切れ目は - 側 → 縁 = 境目 − inset (赤い柱)。
    # 白い柱なら赤は柱の手前で終わるので、縁は境目から赤の側へ inset。
    side = -post_half_mm if edge.red_on_plus else post_half_mm
    if getattr(edge, "post", "red") == "white":
        side = -side
    # 推定の位置から見た縁の位置 (今のセルの中央が 0)
    edge_est = estimated_offset_mm + edge.offset_mm
    # 境目は (k + 1/2)·pitch
    k = round((edge_est - side) / pitch_mm - 0.5)
    boundary = (k + 0.5) * pitch_mm
    return boundary + side - edge.offset_mm
