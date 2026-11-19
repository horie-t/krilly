"""走りながら、今の脚の中のセルの境目に壁が写っていないかを見張る (issue #188)。

``perception.ahead`` が 1 枚から作った行・列ごとの赤の数 (:class:`AheadProfiles`) と、
撮影時点の推定姿勢 (遅れ込み、#186) と、``CellMotion`` の今の脚 (主軸と行き先) から、
**今の脚が通る境目**の位置の赤の割合を見る。経路は開いた境目しか通らないので、
そこに壁が見えたら推定のずれか地図の誤り — 止めるべき。

- 行き先のセルの向こう側の境目 (正当に壁がありうる) は見ない (``end_margin_m``)。
- 1 フレームだけの読みでは止めない (``min_frames``)。
- 試算だけ (``observe_only``) なら止めずに記録する。新しい判定はまずこれで記録を
  とってから使うこと (2026-11-19 の #189 の衝突の教訓)。

保存画像での確かめ (2026-11-19、long8 木の床 EV 0 / 黒い床の廊下 EV −2、0.30 m/s、
9 走行): 経路上の開いた境目 1,094 回の赤の割合は 中央値 0.00 / 99% 0.08 / 最大 0.20
(最大は機体のすぐ近くで半分隠れた所)。壁のある境目は 80〜300 mm のどの距離でも
約 95% のフレームで 0.5 以上。だからしきい値 0.5。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from krilly.perception.ahead import AheadConfig, boundary_fraction


@dataclass(frozen=True)
class AheadWatchConfig:
    threshold: float = 0.5
    #: 行き先のセルの向こう側の境目から、これより手前の境目だけを見る [m]。
    end_margin_m: float = 0.03
    #: 同じ境目でしきい値を超えたフレームがこれだけ続いたら止める。
    min_frames: int = 2
    #: 止めずに記録だけする。
    observe_only: bool = False
    #: 機体が今の脚の通り道 (行き先を通る、主軸に平行な直線) から横にこれ以上離れている
    #: 間は見ない [m]。**止まらずに曲がる (#80) とき、``axis_k`` は機体がまだ前の脚の上に
    #: いるうちに次の向きへ切り替わる** ので、そのまま見ると前の脚の横の壁を「次の脚の
    #: 境目」と取り違える (2026-11-19 の long8、`--chain-legs 4` などで最大 0.30〜0.40)。
    max_lane_offset_m: float = 0.04
    #: これより近い境目は見ない [m]。機体の陰の縁で境目が半分隠れて読みが不安定になり
    #: (黒い床の 3×3 で 70 mm 先に 0.34、2026-11-20)、しかも 0.30 m/s ではもう止まりきれない
    #: (止まるのに 75 mm + 遅れの分 21 mm)。
    min_distance_m: float = 0.08
    pitch_m: float = 0.18
    profile: AheadConfig = field(default_factory=AheadConfig)


@dataclass
class AheadAlarm:
    """今の脚の境目に壁が見えた。"""

    distance_m: float     # 撮影時点で機体から境目まで
    fraction: float       # 境目の位置の赤の割合
    frames: int           # 続けて見えたフレーム数


class AheadWatch:
    """1 動作の間の見張り。動作ごとに :meth:`reset` する。"""

    def __init__(self, config: AheadWatchConfig | None = None) -> None:
        self.cfg = config or AheadWatchConfig()
        self.reset()

    def reset(self) -> None:
        self.alarm: AheadAlarm | None = None
        self.max_fraction = 0.0           # 今の脚の境目で見えた赤の割合の最大 (記録用)
        self.max_at: tuple[float, float] | None = None   # その (境目までの距離, 通り道からの横) [m]
        self.off_lane = 0                 # 通り道から外れていて見なかったフレーム数
        self.checked = 0
        self._streak: dict[tuple[int, int], int] = {}

    def check(self, prof, past_pose: tuple[float, float, float],
              reference: tuple[float, float, float], axis_k: int,
              phi_ref: float) -> AheadAlarm | None:
        """1 フレームを見る。止めるべきなら :class:`AheadAlarm` を返す (``observe_only`` でも)。"""
        cfg = self.cfg
        x, y, _ = past_pose
        ang = phi_ref + axis_k * math.pi / 2.0
        ux, uy = math.cos(ang), math.sin(ang)
        p = cfg.pitch_m
        cx, cy = round(x / p) * p, round(y / p) * p
        along = (x - cx) * ux + (y - cy) * uy            # 今のセルの中央から主軸の向きに
        remaining = (reference[0] - x) * ux + (reference[1] - y) * uy
        lane = -(x - reference[0]) * uy + (y - reference[1]) * ux     # 通り道からの横
        if abs(lane) > cfg.max_lane_offset_m:
            self.off_lane += 1
            self._streak.clear()
            return None
        last = remaining + p / 2.0 - cfg.end_margin_m      # これより手前の境目だけ
        seen: set[tuple[int, int]] = set()
        result = None
        for j in range(3):
            b = (j + 0.5) * p - along
            if b < cfg.min_distance_m or b >= last:
                continue
            frac = boundary_fraction(prof, axis_k, b, cfg.profile)
            if frac is None:
                continue
            self.checked += 1
            if frac > self.max_fraction:
                self.max_fraction = frac
                self.max_at = (b, lane)
            key = (round((x + b * ux) / (p / 2)), round((y + b * uy) / (p / 2)))
            seen.add(key)
            if frac >= cfg.threshold:
                self._streak[key] = self._streak.get(key, 0) + 1
                if self._streak[key] >= cfg.min_frames and result is None:
                    result = AheadAlarm(b, frac, self._streak[key])
            else:
                self._streak[key] = 0
        for key in list(self._streak):
            if key not in seen:
                del self._streak[key]
        if result is not None and self.alarm is None:
            self.alarm = result
        return result
