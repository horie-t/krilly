"""動作の**途中で**異常を見張る (#153)。

指令外の回転のガード (``--max-heading-residual``) と STATUS の読み (#148) は、どちらも
**動作が終わった後**にしか判定しない。2026-10-17 の交流会では、壁にぶつかって機体が
回ったあとも 1 つの動作 (``Sへ2->Wへ4``) が制限時間 18 秒まで指令を出し続け、壁と柱に
当たりながら進んだ (VS を手で切って止めた)。

平行移動では回転を一切指令しないので、**推定方位が基準からずれていくこと自体が異常の
証拠**になる。正常な動作の終了時の残差は 0.9° 以下 (16x16・8x8 の通し走行で最大 0.86°)。
1 tick だけの跳ねで止めないよう、上限を ``persist_ticks`` 回続けて超えたら止める。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RotationWatch:
    """平行移動中の「指令していない回転」を見張る。

    ``update(heading_error_rad)`` を制御周期ごとに呼び、上限を ``persist_ticks`` 回
    続けて超えたら True を返す (= その場で止める)。``peak_rad`` は動作中の最大の
    ずれ (ログ用: 正常な動作でどこまで振れるかの実測になる)。
    """

    limit_rad: float
    persist_ticks: int = 3
    peak_rad: float = field(default=0.0, init=False)
    _over: int = field(default=0, init=False)

    def reset(self) -> None:
        self.peak_rad = 0.0
        self._over = 0

    def update(self, heading_error_rad: float) -> bool:
        err = abs(heading_error_rad)
        self.peak_rad = max(self.peak_rad, err)
        self._over = self._over + 1 if err > self.limit_rad else 0
        return self._over >= max(1, self.persist_ticks)


def motion_timeout_s(cells: int, corners: int = 0, floor_s: float = 3.0) -> float:
    """平行移動 1 動作の制限時間 [s] (#153)。見込み時間の 2 倍、最低 ``floor_s``。

    見込みは 0.30 m/s の実測 (1 セル 0.6s + 固定 0.6s + コーナー 1 回 0.3s、余裕込み)。
    以前は ``max(10s, セル数 x 3s)`` で、1 セルの動作 (実測 1.1s) に 10 秒、6 セルに 18 秒を
    与えていた — 進めなくなった機体が壁を押し続ける時間がそれだけ長かった。
    """
    expected = 0.6 + 0.65 * cells + 0.3 * corners
    return max(floor_s, 2.0 * expected)
