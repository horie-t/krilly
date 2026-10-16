"""走行スクリプトの終了コード。ランチャ (#79) はこれで鳴らす音を変える。

0 は完走。``scripts.speed_run`` などはここの値を返し、:mod:`krilly.app.launcher` は
同じ値を読む (定義を 1 か所にしておかないと、片方だけ変えて音が黙って変わる)。
"""

from __future__ import annotations

import logging
from typing import Callable

from krilly.hal.l6470 import SpiLost

#: 走行が中断で終わった (ガードで止まった、回復できなかった、など)。
EXIT_ABORTED = 2

#: 走行中に SPI の応答が途切れた (#148)。**モータは最後の指令で回り続けている
#: 可能性がある**ので、ランチャは警報を鳴らし、VS 遮断のトグルで止めてもらう。
EXIT_SPI_LOST = 3


def run_reporting_spi_lost(body: Callable[[], int | None], log: logging.Logger) -> int:
    """走行スクリプトの本体を回し、SPI の途切れ (#148) を専用の終了コードに変える。

    :class:`~krilly.hal.l6470.SpiLost` は動作ごとの STATUS 読みから上がってくる。
    ``with`` の後始末 (解放) はここに来る前に済んでいるが、**それも SPI を通るので
    届いたとは限らない**。だから操作者に VS を切ってもらう文言を必ず出す。
    本体が None を返したら 0 (完走) とみなす。
    """
    try:
        return body() or 0
    except SpiLost as e:
        log.error("走行を打ち切った: %s。回復は試みない (通信が無いと指令が届かない)。", e)
        if e.released:
            log.error("  途中で応答が戻ったので解放 (HardHiZ) を送った。"
                      "それでも回っていたら **VS 遮断のトグルを切ること**")
        else:
            # 応答が無いのは「読めない」だけで、送った解放が届いたかは分からない
            # (#148 の実機確認では MISO 側だけ偽装して、実際には HiZ が立っていた)
            log.error("  応答が戻らず、解放が届いたか確認できない。モータは最後の指令で"
                      "回り続けている可能性がある。**VS 遮断のトグルを切ること**")
        log.error("  切り分け: 車輪を浮かせて python -m scripts.spi_watch --seconds 30 "
                  "を回しながら SPI の配線を揺する")
        return EXIT_SPI_LOST
