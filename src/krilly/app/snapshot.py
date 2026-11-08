"""止まったときの画像を残す (#153)。

``--save-frames`` を付けていない走行では画像が 1 枚も残らず、「止まったときに機体が
どうなっていたか」を後から確かめられなかった (2026-10-17 の会場で、壁に当たった後の
状態を見られなかった)。ガードで止まったとき・走行が中断で終わったとき・SPI が
途切れたときに 1 枚ずつ保存し、ログに場所を出す。保存は ``cv2.imwrite`` だけで、
失敗しても走行の処理は止めない (撮影が原因で停止処理が遅れてはいけない)。
"""

from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path

#: 保存先。``logs/`` は launcher の走行ログと同じ場所 (``.gitignore`` 済み)。
SNAPSHOT_DIR = Path("logs") / "snapshots"

log = logging.getLogger(__name__)


def snapshot_path(reason: str, now: datetime.datetime | None = None,
                  directory: Path = SNAPSHOT_DIR) -> Path:
    """``<日時>_<理由>.png`` のパス。理由は英数字とアンダースコアだけに丸める。"""
    now = now or datetime.datetime.now()
    tag = re.sub(r"[^0-9A-Za-z_-]+", "_", reason).strip("_") or "stop"
    return directory / f"{now:%Y%m%d_%H%M%S}_{tag}.png"


def save_snapshot(frame, reason: str, directory: Path = SNAPSHOT_DIR,
                  now: datetime.datetime | None = None) -> Path | None:
    """``frame`` (BGR) を保存してパスを返す。失敗したら None (例外は出さない)。"""
    if frame is None:
        return None
    try:
        import cv2

        path = snapshot_path(reason, now, directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), frame):
            return None
        return path
    except Exception as e:      # 撮影の失敗で停止処理を止めない
        log.warning("停止時の画像を保存できなかった: %s", e)
        return None
