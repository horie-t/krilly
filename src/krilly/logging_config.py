"""Krilly のロギングを一元的にセットアップするモジュール。

アプリケーションやスクリプトの起動時に :func:`setup_logging` を一度だけ呼び出す。
"""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def setup_logging(level: int | str | None = None) -> None:
    """ルートロガーを設定する。

    ログレベルは ``KRILLY_LOG_LEVEL`` 環境変数 (例: ``DEBUG``、``INFO``)
    で上書きできる。デフォルトは ``INFO``。
    """
    if level is None:
        level = os.environ.get("KRILLY_LOG_LEVEL", "INFO")
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)


def get_logger(name: str) -> logging.Logger:
    """モジュール用ロガーを返す ( :func:`logging.getLogger` の薄いラッパー)。"""
    return logging.getLogger(name)
