"""シグナルを受けても必ずモーターを解放する仕組み。

実機で **モーターが回り続けたまま制御を失う**事故が起きた。原因は、

- ``timeout(1)`` が送る **SIGTERM** や、端末が送る **SIGINT** (Ctrl-C) の既定動作では
  Python の ``finally`` / ``with`` の後始末が走らない (SIGTERM は即終了、SIGINT は
  KeyboardInterrupt になるが、C 拡張の中や sleep 中の扱いで取りこぼすことがある)
- したがって ``with L6470Chain(...)`` の終了時 ``hard_hiz`` が呼ばれず、L6470 は
  最後の Run 指令のまま回り続ける

本モジュールはシグナルハンドラで**先にモーターを解放**してから既定動作に戻して
自分にシグナルを再送する。これで ``timeout`` でも Ctrl-C でも確実に止まる。

**SIGKILL (kill -9) は捕捉できない**。その場合はモーター電源 (VS) を落とすか、
``python -m scripts.motor_stop`` で出力を解放すること。
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from typing import Callable, Iterable

# 捕捉するシグナル (プラットフォームに無いものは無視する)
STOP_SIGNALS = tuple(
    s for s in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, s)
)


def release(chain) -> None:
    """停止指令 -> 出力解放。停止処理自体で落ちないよう例外は握りつぶす。

    ``soft_stop_all`` で減速停止させてから ``hard_hiz_all`` でブリッジを開放する
    (順序が逆だと惰性で動く)。chain がどちらを持たなくても安全に無視する。
    """
    for action in ("soft_stop_all", "hard_hiz_all"):
        method = getattr(chain, action, None)
        if method is None:
            continue
        try:
            method()
        except Exception:   # noqa: BLE001  停止経路では何があっても続行する
            pass


@contextmanager
def emergency_stop(
    chain,
    signals: Iterable[str] = STOP_SIGNALS,
    on_stop: Callable[[int], None] | None = None,
):
    """``with`` を抜けるときと、シグナル受信時にモーターを解放する。

    ``signals`` は ``signal`` モジュールの属性名 (既定 SIGINT/SIGTERM/SIGHUP)。
    ``on_stop`` はログ出力などのフック (シグナル番号を受け取る)。
    """
    previous: dict[int, object] = {}

    def handler(signum, _frame):
        release(chain)
        if on_stop is not None:
            try:
                on_stop(signum)
            except Exception:   # noqa: BLE001
                pass
        # 既定動作へ戻して自分に再送する (終了コード・終了理由を保つ)
        signal.signal(signum, previous.get(signum, signal.SIG_DFL))
        signal.raise_signal(signum)

    for name in signals:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # メインスレッド以外では設定できない。その場合は with の後始末のみ
            pass
    try:
        yield
    finally:
        release(chain)
        for sig, prev in previous.items():
            try:
                signal.signal(sig, prev)   # type: ignore[arg-type]
            except (ValueError, OSError, TypeError):
                pass
