"""シグナル時のモーター解放 (emergency_stop) のユニットテスト。"""

import signal

import pytest

from krilly.motion.emergency_stop import STOP_SIGNALS, emergency_stop, release


class FakeChain:
    def __init__(self, fail: str | None = None):
        self.calls: list[str] = []
        self.fail = fail

    def soft_stop_all(self):
        self.calls.append("soft_stop")
        if self.fail == "soft_stop":
            raise RuntimeError("SPI 死亡")

    def hard_hiz_all(self):
        self.calls.append("hard_hiz")
        if self.fail == "hard_hiz":
            raise RuntimeError("SPI 死亡")


# --- release ---------------------------------------------------------------
def test_release_stops_then_frees_the_bridges():
    chain = FakeChain()
    release(chain)
    assert chain.calls == ["soft_stop", "hard_hiz"]   # 順序が逆だと惰性で動く


def test_release_continues_after_a_failure():
    """停止経路では例外を握りつぶして次の手段まで進む。"""
    chain = FakeChain(fail="soft_stop")
    release(chain)
    assert chain.calls == ["soft_stop", "hard_hiz"]   # 失敗しても hard_hiz は送る


def test_release_tolerates_a_chain_without_the_methods():
    release(object())        # 例外にならないこと


# --- コンテキストマネージャ ------------------------------------------------
def test_releases_on_normal_exit():
    chain = FakeChain()
    with emergency_stop(chain, signals=()):
        assert chain.calls == []
    assert chain.calls == ["soft_stop", "hard_hiz"]


def test_releases_on_exception():
    chain = FakeChain()
    with pytest.raises(ValueError):
        with emergency_stop(chain, signals=()):
            raise ValueError("走行中の例外")
    assert chain.calls == ["soft_stop", "hard_hiz"]


def test_installs_and_restores_handlers():
    chain = FakeChain()
    before = signal.getsignal(signal.SIGTERM)
    with emergency_stop(chain, signals=("SIGTERM",)):
        assert signal.getsignal(signal.SIGTERM) is not before
    assert signal.getsignal(signal.SIGTERM) is before


def test_handler_releases_before_reraising(monkeypatch):
    """シグナルを受けたら、まずモーターを解放してから既定動作に戻して再送する。"""
    chain = FakeChain()
    raised: list[int] = []
    monkeypatch.setattr(signal, "raise_signal", raised.append)
    seen: list[int] = []
    with emergency_stop(chain, signals=("SIGTERM",), on_stop=seen.append):
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)          # SIGTERM 相当を直接呼ぶ
        assert chain.calls == ["soft_stop", "hard_hiz"]
        assert raised == [signal.SIGTERM]      # 既定動作へ再送している
        assert seen == [signal.SIGTERM]        # フックも呼ばれる


def test_handler_survives_a_failing_hook(monkeypatch):
    chain = FakeChain()
    monkeypatch.setattr(signal, "raise_signal", lambda _s: None)
    def boom(_signum):
        raise RuntimeError("ログ失敗")
    with emergency_stop(chain, signals=("SIGTERM",), on_stop=boom):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert chain.calls == ["soft_stop", "hard_hiz"]


def test_default_signals_cover_ctrl_c_and_timeout():
    # Ctrl-C = SIGINT、timeout(1) = SIGTERM。SIGKILL は捕捉できない
    assert "SIGINT" in STOP_SIGNALS and "SIGTERM" in STOP_SIGNALS
