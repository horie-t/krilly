"""ボタン 1 個 + ブザーのランチャ (#79)。時刻もボタンも子プロセスもフェイクで回す。"""

import signal

import pytest

from krilly.app.launcher import (
    BEEP_ALARM,
    BEEP_BOOT,
    BEEP_CANCEL,
    BEEP_CONFIG_ERROR,
    BEEP_COUNTDOWN,
    BEEP_DONE,
    BEEP_FAILED,
    BEEP_POWEROFF,
    BEEP_STOPPED,
    BEEP_TICK,
    COUNTDOWN_S,
    Launcher,
    Press,
    PressDetector,
    State,
)

DT = 0.01


class FakeChild:
    def __init__(self, dies_on=(signal.SIGINT,), rc_on_signal=-signal.SIGINT):
        self.signals: list[int] = []
        self.rc: int | None = None
        self.dies_on = dies_on
        self.rc_on_signal = rc_on_signal

    def poll(self):
        return self.rc

    def send_signal(self, sig):
        self.signals.append(sig)
        if sig in self.dies_on:
            self.rc = self.rc_on_signal


class Rig:
    """ランチャと、それを時間で回す道具。"""

    def __init__(self, configured=True, child=None):
        self.t = 0.0
        self.beeps = []
        self.spawned = []
        self.powered_off = False
        self.child = child or FakeChild()

        def spawn():
            self.spawned.append(self.t)
            return self.child

        self.l = Launcher(spawn=spawn if configured else None, beep=self.beeps.append,
                          poweroff=self._off)

    def _off(self):
        self.powered_off = True

    def run(self, seconds, pressed=False):
        for _ in range(int(round(seconds / DT))):
            self.t += DT
            self.l.tick(self.t, pressed)

    def hold(self, seconds):
        self.run(seconds, True)
        self.run(0.1, False)


# --- ボタンのイベント ------------------------------------------------------------

def test_press_detector_debounces_and_classifies():
    d = PressDetector()
    events = []
    t = 0.0
    for pressed, dur in ((True, 0.01), (False, 0.01), (True, 0.5), (False, 0.1),
                         (True, 1.5), (False, 0.1)):
        for _ in range(int(round(dur / DT))):
            t += DT
            events += d.update(t, pressed)
    # 10 ms の瞬断はチャタリングとして捨てる
    assert events == [Press.DOWN, Press.SHORT, Press.DOWN, Press.HOLD_START, Press.LONG]


# --- 待機 -> 発進 -------------------------------------------------------------

def test_boot_beeps_and_a_long_press_starts_after_the_countdown():
    r = Rig()
    assert r.beeps == [BEEP_BOOT]
    r.hold(1.2)
    assert BEEP_TICK in r.beeps and r.beeps[-1] == BEEP_COUNTDOWN
    assert r.l.state is State.COUNTDOWN and not r.spawned
    r.run(COUNTDOWN_S)
    assert r.l.state is State.RUNNING and len(r.spawned) == 1


def test_a_short_press_does_nothing():
    """誤って触れただけで走り出さない。"""
    r = Rig()
    r.hold(0.3)
    r.run(COUNTDOWN_S + 1)
    assert r.l.state is State.IDLE and not r.spawned


def test_the_countdown_can_be_cancelled():
    r = Rig()
    r.hold(1.2)
    r.hold(0.1)
    assert r.l.state is State.IDLE and r.beeps[-1] == BEEP_CANCEL
    r.run(COUNTDOWN_S + 1)
    assert not r.spawned


# --- 走行中 --------------------------------------------------------------------

def test_a_press_while_running_sends_sigint_once():
    """停止は既存の emergency_stop が受ける SIGINT だけ (新しい停止経路を作らない)。"""
    child = FakeChild(dies_on=())
    r = Rig(child=child)
    r.hold(1.2)
    r.run(COUNTDOWN_S)
    r.hold(0.05)
    r.hold(0.05)                                  # 連打しても SIGINT は 1 回
    assert child.signals == [signal.SIGINT]
    assert r.l.state is State.STOPPING


def test_stopped_by_the_button_beeps_differently_from_a_crash():
    r = Rig()
    r.hold(1.2)
    r.run(COUNTDOWN_S)
    r.hold(0.05)
    assert r.l.state is State.IDLE and r.beeps[-1] == BEEP_STOPPED


@pytest.mark.parametrize("rc, beep", [(0, BEEP_DONE), (2, BEEP_FAILED)])
def test_the_exit_code_decides_the_beep(rc, beep):
    r = Rig()
    r.hold(1.2)
    r.run(COUNTDOWN_S)
    r.child.rc = rc
    r.run(0.05)
    assert r.l.state is State.IDLE and r.beeps[-1] == beep and r.l.last_returncode == rc


def test_a_child_that_ignores_sigint_gets_sigterm_then_an_alarm_never_sigkill():
    """SIGKILL は捕まえられず L6470 が回り続けるので送らない。止めるのは VS のトグル。"""
    child = FakeChild(dies_on=())
    r = Rig(child=child)
    r.hold(1.2)
    r.run(COUNTDOWN_S)
    r.hold(0.05)
    r.run(2.5)
    assert child.signals == [signal.SIGINT, signal.SIGTERM]
    r.run(6.0)
    assert BEEP_ALARM in r.beeps
    assert signal.SIGKILL not in child.signals


def test_holding_the_stop_press_does_not_restart_the_run():
    """止めた押下を押し続けたまま走行が終わっても、その押下で再発進しない。"""
    r = Rig()
    r.hold(1.2)
    r.run(COUNTDOWN_S)
    r.hold(2.0)                                   # 押した瞬間に止まり、そのまま 2 秒
    r.run(COUNTDOWN_S + 1)
    assert len(r.spawned) == 1 and r.l.state is State.IDLE


def test_a_long_hold_while_running_does_not_power_off():
    r = Rig(child=FakeChild(dies_on=()))
    r.hold(1.2)
    r.run(COUNTDOWN_S)
    r.hold(6.0)
    assert not r.powered_off


# --- 電源断・設定エラー ---------------------------------------------------------

def test_five_seconds_while_idle_powers_off():
    r = Rig()
    r.hold(5.2)
    assert r.powered_off and r.beeps[-1] == BEEP_POWEROFF and not r.spawned


def test_without_a_config_it_never_runs_but_can_still_power_off():
    r = Rig(configured=False)
    assert r.beeps == [BEEP_CONFIG_ERROR]
    r.hold(1.2)
    r.run(COUNTDOWN_S + 1)
    assert r.l.state is State.CONFIG_ERROR
    r.hold(5.2)
    assert r.powered_off


# --- 起動時の設定の検証 (scripts/launcher.py) -------------------------------------

def test_the_launcher_validates_the_saved_command_with_the_scripts_own_parser(tmp_path):
    from scripts.launcher import check_config

    good = tmp_path / "run.yaml"
    good.write_text('script: speed_run\nargs: [--size, "16", --ev, "-2", --white-tops]\n',
                    encoding="utf-8")
    cfg, command = check_config(good)
    assert command[1:] == ["-m", "scripts.speed_run", "--size", "16", "--ev", "-2",
                           "--white-tops"]

    typo = tmp_path / "typo.yaml"
    typo.write_text('script: speed_run\nargs: [--evv, "-2"]\n', encoding="utf-8")
    assert check_config(typo) == (None, None)
    assert check_config(tmp_path / "missing.yaml") == (None, None)
