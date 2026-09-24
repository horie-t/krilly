"""ボタン 1 個 + ブザーで走行を起動・停止する (issue #79)。

本番の会場ではネットワーク越しに操作できない。**設定は前日の試走で決めてファイルに
保存し** (``speed_run --save-run-config``)、当日は電源を入れてボタンを押すだけにする。

ボタン 1 個で足りるのは、状態によって意味が変わるから:

============  =====================  =========================================
状態          操作                   動作
============  =====================  =========================================
待機          1 秒押して離す         カウントダウン (ピッ ピッ ピッ ピー) の後に発進
待機          5 秒押し続ける         電源を切る (SD カードを守る)
カウントダウン  押す                   取り消し
走行中        押す (押した瞬間)      子プロセスに SIGINT
============  =====================  =========================================

**停止は SIGINT だけ。** 子プロセス (``speed_run``) の ``emergency_stop`` がシグナル
ハンドラの中で ``soft_stop_all`` -> ``hard_hiz_all`` を実行する。**新しい停止経路を
作らない**のが要点で、Ctrl-C と同じ経路をボタンから叩くだけにしてある。
SIGINT で 2 秒たっても終わらなければ SIGTERM (同じハンドラが受ける)。それでも
終わらなければ**警報を鳴らし続ける** — SIGKILL は使わない。SIGKILL は捕まえられず、
L6470 は最後の Run 指令を保持したまま回り続ける (実績あり)。その場合に止めるのは
**ソフトを通らない VS 遮断のトグル**の役目。

このモジュールは純粋なロジックで、時刻・ボタンの状態・子プロセス・ブザーをすべて
外から渡す。実機への配線は ``scripts/launcher.py``。
"""

from __future__ import annotations

import signal
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

#: 鳴らし方: (鳴らす秒, 休む秒) の列。
Pattern = tuple[tuple[float, float], ...]

BEEP_BOOT: Pattern = ((0.1, 0.0),)                               # ピッ (起動した)
BEEP_TICK: Pattern = ((0.04, 0.0),)                              # 長押しの区切り
BEEP_COUNTDOWN: Pattern = ((0.1, 0.9),) * 3 + ((0.6, 0.0),)      # ピッ ピッ ピッ ピー
BEEP_CANCEL: Pattern = ((0.05, 0.05), (0.05, 0.0))               # ピピ (取り消し)
BEEP_DONE: Pattern = ((0.08, 0.06), (0.08, 0.06), (0.2, 0.0))    # ピピピッ (完走)
BEEP_FAILED: Pattern = ((0.5, 0.3), (0.5, 0.0))                  # ピー ピー (中断)
BEEP_STOPPED: Pattern = ((0.3, 0.0),)                            # ピー (ボタンで止めた)
BEEP_CONFIG_ERROR: Pattern = ((0.1, 0.1),) * 5                   # ピピピピピ (設定が読めない)
BEEP_ALARM: Pattern = ((0.2, 0.2),) * 5                          # 止まらない (繰り返す)
BEEP_POWEROFF: Pattern = ((1.0, 0.0),)                           # ピー (電源を切る)

#: チャタリング除去: この時間だけ同じ状態が続いたら確定とみなす [s]。
DEBOUNCE_S = 0.03
START_HOLD_S = 1.0          # 発進に必要な長押し
POWEROFF_HOLD_S = 5.0       # 電源断に必要な長押し
COUNTDOWN_S = 3.0           # 長押しを離してから発進までの時間 (BEEP_COUNTDOWN と同じ長さ)
SIGTERM_AFTER_S = 2.0       # SIGINT で終わらなければ SIGTERM を送るまでの時間
ALARM_AFTER_S = 5.0         # さらにこれだけ終わらなければ警報
ALARM_EVERY_S = 3.0         # 警報を繰り返す間隔


class Press(Enum):
    """ボタンのイベント。"""

    DOWN = "down"               # 押した瞬間
    HOLD_START = "hold_start"   # 押したまま START_HOLD_S に達した
    HOLD_POWEROFF = "hold_poweroff"  # 押したまま POWEROFF_HOLD_S に達した
    SHORT = "short"             # START_HOLD_S 未満で離した
    LONG = "long"               # START_HOLD_S 以上 POWEROFF_HOLD_S 未満で離した


@dataclass
class PressDetector:
    """ボタンの瞬間値からイベントを作る (チャタリング除去込み)。"""

    debounce_s: float = DEBOUNCE_S
    start_hold_s: float = START_HOLD_S
    poweroff_hold_s: float = POWEROFF_HOLD_S
    _raw: bool = field(default=False, init=False)
    _raw_since: float = field(default=0.0, init=False)
    _stable: bool = field(default=False, init=False)
    _down_at: float = field(default=0.0, init=False)
    _fired: set = field(default_factory=set, init=False)

    def update(self, now: float, pressed: bool) -> list[Press]:
        if pressed != self._raw:
            self._raw, self._raw_since = pressed, now
        events: list[Press] = []
        if self._raw != self._stable and now - self._raw_since >= self.debounce_s:
            self._stable = self._raw
            if self._stable:
                self._down_at = self._raw_since
                self._fired = set()
                events.append(Press.DOWN)
            else:
                held = self._raw_since - self._down_at
                if Press.HOLD_POWEROFF not in self._fired:
                    events.append(Press.SHORT if held < self.start_hold_s else Press.LONG)
        if self._stable:
            held = now - self._down_at
            for ev, limit in ((Press.HOLD_START, self.start_hold_s),
                              (Press.HOLD_POWEROFF, self.poweroff_hold_s)):
                if held >= limit and ev not in self._fired:
                    self._fired.add(ev)
                    events.append(ev)
        return events


class Child(Protocol):
    """子プロセス (``subprocess.Popen`` の必要な部分)。"""

    def poll(self) -> int | None: ...
    def send_signal(self, sig: int) -> None: ...


class State(Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    RUNNING = "running"
    STOPPING = "stopping"
    CONFIG_ERROR = "config_error"
    POWEROFF = "poweroff"


@dataclass
class Launcher:
    """ボタンの状態機械。``tick(now, pressed)`` を 100 Hz 程度で呼ぶ。

    ``spawn`` は走行の子プロセスを起動する (起動するコマンドは設定から決まっている)。
    ``spawn`` が None なら設定が読めなかったということで、走行は起動せず、電源断
    だけを受け付ける (押すたびに設定エラーの音を鳴らして知らせる)。
    """

    spawn: Callable[[], Child] | None
    beep: Callable[[Pattern], None]
    poweroff: Callable[[], None]
    log: Callable[[str], None] = lambda _msg: None
    detector: PressDetector = field(default_factory=PressDetector)
    state: State = field(default=State.IDLE, init=False)
    child: Child | None = field(default=None, init=False)
    last_returncode: int | None = field(default=None, init=False)
    _t: float = field(default=0.0, init=False)
    _signalled: list[int] = field(default_factory=list, init=False)
    _last_alarm: float = field(default=-1e9, init=False)
    #: 押し始めたときの状態。**長押し系のイベントは、待機中に押し始めた押下にだけ効く**。
    #: でないと、走行を止めた押下を押し続けたまま走行が終わると、同じ押下が
    #: 「1 秒押して離した」に化けて再発進する。
    _press_began_in: State | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.spawn is None:
            self.state = State.CONFIG_ERROR
            self.beep(BEEP_CONFIG_ERROR)
        else:
            self.beep(BEEP_BOOT)

    # -- 1 周 -------------------------------------------------------------
    def tick(self, now: float, pressed: bool) -> None:
        for ev in self.detector.update(now, pressed):
            self._on_press(now, ev)
        if self.state is State.COUNTDOWN and now - self._t >= COUNTDOWN_S:
            self._start(now)
        if self.state in (State.RUNNING, State.STOPPING):
            self._watch_child(now)

    # -- ボタン -------------------------------------------------------------
    def _on_press(self, now: float, ev: Press) -> None:
        s = self.state
        if ev is Press.DOWN:
            self._press_began_in = s
        elif self._press_began_in is not s:
            return                          # 別の状態で押し始めた押下の続き
        if s in (State.RUNNING, State.STOPPING):
            if ev is Press.DOWN:
                self._stop(now)
            return
        if s is State.COUNTDOWN:
            if ev is Press.DOWN:
                self.state = State.IDLE
                self.beep(BEEP_CANCEL)
                self.log("発進を取り消した")
            return
        if s is State.POWEROFF:
            return
        # 待機中 / 設定エラー
        if ev is Press.HOLD_POWEROFF:
            self.state = State.POWEROFF
            self.beep(BEEP_POWEROFF)
            self.log("長押し 5 秒: 電源を切る")
            self.poweroff()
        elif s is State.CONFIG_ERROR:
            if ev is Press.DOWN:
                self.beep(BEEP_CONFIG_ERROR)
        elif ev is Press.HOLD_START:
            self.beep(BEEP_TICK)            # 「離せば発進する」の合図
        elif ev is Press.LONG:
            self.state = State.COUNTDOWN
            self._t = now
            self.beep(BEEP_COUNTDOWN)
            self.log(f"{COUNTDOWN_S:.0f} 秒後に発進する (もう一度押すと取り消し)")

    # -- 子プロセス -----------------------------------------------------------
    def _start(self, now: float) -> None:
        assert self.spawn is not None
        self.child = self.spawn()
        self._signalled = []
        self._t = now
        self.state = State.RUNNING
        self.log("走行を開始した")

    def _stop(self, now: float) -> None:
        if self.child is None or signal.SIGINT in self._signalled:
            return
        self.child.send_signal(signal.SIGINT)
        self._signalled.append(signal.SIGINT)
        self._t = now
        self.state = State.STOPPING
        self.log("ボタンで停止: SIGINT を送った")

    def _watch_child(self, now: float) -> None:
        assert self.child is not None
        rc = self.child.poll()
        if rc is not None:
            self._finished(rc)
            return
        if self.state is not State.STOPPING:
            return
        waited = now - self._t
        if waited >= SIGTERM_AFTER_S and signal.SIGTERM not in self._signalled:
            self.child.send_signal(signal.SIGTERM)
            self._signalled.append(signal.SIGTERM)
            self.log(f"SIGINT から {SIGTERM_AFTER_S:.0f} 秒たっても終わらない: SIGTERM")
        if (waited >= SIGTERM_AFTER_S + ALARM_AFTER_S
                and now - self._last_alarm >= ALARM_EVERY_S):
            # SIGKILL は送らない (L6470 が最後の Run を保持して回り続ける)。
            # 止めるのは VS 遮断のトグル。
            self._last_alarm = now
            self.beep(BEEP_ALARM)
            self.log("走行プロセスが終わらない。**VS 遮断のトグルでモータを止めること**")

    def _finished(self, rc: int) -> None:
        stopped = bool(self._signalled)
        self.last_returncode = rc
        self.child = None
        self.state = State.IDLE
        if rc == 0:
            self.beep(BEEP_DONE)
            self.log("走行が終わった (終了コード 0)")
        elif stopped:
            self.beep(BEEP_STOPPED)
            self.log(f"ボタンで止めた (終了コード {rc})")
        else:
            self.beep(BEEP_FAILED)
            self.log(f"走行が中断した (終了コード {rc})")


def pattern_seconds(pattern: Pattern) -> float:
    """鳴らし終わるまでの時間 [s]。"""
    return sum(on + off for on, off in pattern)
