#!/usr/bin/env python3
"""機体単体で走らせるランチャ (issue #79)。systemd から起動する。

電源を入れると起動してピッと鳴り、ボタンを待つ。操作と音の意味は
:mod:`krilly.app.launcher` (1 秒押して離す = 発進、走行中に押す = 停止、5 秒 = 電源断)。

走らせるコマンドは ``config/run.yaml`` (前日の試走で ``speed_run --save-run-config``
が完走時に書いたもの)。**起動時に実行するコマンド全体をログに出す** — ファイルの弱点は
見えないことで、特に EV は床で 2 段違う (木 0 / 黒 -2) のに持ち越しやすい。

走行ごとの出力は ``logs/run_<日時>.log`` に残る (当日は SSH できないので、後で読む)。

例:
    # 何が走るかを確かめるだけ (GPIO に触らない)
    python -m scripts.launcher --check
    # 配線の確認: 全部の音を順に鳴らし、ボタンの状態を 10 秒表示する
    python -m scripts.launcher --beep-test
    # 手元で実際に待ち受ける (systemd と同じ)
    python -m scripts.launcher
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from krilly.app import launcher as sounds
from krilly.app.launcher import Launcher, Pattern, State, pattern_seconds
from krilly.config.loader import RUN_CONFIG_PATH, RunConfig, load_run_config
from krilly.hal.gpio_io import DEFAULT_BUTTON_GPIO, DEFAULT_BUZZER_GPIO, Button, Buzzer
from krilly.logging_config import get_logger, setup_logging

log = get_logger("krilly.launcher")

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "logs"
TICK_S = 0.01


def check_config(path: Path) -> tuple[RunConfig | None, list[str] | None]:
    """設定を読み、スクリプト自身の引数解析で検証する。ダメなら (None, None)。

    **起動時に検証する**のは、当日の朝に分かるより試走の夜に分かる方がいいから。
    ``build_parser`` は実機に触らない (``tests/test_run_scripts.py`` が保証している)。
    """
    try:
        cfg = load_run_config(path)
    except FileNotFoundError:
        log.error("走行設定 %s が無い。前日の試走で `speed_run ... --save-run-config` を"
                  "完走させて作ること", path)
        return None, None
    except (ValueError, KeyError, OSError) as e:
        log.error("走行設定 %s が読めない: %s", path, e)
        return None, None
    module = importlib.import_module(f"scripts.{cfg.script}")
    try:
        module.build_parser().parse_args(cfg.args)
    except SystemExit:
        log.error("走行設定の引数を %s が受け付けない: %s", cfg.script, cfg.args)
        return None, None
    command = cfg.command(sys.executable)
    log.info("走行設定: %s (保存 %s)", path, cfg.saved_at or "不明")
    log.info("  実行するコマンド: %s", " ".join(command[1:]))
    ev = cfg.arg_value("--ev")
    log.info("  EV %s / --white-tops %s  ※ 木の床は EV 0、黒い競技用床は -2 (#100)",
             ev or "0 (既定)", "あり" if "--white-tops" in cfg.args else "なし")
    return cfg, command


class BeepPlayer:
    """鳴らし方の型を別スレッドで鳴らす。新しい型は鳴っている型を打ち切る。"""

    def __init__(self, buzzer: Buzzer) -> None:
        self.buzzer = buzzer
        self._pattern: Pattern | None = None
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def play(self, pattern: Pattern) -> None:
        with self._cv:
            self._pattern = pattern
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pattern is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                pattern, self._pattern = self._pattern, None
            # 1 回の失敗で音のスレッドを死なせない。当日は音が唯一の表示なので、
            # 黙って鳴らなくなるのが一番まずい (#79: lgpio の例外で実際にそうなった)
            try:
                for on, off in pattern:
                    self.buzzer.on()
                    if self._sleep(on):
                        break
                    self.buzzer.off()
                    if self._sleep(off):
                        break
                self.buzzer.off()
            except Exception:   # noqa: BLE001
                log.exception("ブザーを鳴らせなかった")

    def _sleep(self, seconds: float) -> bool:
        """``seconds`` 待つ。途中で次の型が来たら True (打ち切り)。"""
        with self._cv:
            self._cv.wait_for(lambda: self._pattern is not None or self._stop, seconds)
            return self._pattern is not None or self._stop

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=1.0)
        self.buzzer.off()


def spawner(command: list[str]):
    def spawn():
        LOG_DIR.mkdir(exist_ok=True)
        path = LOG_DIR / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
        out = open(path, "w", encoding="utf-8")
        log.info("走行を起動: %s (出力は %s)", " ".join(command[1:]), path)
        return subprocess.Popen(command, cwd=REPO, stdout=out, stderr=subprocess.STDOUT)
    return spawn


def poweroff() -> None:
    """電源を切る。sudo はパスワード無しで ``systemctl poweroff`` を許しておくこと。"""
    subprocess.run(["sudo", "-n", "systemctl", "poweroff"], check=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ボタン 1 個 + ブザーで走行を起動・停止する (#79)")
    p.add_argument("--config", default=str(RUN_CONFIG_PATH), help="走行設定の YAML")
    p.add_argument("--check", action="store_true",
                   help="設定を検証して実行するコマンドを表示するだけ (GPIO に触らない)")
    p.add_argument("--beep-test", action="store_true",
                   help="配線の確認: 全部の音を順に鳴らし、ボタンの状態を 10 秒表示する")
    return p


def beep_test(button: Button, player: BeepPlayer) -> None:
    for name in ("BOOT", "TICK", "COUNTDOWN", "CANCEL", "DONE", "STOPPED", "FAILED",
                 "CONFIG_ERROR", "ALARM", "POWEROFF"):
        pattern = getattr(sounds, f"BEEP_{name}")
        log.info("音: %s", name)
        player.play(pattern)
        time.sleep(pattern_seconds(pattern) + 0.8)
    log.info("ボタンを押してみる (10 秒)")
    last = None
    end = time.monotonic() + 10.0
    while time.monotonic() < end:
        now = button.is_pressed()
        if now != last:
            log.info("  ボタン: %s", "押されている" if now else "離れている")
            last = now
        time.sleep(TICK_S)


def main() -> int:
    args = build_parser().parse_args()
    setup_logging()
    cfg, command = check_config(Path(args.config))
    if args.check:
        return 0 if cfg is not None else 1

    button = Button(cfg.button_gpio if cfg and cfg.button_gpio is not None
                    else DEFAULT_BUTTON_GPIO)
    buzzer = Buzzer(cfg.buzzer_gpio if cfg and cfg.buzzer_gpio is not None
                    else DEFAULT_BUZZER_GPIO,
                    passive=cfg.buzzer_passive if cfg else True)
    player = BeepPlayer(buzzer)
    if args.beep_test:
        try:
            beep_test(button, player)
        finally:
            player.close()
            button.close()
            buzzer.close()
        return 0
    launcher = Launcher(spawn=spawner(command) if command else None,
                        beep=player.play, poweroff=poweroff, log=log.info)
    log.info("ボタン GPIO%d / ブザー GPIO%d で待機中 (1 秒押して離すと発進)",
             button.pin, buzzer.pin)

    # systemd の停止 (SIGTERM) やシャットダウンでは、走行中なら子に SIGINT を送って
    # 既存の emergency_stop を通してから抜ける
    quitting = []

    def on_term(signum, _frame):
        quitting.append(signum)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    try:
        while not quitting:
            launcher.tick(time.monotonic(), button.is_pressed())
            time.sleep(TICK_S)
    finally:
        child = launcher.child
        if child is not None and child.poll() is None:
            log.warning("ランチャの終了: 走行中の子に SIGINT を送る")
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                child.send_signal(signal.SIGTERM)
        player.close()
        button.close()
        buzzer.close()
    return 0 if launcher.state is not State.CONFIG_ERROR else 1


if __name__ == "__main__":
    sys.exit(main())
