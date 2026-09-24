"""機体単体で起動・停止するためのボタンとブザー (issue #79)。

本番の会場ではネットワーク越しに操作できないので、**ボタン 1 個 + ブザー**で走らせる
(ボタンの意味は状態で変わる: :mod:`krilly.app.launcher`)。

- ボタン: タクトスイッチを GPIO と GND の間に入れ、**内部プルアップ**で読む
  (押すと 0 = ``active_low``)。外付けの抵抗は要らない
- ブザー: ``passive`` (圧電素子そのもの) なら PWM で鳴らす。``active`` (発振回路内蔵)
  なら H/L だけで鳴る。**passive で鳴らしても active は鳴る** (断続音になるだけ) が、
  逆は鳴らない (カチッと言うだけ) ので、既定は passive

Pi 5 の GPIO は RP1 経由なので ``RPi.GPIO`` は動かない。``lgpio`` を使う
(``docs/setup-pi5.md``)。他の HAL と同じく ``gpio=`` にフェイクを挿せば実機なしで
テストでき、省略したときだけ ``lgpio`` を import してチップを開く。
"""

from __future__ import annotations

#: 既定のピン (BCM 番号)。SPI0 (GPIO 7-11) と I2C1 (GPIO 2, 3) を避けた空きピン。
#: 実機で配線したピンに合わせて ``run.yaml`` で変えられる。
DEFAULT_BUTTON_GPIO = 17        # 物理ピン 11
DEFAULT_BUZZER_GPIO = 18        # 物理ピン 12
DEFAULT_CHIP = 0                # Pi 5 の RP1 (カーネルによっては gpiochip4 と同じもの)
#: passive ブザーを鳴らす周波数 [Hz]。圧電素子の共振 (2-4 kHz) の辺りが一番大きい。
DEFAULT_TONE_HZ = 2700


def open_gpio(chip: int = DEFAULT_CHIP):
    """``lgpio`` を import してチップを開き、``(lgpio モジュール, ハンドル)`` を返す。"""
    import lgpio

    return lgpio, lgpio.gpiochip_open(chip)


class _Pin:
    """``lgpio`` のモジュールとハンドルの組 (注入されたものか、自分で開いたものか)。"""

    def __init__(self, gpio, chip: int) -> None:
        if gpio is None:
            self.lg, self.handle = open_gpio(chip)
            self._owned = True
        else:
            self.lg, self.handle = gpio
            self._owned = False

    def close(self) -> None:
        if self._owned:
            self.lg.gpiochip_close(self.handle)
            self._owned = False


class Button:
    """タクトスイッチ 1 個。``is_pressed()`` は瞬間値 (チャタリング除去は呼び出し側)。"""

    def __init__(self, pin: int = DEFAULT_BUTTON_GPIO, gpio=None,
                 chip: int = DEFAULT_CHIP, active_low: bool = True) -> None:
        self.pin = pin
        self.active_low = active_low
        self._io = _Pin(gpio, chip)
        flags = self._io.lg.SET_PULL_UP if active_low else self._io.lg.SET_PULL_DOWN
        self._io.lg.gpio_claim_input(self._io.handle, pin, flags)

    def is_pressed(self) -> bool:
        level = self._io.lg.gpio_read(self._io.handle, self.pin)
        return (level == 0) if self.active_low else (level == 1)

    def close(self) -> None:
        try:
            self._io.lg.gpio_free(self._io.handle, self.pin)
        finally:
            self._io.close()


class Buzzer:
    """圧電ブザー。``on()`` / ``off()`` だけを持ち、鳴らし方の型は呼び出し側が決める。"""

    def __init__(self, pin: int = DEFAULT_BUZZER_GPIO, gpio=None,
                 chip: int = DEFAULT_CHIP, passive: bool = True,
                 tone_hz: int = DEFAULT_TONE_HZ) -> None:
        self.pin = pin
        self.passive = passive
        self.tone_hz = tone_hz
        self._io = _Pin(gpio, chip)
        self._io.lg.gpio_claim_output(self._io.handle, pin, 0)

    def on(self) -> None:
        if self.passive:
            self._io.lg.tx_pwm(self._io.handle, self.pin, self.tone_hz, 50)
        else:
            self._io.lg.gpio_write(self._io.handle, self.pin, 1)

    def off(self) -> None:
        if self.passive:
            self._io.lg.tx_pwm(self._io.handle, self.pin, 0, 0)
        self._io.lg.gpio_write(self._io.handle, self.pin, 0)

    def close(self) -> None:
        try:
            self.off()
            self._io.lg.gpio_free(self._io.handle, self.pin)
        finally:
            self._io.close()
