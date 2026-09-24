"""ボタンとブザーの HAL (#79)。lgpio をフェイクに差し替えて呼び出し列を確かめる。"""

from krilly.hal.gpio_io import Button, Buzzer


class FakeLgpio:
    SET_PULL_UP = 0x20
    SET_PULL_DOWN = 0x40

    def __init__(self):
        self.calls = []
        self.level = 1

    def gpio_claim_input(self, h, pin, flags):
        self.calls.append(("claim_input", pin, flags))

    def gpio_claim_output(self, h, pin, level):
        self.calls.append(("claim_output", pin, level))

    def gpio_read(self, h, pin):
        return self.level

    def gpio_write(self, h, pin, level):
        self.calls.append(("write", pin, level))

    def tx_pwm(self, h, pin, freq, duty):
        self.calls.append(("pwm", pin, freq, duty))

    def gpio_free(self, h, pin):
        self.calls.append(("free", pin))

    def gpiochip_close(self, h):
        self.calls.append(("close",))


def test_button_is_active_low_with_the_internal_pull_up():
    lg = FakeLgpio()
    b = Button(17, gpio=(lg, 0))
    assert lg.calls[0] == ("claim_input", 17, lg.SET_PULL_UP)
    lg.level = 1
    assert not b.is_pressed()
    lg.level = 0
    assert b.is_pressed()
    b.close()
    assert ("free", 17) in lg.calls
    assert ("close",) not in lg.calls              # 注入されたチップは閉じない


def test_passive_buzzer_is_driven_with_pwm():
    lg = FakeLgpio()
    z = Buzzer(18, gpio=(lg, 0), tone_hz=2700)
    z.on()
    z.off()
    assert ("pwm", 18, 2700, 50) in lg.calls
    assert lg.calls[-2:] == [("pwm", 18, 0, 0), ("write", 18, 0)]


def test_active_buzzer_is_just_high_and_low():
    lg = FakeLgpio()
    z = Buzzer(18, gpio=(lg, 0), passive=False)
    z.on()
    z.off()
    assert [c for c in lg.calls if c[0] == "pwm"] == []
    assert lg.calls[-2:] == [("write", 18, 1), ("write", 18, 0)]
