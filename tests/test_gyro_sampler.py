"""ジャイロの連続サンプリング (#81) のユニットテスト。

**この層で確かめたいのは 1 つ**: 制御周期の中で角速度が変化しているとき、
点サンプル (今の実装) と連続積分で結果が食い違うこと。食い違うなら、実機の
「ジャイロに見えない回転」の候補としてサンプリングを疑う根拠になる。
"""

import math

import pytest

from krilly.hal.gyro_sampler import GyroSampler


class FakeImu:
    """時刻を関数にした角速度を返すフェイク (z だけ使う)。"""

    def __init__(self, rate_dps, clock):
        self._rate = rate_dps
        self._clock = clock

    @property
    def gyro(self):
        return (0.0, 0.0, self._rate(self._clock()))


class Clock:
    """手で進める時計 (スレッドを使わず決定的にテストするため)。"""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def sample_at(rate_dps, step, duration, **kwargs):
    """``step`` 間隔で ``duration`` 秒ぶんサンプルし、最後の :meth:`take` を返す。"""
    clock = Clock()
    sampler = GyroSampler(FakeImu(rate_dps, clock), clock=clock, **kwargs)
    n = int(round(duration / step))
    for _ in range(n + 1):
        sampler.poll()
        clock.advance(step)
    return sampler.take()


def test_a_constant_rate_integrates_to_rate_times_time():
    s = sample_at(lambda t: 30.0, step=0.005, duration=1.0)
    assert s.dt_s == pytest.approx(1.0)
    assert math.degrees(s.delta_rad) == pytest.approx(30.0, abs=0.2)
    assert math.degrees(s.rate_rad_s) == pytest.approx(30.0, abs=0.2)
    assert s.spread_dps == pytest.approx(0.0)


def test_bias_scale_and_sign_are_applied_like_the_run_scripts():
    """``(raw - bias) * sign * scale`` の順で適用すること。

    順序が違うと bias が scale 倍されてしまい、静止時に 0 にならない。
    """
    s = sample_at(lambda t: 10.0, step=0.01, duration=1.0,
                  bias_dps=2.0, scale=0.97, sign=-1.0)
    assert math.degrees(s.rate_rad_s) == pytest.approx((10.0 - 2.0) * -1.0 * 0.97, abs=0.1)


def test_a_still_machine_integrates_to_nothing():
    s = sample_at(lambda t: 0.25, step=0.01, duration=2.0, bias_dps=0.25)
    assert math.degrees(s.delta_rad) == pytest.approx(0.0, abs=1e-9)


def test_point_sampling_and_continuous_integration_disagree_on_fast_content():
    """**これが #81 の仮説そのもの。**

    制御周期 (20ms) より速い成分があると、20ms に 1 回の点サンプルは位相次第で
    偏り、連続積分と食い違う。ここでは 50Hz の振動 (平均 0) を、50Hz の点サンプルで
    見た場合と連続積分した場合で比べる。**振動の平均は 0 なので、正しい答えは 0。**
    """
    # 50Hz の正弦波。制御周期 20ms とちょうど同期しているので、点サンプルは
    # 毎回同じ位相を引き、平均 0 のはずの振動が「一定のレート」に化ける。
    rate = lambda t: 40.0 * math.sin(2 * math.pi * 50.0 * t + 0.5)
    fast = sample_at(rate, step=0.001, duration=2.0)
    point = sample_at(rate, step=0.02, duration=2.0)
    assert math.degrees(fast.delta_rad) == pytest.approx(0.0, abs=0.5)   # 振動は消える
    assert abs(math.degrees(point.delta_rad)) > 20.0                     # 点サンプルは化ける
    # 振れの大きさは連続サンプルでしか見えない (点サンプルは同じ位相しか見ない)
    assert fast.spread_dps > 70.0
    assert point.spread_dps < 1.0


def test_take_resets_the_accumulator():
    clock = Clock()
    sampler = GyroSampler(FakeImu(lambda t: 60.0, clock), clock=clock)
    for _ in range(11):
        sampler.poll()
        clock.advance(0.01)
    first = sampler.take()
    assert math.degrees(first.delta_rad) == pytest.approx(6.0, abs=0.1)
    empty = sampler.take()
    assert empty.count == 0 and empty.delta_rad == 0.0
    assert empty.rate_rad_s == 0.0 and empty.spread_dps == 0.0   # 0 除算しない


def test_the_first_sample_only_sets_the_clock():
    """最初のサンプルには「前回」が無いので積分に入れない。"""
    clock = Clock()
    sampler = GyroSampler(FakeImu(lambda t: 60.0, clock), clock=clock)
    assert sampler.poll() is None
    assert sampler.take().count == 0


def test_uneven_sampling_uses_the_real_elapsed_time():
    """I2C の揺らぎで間隔がばらついても、実時刻の差で積分すること。"""
    clock = Clock()
    sampler = GyroSampler(FakeImu(lambda t: 36.0, clock), clock=clock)
    sampler.poll()
    for dt in (0.004, 0.020, 0.006, 0.030, 0.005):     # 合計 0.065s
        clock.advance(dt)
        sampler.poll()
    s = sampler.take()
    assert s.dt_s == pytest.approx(0.065)
    assert math.degrees(s.delta_rad) == pytest.approx(36.0 * 0.065, abs=1e-6)


def test_the_thread_samples_and_stops():
    """スレッドで回しても積分が進み、停止できること (実時計を使う)。"""
    import time as _time

    sampler = GyroSampler(FakeImu(lambda t: 90.0, _time.monotonic),
                          interval_s=0.001)
    with sampler:
        _time.sleep(0.1)
        s = sampler.take()
    assert s.count > 10
    assert math.degrees(s.delta_rad) == pytest.approx(90.0 * s.dt_s, abs=0.5)
    assert sampler._thread is None


def test_a_failing_read_stops_the_thread_and_is_reported():
    """**黙って 0 を返し続けないこと。**

    I2C が落ちたのに角速度 0 を返し続けると、走行側は「回っていない」と誤解した
    まま走り、方位の誤差が溜まる。落ちたことが分かるようにしておく。
    """
    import time as _time

    class BrokenImu:
        @property
        def gyro(self):
            raise OSError("I2C read failed")

    sampler = GyroSampler(BrokenImu(), interval_s=0.001)
    with sampler:
        _time.sleep(0.05)
        assert not sampler.alive
    assert isinstance(sampler.error, OSError)


def test_last_rate_reproduces_the_old_point_sample():
    """``last_rate_rad_s`` は「その瞬間の 1 サンプル」であること (A/B の基準)。"""
    clock = Clock()
    sampler = GyroSampler(FakeImu(lambda t: 100.0 * t, clock), clock=clock)
    clock.advance(0.5)
    sampler.poll()
    assert math.degrees(sampler.last_rate_rad_s) == pytest.approx(50.0)
