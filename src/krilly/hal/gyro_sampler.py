"""ジャイロ z を取りこぼさずに積分する (issue #81)。

**問題**: 制御ループは 20ms に 1 回 ``imu.gyro[2]`` を読み、その値を「その 20ms 間の
平均角速度」として扱っている。ところが BNO055 は NDOF モードで **100Hz** (10ms 周期) で
出力しているので、**チップが出すサンプルの半分は読まれずに捨てられている**。捨てた側と
読んだ側の統計が同じなら平均としては正しいが、動きの過渡が制御周期と同期していると
偏る。これは「実際に回っているのにジャイロが報告しない」形の誤差になり、方位保持は
ジャイロで閉じているので**補正されないまま横流れになる**。

**対処**: 別スレッドでチップの出力周期より速くサンプルし、``rate * dt`` を積み上げる。
制御ループは 1 tick 分の積分値を受け取るだけでよい。速く読みすぎても害は無い —
チップは次の更新まで同じ値を保持するので、重複を積分しても零次ホールドの積分に一致する。

**測るための道具でもある。** 1 tick の中で角速度がどれだけ振れているか
(:attr:`GyroSample.spread_dps`) が分かるので、「速い成分があるのか」という
問い自体に答えが出る。振れていなければ原因は別のところにある。

スレッドは :meth:`GyroSampler.poll` を回すだけなので、テストは ``poll`` を直接
呼んで同期的に検証できる (時刻も注入できる)。
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GyroSample:
    """1 区間 (前回 :meth:`GyroSampler.take` からの分) の積分結果。"""

    delta_rad: float      # 積分した回転角 [rad] (バイアス・スケール・符号を適用済み)
    dt_s: float           # 積分した実時間 [s]
    count: int            # 使ったサンプル数
    min_dps: float        # 区間内の角速度の最小 [deg/s] (バイアス減算後)
    max_dps: float        # 同 最大

    @property
    def rate_rad_s(self) -> float:
        """区間の平均角速度 [rad/s]。``DeadReckoning`` にはこの形で渡す。"""
        return self.delta_rad / self.dt_s if self.dt_s > 0.0 else 0.0

    @property
    def spread_dps(self) -> float:
        """区間内で角速度がどれだけ振れたか [deg/s]。

        **これが小さいのに方位がずれるなら、原因はサンプリングではない。**
        制御周期の中で大きく振れているなら、点サンプルは当たり外れの大きい賭けに
        なっている (=取りこぼしが効きうる)。
        """
        return self.max_dps - self.min_dps if self.count else 0.0


class GyroSampler:
    """ジャイロ z を連続サンプルして区間ごとの回転角を返す。

    ``imu`` は ``gyro`` プロパティ (x, y, z の deg/s) を持つオブジェクト。
    ``bias_dps`` は静止時に測ったバイアス、``scale`` は :data:`gyro_scale_z`、
    ``sign`` は取付の符号。**適用の順序は実機スクリプトと同じ**:
    ``(raw - bias) * sign * scale``。

    使い方::

        with GyroSampler(imu, bias_dps=bias, scale=cfg.gyro_scale_z) as sampler:
            while ...:
                time.sleep(dt)
                sample = sampler.take()
                motion.update(dt, gyro_rate=sample.rate_rad_s)
    """

    def __init__(self, imu, bias_dps: float = 0.0, scale: float = 1.0,
                 sign: float = 1.0, interval_s: float = 0.005,
                 clock=time.monotonic) -> None:
        self.imu = imu
        self.bias_dps = bias_dps
        self.scale = scale
        self.sign = sign
        self.interval_s = interval_s
        self._clock = clock
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_t: float | None = None
        self._reset_accumulator()
        #: サンプルの総数と、``take`` で切り出した区間の数 (診断用)。
        self.total_samples = 0
        self.intervals = 0
        #: スレッドが落ちた原因 (I2C エラーなど)。**None でなければ方位が更新されて
        #: いない**ので、走行スクリプトは気づいて警告すること — 角速度 0 を返し続けると
        #: 「回っていない」と誤解したまま走り続け、方位の誤差が黙って溜まる。
        self.error: BaseException | None = None
        #: 直近 1 サンプルの角速度 [rad/s] (バイアス等を適用済み)。
        #: **従来の点サンプルを再現するため**にある: 制御 tick の瞬間にこれを読んで
        #: ``rate * dt`` を積めば、20ms に 1 回だけ読んでいた頃と同じ積分になる。
        #: 連続積分と並べて 1 回の走行で比べられるので、走行差が入らない。
        self.last_rate_rad_s = 0.0

    def _reset_accumulator(self) -> None:
        self._delta_rad = 0.0
        self._dt = 0.0
        self._count = 0
        self._min = math.inf
        self._max = -math.inf

    # -- サンプリング -------------------------------------------------------
    def poll(self) -> float | None:
        """1 サンプル読んで積分する。積分した角度 [rad] を返す (初回は None)。

        **時間は読んだ時刻の差で測る** (指定間隔ではなく)。I2C の読み出しは
        900us 前後かかり、スレッドのスケジューリングも揺れるので、名目の間隔で
        積分すると実時間とずれていく。
        """
        dps = (self.imu.gyro[2] - self.bias_dps) * self.sign * self.scale
        now = self._clock()
        with self._lock:
            self.total_samples += 1
            self.last_rate_rad_s = math.radians(dps)
            previous, self._last_t = self._last_t, now
            if previous is None:
                return None
            dt = now - previous
            delta = math.radians(dps) * dt
            self._delta_rad += delta
            self._dt += dt
            self._count += 1
            self._min = min(self._min, dps)
            self._max = max(self._max, dps)
            return delta

    def take(self) -> GyroSample:
        """前回の :meth:`take` からの積分値を返し、蓄積をリセットする。"""
        with self._lock:
            sample = GyroSample(
                delta_rad=self._delta_rad, dt_s=self._dt, count=self._count,
                min_dps=0.0 if not self._count else self._min,
                max_dps=0.0 if not self._count else self._max,
            )
            self._reset_accumulator()
            self.intervals += 1
        return sample

    # -- スレッド -----------------------------------------------------------
    def start(self) -> "GyroSampler":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gyro-sampler",
                                        daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except BaseException as exc:      # I2C の一時的な失敗でも黙って止まらない
                self.error = exc
                return
            if self.interval_s > 0.0:
                self._stop.wait(self.interval_s)

    @property
    def alive(self) -> bool:
        """スレッドが生きているか (落ちていたら方位は更新されていない)。"""
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)

    def __enter__(self) -> "GyroSampler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
