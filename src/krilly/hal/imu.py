"""BNO055 9軸 IMU を **I2C** 経由で扱う (issue #6)。

ボード (秋月 AE-BNO055-BO) は I2C モード (アドレス 0x28) で出荷され、
ジャンパの変更は不要なので、I2C を使えば小さな UART 選択用のはんだパッドを
触らずに済む。

クロックストレッチについての注意: 従来の Broadcom BSC I2C (Pi 1-4) は
BNO055 のクロックストレッチを正しく扱えないが、**Pi 5 は RP1 (DesignWare) I2C を
使用しており、これを正しく処理できる** — 100 kHz で実機上の安定動作が確認済みなので、
バス速度を下げる必要はない。念のため保険として、各転送は OSError 時にリトライする。
もし遅い Pi で読み取りが正しく行えない場合は、``dtparam=i2c_arm_baudrate`` で
バス速度を下げること (docs/setup-pi5.md を参照)。

レジスタプロトコル: 標準的な I2C レジスタ読み書き (レジスタポインタを書き込み、
repeated-start で N バイト読み取る)。マルチバイト値はすべて符号付き 16-bit
リトルエンディアン。デフォルトの単位: Euler 16 LSB/deg、gyro 16 LSB/deg-per-sec、
quaternion 2^14 LSB/unit。使用するレジスタはすべて page 0 (デフォルト) にある。
"""

from __future__ import annotations

import time

# --- レジスタ (page 0) ------------------------------------------------------
CHIP_ID = 0x00
GYR_DATA_X_LSB = 0x14
EUL_HEADING_LSB = 0x1A
QUA_DATA_W_LSB = 0x20
CALIB_STAT = 0x35
UNIT_SEL = 0x3B
OPR_MODE = 0x3D
PWR_MODE = 0x3E
SYS_TRIGGER = 0x3F

CHIP_ID_VALUE = 0xA0
DEFAULT_ADDRESS = 0x28  # AE-BNO055-BO の工場出荷デフォルト (J1 オープン)

# 動作モード
MODE_CONFIG = 0x00
MODE_NDOF = 0x0C  # 9-DOF センサフュージョン、絶対姿勢

# デフォルトの単位選択に対応するスケールファクタ
_EULER_LSB_PER_DEG = 16.0
_GYRO_LSB_PER_DPS = 16.0
_QUAT_LSB = float(1 << 14)  # 16384


def _s16(lo: int, hi: int) -> int:
    """2 バイトを符号付き 16-bit リトルエンディアン整数として解釈する。"""
    value = lo | (hi << 8)
    return value - 0x10000 if value & 0x8000 else value


class Bno055Imu:
    """BNO055 を I2C 経由で扱う。

    ``bus`` はテスト用に注入できる (``read_i2c_block_data`` /
    ``write_i2c_block_data`` / ``close`` を持つオブジェクト、例えば
    ``smbus2.SMBus`` など)。注入しない場合は ``bus_id`` 上で ``smbus2.SMBus`` を開く。
    """

    def __init__(
        self,
        address: int = DEFAULT_ADDRESS,
        bus_id: int = 1,
        bus=None,
        retries: int = 5,
    ) -> None:
        if bus is None:
            from smbus2 import SMBus  # 遅延インポート

            bus = SMBus(bus_id)
        self._bus = bus
        self._addr = address
        self._retries = retries

    # -- 低レベルなレジスタアクセス (I2C エラー / クロックストレッチ時にリトライ) --
    def _read_register(self, reg: int, length: int) -> bytes:
        last = None
        for _ in range(self._retries):
            try:
                return bytes(self._bus.read_i2c_block_data(self._addr, reg, length))
            except OSError as exc:  # バスエラー / クロックストレッチによる破損
                last = exc
        raise IOError(f"BNO055 I2C read reg 0x{reg:02X} failed: {last}")

    def _write_register(self, reg: int, data: bytes) -> None:
        last = None
        for _ in range(self._retries):
            try:
                self._bus.write_i2c_block_data(self._addr, reg, list(data))
                return
            except OSError as exc:
                last = exc
        raise IOError(f"BNO055 I2C write reg 0x{reg:02X} failed: {last}")

    def _read_vector(self, reg: int, count: int, scale: float) -> tuple[float, ...]:
        data = self._read_register(reg, count * 2)
        return tuple(_s16(data[2 * i], data[2 * i + 1]) / scale for i in range(count))

    # -- セットアップ -------------------------------------------------------
    def set_mode(self, mode: int) -> None:
        self._write_register(OPR_MODE, bytes([mode]))

    def begin(self, mode: int = MODE_NDOF, use_external_crystal: bool = False) -> None:
        """チップを確認し、動作モード (デフォルトは NDOF) に移行する。

        ボードに 32.768 kHz の水晶発振子がある場合 (AE-BNO055-BO にはある) は
        ``use_external_crystal=True`` を指定する。heading の安定性が向上する。
        """
        chip = self._read_register(CHIP_ID, 1)[0]
        if chip != CHIP_ID_VALUE:
            raise IOError(f"unexpected BNO055 chip id 0x{chip:02X} (expected 0xA0)")
        self.set_mode(MODE_CONFIG)
        time.sleep(0.025)  # operating -> CONFIG には約 19 ms 必要
        if use_external_crystal:
            self._write_register(SYS_TRIGGER, bytes([0x80]))
            time.sleep(0.01)
        self.set_mode(mode)
        time.sleep(0.02)  # CONFIG -> operating には約 7 ms 必要

    # -- フュージョン結果 / 生の測定値 -------------------------------------
    @property
    def euler(self) -> tuple[float, float, float]:
        """フュージョンした絶対姿勢 (heading, roll, pitch) を度単位で返す。"""
        return self._read_vector(EUL_HEADING_LSB, 3, _EULER_LSB_PER_DEG)  # type: ignore[return-value]

    @property
    def heading_deg(self) -> float:
        return self.euler[0]

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        """フュージョンした姿勢のクォータニオン (w, x, y, z)。"""
        return self._read_vector(QUA_DATA_W_LSB, 4, _QUAT_LSB)  # type: ignore[return-value]

    @property
    def gyro(self) -> tuple[float, float, float]:
        """角速度 (x, y, z) を deg/s 単位で返す。"""
        return self._read_vector(GYR_DATA_X_LSB, 3, _GYRO_LSB_PER_DPS)  # type: ignore[return-value]

    @property
    def calibration_status(self) -> tuple[int, int, int, int]:
        """キャリブレーションレベル (sys, gyro, accel, mag)。各値は 0 (未較正) .. 3 (完了)。"""
        c = self._read_register(CALIB_STAT, 1)[0]
        return ((c >> 6) & 3, (c >> 4) & 3, (c >> 2) & 3, c & 3)

    def measure_gyro_bias(
        self, samples: int = 100, delay: float = 0.01
    ) -> tuple[float, float, float]:
        """ロボットを静止させた状態で gyro 出力を平均する (deg/s)。

        デッドレコニングでは、以降の gyro 読み取り値からこのバイアスを差し引く。
        計測の間はロボットを動かさないこと。
        """
        sx = sy = sz = 0.0
        for _ in range(samples):
            x, y, z = self.gyro
            sx += x
            sy += y
            sz += z
            time.sleep(delay)
        return (sx / samples, sy / samples, sz / samples)

    def close(self) -> None:
        self._bus.close()

    def __enter__(self) -> "Bno055Imu":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
