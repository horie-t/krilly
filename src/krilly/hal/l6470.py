"""L6470 (dSPIN) ステッピングドライバ — SPI 経由の単体ドライバ制御。

これは **単体ドライバ** の立ち上げ経路 (issue #5)。1 個の L6470 を専用の
chip-select に接続し、``spidev`` (bus, device) 経由でアクセスする。3 個の
ドライバを共有 CS でデイジーチェーン接続するのは後の段階 (issue #25)。

配線・プロトコルに関する注意 (docs/setup-pi5.md 参照):
- SPI は **mode 3** (CPOL=1, CPHA=1)、MSB ファースト、約 5 MHz。
- L6470 は **CS パルスごとに 1 バイト** をラッチする。各バイトはそれぞれ
  独立した SPI トランザクションであり、バイト間で CS がトグルする。
  ``spidev.xfer2([b])`` は転送の前後で CS をパルスするため、1 バイトずつ
  送信する。
- L6470 は内蔵のモーションエンジンを持つため、個々のステップパルスではなく
  高レベルなコマンド (Run / Move / Stop) を発行する。
- 電源投入直後は STATUS register に UVLO/OCD フラグが立っている。init 時に
  STATUS を 1 回読み出す (読み出しでクリアされる) こと。そうしないと
  ドライバは動作を拒否する。

純粋な register 変換ヘルパ (steps/s ⇄ register 値) はハードウェア無しで
ユニットテストできるよう、モジュールレベルに配置している。
"""

from __future__ import annotations

from dataclasses import dataclass

# --- L6470 タイミング定数 --------------------------------------------------
# デバイス内部の tick は 250 ns。速度・加速度の register 式はすべてこれを
# 基に導出される (データシート "Programming manual" の §"Specific registers")。
_TICK_S = 250e-9

# 変換係数 = (2 ** shift) * tick
_SPEED_COEF = (2 ** 28) * _TICK_S       # Run SPEED register   (20-bit)
_MAX_SPEED_COEF = (2 ** 18) * _TICK_S   # MAX_SPEED / FS_SPD   (10-bit)
_MIN_SPEED_COEF = (2 ** 24) * _TICK_S   # MIN_SPEED            (12-bit)
_ACC_COEF = (2 ** 40) * (_TICK_S ** 2)  # ACC / DEC            (12-bit)


# --- Register アドレス ------------------------------------------------------
class Reg:
    ABS_POS = 0x01
    EL_POS = 0x02
    MARK = 0x03
    SPEED = 0x04
    ACC = 0x05
    DEC = 0x06
    MAX_SPEED = 0x07
    MIN_SPEED = 0x08
    KVAL_HOLD = 0x09
    KVAL_RUN = 0x0A
    KVAL_ACC = 0x0B
    KVAL_DEC = 0x0C
    INT_SPEED = 0x0D
    ST_SLP = 0x0E
    FN_SLP_ACC = 0x0F
    FN_SLP_DEC = 0x10
    K_THERM = 0x11
    ADC_OUT = 0x12
    OCD_TH = 0x13
    STALL_TH = 0x14
    FS_SPD = 0x15
    STEP_MODE = 0x16
    ALARM_EN = 0x17
    CONFIG = 0x18
    STATUS = 0x19


# 各 register のペイロードのバイト幅 (ceil(bits / 8))。
_REG_BYTES = {
    Reg.ABS_POS: 3, Reg.EL_POS: 2, Reg.MARK: 3, Reg.SPEED: 3,
    Reg.ACC: 2, Reg.DEC: 2, Reg.MAX_SPEED: 2, Reg.MIN_SPEED: 2,
    Reg.KVAL_HOLD: 1, Reg.KVAL_RUN: 1, Reg.KVAL_ACC: 1, Reg.KVAL_DEC: 1,
    Reg.INT_SPEED: 2, Reg.ST_SLP: 1, Reg.FN_SLP_ACC: 1, Reg.FN_SLP_DEC: 1,
    Reg.K_THERM: 1, Reg.ADC_OUT: 1, Reg.OCD_TH: 1, Reg.STALL_TH: 1,
    Reg.FS_SPD: 2, Reg.STEP_MODE: 1, Reg.ALARM_EN: 1, Reg.CONFIG: 2,
    Reg.STATUS: 2,
}


# --- Command opcode ---------------------------------------------------------
class Cmd:
    NOP = 0x00
    SET_PARAM = 0x00      # | register
    GET_PARAM = 0x20      # | register
    RUN = 0x50            # | dir
    STEP_CLOCK = 0x58     # | dir
    MOVE = 0x40           # | dir
    GOTO = 0x60
    GOTO_DIR = 0x68       # | dir
    GO_UNTIL = 0x82       # | act | dir
    RELEASE_SW = 0x92     # | act | dir
    GO_HOME = 0x70
    GO_MARK = 0x78
    RESET_POS = 0xD8
    RESET_DEVICE = 0xC0
    SOFT_STOP = 0xB0
    HARD_STOP = 0xB8
    SOFT_HIZ = 0xA0
    HARD_HIZ = 0xA8
    GET_STATUS = 0xD0


# Direction ビット
REV = 0
FWD = 1

# Step-mode の STEP_SEL フィールド (full 〜 1/128)
STEP_MODE_FULL = 0x00
STEP_MODE_HALF = 0x01
STEP_MODE_1_4 = 0x02
STEP_MODE_1_8 = 0x03
STEP_MODE_1_16 = 0x04
STEP_MODE_1_32 = 0x05
STEP_MODE_1_64 = 0x06
STEP_MODE_1_128 = 0x07


# --- 純粋な変換 (steps/s, steps/s^2 ⇄ register 値) --------------------------
def _clamp(value: int, bits: int) -> int:
    return max(0, min(value, (1 << bits) - 1))


def speed_to_run_register(steps_per_sec: float) -> int:
    """Run 速度 [steps/s] を 20-bit の SPEED register 値に変換する。"""
    return _clamp(round(steps_per_sec * _SPEED_COEF), 20)


def run_register_to_speed(reg_value: int) -> float:
    """:func:`speed_to_run_register` の逆変換 [steps/s]。"""
    return reg_value / _SPEED_COEF


def speed_to_max_speed_register(steps_per_sec: float) -> int:
    """最大速度 [steps/s] を 10-bit の MAX_SPEED register 値に変換する。"""
    return _clamp(round(steps_per_sec * _MAX_SPEED_COEF), 10)


def speed_to_min_speed_register(steps_per_sec: float) -> int:
    """最小速度 [steps/s] を 12-bit の MIN_SPEED register 値に変換する。"""
    return _clamp(round(steps_per_sec * _MIN_SPEED_COEF), 12)


def speed_to_fs_spd_register(steps_per_sec: float) -> int:
    """フルステップ切り替え速度 [steps/s] を FS_SPD (10-bit) に変換する。"""
    return _clamp(round(steps_per_sec * _MAX_SPEED_COEF - 0.5), 10)


def accel_to_register(steps_per_sec2: float) -> int:
    """加速度・減速度 [steps/s^2] を 12-bit の register 値に変換する。"""
    return _clamp(round(steps_per_sec2 * _ACC_COEF), 12)


def decode_status(status: int, first_read: bool = False) -> str:
    """16-bit の STATUS register を人間が読める形に要約する。

    フォールトビット (UVLO/TH_WRN/TH_SD/OCD/STEP_LOSS) は active-low
    (0 = イベント発生)。応答が全ビット 0 / 全ビット 1 の場合は SPI 通信が
    確立していないことを意味する。``first_read=True`` の場合は UVLO を
    電源投入時の正常なラッチとして注記する (UVLO は電源投入時に常にセット
    され、最初の GetStatus でクリアされる)。
    """
    if status in (0x0000, 0xFFFF):
        return ("★SPI通信不可の可能性 (応答が全ビット %s)。正常なら 0x7C03 付近。"
                "配線(MISO/MOSI/SCK/CS/GND)・電源(VDD/VS)・SPI有効化・CE番号を確認"
                % ("0" if status == 0x0000 else "1"))
    flags = []
    if not (status >> 9) & 1:
        flags.append("UVLO(初回読み出しなら電源投入時の正常フラグ)"
                     if first_read else "UVLO(低電圧)")
    if not (status >> 10) & 1:
        flags.append("TH_WRN(熱警告)")
    if not (status >> 11) & 1:
        flags.append("TH_SD(熱遮断)")
    if not (status >> 12) & 1:
        flags.append("OCD(過電流)")
    if not (status >> 13) & 1:
        flags.append("STEP_LOSS_A")
    if not (status >> 14) & 1:
        flags.append("STEP_LOSS_B")
    if status & 1:
        flags.append("HiZ(出力停止)")
    return ", ".join(flags) if flags else "フォールトなし"


@dataclass(frozen=True)
class L6470Profile:
    """init 時に適用するモーション・トルク設定。

    デフォルト値は保守的 (KVAL トルクは低め、速度は中程度) で、初回電源投入時の
    ベンチテストでも安全。実機でのチューニングは M2/M6 で行う。
    """

    step_mode: int = STEP_MODE_1_16
    max_speed_steps_s: float = 400.0      # 200 step/rev で約 2 rev/s
    acc_steps_s2: float = 1000.0
    dec_steps_s2: float = 1000.0
    kval_hold: int = 0x20                 # Vs の約 12.5 %
    kval_run: int = 0x40                  # 約 25 %
    kval_acc: int = 0x40
    kval_dec: int = 0x40


class L6470:
    """1 個の chip-select に接続された単体の L6470 ドライバ。

    テスト用に ``spi`` を注入できる (``xfer2`` と ``close`` を持つ任意の
    オブジェクト)。指定しない場合は (bus, device) で ``spidev.SpiDev`` を
    オープンする。
    """

    def __init__(
        self,
        bus: int = 0,
        device: int = 0,
        spi_speed_hz: int = 5_000_000,
        spi=None,
    ) -> None:
        if spi is None:
            import spidev  # 遅延 import: 実機専用の依存 (aarch64)

            spi = spidev.SpiDev()
            spi.open(bus, device)
            spi.max_speed_hz = spi_speed_hz
            spi.mode = 0b11  # SPI mode 3
        self._spi = spi

    # -- 低レベル: CS パルスごとに 1 バイト ---------------------------------
    def _send_byte(self, value: int) -> int:
        return self._spi.xfer2([value & 0xFF])[0]

    def _send_command(self, opcode: int, payload: bytes = b"") -> None:
        self._send_byte(opcode)
        for b in payload:
            self._send_byte(b)

    # -- パラメータアクセス -------------------------------------------------
    def set_param(self, reg: int, value: int) -> None:
        nbytes = _REG_BYTES[reg]
        payload = value.to_bytes(nbytes, "big")
        self._send_command(Cmd.SET_PARAM | reg, payload)

    def get_param(self, reg: int) -> int:
        nbytes = _REG_BYTES[reg]
        self._send_byte(Cmd.GET_PARAM | reg)
        result = 0
        for _ in range(nbytes):
            result = (result << 8) | self._send_byte(Cmd.NOP)
        return result

    # -- ステータス ---------------------------------------------------------
    def get_status(self) -> int:
        """16-bit の STATUS register を読み出す (同時にクリアする)。"""
        self._send_byte(Cmd.GET_STATUS)
        hi = self._send_byte(Cmd.NOP)
        lo = self._send_byte(Cmd.NOP)
        return (hi << 8) | lo

    # -- モーションコマンド -------------------------------------------------
    def run(self, direction: int, steps_per_sec: float) -> None:
        """停止するまで一定速度で回転させる。"""
        speed = speed_to_run_register(steps_per_sec)
        self._send_command(Cmd.RUN | (direction & 1), speed.to_bytes(3, "big"))

    def move(self, direction: int, microsteps: int) -> None:
        """相対的な (マイクロ)ステップ数だけ移動して停止する。モーターは停止している必要がある。"""
        self._send_command(
            Cmd.MOVE | (direction & 1), (microsteps & 0x3FFFFF).to_bytes(3, "big")
        )

    def soft_stop(self) -> None:
        self._send_command(Cmd.SOFT_STOP)

    def hard_stop(self) -> None:
        self._send_command(Cmd.HARD_STOP)

    def soft_hiz(self) -> None:
        """減速後にブリッジをハイインピーダンス状態にする (フリースピン)。"""
        self._send_command(Cmd.SOFT_HIZ)

    def hard_hiz(self) -> None:
        self._send_command(Cmd.HARD_HIZ)

    def reset_device(self) -> None:
        self._send_command(Cmd.RESET_DEVICE)

    # -- セットアップ -------------------------------------------------------
    def configure(self, profile: L6470Profile | None = None) -> int:
        """リセットし、モーション・トルクプロファイルを適用し、電源投入時のフラグをクリアする。

        設定後に読み出した STATUS を返す (フラグはクリア済み)。
        """
        profile = profile or L6470Profile()
        self.reset_device()
        self.set_param(Reg.STEP_MODE, profile.step_mode)
        self.set_param(Reg.MAX_SPEED, speed_to_max_speed_register(profile.max_speed_steps_s))
        self.set_param(Reg.ACC, accel_to_register(profile.acc_steps_s2))
        self.set_param(Reg.DEC, accel_to_register(profile.dec_steps_s2))
        self.set_param(Reg.KVAL_HOLD, profile.kval_hold)
        self.set_param(Reg.KVAL_RUN, profile.kval_run)
        self.set_param(Reg.KVAL_ACC, profile.kval_acc)
        self.set_param(Reg.KVAL_DEC, profile.kval_dec)
        return self.get_status()  # 電源投入時の UVLO/OCD フラグをクリアする

    def close(self) -> None:
        self._spi.close()

    def __enter__(self) -> "L6470":
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.hard_hiz()
        finally:
            self.close()
