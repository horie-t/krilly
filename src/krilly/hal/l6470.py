"""L6470 (dSPIN) stepper driver — single-driver control over SPI.

This is the **single-driver** bring-up path (issue #5): one L6470 on its own
chip-select, addressed via ``spidev`` (bus, device). Daisy-chaining all three
drivers on a shared CS is a later step (issue #25).

Wiring / protocol notes (see docs/setup-pi5.md):
- SPI **mode 3** (CPOL=1, CPHA=1), MSB-first, ~5 MHz.
- The L6470 latches **one byte per CS pulse**: every byte is its own SPI
  transaction, so CS toggles between bytes. ``spidev.xfer2([b])`` pulses CS
  around the transfer, so we send byte-by-byte.
- The L6470 has an on-board motion engine: we issue high-level commands
  (Run / Move / Stop), not individual step pulses.
- On power-up the STATUS register holds UVLO/OCD flags; read STATUS once at
  init (it clears on read) or the driver refuses to move.

The pure register-conversion helpers (steps/s ⇄ register value) live at module
level so they can be unit-tested without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- L6470 timing constant -------------------------------------------------
# The device's internal tick is 250 ns. All speed/accel register formulas are
# derived from it (datasheet "Programming manual", §"Specific registers").
_TICK_S = 250e-9

# Conversion coefficients = (2 ** shift) * tick
_SPEED_COEF = (2 ** 28) * _TICK_S       # Run SPEED register   (20-bit)
_MAX_SPEED_COEF = (2 ** 18) * _TICK_S   # MAX_SPEED / FS_SPD   (10-bit)
_MIN_SPEED_COEF = (2 ** 24) * _TICK_S   # MIN_SPEED            (12-bit)
_ACC_COEF = (2 ** 40) * (_TICK_S ** 2)  # ACC / DEC            (12-bit)


# --- Register addresses ----------------------------------------------------
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


# Byte width of each register's payload (ceil(bits / 8)).
_REG_BYTES = {
    Reg.ABS_POS: 3, Reg.EL_POS: 2, Reg.MARK: 3, Reg.SPEED: 3,
    Reg.ACC: 2, Reg.DEC: 2, Reg.MAX_SPEED: 2, Reg.MIN_SPEED: 2,
    Reg.KVAL_HOLD: 1, Reg.KVAL_RUN: 1, Reg.KVAL_ACC: 1, Reg.KVAL_DEC: 1,
    Reg.INT_SPEED: 2, Reg.ST_SLP: 1, Reg.FN_SLP_ACC: 1, Reg.FN_SLP_DEC: 1,
    Reg.K_THERM: 1, Reg.ADC_OUT: 1, Reg.OCD_TH: 1, Reg.STALL_TH: 1,
    Reg.FS_SPD: 2, Reg.STEP_MODE: 1, Reg.ALARM_EN: 1, Reg.CONFIG: 2,
    Reg.STATUS: 2,
}


# --- Command opcodes -------------------------------------------------------
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


# Direction bit
REV = 0
FWD = 1

# Step-mode STEP_SEL field (full .. 1/128)
STEP_MODE_FULL = 0x00
STEP_MODE_HALF = 0x01
STEP_MODE_1_4 = 0x02
STEP_MODE_1_8 = 0x03
STEP_MODE_1_16 = 0x04
STEP_MODE_1_32 = 0x05
STEP_MODE_1_64 = 0x06
STEP_MODE_1_128 = 0x07


# --- Pure conversions (steps/s, steps/s^2 ⇄ register value) ----------------
def _clamp(value: int, bits: int) -> int:
    return max(0, min(value, (1 << bits) - 1))


def speed_to_run_register(steps_per_sec: float) -> int:
    """Convert a Run speed [steps/s] to the 20-bit SPEED register value."""
    return _clamp(round(steps_per_sec * _SPEED_COEF), 20)


def run_register_to_speed(reg_value: int) -> float:
    """Inverse of :func:`speed_to_run_register` [steps/s]."""
    return reg_value / _SPEED_COEF


def speed_to_max_speed_register(steps_per_sec: float) -> int:
    """Convert a max speed [steps/s] to the 10-bit MAX_SPEED register value."""
    return _clamp(round(steps_per_sec * _MAX_SPEED_COEF), 10)


def speed_to_min_speed_register(steps_per_sec: float) -> int:
    """Convert a min speed [steps/s] to the 12-bit MIN_SPEED register value."""
    return _clamp(round(steps_per_sec * _MIN_SPEED_COEF), 12)


def speed_to_fs_spd_register(steps_per_sec: float) -> int:
    """Convert the full-step switch-over speed [steps/s] to FS_SPD (10-bit)."""
    return _clamp(round(steps_per_sec * _MAX_SPEED_COEF - 0.5), 10)


def accel_to_register(steps_per_sec2: float) -> int:
    """Convert acceleration/deceleration [steps/s^2] to the 12-bit register."""
    return _clamp(round(steps_per_sec2 * _ACC_COEF), 12)


def decode_status(status: int, first_read: bool = False) -> str:
    """Human-readable summary of the 16-bit STATUS register.

    Fault bits (UVLO/TH_WRN/TH_SD/OCD/STEP_LOSS) are active-low (0 = event).
    An all-zero / all-one response means no SPI communication was established.
    ``first_read=True`` annotates UVLO as the normal power-up latch (UVLO is
    always set at power-up and cleared by the first GetStatus).
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
    """Motion/torque configuration applied at init.

    Defaults are conservative (low KVAL torque, moderate speed) — safe for
    first power-on bench testing. Tune on hardware in M2/M6.
    """

    step_mode: int = STEP_MODE_1_16
    max_speed_steps_s: float = 400.0      # ~2 rev/s at 200 step/rev
    acc_steps_s2: float = 1000.0
    dec_steps_s2: float = 1000.0
    kval_hold: int = 0x20                 # ~12.5 % of Vs
    kval_run: int = 0x40                  # ~25 %
    kval_acc: int = 0x40
    kval_dec: int = 0x40


class L6470:
    """Single L6470 driver on one chip-select.

    ``spi`` may be injected (any object with ``xfer2`` and ``close``) for
    testing; otherwise a ``spidev.SpiDev`` is opened on (bus, device).
    """

    def __init__(
        self,
        bus: int = 0,
        device: int = 0,
        spi_speed_hz: int = 5_000_000,
        spi=None,
    ) -> None:
        if spi is None:
            import spidev  # lazy: hardware-only dependency (aarch64)

            spi = spidev.SpiDev()
            spi.open(bus, device)
            spi.max_speed_hz = spi_speed_hz
            spi.mode = 0b11  # SPI mode 3
        self._spi = spi

    # -- low-level: one byte per CS pulse -----------------------------------
    def _send_byte(self, value: int) -> int:
        return self._spi.xfer2([value & 0xFF])[0]

    def _send_command(self, opcode: int, payload: bytes = b"") -> None:
        self._send_byte(opcode)
        for b in payload:
            self._send_byte(b)

    # -- parameter access ---------------------------------------------------
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

    # -- status -------------------------------------------------------------
    def get_status(self) -> int:
        """Read (and clear) the 16-bit STATUS register."""
        self._send_byte(Cmd.GET_STATUS)
        hi = self._send_byte(Cmd.NOP)
        lo = self._send_byte(Cmd.NOP)
        return (hi << 8) | lo

    # -- motion commands ----------------------------------------------------
    def run(self, direction: int, steps_per_sec: float) -> None:
        """Spin at constant speed until stopped."""
        speed = speed_to_run_register(steps_per_sec)
        self._send_command(Cmd.RUN | (direction & 1), speed.to_bytes(3, "big"))

    def move(self, direction: int, microsteps: int) -> None:
        """Move a relative number of (micro)steps, then stop. Motor must be stopped."""
        self._send_command(
            Cmd.MOVE | (direction & 1), (microsteps & 0x3FFFFF).to_bytes(3, "big")
        )

    def soft_stop(self) -> None:
        self._send_command(Cmd.SOFT_STOP)

    def hard_stop(self) -> None:
        self._send_command(Cmd.HARD_STOP)

    def soft_hiz(self) -> None:
        """Decelerate then put the bridges in high-impedance (free spin)."""
        self._send_command(Cmd.SOFT_HIZ)

    def hard_hiz(self) -> None:
        self._send_command(Cmd.HARD_HIZ)

    def reset_device(self) -> None:
        self._send_command(Cmd.RESET_DEVICE)

    # -- setup --------------------------------------------------------------
    def configure(self, profile: L6470Profile | None = None) -> int:
        """Reset, apply a motion/torque profile, and clear power-up flags.

        Returns the STATUS read after configuration (flags cleared).
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
        return self.get_status()  # clears UVLO/OCD power-up flags

    def close(self) -> None:
        self._spi.close()

    def __enter__(self) -> "L6470":
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.hard_hiz()
        finally:
            self.close()
