"""L6470 (dSPIN) daisy-chain control — 3 drivers on one shared chip-select.

This is the multi-driver path (issue #25). It builds on the single-driver
register map / command set / conversions in :mod:`krilly.hal.l6470`.

Daisy-chain topology (shared SCLK + shared CS):

    Pi.MOSI -> dev0.SDI, dev0.SDO -> dev1.SDI, dev1.SDO -> dev2.SDI,
    dev2.SDO -> Pi.MISO

So device **index 0 is nearest MOSI** and index ``n-1`` is nearest MISO.
Set ``mosi_is_index0=False`` if your wiring numbers them the other way.

Framing model (important):
- The chain is one big shift register (n bytes = n*8 bits). In **one CS pulse**
  we clock ``n`` bytes — exactly one byte lands in each device, and each device
  latches its byte on the CS rising edge.
- A K-byte command (opcode + payload) therefore needs **K CS pulses**, and in
  each pulse we send one byte per device (NOP for devices we aren't commanding).
- Because the first byte clocked out travels the *furthest* down the chain, the
  per-pulse transmit order is reversed relative to device index (handled in
  :meth:`_pulse`). The same reversal maps the MISO response bytes back to index
  order.

Read-back caveat: GetParam/GetStatus responses in a daisy chain are shifted by
the chain and appear on the transfer following the opcode. The read helpers here
implement the common scheme, but **the exact read framing must be validated on
hardware** — spin/write is the primary M1 goal; treat reads as best-effort until
confirmed on the bench.
"""

from __future__ import annotations

from krilly.hal.l6470 import (
    FWD,
    REV,
    Cmd,
    L6470Profile,
    Reg,
    _REG_BYTES,
    accel_to_register,
    speed_to_max_speed_register,
    speed_to_run_register,
)

__all__ = ["L6470Chain", "FWD", "REV"]


class L6470Chain:
    """Control ``num_devices`` L6470 drivers daisy-chained on one CS."""

    def __init__(
        self,
        num_devices: int = 3,
        bus: int = 0,
        device: int = 0,
        spi_speed_hz: int = 5_000_000,
        spi=None,
        mosi_is_index0: bool = True,
    ) -> None:
        if num_devices < 1:
            raise ValueError("num_devices must be >= 1")
        self.n = num_devices
        self._mosi_first = mosi_is_index0
        if spi is None:
            import spidev  # lazy: hardware-only dependency (aarch64)

            spi = spidev.SpiDev()
            spi.open(bus, device)
            spi.max_speed_hz = spi_speed_hz
            spi.mode = 0b11  # SPI mode 3
        self._spi = spi

    # -- low-level: one CS pulse transfers one byte per device --------------
    def _pulse(self, values_by_index: list[int]) -> list[int]:
        """Clock one byte per device in a single CS pulse.

        ``values_by_index[i]`` is the byte destined for device ``i``. Returns
        the response bytes mapped back to device index order.
        """
        if len(values_by_index) != self.n:
            raise ValueError(f"expected {self.n} bytes, got {len(values_by_index)}")
        tx = list(values_by_index)
        if self._mosi_first:
            tx.reverse()  # index n-1 (nearest MISO) must be clocked out first
        rx = self._spi.xfer2([b & 0xFF for b in tx])
        if self._mosi_first:
            rx = list(reversed(rx))
        return list(rx)

    # -- command framing ----------------------------------------------------
    def send_commands(self, commands: list[bytes]) -> list[list[int]]:
        """Send a (possibly different) command to each device simultaneously.

        ``commands[i]`` is the full byte string (opcode + payload) for device
        ``i``. All commands must be the same length; pad shorter ones with NOP.
        Returns, per device, the list of response bytes (one per pulse).
        """
        if len(commands) != self.n:
            raise ValueError(f"expected {self.n} commands, got {len(commands)}")
        length = len(commands[0])
        if any(len(c) != length for c in commands):
            raise ValueError("all commands must be the same byte length")
        responses: list[list[int]] = [[] for _ in range(self.n)]
        for j in range(length):
            rx = self._pulse([commands[i][j] for i in range(self.n)])
            for i in range(self.n):
                responses[i].append(rx[i])
        return responses

    def broadcast(self, opcode: int, payload: bytes = b"") -> list[list[int]]:
        """Send the same command to every device."""
        cmd = bytes([opcode]) + payload
        return self.send_commands([cmd] * self.n)

    def send_to(self, index: int, opcode: int, payload: bytes = b"") -> list[list[int]]:
        """Command a single device; all others receive NOP."""
        if not 0 <= index < self.n:
            raise IndexError(index)
        cmd = bytes([opcode]) + payload
        nop = bytes([Cmd.NOP]) * len(cmd)
        commands = [cmd if i == index else nop for i in range(self.n)]
        return self.send_commands(commands)

    # -- parameter / status -------------------------------------------------
    def set_param_all(self, reg: int, value: int) -> None:
        payload = value.to_bytes(_REG_BYTES[reg], "big")
        self.broadcast(Cmd.SET_PARAM | reg, payload)

    def set_param(self, index: int, reg: int, value: int) -> None:
        payload = value.to_bytes(_REG_BYTES[reg], "big")
        self.send_to(index, Cmd.SET_PARAM | reg, payload)

    def get_status_all(self) -> list[int]:
        """Read the 16-bit STATUS of every device (best-effort; verify on HW)."""
        resp = self.broadcast(Cmd.GET_STATUS, bytes([Cmd.NOP, Cmd.NOP]))
        # resp[i] = [dummy(opcode xfer), hi, lo]
        return [(r[1] << 8) | r[2] for r in resp]

    def get_param_all(self, reg: int) -> list[int]:
        nbytes = _REG_BYTES[reg]
        resp = self.broadcast(Cmd.GET_PARAM | reg, bytes([Cmd.NOP]) * nbytes)
        out = []
        for r in resp:
            value = 0
            for b in r[1:]:  # skip the opcode-transfer byte
                value = (value << 8) | b
            out.append(value)
        return out

    # -- motion (per device) ------------------------------------------------
    def run(self, index: int, direction: int, steps_per_sec: float) -> None:
        speed = speed_to_run_register(steps_per_sec)
        self.send_to(index, Cmd.RUN | (direction & 1), speed.to_bytes(3, "big"))

    def move(self, index: int, direction: int, microsteps: int) -> None:
        self.send_to(
            index, Cmd.MOVE | (direction & 1), (microsteps & 0x3FFFFF).to_bytes(3, "big")
        )

    # -- motion (all devices at once) ---------------------------------------
    def run_all(self, directions: list[int], speeds: list[float]) -> None:
        """Start all devices at their own direction/speed in aligned pulses."""
        if len(directions) != self.n or len(speeds) != self.n:
            raise ValueError("directions/speeds length must match num_devices")
        commands = [
            bytes([Cmd.RUN | (directions[i] & 1)])
            + speed_to_run_register(speeds[i]).to_bytes(3, "big")
            for i in range(self.n)
        ]
        self.send_commands(commands)

    def soft_stop_all(self) -> None:
        self.broadcast(Cmd.SOFT_STOP)

    def hard_stop_all(self) -> None:
        self.broadcast(Cmd.HARD_STOP)

    def hard_hiz_all(self) -> None:
        self.broadcast(Cmd.HARD_HIZ)

    def reset_all(self) -> None:
        self.broadcast(Cmd.RESET_DEVICE)

    # -- setup --------------------------------------------------------------
    def configure_all(self, profile: L6470Profile | None = None) -> list[int]:
        """Reset + apply the same profile to every device; clear power-up flags.

        Returns each device's STATUS after configuration.
        """
        profile = profile or L6470Profile()
        self.reset_all()
        self.set_param_all(Reg.STEP_MODE, profile.step_mode)
        self.set_param_all(Reg.MAX_SPEED, speed_to_max_speed_register(profile.max_speed_steps_s))
        self.set_param_all(Reg.ACC, accel_to_register(profile.acc_steps_s2))
        self.set_param_all(Reg.DEC, accel_to_register(profile.dec_steps_s2))
        self.set_param_all(Reg.KVAL_HOLD, profile.kval_hold)
        self.set_param_all(Reg.KVAL_RUN, profile.kval_run)
        self.set_param_all(Reg.KVAL_ACC, profile.kval_acc)
        self.set_param_all(Reg.KVAL_DEC, profile.kval_dec)
        return self.get_status_all()

    def close(self) -> None:
        self._spi.close()

    def __enter__(self) -> "L6470Chain":
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.hard_hiz_all()
        finally:
            self.close()
