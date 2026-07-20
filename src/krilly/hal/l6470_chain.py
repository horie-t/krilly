"""L6470 (dSPIN) デイジーチェーン制御 — 1 個の共有 chip-select に 3 個のドライバ。

これは複数ドライバの経路 (issue #25)。:mod:`krilly.hal.l6470` の単体ドライバ用の
register マップ・コマンドセット・変換の上に構築される。

デイジーチェーンのトポロジ (SCLK 共有 + CS 共有):

    Pi.MOSI -> dev0.SDI, dev0.SDO -> dev1.SDI, dev1.SDO -> dev2.SDI,
    dev2.SDO -> Pi.MISO

したがって **index 0 のデバイスが MOSI に最も近く**、index ``n-1`` が MISO に
最も近い。配線が逆の番号付けになっている場合は ``mosi_is_index0=False`` を
指定する。

フレーミングモデル (重要):
- チェーン全体が 1 個の大きなシフトレジスタ (n バイト = n*8 ビット)。**1 回の
  CS パルス** で ``n`` バイトをクロックすると、各デバイスにちょうど 1 バイトずつ
  行き渡り、各デバイスは CS の立ち上がりエッジで自分のバイトをラッチする。
- したがって K バイトのコマンド (opcode + payload) には **K 回の CS パルス** が
  必要で、各パルスでデバイスごとに 1 バイトずつ送信する (コマンドを送らない
  デバイスには NOP)。
- 最初にクロックアウトされたバイトはチェーンの *最も奥* まで進むため、パルス
  ごとの送信順序はデバイスの index に対して逆順になる (:meth:`_pulse` で処理)。
  MISO の応答バイトを index の順序に戻すのにも同じ逆転を用いる。

読み出しに関する注意: デイジーチェーンでは GetParam/GetStatus の応答はチェーン
分だけシフトされ、opcode の次の転送で現れる。ここの読み出しヘルパは一般的な
方式を実装しているが、**正確な読み出しフレーミングは実機で検証する必要がある**。
回転・書き込みが M1 の主目的なので、読み出しはベンチで確認できるまで
best-effort として扱う。
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
    """1 個の CS にデイジーチェーン接続された ``num_devices`` 個の L6470 ドライバを制御する。"""

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
            import spidev  # 遅延 import: 実機専用の依存 (aarch64)

            spi = spidev.SpiDev()
            spi.open(bus, device)
            spi.max_speed_hz = spi_speed_hz
            spi.mode = 0b11  # SPI mode 3
        self._spi = spi

    # -- 低レベル: 1 回の CS パルスでデバイスごとに 1 バイト転送 ------------
    def _pulse(self, values_by_index: list[int]) -> list[int]:
        """1 回の CS パルスでデバイスごとに 1 バイトをクロックする。

        ``values_by_index[i]`` はデバイス ``i`` 宛てのバイト。応答バイトを
        デバイスの index の順序に戻して返す。
        """
        if len(values_by_index) != self.n:
            raise ValueError(f"expected {self.n} bytes, got {len(values_by_index)}")
        tx = list(values_by_index)
        if self._mosi_first:
            tx.reverse()  # index n-1 (MISO に最も近い) を最初にクロックアウトする必要がある
        rx = self._spi.xfer2([b & 0xFF for b in tx])
        if self._mosi_first:
            rx = list(reversed(rx))
        return list(rx)

    # -- コマンドフレーミング -----------------------------------------------
    def send_commands(self, commands: list[bytes]) -> list[list[int]]:
        """各デバイスに (それぞれ異なりうる) コマンドを同時に送信する。

        ``commands[i]`` はデバイス ``i`` 向けの完全なバイト列 (opcode +
        payload)。すべてのコマンドは同じ長さでなければならず、短いものは NOP で
        パディングする。デバイスごとに応答バイトのリスト (パルスごとに 1 つ) を
        返す。
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
        """すべてのデバイスに同じコマンドを送信する。"""
        cmd = bytes([opcode]) + payload
        return self.send_commands([cmd] * self.n)

    def send_to(self, index: int, opcode: int, payload: bytes = b"") -> list[list[int]]:
        """単一のデバイスにコマンドを送る。他のすべてには NOP が送られる。"""
        if not 0 <= index < self.n:
            raise IndexError(index)
        cmd = bytes([opcode]) + payload
        nop = bytes([Cmd.NOP]) * len(cmd)
        commands = [cmd if i == index else nop for i in range(self.n)]
        return self.send_commands(commands)

    # -- パラメータ / ステータス --------------------------------------------
    def set_param_all(self, reg: int, value: int) -> None:
        payload = value.to_bytes(_REG_BYTES[reg], "big")
        self.broadcast(Cmd.SET_PARAM | reg, payload)

    def set_param(self, index: int, reg: int, value: int) -> None:
        payload = value.to_bytes(_REG_BYTES[reg], "big")
        self.send_to(index, Cmd.SET_PARAM | reg, payload)

    def get_status_all(self) -> list[int]:
        """すべてのデバイスの 16-bit STATUS を読み出す (best-effort。実機で検証すること)。"""
        resp = self.broadcast(Cmd.GET_STATUS, bytes([Cmd.NOP, Cmd.NOP]))
        # resp[i] = [ダミー(opcode 転送), hi, lo]
        return [(r[1] << 8) | r[2] for r in resp]

    def get_param_all(self, reg: int) -> list[int]:
        nbytes = _REG_BYTES[reg]
        resp = self.broadcast(Cmd.GET_PARAM | reg, bytes([Cmd.NOP]) * nbytes)
        out = []
        for r in resp:
            value = 0
            for b in r[1:]:  # opcode 転送のバイトをスキップする
                value = (value << 8) | b
            out.append(value)
        return out

    # -- モーション (デバイス個別) ------------------------------------------
    def run(self, index: int, direction: int, steps_per_sec: float) -> None:
        speed = speed_to_run_register(steps_per_sec)
        self.send_to(index, Cmd.RUN | (direction & 1), speed.to_bytes(3, "big"))

    def move(self, index: int, direction: int, microsteps: int) -> None:
        self.send_to(
            index, Cmd.MOVE | (direction & 1), (microsteps & 0x3FFFFF).to_bytes(3, "big")
        )

    # -- モーション (全デバイス同時) ----------------------------------------
    def run_all(self, directions: list[int], speeds: list[float]) -> None:
        """整列したパルスで、全デバイスをそれぞれの方向・速度で始動させる。"""
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

    # -- セットアップ -------------------------------------------------------
    def configure_all(self, profile: L6470Profile | None = None) -> list[int]:
        """リセットし、全デバイスに同じプロファイルを適用し、電源投入時のフラグをクリアする。

        設定後の各デバイスの STATUS を返す。
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
