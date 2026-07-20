#!/usr/bin/env python3
"""キーボードによるテレオペ (issue #10).

キー入力でボディ速度指令 (vx, vy, omega) を増減し、加減速ランプ付きの
:class:`VelocityDriver` で 3輪を駆動する。curses でノンブロッキングに
キーを読み、一定周期で ``driver.update(dt)`` を呼ぶ。

キー操作:
    w / s : 前進(+x) / 後退(-x)
    a / d : 左(+y) / 右(-y)
    q / e : 左旋回(+omega/CCW) / 右旋回(-omega/CW)
    space : 停止 (指令を 0 に)
    x     : 終了 (減速して停止)

例:
    python -m scripts.teleop
    python -m scripts.teleop --v-step 0.02 --v-max 0.3 --omega-max 3.0

座標系は docs/coordinate-frames.md (+x前 / +y左 / +omega=CCW)。
"""

from __future__ import annotations

import argparse
import curses
import time

from krilly.hal.l6470 import L6470Profile
from krilly.hal.l6470_chain import L6470Chain
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.motion.velocity_driver import VelocityDriver


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def apply_key(
    ch: int,
    vx: float,
    vy: float,
    omega: float,
    v_step: float,
    omega_step: float,
    v_max: float,
    omega_max: float,
) -> tuple[float, float, float]:
    """押されたキー ``ch`` に応じて目標速度 (vx, vy, omega) を更新して返す。

    未対応キーは現状維持。並進は ±v_max、旋回は ±omega_max にクランプ。
    """
    if ch == ord("w"):
        vx = _clamp(vx + v_step, v_max)
    elif ch == ord("s"):
        vx = _clamp(vx - v_step, v_max)
    elif ch == ord("a"):
        vy = _clamp(vy + v_step, v_max)
    elif ch == ord("d"):
        vy = _clamp(vy - v_step, v_max)
    elif ch == ord("q"):
        omega = _clamp(omega + omega_step, omega_max)
    elif ch == ord("e"):
        omega = _clamp(omega - omega_step, omega_max)
    elif ch == ord(" "):
        vx = vy = omega = 0.0
    return (vx, vy, omega)


_HELP = (
    "w/s:前後(+x/-x)  a/d:左右(+y/-y)  q/e:左右旋回(+/-omega)  "
    "space:停止  x:終了"
)


def _run(stdscr, args) -> str | None:
    curses.curs_set(0)
    stdscr.timeout(int(args.dt * 1000))  # getch は最大 dt[ms] ブロック

    with L6470Chain(num_devices=args.devices) as chain:
        statuses = chain.configure_all(L6470Profile())
        if any(s in (0x0000, 0xFFFF) for s in statuses):
            return f"SPI 応答異常 STATUS={[f'0x{s:04X}' for s in statuses]}。配線/電源を確認。"

        driver = VelocityDriver(chain, KiwiKinematics())
        vx = vy = omega = 0.0
        last = time.monotonic()
        while True:
            ch = stdscr.getch()
            now = time.monotonic()
            dt = now - last
            last = now

            if ch in (ord("x"), 27):  # x または ESC で終了
                break
            if ch != -1:
                vx, vy, omega = apply_key(
                    ch, vx, vy, omega,
                    args.v_step, args.omega_step, args.v_max, args.omega_max,
                )
                driver.set_velocity(vx, vy, omega)

            cur = driver.update(dt)

            stdscr.erase()
            stdscr.addstr(0, 0, "krilly テレオペ")
            stdscr.addstr(1, 0, _HELP)
            stdscr.addstr(3, 0, f"目標  vx={vx:+.2f} vy={vy:+.2f} omega={omega:+.2f}")
            stdscr.addstr(4, 0, f"現在  vx={cur[0]:+.2f} vy={cur[1]:+.2f} omega={cur[2]:+.2f}")
            stdscr.refresh()

        # 終了: 減速して停止
        driver.stop()
        deadline = time.monotonic() + 1.5
        while not driver.at_target(1e-3) and time.monotonic() < deadline:
            now = time.monotonic()
            driver.update(now - last)
            last = now
            time.sleep(args.dt)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="キーボード テレオペ")
    p.add_argument("--devices", type=int, default=3, help="連結台数")
    p.add_argument("--v-step", type=float, default=0.02, help="並進の 1 押下あたり増分 [m/s]")
    p.add_argument("--omega-step", type=float, default=0.2, help="旋回の 1 押下あたり増分 [rad/s]")
    p.add_argument("--v-max", type=float, default=0.3, help="並進速度の上限 [m/s]")
    p.add_argument("--omega-max", type=float, default=3.0, help="旋回角速度の上限 [rad/s]")
    p.add_argument("--dt", type=float, default=0.02, help="制御周期 [s]")
    args = p.parse_args()

    err = curses.wrapper(_run, args)
    if err:
        print(err)


if __name__ == "__main__":
    main()
