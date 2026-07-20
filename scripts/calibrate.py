#!/usr/bin/env python3
"""運動学の実機キャリブレーション (issue #11).

既知の指令量だけ L6470 の Move (位置指令) で動かし、実測値を入力すると
補正後の config 値を算出する。

- ``--straight D`` : 前進を D[m] 指令 → 実測距離から wheel_diameter_m を補正。
- ``--rotate T``   : その場回転を T[回転] 指令 → 実測回転から center_to_wheel_m(L) を補正。

補正手順: まず --straight で車輪径を確定してから --rotate で L を確定する
(回転は車輪径が正しい前提で L に誤差を帰属させるため)。

例:
    python -m scripts.calibrate --straight 0.5
    python -m scripts.calibrate --rotate 2

算出した値は自動では書き込まない。表示された値で config/robot.yaml を更新すること。
座標系は docs/coordinate-frames.md (+x前 / +y左 / +omega=CCW)。
"""

from __future__ import annotations

import argparse
import math
import time

from krilly.config import load_robot_config
from krilly.hal.l6470 import FWD, REV, L6470Profile
from krilly.hal.l6470_chain import L6470Chain
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.logging_config import get_logger, setup_logging

log = get_logger("krilly.calibrate")


# --- 純粋な補正計算 (テスト可能) ------------------------------------------
def corrected_wheel_diameter(current_d: float, commanded_m: float, measured_m: float) -> float:
    """実測距離から補正後の車輪径を返す。measured/commanded に比例。"""
    return current_d * measured_m / commanded_m


def corrected_center_to_wheel(current_L: float, commanded_turns: float, measured_turns: float) -> float:
    """実測回転から補正後の L を返す。commanded/measured に比例 (不足回転→L大)。"""
    return current_L * commanded_turns / measured_turns


def wheel_moves_for_body(
    kin: KiwiKinematics, dx: float, dy: float, dphi: float
) -> list[tuple[int, int]]:
    """ボディ変位 (dx, dy, dphi) を各輪の (向き, マイクロステップ数) に変換する。"""
    dists = kin.body_to_wheels(dx, dy, dphi)  # 各輪の転がり距離 [m] (符号付き)
    moves = []
    for s in dists:
        direction = FWD if s >= 0 else REV
        moves.append((direction, round(kin.distance_to_microsteps(abs(s)))))
    return moves


# --- 実機動作 --------------------------------------------------------------
def _execute_moves(chain: L6470Chain, moves: list[tuple[int, int]], profile: L6470Profile,
                   microstep: int) -> None:
    """各輪に Move を送り、完了予測時間だけ待つ。"""
    for i, (direction, micro) in enumerate(moves):
        chain.move(i, direction, micro)
    microstep_speed = profile.max_speed_steps_s * microstep  # マイクロステップ/s
    max_micro = max((m for _, m in moves), default=0)
    wait = max_micro / microstep_speed + 2.0  # 加減速+整定のマージン
    log.info("移動中… 約 %.1f 秒", wait)
    time.sleep(wait)


def main() -> None:
    p = argparse.ArgumentParser(description="運動学キャリブレーション")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--straight", type=float, metavar="D_M", help="前進距離 [m] を指令")
    g.add_argument("--rotate", type=float, metavar="TURNS", help="その場回転 [回転] を指令")
    p.add_argument("--devices", type=int, default=3, help="連結台数")
    args = p.parse_args()

    setup_logging()
    cfg = load_robot_config()
    kin = KiwiKinematics(cfg)
    profile = L6470Profile()

    with L6470Chain(num_devices=args.devices) as chain:
        statuses = chain.configure_all(profile)
        if any(s in (0x0000, 0xFFFF) for s in statuses):
            log.error("SPI 応答異常 STATUS=%s。配線/電源を確認。中止。",
                      [f"0x{s:04X}" for s in statuses])
            return

        if args.straight is not None:
            d = args.straight
            log.info("前進 %.3f m を指令します", d)
            _execute_moves(chain, wheel_moves_for_body(kin, d, 0.0, 0.0), profile, cfg.microstep)
            measured = float(input(f"実際の移動距離を測って [m] で入力: "))
            new_d = corrected_wheel_diameter(cfg.wheel_diameter_m, d, measured)
            log.info("現在 wheel_diameter_m=%.5f → 補正値 %.5f", cfg.wheel_diameter_m, new_d)
            log.info("config/robot.yaml の wheel_diameter_m を %.5f に更新してください", new_d)
        else:
            turns = args.rotate
            log.info("その場回転 %.2f 回転 (%.0f度) を指令します", turns, turns * 360)
            _execute_moves(chain, wheel_moves_for_body(kin, 0.0, 0.0, turns * 2 * math.pi),
                           profile, cfg.microstep)
            measured_deg = float(input("実際の回転量を測って [度] で入力: "))
            new_L = corrected_center_to_wheel(cfg.center_to_wheel_m, turns * 360, measured_deg)
            log.info("現在 center_to_wheel_m=%.5f → 補正値 %.5f", cfg.center_to_wheel_m, new_L)
            log.info("config/robot.yaml の center_to_wheel_m を %.5f に更新してください", new_L)


if __name__ == "__main__":
    main()
