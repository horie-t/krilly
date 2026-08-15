#!/usr/bin/env python3
"""1セル前進・90°ターン (位置連動) の実機確認スクリプト (issue #17).

:class:`CellMotion` で「1 セル前進」「90°ターン」を **自己位置推定の残量で
終端判定しながら** 実行する。各プリミティブごとに、理想格子上の基準姿勢と
推定姿勢・残差を表示するので、実機の停止位置とセル中心のズレを見比べて
ゲイン (CellMotionConfig) を詰められる。

``--camera-yaw`` を付けると、動作の**前後でカメラのフレームから軸角を実測**し
(:mod:`krilly.perception.axis_yaw`)、ジャイロ由来の推定方位と突き合わせる。
シーケンスの総回転が 90° の倍数なら、カメラ側の差分がそのまま
**理想からの行き過ぎ量 (+ = CCW 側)** になるので、方位の真値として使える。

例:
    python -m scripts.cell_move_demo                       # 既定 F,L,F,R (L字)
    python -m scripts.cell_move_demo --seq F,F,L,F         # 任意のシーケンス
    python -m scripts.cell_move_demo --seq U --no-imu      # 180°ターンをオドメトリのみで
    python -m scripts.cell_move_demo --v 0.08 --omega 1.0  # ゆっくり
    python -m scripts.cell_move_demo --seq L --camera-yaw   # 90°ターンの方位をカメラで実測
    python -m scripts.cell_move_demo --seq F4               # 4セルを 1 動作で (速度調整用)
    python -m scripts.cell_move_demo --v 0.24 --accel 0.8 --decel 0.8 --max-speed 600

シーケンスのトークン: F=前進 / B=後退 / L=左90° / R=右90° / U=180°。
数字を付けると 1 動作でその回数ぶん動く (``F4``=4セルを 1 動作、``L2``=180°)。
``F,F,F,F`` (4 動作) と ``F4`` (1 動作) の比較が、距離誤差がスケール由来か
動作あたりの固定オーバーラン由来かの切り分けになる (#21)。

配線: L6470×3 デイジーチェーン + BNO055 (I2C 0x28)。座標系 +x前 / +y左 / +omega=CCW。
"""

from __future__ import annotations

import argparse
import contextlib
import math
import re
import time
from dataclasses import dataclass

from krilly.hal.imu import Bno055Imu
from krilly.hal.l6470_chain import L6470Chain
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.localization.estimator import DeadReckoning
from krilly.logging_config import get_logger, setup_logging
from krilly.motion.cell_motion import CellMotion
from krilly.motion.emergency_stop import emergency_stop
from krilly.motion.tuning import add_tuning_args, build_tuning, check_limits, describe_faults
from krilly.motion.velocity_driver import VelocityDriver
from krilly.perception.axis_yaw import (
    AxisYaw,
    calibrated_axis_yaw_config,
    fold_deg,
    median_axis_yaw,
    yaw_delta_rad,
)
from krilly.perception.cell_pose import CellOffset, cell_offset
from krilly.perception.wall_detect import WallDetector, calibrated_config

log = get_logger("krilly.cell_move_demo")

TOKENS = {
    "F": ("%dセル前進", lambda m, n: m.start_forward_cells(n)),
    "B": ("%dセル後退", lambda m, n: m.start_forward_cells(-n)),
    "L": ("左%d°", lambda m, n: m.start_turn_left(n)),
    "R": ("右%d°", lambda m, n: m.start_turn_right(n)),
}
# U は L2 (180°) の別名。過去のシーケンス表記との互換のため残す。
ALIASES = {"U": ("L", 2)}


@dataclass(frozen=True)
class Move:
    """1 動作 (トークンと回数)。``F4`` なら 4 セルを **1 動作で** 進む。"""

    token: str
    count: int = 1

    @property
    def label(self) -> str:
        fmt = TOKENS[self.token][0]
        # 回転は角度で書いた方が読みやすい (L2 -> 左180°)
        return fmt % (self.count * 90 if self.token in ("L", "R") else self.count)

    def start(self, motion) -> None:
        TOKENS[self.token][1](motion, self.count)


ALONG, CROSS = "前後", "左右"

# 基準姿勢から見た車体軸の向き (理想格子上なので 90° の倍数) -> (軸名, 符号)
GRID_AXIS = {(1, 0): (ALONG, 1), (-1, 0): (ALONG, -1),
             (0, 1): (CROSS, 1), (0, -1): (CROSS, -1)}


def world_components(
    forward_m: float | None, left_m: float | None, phi: float, base_phi: float = 0.0
) -> dict[str, float]:
    """車体フレームのずれ (m) を ``base_phi`` の車体軸へ直す。

    軸名は ``base_phi`` (ふつうはシーケンス開始時の姿勢) から見た **前後 / 左右**。
    このスクリプトの推定器は φ=0 で始まる (基準は迷路の北ではなく**置いたときの
    機体の向き**) ので、迷路の東西南北で呼ぶと意味が逆になる。

    ``phi`` は**理想格子上の**方位なので ``base_phi`` との差は必ず 90° の倍数で、
    車体軸はそのまま基準軸に対応する (成分が混ざらない)。測れなかった軸はキーごと
    落とすので、動作の前後で共通する軸だけを引き算すれば「測ってもいない方向」の
    誤差をでっち上げずに済む。
    """
    rel = phi - base_phi
    c, s = round(math.cos(rel)), round(math.sin(rel))
    out: dict[str, float] = {}
    for value, vec in ((forward_m, (c, s)), (left_m, (-s, c))):
        if value is None:
            continue
        name, sign = GRID_AXIS[vec]
        out[name] = value * sign
    return out


def wrapped_deg(rad: float) -> float:
    """表示用: 角度を (-180, 180] deg に正規化する。

    推定φは積算値なので 360° を超えて伸びる (2周すれば +720°)。基準φは
    (-180, 180] に正規化してあるため、並べて読むには表示側も揃える必要がある。
    """
    return math.degrees((rad + math.pi) % (2 * math.pi) - math.pi)


def parse_seq(text: str) -> list[Move]:
    """"F,L,F" / "FLF" / "F4,L" いずれの書き方も受け付けて動作列にする。"""
    cleaned = "".join(text.upper().split()).replace(",", "")
    if not re.fullmatch(r"(?:[A-Z]\d*)+", cleaned):
        raise SystemExit(f"シーケンスの書式が不正: {text!r} (例: F,L,F / FLF / F4,L)")
    moves: list[Move] = []
    for token, digits in re.findall(r"([A-Z])(\d*)", cleaned):
        token, factor = ALIASES.get(token, (token, 1))
        if token not in TOKENS:
            raise SystemExit(
                f"未知のトークン {token!r} "
                f"(使えるのは {'/'.join(TOKENS)}{'/' + '/'.join(ALIASES)})"
            )
        count = (int(digits) if digits else 1) * factor
        if count < 1:
            raise SystemExit(f"回数は 1 以上にすること: {token}{digits}")
        moves.append(Move(token, count))
    if not moves:
        raise SystemExit(f"シーケンスが空: {text!r}")
    return moves


def main() -> None:
    p = argparse.ArgumentParser(description="1セル前進・90°ターンの位置連動デモ")
    p.add_argument("--devices", type=int, default=3, help="連結台数")
    p.add_argument("--bus", type=int, default=0, help="SPI バス (既定 0)")
    p.add_argument("--device", type=int, default=0, help="SPI デバイス/CE (既定 0)")
    p.add_argument("--seq", default="F,L,F,R", help="動作シーケンス (F/B/L/R/U、数字で回数)")
    add_tuning_args(p, omega=1.5)
    p.add_argument("--dt", type=float, default=0.02, help="制御周期 [s]")
    p.add_argument("--pause", type=float, default=0.5, help="プリミティブ間の停止秒数")
    p.add_argument("--no-imu", action="store_true", help="ジャイロ融合せずオドメトリのみ")
    p.add_argument("--gyro-sign", type=float, default=1.0, help="ジャイロz符号 (+1/-1)")
    p.add_argument("--gyro-scale", type=float, default=None,
                   help="ジャイロzスケール補正 (既定 robot.yaml の gyro_scale_z)")
    p.add_argument("--timeout", type=float, default=10.0, help="1プリミティブの上限秒数")
    p.add_argument("--camera-yaw", action="store_true",
                   help="動作前後の方位をカメラで実測してジャイロ推定と突き合わせる")
    p.add_argument("--camera-pose", action="store_true",
                   help="動作前後のセル内位置をカメラで実測し、理想量とのズレを出す")
    p.add_argument("--yaw-samples", type=int, default=5, help="カメラ実測のフレーム数 (中央値)")
    p.add_argument("--save-frames", default=None,
                   help="カメラ実測フレームの保存先プレフィクス 例: /tmp/yaw")
    args = p.parse_args()

    setup_logging()
    seq = parse_seq(args.seq)
    kin = KiwiKinematics()
    tuning = build_tuning(args)
    gyro_scale = args.gyro_scale if args.gyro_scale is not None else kin.cfg.gyro_scale_z
    log.info("チューニング: %s", tuning.describe())
    for warning in check_limits(tuning, kin):
        log.warning("%s", warning)

    with contextlib.ExitStack() as stack:
        chain = stack.enter_context(
            L6470Chain(num_devices=args.devices, bus=args.bus, device=args.device)
        )
        stack.enter_context(
            emergency_stop(chain, on_stop=lambda sig: log.warning(
                "シグナル %s を受信。モーターを解放した。", sig))
        )
        statuses = chain.configure_all(tuning.profile)
        if any(s in (0x0000, 0xFFFF) for s in statuses):
            log.error("SPI 応答異常 (STATUS=%s)。配線/電源を確認。中止。",
                      [f"0x{s:04X}" for s in statuses])
            return

        imu = None
        bias_z = 0.0
        if not args.no_imu:
            imu = stack.enter_context(Bno055Imu())
            imu.begin()
            log.info("静止のままジャイロバイアス計測中…")
            bias_z = imu.measure_gyro_bias()[2]
            log.info("ジャイロバイアス z=%.3f deg/s / スケール %.4f", bias_z, gyro_scale)

        camera = None
        if args.camera_yaw or args.camera_pose:
            from krilly.hal.camera import Camera   # 遅延 import (実機専用の依存)

            camera = stack.enter_context(Camera())
        yaw_cfg = calibrated_axis_yaw_config()
        detector = WallDetector(calibrated_config())

        driver = VelocityDriver(chain, kin, limits=tuning.limits)
        est = DeadReckoning(kin)
        motion = CellMotion(driver, est, config=tuning.motion)

        def gyro_rate() -> float | None:
            """バイアス減算・符号・スケール補正を掛けた角速度 [rad/s]。"""
            if imu is None:
                return None
            return math.radians(imu.gyro[2] - bias_z) * args.gyro_sign * gyro_scale

        def measure_yaw(tag: str) -> AxisYaw | None:
            """カメラで軸角を実測する (複数フレームの中央値)。"""
            if camera is None:
                return None
            frames = [camera.capture() for _ in range(args.yaw_samples)]
            result = median_axis_yaw(frames, yaw_cfg)
            if args.save_frames:
                import cv2

                from krilly.perception.axis_yaw import annotate

                cv2.imwrite(f"{args.save_frames}_{tag}.png", frames[-1])
                cv2.imwrite(f"{args.save_frames}_{tag}_yaw.png", annotate(frames[-1], yaw_cfg))
            if result is None:
                log.warning("カメラ実測 (%s): 赤い壁エッジが足りず測定不能", tag)
            else:
                log.info("カメラ実測 (%s): 軸角 %+.3f° (線分 %d 本, 総長 %.0fpx)",
                         tag, result.angle_deg, result.segments, result.total_length_px)
            return result

        def measure_pose(tag: str) -> tuple[CellOffset, float] | None:
            """カメラでセル内の位置ずれを実測する (そのときの理想方位と一緒に返す)。"""
            if camera is None or not args.camera_pose:
                return None
            off = cell_offset(camera.capture(), detector)
            if not off.measured:
                log.warning("カメラ実測 (%s): 位置測定に足る赤帯が無い "
                            "(壁の無いセルでは測れない)", tag)
                return None
            log.info("カメラ実測 (%s): セル内のずれ 前後=%s 左右=%s (根拠の壁 前後%d枚/左右%d枚)",
                     tag,
                     "測定不能" if off.forward_m is None else "%+.1fmm" % (off.forward_m * 1e3),
                     "測定不能" if off.left_m is None else "%+.1fmm" % (off.left_m * 1e3),
                     off.walls_y, off.walls_x)
            return (off, motion.reference[2])

        def run_primitive(label: str, timeout: float) -> None:
            """完了 (または timeout) までループを回す。update は純計算なので寝るのはここ。"""
            t0 = time.monotonic()
            last = t0
            while True:
                time.sleep(args.dt)
                now = time.monotonic()
                dt = now - last
                last = now
                if motion.update(dt, gyro_rate=gyro_rate()):
                    break
                if now - t0 > timeout:
                    log.warning("%s: %.1fs で終わらず打ち切り (残量 %.4f)",
                                label, timeout, motion.remaining)
                    motion.abort()
                    break
            faults = describe_faults(chain.get_status_all(), ignore=("UVLO",))
            if faults:
                log.warning("%s: L6470 フォールト %s "
                            "(STEP_LOSS/OCD なら速度・加速度に対してトルクが不足)",
                            label, faults)
            x, y, phi = est.pose
            rx, ry, rphi = motion.reference
            along, cross, dphi = motion.residual()
            log.info(
                "%s 完了 %.2fs: 推定 X=%.4f Y=%.4f φ=%+.2f° / 基準 X=%.4f Y=%.4f φ=%+.2f°"
                " / 残差 前後=%+.4fm 左右=%+.4fm 方位=%+.2f°",
                label, time.monotonic() - t0, x, y, wrapped_deg(phi),
                rx, ry, wrapped_deg(rphi), along, cross, math.degrees(dphi),
            )

        def coast(seconds: float) -> None:
            """停止指令のまま update を回して減速しきる (推定も継続)。"""
            last = time.monotonic()
            deadline = last + seconds
            while time.monotonic() < deadline:
                time.sleep(args.dt)
                now = time.monotonic(); dt = now - last; last = now
                motion.update(dt, gyro_rate=gyro_rate())

        log.info("シーケンス %s を実行 (1セル=%.3fm)",
                 " ".join(m.label for m in seq), motion.cell_pitch_m)
        chain.get_status_all()   # 電源投入時の UVLO ラッチを捨てて以降の差分を見る
        yaw_before = measure_yaw("before")
        pose_before = measure_pose("before")
        phi_before = est.phi
        for i, move in enumerate(seq, 1):
            log.info("[%d/%d] %s 開始", i, len(seq), move.label)
            move.start(motion)
            # 複数セル・複数回転を 1 動作で回すぶんだけ打ち切り時間も延ばす
            run_primitive(f"[{i}/{len(seq)}] {move.label}", args.timeout * move.count)
            coast(args.pause)
        yaw_after = measure_yaw("after")
        pose_after = measure_pose("after")

        along, cross, dphi = motion.residual()
        log.info("完了。理想との総残差: 前後=%+.4fm 左右=%+.4fm 方位=%+.2f°",
                 along, cross, math.degrees(dphi))
        if pose_before is not None and pose_after is not None:
            # セル内のずれを世界成分に直して引き算する。理想格子上の移動量は
            # 定義どおり正確なので、差分がそのまま「理想との差 = 実機の誤差」になる。
            # 前後で共通して測れた軸だけを比べる (片方しか無い軸は根拠が無い)。
            base_phi = pose_before[1]
            before = world_components(pose_before[0].forward_m, pose_before[0].left_m,
                                      base_phi, base_phi)
            after = world_components(pose_after[0].forward_m, pose_after[0].left_m,
                                     pose_after[1], base_phi)
            common = [k for k in (ALONG, CROSS) if k in before and k in after]
            if common:
                log.info("理想格子からのずれ (カメラ実測の差分、開始時の機体軸で): %s"
                         "  (前後 + = 行き過ぎ / 左右 + = 左へずれた)",
                         " ".join("%s %+.1fmm" % (k, (after[k] - before[k]) * 1e3)
                                  for k in common))
            missing = [k for k in (ALONG, CROSS) if k not in common]
            if missing:
                log.info("  %s 方向は動作の前後どちらかで測れなかったので比較しない",
                         "/".join(missing))
        if yaw_before is not None and yaw_after is not None:
            # 軸は 90° 周期なので、ジャイロ側の総回転も同じ折り返しで比べる
            gyro_delta = fold_deg(math.degrees(est.phi - phi_before))
            cam_delta = math.degrees(yaw_delta_rad(yaw_before, yaw_after))
            log.info(
                "方位の実測 (90°の倍数を除いた差分): カメラ %+.3f° / ジャイロ推定 %+.3f°"
                " / 差 %+.3f°  (+ = CCW 側へ行き過ぎ)",
                cam_delta, gyro_delta, cam_delta - gyro_delta,
            )
        # with 終了で hard_hiz により出力を解放


if __name__ == "__main__":
    main()
