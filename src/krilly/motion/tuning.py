"""走行チューニングのつまみを 1 か所に集約する (issue #21)。

速度・加速度・KVAL トルクは **どのスクリプトで測っても同じ値でなければ意味がない**。
時間定数 (:class:`krilly.app.run_manager.RunManager` の見積もり) は速度で変わり、
脱調はトルクで変わるので、``cell_move_demo`` で詰めた値がそのまま ``search_run`` /
``speed_run`` に効かないと調整が噛み合わない。CLI 引数の定義と、そこから 3 つの
設定オブジェクト (:class:`L6470Profile` / :class:`RampLimits` /
:class:`CellMotionConfig`) を組み立てる処理をここに集約する。

3 つの設定は独立ではなく、**下の層が上の層の指令を追従できる**必要がある:

- :class:`CellMotionConfig` の減速度は :class:`RampLimits` のランプ上限以下に
  丸められる (:class:`~krilly.motion.cell_motion.CellMotion` 側で clamp)。
  つまり減速を強めるにはランプ上限も上げる必要がある。
- :class:`RampLimits` のランプは L6470 の ACC/DEC register より緩くなければ、
  実際の加減速はドライバ側の値に律速される (ソフトのランプが空振りする)。
- 指令速度のピークが L6470 の MAX_SPEED register を超えると、そこで頭打ちになり
  3 輪の速度比が崩れる (= 経路が曲がる)。

:func:`check_limits` がこの 3 つを実際の運動学で検算して警告文を返すので、
速度を上げるときは必ず通すこと。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from krilly.hal.l6470 import L6470Profile, fault_flags
from krilly.kinematics.kiwi import KiwiKinematics
from krilly.motion.corner import (
    blend_distance_m,
    corridor_clearance_m,
    min_post_clearance_m,
)
from krilly.motion.cell_motion import CellMotionConfig
from krilly.motion.velocity_driver import RampLimits


@dataclass(frozen=True)
class TuningProfile:
    """1 回の走行で使う速度・加速度・トルクの一式。"""

    profile: L6470Profile        # L6470 のレジスタ設定 (トルク・上限)
    limits: RampLimits           # ボディ速度のランプ上限
    motion: CellMotionConfig     # 1 セル動作のプロファイル

    def describe(self) -> str:
        """ログ 1 行用の要約 (走行のたびに記録して実測値と結び付けるため)。"""
        return (
            "v=%.3fm/s omega=%.2frad/s accel=%.2f/%.1f decel=%.2f/%.1f dwell=%.2fs "
            "KVAL run=0x%02X hold=0x%02X MAX_SPEED=%.0fstep/s"
            % (
                self.motion.v_max, self.motion.omega_max,
                self.limits.max_linear_accel_mps2, self.limits.max_angular_accel_radps2,
                self.motion.decel_mps2, self.motion.angular_decel_radps2,
                self.motion.settle_dwell_s,
                self.profile.kval_run, self.profile.kval_hold,
                self.profile.max_speed_steps_s,
            )
        )


def add_tuning_args(parser, *, v: float | None = None, omega: float | None = None) -> None:
    """速度・加速度・トルクの CLI 引数を ``parser`` に追加する。

    既定値は各設定クラスの既定 (= #21 で決めた採用点)。``v`` / ``omega`` を渡すと
    そのスクリプトだけ違う既定にできる。
    """
    d_limits = RampLimits()
    d_motion = CellMotionConfig()
    d_profile = L6470Profile()
    parser.add_argument("--v", type=float, default=v if v is not None else d_motion.v_max,
                        help="前進の最大速度 [m/s]")
    parser.add_argument("--omega", type=float,
                        default=omega if omega is not None else d_motion.omega_max,
                        help="旋回の最大角速度 [rad/s]")
    parser.add_argument("--accel", type=float, default=d_limits.max_linear_accel_mps2,
                        help="並進のランプ上限 [m/s^2]")
    parser.add_argument("--angular-accel", type=float,
                        default=d_limits.max_angular_accel_radps2,
                        help="旋回のランプ上限 [rad/s^2]")
    parser.add_argument("--decel", type=float, default=d_motion.decel_mps2,
                        help="前進の減速エンベロープ [m/s^2] (--accel 以下に丸められる)")
    parser.add_argument("--angular-decel", type=float,
                        default=d_motion.angular_decel_radps2,
                        help="旋回の減速エンベロープ [rad/s^2] (--angular-accel 以下に丸められる)")
    parser.add_argument("--settle-dwell", type=float, default=d_motion.settle_dwell_s,
                        help="動作を止めてから残差を判定するまでの待ち [s]")
    parser.add_argument("--retry-angle-tol", type=float,
                        default=math.degrees(d_motion.retry_angle_tol_rad),
                        help="旋回をやり直す残角のしきい値 [deg] "
                             "(ガタの帯より内側は追いかけても詰まらない)")
    parser.add_argument("--kval", type=lambda s: int(s, 0), default=d_profile.kval_run,
                        help="L6470 の KVAL_RUN/ACC/DEC (0x00-0xFF, 0x80=Vs の 50%%)")
    parser.add_argument("--kval-hold", type=lambda s: int(s, 0),
                        default=d_profile.kval_hold, help="L6470 の KVAL_HOLD")
    parser.add_argument("--max-speed", type=float, default=d_profile.max_speed_steps_s,
                        help="L6470 の MAX_SPEED [フルステップ/s]")
    parser.add_argument("--driver-accel", type=float, default=d_profile.acc_steps_s2,
                        help="L6470 の ACC/DEC [フルステップ/s^2]")


def build_tuning(args, motion: CellMotionConfig | None = None) -> TuningProfile:
    """:func:`add_tuning_args` で解析した ``args`` から設定一式を組み立てる。

    ``motion`` を渡すと、速度・減速度以外のゲイン (P ゲインや許容値) をその値から
    引き継ぐ。
    """
    base = motion or CellMotionConfig()
    return TuningProfile(
        profile=L6470Profile(
            max_speed_steps_s=args.max_speed,
            acc_steps_s2=args.driver_accel,
            dec_steps_s2=args.driver_accel,
            kval_hold=args.kval_hold,
            kval_run=args.kval,
            kval_acc=args.kval,
            kval_dec=args.kval,
        ),
        limits=RampLimits(
            max_linear_accel_mps2=args.accel,
            max_angular_accel_radps2=args.angular_accel,
        ),
        motion=replace(
            base,
            v_max=args.v,
            omega_max=args.omega,
            decel_mps2=args.decel,
            angular_decel_radps2=args.angular_decel,
            settle_dwell_s=args.settle_dwell,
            retry_angle_tol_rad=math.radians(args.retry_angle_tol),
        ),
    )


def _wheel_rows(kin: KiwiKinematics) -> list[tuple[float, float, float]]:
    """各輪の (vx, vy, omega) に対する係数行 (逆運動学の行列を公開 API から復元)。"""
    cx = kin.body_to_wheels(1.0, 0.0, 0.0)
    cy = kin.body_to_wheels(0.0, 1.0, 0.0)
    cw = kin.body_to_wheels(0.0, 0.0, 1.0)
    return list(zip(cx, cy, cw))


def peak_wheel_value(
    kin: KiwiKinematics, vx: float, vy: float, omega: float
) -> float:
    """ボディ指令の絶対値上限 (vx, vy, omega) に対する車輪値のピーク。

    運動学は線形なので、符号の最悪の組み合わせは各項の絶対値の和になる。
    速度を入れれば車輪速度 [m/s]、加速度を入れれば車輪加速度 [m/s^2] が出る。
    """
    return max(
        abs(a) * abs(vx) + abs(b) * abs(vy) + abs(c) * abs(omega)
        for a, b, c in _wheel_rows(kin)
    )


def check_limits(tuning: TuningProfile, kin: KiwiKinematics | None = None,
                 maze_pitch_m: float = 0.180) -> list[str]:
    """設定の整合性を実際の運動学で検算し、警告文のリストを返す (空なら健全)。

    見るのは 4 点:

    1. 指令速度のピークが L6470 の MAX_SPEED を超えないか (超えると頭打ちになり、
       3 輪の速度比が崩れて経路が曲がる)。
    2. ソフトのランプが L6470 の ACC/DEC を超えないか (超えるとドライバ側が律速し、
       減速が指令より遅れて停止位置が行き過ぎる)。
    3. 減速エンベロープがランプ上限を超えていないか (:class:`CellMotion` 側で
       丸められるので危険ではないが、指定した値が効かない)。
    4. 止まらない方向転換 (#80) のコーナーが幾何に収まるか。
    """
    kin = kin or KiwiKinematics()
    m = m_cfg = tuning.motion
    warnings: list[str] = []

    # 1. 速度のピーク: 前進 / **横移動** / 旋回のうち最大。
    #    横移動 (#76) は主軸が vy になるので vx と vy が入れ替わる。純 vy の係数は
    #    純 vx より大きい (スポーク角が [0,120,240] 近辺なので max|cos| > max|sin|) ため、
    #    **同じ体速度でも横の方が約 4% 高い輪速を要求する**。しかも飽和するのは W0
    #    (スポーク角がほぼ 0° = 駆動方向が真横の車輪) なので、見落とすと横移動の距離だけが
    #    縮んで前後へ流れる。オドメトリは指令から積分するので気づけない。
    forward = peak_wheel_value(kin, m.v_max, m.v_cross_max, m.omega_hold_max)
    lateral = peak_wheel_value(kin, m.v_cross_max, m.v_max, m.omega_hold_max)
    turning = peak_wheel_value(kin, m.v_hold_max, m.v_hold_max, m.omega_max)
    peak_hz = kin.wheel_speed_to_step_hz(max(forward, lateral, turning))
    if peak_hz > tuning.profile.max_speed_steps_s:
        worst = max((forward, "前進"), (lateral, "横移動"), (turning, "旋回"))[1]
        warnings.append(
            "指令速度のピーク %.0f step/s (%s) が MAX_SPEED %.0f step/s を超える。"
            "--max-speed を上げること (超えると 3 輪の速度比が崩れて経路が曲がる)。"
            % (peak_hz, worst, tuning.profile.max_speed_steps_s)
        )

    # 2. 加速度のピーク: 並進のランプと旋回のランプの大きい方。3 軸が同時に最大レートで
    #    変化する組み合わせは 1 tick の過渡でしか起きないので、ここでは見ない。
    accel_hz = kin.wheel_speed_to_step_hz(max(
        peak_wheel_value(kin, tuning.limits.max_linear_accel_mps2,
                         tuning.limits.max_linear_accel_mps2, 0.0),
        peak_wheel_value(kin, 0.0, 0.0, tuning.limits.max_angular_accel_radps2),
    ))
    if accel_hz > tuning.profile.acc_steps_s2:
        warnings.append(
            "ソフトのランプ %.0f step/s^2 が L6470 の ACC/DEC %.0f step/s^2 を超える。"
            "--driver-accel を上げること (超えるとドライバ側が律速し停止が行き過ぎる)。"
            % (accel_hz, tuning.profile.acc_steps_s2)
        )

    # 3. 減速エンベロープがランプ上限を超えていないか (CellMotion 側で丸められる)
    if m.decel_mps2 > tuning.limits.max_linear_accel_mps2:
        warnings.append(
            "--decel %.2f は --accel %.2f に丸められる (減速を強めるなら両方上げる)。"
            % (m.decel_mps2, tuning.limits.max_linear_accel_mps2)
        )
    if m.angular_decel_radps2 > tuning.limits.max_angular_accel_radps2:
        warnings.append(
            "--angular-decel %.1f は --angular-accel %.1f に丸められる。"
            % (m.angular_decel_radps2, tuning.limits.max_angular_accel_radps2)
        )
    # 4. 止まらない方向転換 (#80): コーナーの膨らみが幾何に収まるか。
    #    ブレンド距離は v^2/(2a) なので、**速度を上げるか減速度を下げると二乗で伸びる**。
    #    半セルを超えると機体が前のセルの廊下に居るうちに横へ動き出すことになり、
    #    「膨らむ先は開いている象限」という前提そのものが崩れる。
    blend = blend_distance_m(m_cfg.v_max, min(m_cfg.decel_mps2,
                                              tuning.limits.max_linear_accel_mps2))
    half_cell = maze_pitch_m / 2.0
    if blend > half_cell:
        warnings.append(
            "コーナーのブレンド距離 %.0fmm が半セル %.0fmm を超える。前のセルの廊下に"
            "居るうちに横へ動き出す (--v を下げるか --decel を上げること)。"
            % (blend * 1000, half_cell * 1000)
        )
    elif min_post_clearance_m(blend, maze_pitch_m) < corridor_clearance_m(maze_pitch_m):
        warnings.append(
            "コーナーの柱との余裕 %.0fmm が廊下の余裕 %.0fmm を下回る "
            "(ブレンド距離 %.0fmm)。丸め方が通行の制約になる。"
            % (min_post_clearance_m(blend, maze_pitch_m) * 1000,
               corridor_clearance_m(maze_pitch_m) * 1000, blend * 1000)
        )

    return warnings


def describe_faults(statuses: list[int], ignore: tuple[str, ...] = ()) -> str | None:
    """デバイスごとの STATUS からフォールトの要約を作る (無ければ None)。

    ``GetStatus`` は読み出しでフラグをクリアするので、動作のたびに呼べば
    「その動作の間に起きたか」が分かる。速度・トルクを上げるときの安全弁。
    """
    parts = []
    for index, status in enumerate(statuses):
        flags = sorted(fault_flags(status) - set(ignore))
        if flags:
            parts.append("M%d:%s" % (index, "/".join(flags)))
    return " ".join(parts) if parts else None
