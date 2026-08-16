"""1セル前進・90°ターンの位置連動 (issue #17)。

「指定時間だけ速度を出す」オープンループ (M2 #9 の :class:`VelocityDriver` 単体)
では、スリップ・加減速の丸め・ジャイロ融合の差分が毎回残り、セルを跨ぐたびに
誤差が積み上がる。本モジュールは **自己位置推定 (#12-#14) を閉ループに入れて**
1 セル前進・90°ターンを終端条件で打ち切る動作プリミティブを提供する。

要点:

- **理想格子上の基準姿勢 (x_ref, y_ref, phi_ref) を内部に持つ**。プリミティブの
  開始時に「終わった時にあるべき姿勢」を基準に加算し、推定姿勢をそこへ寄せる。
  各回の残差は次のプリミティブの残量に含まれるため、**誤差がセル間で累積しない**
  (推定姿勢から相対的に 180mm 進む実装だと残差が積み上がる)。
- **主軸は台形プロファイル**: 残量 s に対し ``v = min(v_max, sqrt(2*a*s))``。
  減速度 a は :class:`VelocityDriver` のランプ上限以下に丸めるので、指令は必ず
  ランプで追従できる (追従できない指令を出すとオーバーシュートになる)。
- **主軸以外は P 制御で保持**: 前進中は横ずれ (vy) と方位ずれ (omega)、旋回中は
  位置ずれ (vx, vy) を basis 姿勢へ引き戻す。ホロノミックの利点をそのまま使う。
- **終了判定は残量**: 残量が許容値以内になったら 0 指令へランプダウン (整定) し、
  整定後になお残っていれば低速でやり直す (既定 1 回)。

``update(dt)`` は :class:`VelocityDriver` と同様に **純粋な計算のみ** (time.sleep
を含まない) で、推定器の積分もこの中で行う。ハードウェアは注入された driver
経由でのみ触るため、フェイク chain でユニットテストできる。

座標系は docs/coordinate-frames.md (+x 前 / +y 左 / +omega=CCW)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from krilly.config import MazeConfig, load_maze_config
from krilly.localization.estimator import DeadReckoning
from krilly.motion.velocity_driver import VelocityDriver


def _wrap_angle(a: float) -> float:
    """角度を [-π, π) に正規化する (最短回転方向で角度差を扱うため)。"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _clamp(value: float, limit: float) -> float:
    """``value`` を [-limit, +limit] に収める (limit >= 0)。"""
    return max(-limit, min(limit, value))


def _envelope(remaining: float, v_max: float, decel: float, dt: float) -> float:
    """残量に対する台形プロファイルの速度指令 (符号付き)。

    ``sqrt(2*decel*|残量|)`` が減速エンベロープ (残量ぴったりで停止できる速度)。
    さらに ``|残量|/dt`` で抑えて、1 tick で残量を超えて進まないようにする。
    加速側は :class:`VelocityDriver` のレート制限が受け持つ。
    """
    r = abs(remaining)
    v = min(v_max, math.sqrt(2.0 * decel * r))
    if dt > 0.0:
        v = min(v, r / dt)
    return math.copysign(v, remaining)


class Phase(Enum):
    """プリミティブの進行状態。"""

    IDLE = "idle"        # 指令待ち
    RUN = "run"          # 台形プロファイルで主軸を詰めている
    SETTLE = "settle"    # 0 指令へランプダウン中 (整定)
    DONE = "done"        # 完了 (残差は residual() で確認)


class Kind(Enum):
    """プリミティブの種類 (主軸がどれか)。"""

    FORWARD = "forward"  # 主軸 = 進行方向の距離
    TURN = "turn"        # 主軸 = 方位角


@dataclass(frozen=True)
class CellMotionConfig:
    """1セル動作のチューニング定数 (既定値は脱調しにくい保守的な値)。"""

    # 主軸の速度上限と減速エンベロープ (#21 の実機ラダーで決めた採用点)。
    # 直進: v=0.12/0.18/0.24/0.30 を 4 セル直進で比較。所要は理論値どおり縮み
    # (6.25/4.27/3.29/2.69s)、距離誤差は v>=0.24 で 720mm あたり平均 -5mm・幅 ±4mm、
    # v<=0.18 では ±2mm。0.30 は 0.24 に対し 4 セルで 0.5s しか稼げず MAX_SPEED の
    # 余裕も衝突エネルギーも悪くなるので 0.24 を採る。
    # 旋回: omega=1.5/2.0/2.5 で 1 回あたり 1.83/1.56/1.67s、方位誤差 (4 回転計) は
    # -0.60/-0.36/-0.50°。2.0 が最速かつ最も正確で、2.5 は速くならずやり直しも増える。
    v_max: float = 0.24                    # 前進の最大速度 [m/s]
    omega_max: float = 2.0                 # 旋回の最大角速度 [rad/s]
    decel_mps2: float = 0.8                # 前進の減速度 [m/s^2] (driver ランプ以下に丸める)
    angular_decel_radps2: float = 7.0      # 旋回の角減速度 [rad/s^2] (同上)

    # 主軸以外の保持ゲイン (P 制御)
    k_cross: float = 3.0                   # 横ずれ[m] -> vy [1/s]
    v_cross_max: float = 0.04              # 横ずれ補正の速度上限 [m/s]
    k_heading: float = 4.0                 # 方位ずれ[rad] -> omega [1/s]
    omega_hold_max: float = 0.8            # 方位保持の角速度上限 [rad/s]
    k_position: float = 3.0                # 旋回中の位置ずれ[m] -> vx,vy [1/s]
    v_hold_max: float = 0.04               # 位置保持の速度上限 [m/s]

    # 主軸の下限速度 (エンベロープの終端で静止摩擦に負けて止まらないように)
    # 許容値を詰めるとエンベロープの終端指令が小さくなり、実機では動かないまま
    # creep し続けることがある。残量が許容外の間は下限を保証する。
    # ``下限 * dt <= 2 * 許容値`` なら 1 tick で許容帯を跳び越さない (既定は満たす)。
    min_v: float = 0.010                   # 前進の下限速度 [m/s]
    min_omega: float = 0.15                # 旋回の下限角速度 [rad/s]

    # 終了判定 (実機 #17: 残差はほぼ許容値そのものになるので小さめに取る)
    pos_tol_m: float = 0.0015              # 前進の残距離許容 [m]
    angle_tol_rad: float = math.radians(0.3)   # 旋回の残角許容 [rad]
    settled_v: float = 1e-3                # 整定判定の速度しきい値 [m/s]
    settled_omega: float = 1e-2            # 整定判定の角速度しきい値 [rad/s]
    # 指令が 0 になってから残差を判定するまで待つ秒数。整定判定が見ているのは
    # **指令値**なので、指令が 0 になった時点では機体はまだ動いている。指令のランプが
    # 実際に効き終わるまでの分だけ待つ。
    # (#21: 当初これを 0.25s にして「ガタの揺れ戻りを残量と誤読している」仮説を試したが、
    #  実機ではやり直しが 12/12 回そのまま発生し時間だけ悪化した。揺れは往復して 0 に
    #  戻る振動ではなく、機体がガタの帯のどこかに落ち着いて留まる。待っても消えない。)
    settle_dwell_s: float = 0.10

    # **やり直し**の判定に使う残量。上の許容値より緩くする。
    #
    # 旋回はどの角速度でも残差 0.9-1.45° に落ち着き、やり直しは 12/12 回発生して
    # 1 回も改善せず、1 旋回あたり 0.8s を捨てていた。詰まらないものは追いかけない。
    #
    # 当初この 1.6° はオムニホイールのローラーの遊び (0.5-1.0mm -> 車体 0.64-1.27°) の
    # 帯のすぐ外側として根拠づけていたが、**それは誤りだった** (#72)。実際は前輪 W0 の
    # 六角ハブのネジが緩んでおり、締めたら励磁時の車体の跳ねは +2.06° -> -0.03 ± 0.26°
    # (ゼロと同じ) になった。ところが**旋回の残差 (1.04-1.40°) と行き過ぎ (2.3-3.2°) は
    # 締めても変わらなかった**ので、これらは機械的な遊びではない。原因は未特定で、
    # ジャイロの遅れを疑っている (#73)。値自体は実測の残差と合っているので据え置く。
    retry_pos_tol_m: float = 0.004                 # 前進のやり直し判定 [m]
    retry_angle_tol_rad: float = math.radians(1.6)  # 旋回のやり直し判定 [rad]
    retry_v_max: float = 0.03              # やり直し時の速度上限 [m/s]
    retry_omega_max: float = 0.4           # やり直し時の角速度上限 [rad/s]
    max_retries: int = 1                   # 整定後に残っていた場合のやり直し回数


class CellMotion:
    """自己位置推定と連動した 1 セル前進・90°ターンの動作プリミティブ。

    ``driver`` は :class:`VelocityDriver`、``estimator`` は :class:`DeadReckoning`。
    ``update(dt, gyro_rate=...)`` を制御ループから一定レートで呼び、``done`` が
    True になったら次のプリミティブを ``start_*`` で開始する。
    """

    def __init__(
        self,
        driver: VelocityDriver,
        estimator: DeadReckoning,
        config: CellMotionConfig | None = None,
        maze: MazeConfig | None = None,
    ) -> None:
        self.driver = driver
        self.est = estimator
        self.cfg = config or CellMotionConfig()
        self.kin = driver.kin
        self.cell_pitch_m = (maze or load_maze_config()).cell_pitch_m
        # 減速エンベロープは driver のランプ上限を超えられない (超えると追従不能)
        self._decel = min(self.cfg.decel_mps2, driver.limits.max_linear_accel_mps2)
        self._angular_decel = min(
            self.cfg.angular_decel_radps2, driver.limits.max_angular_accel_radps2
        )
        # 理想格子上の基準姿勢 = 「今のプリミティブが終わった時にあるべき姿勢」
        self.x_ref, self.y_ref, self.phi_ref = estimator.pose
        self._kind: Kind | None = None
        self._phase = Phase.IDLE
        self._angle_remaining = 0.0   # 旋回の残角 [rad] (実回転分を差し引いていく)
        self._phi_prev = estimator.phi
        self._retries = 0
        self._settle_elapsed = 0.0    # 指令が 0 になってからの経過 [s]
        self._retry_remaining = 0.0   # やり直しを決めた時点の残量 (不感帯の調整用)

    @property
    def retries(self) -> int:
        """このプリミティブでやり直した回数 (0 が正常。増えるなら整定が足りない)。"""
        return self._retries

    @property
    def retry_remaining(self) -> float:
        """やり直しを決めた時点の残量 (最後の 1 回)。不感帯を決めるための実測値。

        完了時の残差はやり直し**後**の値なので、これを見ないと
        ``retry_pos_tol_m`` / ``retry_angle_tol_rad`` を実測から詰められない。
        """
        return self._retry_remaining

    # -- 基準姿勢 -----------------------------------------------------------
    @property
    def reference(self) -> tuple[float, float, float]:
        """理想格子上の基準姿勢 (x_ref[m], y_ref[m], phi_ref[rad])。"""
        return (self.x_ref, self.y_ref, self.phi_ref)

    def set_reference(
        self, x: float | None = None, y: float | None = None, phi: float | None = None
    ) -> tuple[float, float, float]:
        """基準姿勢を明示的に設定する (スタートセル投入時やグリッド補正後に使う)。"""
        if x is not None:
            self.x_ref = x
        if y is not None:
            self.y_ref = y
        if phi is not None:
            self.phi_ref = phi
        return self.reference

    def sync_reference_to_estimate(self) -> tuple[float, float, float]:
        """基準姿勢を現在の推定姿勢に合わせる (誤差を「無かったこと」にする)。"""
        self.x_ref, self.y_ref, self.phi_ref = self.est.pose
        return self.reference

    # -- プリミティブの開始 -------------------------------------------------
    def start_forward(self, distance_m: float) -> None:
        """現在の基準方位へ ``distance_m`` 前進する (基準点を進める)。"""
        self.x_ref += distance_m * math.cos(self.phi_ref)
        self.y_ref += distance_m * math.sin(self.phi_ref)
        self._begin(Kind.FORWARD)

    def start_forward_cells(self, cells: float = 1.0) -> None:
        """``cells`` セル分 (既定 1 セル = 180mm) 前進する。"""
        self.start_forward(cells * self.cell_pitch_m)

    def start_turn(self, delta_rad: float) -> None:
        """基準方位を ``delta_rad`` 回して、その絶対方位まで旋回する (+ = CCW)。

        残角には開始時点の方位誤差 (基準方位 - 推定方位) も含めるので、旋回が
        終わった時点で推定方位は基準方位に一致する。
        """
        heading_error = _wrap_angle(self.phi_ref - self.est.phi)
        self.phi_ref = _wrap_angle(self.phi_ref + delta_rad)
        self._angle_remaining = delta_rad + heading_error
        self._begin(Kind.TURN)

    def start_turn_left(self, quarter_turns: int = 1) -> None:
        """左 (CCW) へ 90°×``quarter_turns`` 旋回する。"""
        self.start_turn(quarter_turns * math.pi / 2.0)

    def start_turn_right(self, quarter_turns: int = 1) -> None:
        """右 (CW) へ 90°×``quarter_turns`` 旋回する。"""
        self.start_turn(-quarter_turns * math.pi / 2.0)

    def _begin(self, kind: Kind) -> None:
        self._kind = kind
        self._phase = Phase.RUN
        self._retries = 0
        self._phi_prev = self.est.phi

    def abort(self) -> None:
        """進行中のプリミティブを打ち切って停止指令にする (基準姿勢は触らない)。"""
        self._kind = None
        self._phase = Phase.IDLE
        self.driver.stop()

    # -- 状態 ---------------------------------------------------------------
    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def kind(self) -> Kind | None:
        return self._kind

    @property
    def done(self) -> bool:
        """プリミティブが完了 (または未開始) なら True。"""
        return self._phase in (Phase.DONE, Phase.IDLE)

    @property
    def remaining(self) -> float:
        """主軸の残量 (前進なら [m]、旋回なら [rad]、符号付き)。"""
        if self._kind is Kind.FORWARD:
            return self._along_remaining()
        if self._kind is Kind.TURN:
            return self._angle_remaining
        return 0.0

    def residual(self) -> tuple[float, float, float]:
        """基準姿勢に対する推定姿勢の残差 (前後[m], 左右[m], 方位[rad])。

        前後・左右は基準方位のフレームで見た「あと進むべき量」(基準 - 推定)。
        """
        return (self._along_remaining(), self._cross_error(), self._heading_error())

    # -- 誤差の計算 (基準方位フレーム) --------------------------------------
    def _along_remaining(self) -> float:
        dx, dy = self.x_ref - self.est.x, self.y_ref - self.est.y
        return dx * math.cos(self.phi_ref) + dy * math.sin(self.phi_ref)

    def _cross_error(self) -> float:
        dx, dy = self.x_ref - self.est.x, self.y_ref - self.est.y
        return -dx * math.sin(self.phi_ref) + dy * math.cos(self.phi_ref)

    def _heading_error(self) -> float:
        return _wrap_angle(self.phi_ref - self.est.phi)

    def _position_error_body(self) -> tuple[float, float]:
        """基準位置への誤差を **推定方位の車体フレーム** で見た (前後, 左右)。"""
        dx, dy = self.x_ref - self.est.x, self.y_ref - self.est.y
        c, s = math.cos(self.est.phi), math.sin(self.est.phi)
        return (dx * c + dy * s, -dx * s + dy * c)

    # -- 制御ループから一定 dt で呼ぶ ---------------------------------------
    def update(self, dt: float, gyro_rate: float | None = None) -> bool:
        """1 tick 進める: 指令生成 -> driver 駆動 -> 推定器の積分。

        ``gyro_rate`` [rad/s] (バイアス減算済) を渡すと、方位はジャイロ、並進は
        車輪から積分する (#13 の融合方針)。省略時は車輪のみ。
        完了 (``done``) を返す。
        """
        self._command(dt)
        cur = self.driver.update(dt)
        wheel_mps = self.kin.body_to_wheels(*cur)
        if gyro_rate is None:
            self.est.update_wheel_speeds(wheel_mps, dt)
        else:
            self.est.update_with_gyro_rate(wheel_mps, gyro_rate, dt)
        self._advance_phase(cur, dt)
        return self.done

    def _command(self, dt: float) -> None:
        """現在の推定姿勢と基準姿勢から目標ボディ速度を決める。"""
        if self._phase is not Phase.RUN:
            self.driver.stop()   # IDLE / SETTLE / DONE は 0 指令 (ランプで減速)
            return
        cfg = self.cfg
        retry = self._retries > 0
        if self._kind is Kind.FORWARD:
            v_max = cfg.retry_v_max if retry else cfg.v_max
            remaining = self._along_remaining()
            vx = _envelope(remaining, v_max, self._decel, dt)
            vx = self._with_floor(vx, cfg.min_v, remaining)
            vy = _clamp(cfg.k_cross * self._cross_error(), cfg.v_cross_max)
            omega = _clamp(cfg.k_heading * self._heading_error(), cfg.omega_hold_max)
        else:  # Kind.TURN
            omega_max = cfg.retry_omega_max if retry else cfg.omega_max
            omega = _envelope(self._angle_remaining, omega_max, self._angular_decel, dt)
            omega = self._with_floor(omega, cfg.min_omega, self._angle_remaining)
            e_fwd, e_left = self._position_error_body()
            vx = _clamp(cfg.k_position * e_fwd, cfg.v_hold_max)
            vy = _clamp(cfg.k_position * e_left, cfg.v_hold_max)
        self.driver.set_velocity(vx, vy, omega)

    def _with_floor(self, v: float, floor: float, remaining: float) -> float:
        """主軸の指令に下限速度をかける (残量が許容内なら素通し)。

        許容値を詰めるとエンベロープ終端の指令が静止摩擦を下回り、実機では
        動かないまま creep し続けることがあるため、まだ詰めるべき残量がある間は
        ``floor`` 以上を保証する。許容帯に入ったら素通しして、そのまま整定へ渡す。
        """
        if abs(remaining) <= self._tolerance():
            return v
        return math.copysign(max(abs(v), floor), remaining)

    def _advance_phase(self, cur: tuple[float, float, float], dt: float = 0.0) -> None:
        """残量・整定状況から状態遷移する (旋回の残角もここで減らす)。"""
        # 実回転分を残角から差し引く (180°以上の旋回でも符号が反転しないように
        # 絶対方位の差ではなく積算で持つ)
        dphi = _wrap_angle(self.est.phi - self._phi_prev)
        self._phi_prev = self.est.phi
        if self._kind is Kind.TURN:
            self._angle_remaining -= dphi

        if self._phase is Phase.RUN:
            if abs(self.remaining) <= self._tolerance():
                self._phase = Phase.SETTLE
                self._settle_elapsed = 0.0
                self.driver.stop()
            return
        if self._phase is Phase.SETTLE:
            vx, vy, omega = cur
            settled = (
                max(abs(vx), abs(vy)) <= self.cfg.settled_v
                and abs(omega) <= self.cfg.settled_omega
            )
            if not settled:
                self._settle_elapsed = 0.0
                return
            # 指令が 0 でも機体はしばらく揺れている。揺れが収まるまで残差を見ない
            # (見るとガタの弾性的な戻りを「残量」と誤読してやり直しに入る)。
            self._settle_elapsed += dt
            if self._settle_elapsed < self.cfg.settle_dwell_s:
                return
            # 整定後にまだ残っていれば低速でやり直す (惰行分の詰め直し)
            if (abs(self.remaining) > self._retry_tolerance()
                    and self._retries < self.cfg.max_retries):
                self._retry_remaining = self.remaining
                self._retries += 1
                self._settle_elapsed = 0.0
                self._phase = Phase.RUN
            else:
                self._phase = Phase.DONE

    def _tolerance(self) -> float:
        """主軸を詰めるのをやめる (整定へ移る) 残量。"""
        return self.cfg.pos_tol_m if self._kind is Kind.FORWARD else self.cfg.angle_tol_rad

    def _retry_tolerance(self) -> float:
        """やり直すかどうかの残量。機械の不感帯より内側は追いかけない。"""
        return (self.cfg.retry_pos_tol_m if self._kind is Kind.FORWARD
                else self.cfg.retry_angle_tol_rad)
