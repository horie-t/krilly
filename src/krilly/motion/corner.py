"""止まらずに方向転換するときの幾何 (issue #80)。

旋回レス走行 (#76) では「曲がる」= 進行軸を変えることなので、**機体を回さずに
vx と vy を重ねればコーナーは丸められる**。区間ごとの完全停止 (実測 0.81s) が
そのぶん要らなくなる。ここではその軌跡と、通れるかどうかの余裕を計算する。

軌跡は「片方の軸を一定減速で 0 へ、もう片方を同じ一定加速で v へ」の重ね合わせ。
セル中心を原点、入りを +y (北)、出を +x (東) に取ると、ブレンド距離
``d = v^2 / (2a)`` に対して

    x(s) = d * s^2 ,  y(s) = -d * (1-s)^2      (s = 0..1)

で、始点 (0, -d) を +y で通過し、終点 (d, 0) を +x で抜ける。**速度指令は継ぎ目で
連続**になる (切り替え時点の残量がちょうど d なので、減速エンベロープ
``sqrt(2*a*d)`` が v にぴったり一致する)。

要点は 3 つ:

- 膨らむ先は「入ってきた辺」と「出ていく辺」に挟まれた**開いている象限**。壁は
  無く、柱が 1 本あるだけなので、余裕は直線の廊下より広い。
- 経路全体の最小余裕は**ブレンドの外**で決まる。廊下を柱の真横で通り抜けるとき
  (余裕 21.4mm) が最も狭く、そこはコーナーを丸めても丸めなくても通る。
  :func:`min_post_clearance_m` >= :func:`corridor_clearance_m` は d によらず成り立つ。
- ブレンド中は速度ベクトルの大きさが **0.707*v** まで落ちる (両軸が入れ替わるので)。
  つまり輪速のピークは直線より小さく、駆動側の余裕はむしろ増える。
"""

from __future__ import annotations

import math

#: 機体の外接円の半径 [m] (#80)。``sqrt((45.07 + 12.75)^2 + 24^2)`` = 62.6mm。
#: 内訳は 中心-車輪 45.07mm (``center_to_wheel_m``) + 車輪の半幅 12.75mm を車軸方向、
#: 車輪半径 24mm を進行方向に取ったもの = 車輪の最も外側の点。
#: **実体は形のある三角形**なので、円で見るのは安全側の近似。
MACHINE_RADIUS_M = 0.0626


def blend_distance_m(v_max: float, decel: float) -> float:
    """コーナーを丸め始める残量 [m]。``v^2 / (2a)`` = 一定減速で止まれる距離。

    この距離で次の区間へ移ると、古い軸の減速エンベロープが出す速度がちょうど
    ``v_max`` になるので、**指令が跳ねない**。
    """
    if decel <= 0.0:
        return 0.0
    return v_max * v_max / (2.0 * decel)


def blend_duration_s(v_max: float, decel: float) -> float:
    """ブレンドに掛かる時間 [s] (``v/a``)。"""
    return v_max / decel if decel > 0.0 else 0.0


def corner_path(blend_m: float, samples: int = 33) -> list[tuple[float, float]]:
    """コーナーの軌跡 (セル中心が原点、入り +y / 出 +x)。"""
    return [(blend_m * s * s, -blend_m * (1.0 - s) * (1.0 - s))
            for s in (i / (samples - 1) for i in range(samples))]


def corner_bulge_m(blend_m: float) -> float:
    """内側の角に最も寄る点の、各中心線からのずれ [m] (``d/4``)。

    s=1/2 で ``(d/4, -d/4)``。この点が「コーナーをどれだけ内側へ切ったか」。
    """
    return blend_m / 4.0


def _post_corner_m(pitch_m: float, wall_thickness_m: float) -> float:
    """柱の内側の角までの距離 [m] (セル中心から、各軸方向に)。"""
    return pitch_m / 2.0 - wall_thickness_m / 2.0


def corridor_clearance_m(pitch_m: float = 0.180, wall_thickness_m: float = 0.012,
                         radius_m: float = MACHINE_RADIUS_M) -> float:
    """直線の廊下を中心線上で通るときの、壁との余裕 [m]。

    実測の基準線。**コーナーの余裕はこれと比べる**もので、これより広ければ
    「丸めても、通れるかどうかを決めているのは相変わらず廊下の方」となる。
    """
    return _post_corner_m(pitch_m, wall_thickness_m) - radius_m


def min_post_clearance_m(blend_m: float, pitch_m: float = 0.180,
                         wall_thickness_m: float = 0.012,
                         radius_m: float = MACHINE_RADIUS_M,
                         samples: int = 33) -> float:
    """ブレンド中、内側の柱に最も近づいたときの余裕 [m]。

    内側の柱は開いている 2 辺の間、``(h, -h)`` (h = 半ピッチ - 壁厚/2) にある。
    """
    h = _post_corner_m(pitch_m, wall_thickness_m)
    return min(math.hypot(h - x, -h - y)
               for x, y in corner_path(blend_m, samples)) - radius_m


def peak_speed_ratio() -> float:
    """ブレンド中の速度ベクトルの最小の大きさ / ``v_max`` (= 1/√2)。

    両軸が入れ替わる途中で ``|v| = v/sqrt(2)`` まで落ちる。つまりコーナーでは
    **輪速のピークは上がらない**。上がるのは加速度の方 (両軸が同時に変化する)。
    """
    return 1.0 / math.sqrt(2.0)
