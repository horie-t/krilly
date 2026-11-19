"""走行中のカメラ測定を 1 フレームずつ記録する (issue #184)。

止まらずに補正する設計 (M7) の前提になる数字 — 1 フレームの処理時間、撮影から
使えるまでの遅れ、走りながら壁が読めるか、機体の揺れ — を実機で測るための層。

- :func:`probe_frame`: 1 枚から、壁の帯 (割合・ずれ・飽和)、セル内のずれ
  (``cell_offset``)、軸角 (``axis_yaw``)、消失点から読むカメラの傾き
  (``camera_tilt``) を出し、**各段の処理時間**も一緒に返す。純計算。
- :class:`FrameLogger`: 別スレッドで撮影 → :func:`probe_frame` → CSV に 1 行。
  撮影時刻 (``SensorTimestamp``) と受け取り・処理完了の時刻を並べて残す。
- :class:`TickLog`: 制御ループ側の 1 tick を CSV に 1 行 (推定姿勢・指令速度・
  ジャイロ)。フレームの撮影時刻と突き合わせて、揺れと加速度の関係や遅れを出す。

**揺れの見分け方** (#184 の本文): 横ずれが振れているのに傾きがほぼ 0 なら枠の
横ずれ、横ずれ ≈ (壁の上面からの高さ) × 傾き なら機体全体の傾き、``axis_yaw`` だけが
振れるなら天板のねじれ。そのために 3 つを**同じフレームから**出す。

**このスレッド自体が制御ループの邪魔をする** (GIL)。tick の間隔 (``dt``) も
:class:`TickLog` に残すので、記録あり/なしの比較で影響の大きさが分かる — それが
#185 (画像処理を別プロセスに) の根拠になる。
"""

from __future__ import annotations

import csv
import dataclasses
import math
import os
import threading
import time
from dataclasses import dataclass

import numpy as np

from krilly.logging_config import get_logger
from krilly.perception.axis_yaw import AxisYawConfig, axis_yaw
from krilly.perception.camera_tilt import measure_tilt
from krilly.perception.cell_pose import cell_offset
from krilly.perception.red_wall import red_mask
from krilly.perception.wall_detect import (
    BACK,
    CALIBRATED_GEOMETRY,
    CALIBRATED_RED,
    FRONT,
    LEFT,
    RIGHT,
    WallDetector,
)

log = get_logger("krilly.frame_log")

EDGES = (FRONT, BACK, LEFT, RIGHT)

#: カメラ (レンズの先端) の床からの高さ [mm]。CAD では 380.1、#140 の組み直し後の
#: 実測は 378 (``docs/claude/hardware-sessions.md``)。傾きの角度はこれに比例する。
CAMERA_HEIGHT_MM = 378.0
WALL_HEIGHT_MM = 50.0


def focal_px(camera_height_mm: float = CAMERA_HEIGHT_MM,
             wall_height_mm: float = WALL_HEIGHT_MM) -> float:
    """壁の上面の平面に対する焦点距離 [px] (``scripts/camera_fov`` と同じ式)。"""
    return CALIBRATED_GEOMETRY.px_per_mm_x * (camera_height_mm - wall_height_mm)


FRAME_FIELDS = (
    "seq", "sensor_ts_ns", "recv_ns", "done_ns", "exposure_us", "gain", "frame_duration_us",
    "t_measure_ms", "t_yaw_ms", "t_tilt_ms", "t_edges_ms", "t_ahead_ms",
    *(f"frac_{e}" for e in EDGES), *(f"shift_{e}_px" for e in EDGES),
    *(f"sat_{e}" for e in EDGES),
    "forward_mm", "left_mm", "walls_x", "walls_y",
    "yaw_deg", "yaw_len_px",
    "roll_deg", "pitch_deg", "roll_resid_deg", "pitch_resid_deg", "bands_h", "bands_v",
    "saved",
    # #189: 壁の切れ目の縁 (perception.wall_end.encode_edges の文字列)
    "edges",
)

TICK_FIELDS = (
    "t_ns", "dt_s", "x_m", "y_m", "phi_rad", "ref_x_m", "ref_y_m", "ref_phi_rad",
    "cmd_vx", "cmd_vy", "cmd_omega", "gyro_rad_s", "label",
    # #186: この tick で融合したカメラの補正 (世界座標) と、使った測定の数
    "fuse_dx_mm", "fuse_dy_mm", "fuse_dphi_deg", "fuse_n",
)


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


@dataclass
class ProbeConfig:
    """:func:`probe_frame` の設定。各段は個別に切れる (処理時間の内訳を見るため)。"""

    focal_px: float
    yaw: AxisYawConfig | None = None
    do_yaw: bool = True
    do_tilt: bool = True
    tilt_min_length: int = 120
    #: 壁の切れ目の縁を探すか (#189)。左右の帯だけなら 1 枚 3.3 ms、4 本で 5.7 ms。
    do_edges: bool = False
    edge_bands: tuple = ("left", "right", "front", "back")
    #: 柱の上面の色 ("auto" / "red" / "white"、``wall_end.EdgeConfig.post_top``)。
    post_top: str = "auto"
    #: 前方の境目の見張り用に、行・列ごとの赤の数を作るか (#188)。
    do_ahead: bool = False


def probe_frame(bgr: np.ndarray, detector: WallDetector, cfg: ProbeConfig) -> dict:
    """1 枚から壁の帯・セル内のずれ・軸角・傾きを測り、各段の処理時間を添える。

    返す dict のキーは :data:`FRAME_FIELDS` の一部 (時刻とメタデータは呼び出し側)。
    測れなかった値は None。
    """
    row: dict = {}
    t0 = time.perf_counter()
    measured = detector.measure(bgr)
    off = cell_offset(bgr, detector, measured=measured)
    t1 = time.perf_counter()
    for e in EDGES:
        fraction, shift, saturated = measured[e]
        row[f"frac_{e}"] = float(fraction)
        row[f"shift_{e}_px"] = int(shift)
        row[f"sat_{e}"] = bool(saturated)
    row["forward_mm"] = None if off.forward_m is None else off.forward_m * 1000.0
    row["left_mm"] = None if off.left_m is None else off.left_m * 1000.0
    row["walls_x"], row["walls_y"] = off.walls_x, off.walls_y
    row["t_measure_ms"] = (t1 - t0) * 1000.0

    row["yaw_deg"] = row["yaw_len_px"] = None
    if cfg.do_yaw:
        t2 = time.perf_counter()
        yaw = axis_yaw(bgr, cfg.yaw)
        row["t_yaw_ms"] = (time.perf_counter() - t2) * 1000.0
        if yaw is not None:
            row["yaw_deg"], row["yaw_len_px"] = yaw.angle_deg, yaw.total_length_px

    row["edges"] = ""
    if cfg.do_edges:
        from krilly.perception.wall_end import encode_edges, find_band_edges

        t4 = time.perf_counter()
        shifts = {e: int(measured[e][1]) for e in EDGES}
        from krilly.perception.wall_end import EdgeConfig

        row["edges"] = encode_edges(find_band_edges(bgr, bands=cfg.edge_bands,
                                                    shifts_px=shifts,
                                                    config=EdgeConfig(post_top=cfg.post_top)))
        row["t_edges_ms"] = (time.perf_counter() - t4) * 1000.0

    if cfg.do_ahead:
        from krilly.perception.ahead import ahead_profiles

        t5 = time.perf_counter()
        row["ahead_profiles"] = ahead_profiles(
            bgr, {e: int(measured[e][1]) for e in EDGES})
        row["t_ahead_ms"] = (time.perf_counter() - t5) * 1000.0

    for k in ("roll_deg", "pitch_deg", "roll_resid_deg", "pitch_resid_deg"):
        row[k] = None
    if cfg.do_tilt:
        t3 = time.perf_counter()
        tilt = measure_tilt(red_mask(bgr, CALIBRATED_RED), cfg.focal_px,
                            min_length=cfg.tilt_min_length)
        row["t_tilt_ms"] = (time.perf_counter() - t3) * 1000.0
        row["roll_deg"], row["pitch_deg"] = tilt.roll_deg, tilt.pitch_deg
        row["roll_resid_deg"], row["pitch_resid_deg"] = tilt.residual_h_deg, tilt.residual_v_deg
        row["bands_h"], row["bands_v"] = len(tilt.horizontal), len(tilt.vertical)
    return row


class FrameLogger:
    """別スレッドで撮り続け、1 フレーム 1 行を CSV に書く。

    ``camera`` は ``capture_with_metadata()`` を持つもの (:class:`krilly.hal.camera.Camera`
    か偽物)。**動いている間、他のコードはカメラに触らないこと** — 停止中の測定
    (``measure_yaw`` など) は :meth:`stop` の後に行う。

    処理が撮影より遅ければフレームは捨てられる (picamera2 が最新を渡す)。捨てた
    数は撮影時刻の飛びから後で分かる。

    ``save_every`` > 0 なら N 枚に 1 枚を ``save_dir`` に JPEG で残す (ブレを目で
    見るため)。**書き出しは処理時間を食う**ので、時間を測る走行では 0 にすること。

    ``path`` が None なら CSV は書かない (融合だけに使うとき)。``on_measure`` を
    与えると、1 枚測るたびに ``on_measure(row)`` を**このスレッドから**呼ぶ (#186:
    制御ループへ測定を渡す。受け手はキューに積むだけにすること)。

    ``yaw_every`` > 0 なら ``axis_yaw`` は N 枚に 1 枚だけ測る (``probe.do_yaw`` が
    True のとき)。#184 の実測で ``axis_yaw`` は 23〜41 ms かかり、毎フレーム回すと
    30 fps が保てない。
    """

    def __init__(self, camera, detector: WallDetector, probe: ProbeConfig,
                 path: str | None, save_every: int = 0, save_dir: str | None = None,
                 clock_ns=time.monotonic_ns, on_measure=None, yaw_every: int = 0) -> None:
        self.camera = camera
        self.detector = detector
        self.probe = probe
        self.path = path
        self.save_every = save_every
        self.save_dir = save_dir
        self.clock_ns = clock_ns
        self.on_measure = on_measure
        self.yaw_every = yaw_every
        self.frames = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._file = None
        self._writer = None
        self._lock = threading.Lock()

    def start(self) -> "FrameLogger":
        if self.path is not None:
            new = not os.path.exists(self.path)
            self._file = open(self.path, "a", newline="")
            self._writer = csv.writer(self._file)
            if new:
                self._writer.writerow(FRAME_FIELDS)
        if self.save_every > 0 and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="frame-logger", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "FrameLogger":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def step(self) -> dict:
        """1 枚撮って測り、1 行書く (スレッドから呼ばれる。テストでは直接呼べる)。"""
        frame, meta = self.camera.capture_with_metadata()
        recv = self.clock_ns()
        probe = self.probe
        if self.yaw_every > 0 and probe.do_yaw and self.frames % self.yaw_every:
            probe = dataclasses.replace(probe, do_yaw=False)
        row = probe_frame(frame, self.detector, probe)
        done = self.clock_ns()
        row.update(
            seq=self.frames, sensor_ts_ns=meta.get("SensorTimestamp"), recv_ns=recv,
            done_ns=done, exposure_us=meta.get("ExposureTime"),
            gain=meta.get("AnalogueGain"), frame_duration_us=meta.get("FrameDuration"),
            saved="",
        )
        if self.save_every > 0 and self.save_dir and self.frames % self.save_every == 0:
            import cv2

            name = f"frame_{self.frames:05d}.jpg"
            cv2.imwrite(os.path.join(self.save_dir, name), frame)
            row["saved"] = name
        with self._lock:
            if self._writer is not None:
                self._writer.writerow([_fmt(row.get(k)) for k in FRAME_FIELDS])
        self.frames += 1
        if self.on_measure is not None:
            self.on_measure(row)
        return row

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:                      # noqa: BLE001 — 記録は走行を止めない
                self.errors += 1
                if self.errors <= 3:
                    log.exception("フレームの記録に失敗 (走行は続ける)")
                time.sleep(0.01)


class TickLog:
    """制御ループの 1 tick を CSV に 1 行書く。:meth:`record` はループから呼ぶ。

    ``label`` には動作の名前 (``[1/2] 右へ6セル`` など) を入れておくと、後で場面
    (止まっている/走っている) を分けやすい。
    """

    def __init__(self, path: str, clock_ns=time.monotonic_ns) -> None:
        self.path = path
        self.clock_ns = clock_ns
        new = not os.path.exists(path)
        self._file = open(path, "a", newline="")
        self._writer = csv.writer(self._file)
        if new:
            self._writer.writerow(TICK_FIELDS)
        self.label = ""

    def record(self, dt: float, pose, reference, command, gyro: float | None,
               fused: tuple[float, float, float, int] | None = None) -> None:
        """``fused`` = この tick の融合の (dx [m], dy [m], dphi [rad], 測定の数) (#186)。"""
        x, y, phi = pose
        rx, ry, rphi = reference
        vx, vy, w = command
        self._writer.writerow([
            self.clock_ns(), _fmt(dt, 5), _fmt(x, 5), _fmt(y, 5), _fmt(phi, 6),
            _fmt(rx, 5), _fmt(ry, 5), _fmt(rphi, 6),
            _fmt(vx, 4), _fmt(vy, 4), _fmt(w, 4),
            "" if gyro is None else _fmt(gyro, 5), self.label,
            *(("", "", "", "") if fused is None else
              (_fmt(fused[0] * 1000.0), _fmt(fused[1] * 1000.0),
               _fmt(math.degrees(fused[2]), 4), fused[3])),
        ])

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TickLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def measurement_from_row(row: dict):
    """:func:`probe_frame` の 1 行を融合の測定に直す (撮影時刻が無ければ None)。"""
    from krilly.localization.delayed_fusion import CameraMeasurement

    ts = row.get("sensor_ts_ns")
    if ts is None:
        return None
    yaw = row.get("yaw_deg")
    from krilly.perception.wall_end import decode_edges

    return CameraMeasurement(
        int(ts),
        forward_m=None if row.get("forward_mm") is None else row["forward_mm"] / 1000.0,
        left_m=None if row.get("left_mm") is None else row["left_mm"] / 1000.0,
        axis_rad=None if yaw is None else math.radians(yaw),
        edges=tuple(decode_edges(row.get("edges"))),
        ahead=row.get("ahead_profiles"),
    )


def read_csv(path: str) -> list[dict]:
    """記録した CSV を読む。数値は float に、空欄は None にする (``label`` などは文字列)。"""
    out = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            row = {}
            for k, v in raw.items():
                if v == "" or v is None:
                    row[k] = None
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            out.append(row)
    return out


def percentile(values: list[float], q: float) -> float | None:
    """``values`` の q パーセンタイル (空なら None)。"""
    return None if not values else float(np.percentile(values, q))


def wall_top_lever_mm(camera_height_mm: float = CAMERA_HEIGHT_MM) -> float:
    """傾きを位置のずれに拡大するてこの長さ [mm] (壁の上面からカメラまで)。"""
    return camera_height_mm - WALL_HEIGHT_MM


def tilt_to_shift_mm(tilt_deg: float, camera_height_mm: float = CAMERA_HEIGHT_MM) -> float:
    """カメラが ``tilt_deg`` 傾いたときの、壁の上面での見かけのずれ [mm]。"""
    return wall_top_lever_mm(camera_height_mm) * math.tan(math.radians(tilt_deg))
