"""走行中のカメラ記録 (#184) のテスト: 1 枚の測定、記録のスレッド、解析。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from krilly.hal.camera import Camera
from krilly.perception.axis_yaw import calibrated_axis_yaw_config
from krilly.perception.frame_log import (
    FRAME_FIELDS,
    FrameLogger,
    ProbeConfig,
    TickLog,
    focal_px,
    probe_frame,
    read_csv,
    tilt_to_shift_mm,
)
from krilly.perception.wall_detect import (
    CALIBRATED_BANDS,
    CALIBRATED_GEOMETRY,
    DEFAULT_FRAME_SIZE,
    FRONT,
    LEFT,
    RIGHT,
    BACK,
    WallDetector,
    calibrated_config,
)
from scripts import frame_log_report as rep

RED = (0, 0, 255)


def frame_with_bands(shift_x: int = 0, shift_y: int = 0,
                     edges=(LEFT, RIGHT, FRONT, BACK)) -> np.ndarray:
    img = np.zeros((DEFAULT_FRAME_SIZE[1], DEFAULT_FRAME_SIZE[0], 3), dtype=np.uint8)
    for edge in edges:
        lo, hi = CALIBRATED_BANDS[edge]
        if edge in (LEFT, RIGHT):
            img[:, lo + shift_x: hi + shift_x] = RED
        else:
            img[lo + shift_y: hi + shift_y, :] = RED
    return img


@pytest.fixture
def detector():
    return WallDetector(calibrated_config())


def probe_cfg(**kw):
    return ProbeConfig(focal_px=focal_px(), yaw=calibrated_axis_yaw_config(), **kw)


def test_probe_reads_offset_and_reports_stage_times(detector):
    shift = 10
    row = probe_frame(frame_with_bands(shift_x=shift), detector, probe_cfg())
    assert row["left_mm"] == pytest.approx(shift / CALIBRATED_GEOMETRY.px_per_mm_x, abs=0.7)
    assert row["walls_x"] == 2
    for k in ("t_measure_ms", "t_yaw_ms", "t_tilt_ms"):
        assert row[k] >= 0.0
    # 平行な帯なので傾きはほぼ 0、軸角も 0
    assert row["yaw_deg"] == pytest.approx(0.0, abs=0.5)
    if row["pitch_deg"] is not None:
        assert abs(row["pitch_deg"]) < 0.5


def test_probe_stages_can_be_switched_off(detector):
    row = probe_frame(frame_with_bands(), detector, probe_cfg(do_yaw=False, do_tilt=False))
    assert "t_yaw_ms" not in row and "t_tilt_ms" not in row
    assert row["yaw_deg"] is None and row["roll_deg"] is None


class FakePicam:
    """``capture_request`` を持つ偽の picamera2。撮影時刻は 33ms 刻み。"""

    def __init__(self, frame):
        self.frame = frame
        self.n = 0

    def capture_request(self):
        fake = self

        class Req:
            def make_array(self, name):
                return fake.frame

            def get_metadata(self):
                fake.n += 1
                return {"SensorTimestamp": 1_000_000_000 + fake.n * 33_333_333,
                        "ExposureTime": 30000, "AnalogueGain": 1.7, "FrameDuration": 33333}

            def release(self):
                pass

        return Req()

    def stop(self):
        pass

    def close(self):
        pass


def test_capture_with_metadata_uses_one_request():
    frame = frame_with_bands()
    cam = Camera(picam2=FakePicam(frame))
    got, meta = cam.capture_with_metadata()
    assert got is frame
    assert meta["SensorTimestamp"] == 1_000_000_000 + 33_333_333


def test_capture_with_metadata_falls_back_without_requests():
    class Old:
        def capture_array(self):
            return "frame"

        def stop(self):
            pass

        def close(self):
            pass

    assert Camera(picam2=Old()).capture_with_metadata() == ("frame", {})


def test_logger_writes_rows_with_timestamps(tmp_path, detector):
    cam = Camera(picam2=FakePicam(frame_with_bands(shift_x=5)))
    clock = iter(range(2_000_000_000, 3_000_000_000, 10_000_000))
    path = str(tmp_path / "x_frames.csv")
    logger = FrameLogger(cam, detector, probe_cfg(do_tilt=False), path,
                         clock_ns=lambda: next(clock))
    logger._file = open(path, "w", newline="")      # スレッドを起こさずに step を直接呼ぶ
    import csv
    logger._writer = csv.writer(logger._file)
    logger._writer.writerow(FRAME_FIELDS)
    for _ in range(3):
        logger.step()
    logger._file.close()
    rows = read_csv(path)
    assert len(rows) == 3
    assert rows[1]["sensor_ts_ns"] - rows[0]["sensor_ts_ns"] == pytest.approx(33_333_333)
    assert rows[0]["done_ns"] > rows[0]["recv_ns"]
    assert rows[0]["exposure_us"] == 30000


def test_logger_thread_starts_and_stops(tmp_path, detector):
    cam = Camera(picam2=FakePicam(frame_with_bands()))
    path = str(tmp_path / "t_frames.csv")
    with FrameLogger(cam, detector, probe_cfg(do_tilt=False, do_yaw=False), path) as lg:
        import time
        deadline = time.monotonic() + 2.0
        while lg.frames < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert lg.frames >= 2 and lg.errors == 0
    assert len(read_csv(path)) == lg.frames


def test_tilt_to_shift_uses_wall_top_lever():
    # 378 - 50 = 328mm のてこで 1° -> 5.7mm
    assert tilt_to_shift_mm(1.0) == pytest.approx(328 * math.tan(math.radians(1.0)))


# --- 解析 -----------------------------------------------------------------------

def test_body_offset_from_estimate_folds_to_nearest_cell():
    # 東向き (φ=0) で x=0.185 -> 隣のセルの中心から +5mm 前
    f, l = rep.body_offset_from_estimate(0.185, -0.003, 0.0)
    assert f == pytest.approx(5.0) and l == pytest.approx(-3.0)
    # 北向き (φ=90°) で世界の +x は機体の右 = 左が負
    f, l = rep.body_offset_from_estimate(0.004, 0.0, math.pi / 2)
    assert f == pytest.approx(0.0, abs=1e-9) and l == pytest.approx(-4.0)


def make_logs(tmp_path, delay_ms: float = 0.0, sway_mm: float = 0.0):
    """東へ 0.2m/s で 1s 走って止まる合成ログ。カメラは ``delay_ms`` 遅れた絵を返す。"""
    ticks = TickLog(str(tmp_path / "s_ticks.csv"), clock_ns=lambda: 0)
    import csv
    t, x, v = 0.0, 0.0, 0.0
    tick_rows = []
    for i in range(150):          # 3s、20ms
        t = i * 0.02
        target = 0.2 if 0.2 <= t < 1.6 else 0.0
        v = min(target, v + 0.02) if target > v else max(target, v - 0.02)
        x += v * 0.02
        tick_rows.append((t, x, v))
    ticks.close()
    with open(tmp_path / "s_ticks.csv", "w", newline="") as f:
        w = csv.writer(f)
        from krilly.perception.frame_log import TICK_FIELDS
        w.writerow(TICK_FIELDS)
        for t, x, v in tick_rows:
            w.writerow([int(t * 1e9), 0.02, x, 0.0, 0.0, x, 0.0, 0.0, v, 0.0, 0.0, 0.0,
                        "走行" if v else "停止"])
    ts = np.array([r[0] for r in tick_rows])
    xs = np.array([r[1] for r in tick_rows])
    with open(tmp_path / "s_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FRAME_FIELDS)
        for k in range(1, 85):
            tf = k * 0.0333
            true_x = float(np.interp(tf - delay_ms / 1000.0, ts, xs))
            fwd = (true_x - round(true_x / 0.18) * 0.18) * 1000.0
            left = sway_mm * math.sin(2 * math.pi * 8 * tf)
            row = {k2: "" for k2 in FRAME_FIELDS}
            row.update(seq=k, sensor_ts_ns=int(tf * 1e9), recv_ns=int((tf + 0.02) * 1e9),
                       done_ns=int((tf + 0.05) * 1e9), forward_mm=f"{fwd:.3f}",
                       left_mm=f"{left:.3f}", frame_duration_us=33333)
            w.writerow([row[k2] for k2 in FRAME_FIELDS])
    return str(tmp_path / "s")


def test_phases_and_zero_residual_when_camera_matches(tmp_path):
    prefix = make_logs(tmp_path)
    frames = read_csv(prefix + "_frames.csv")
    ticks = rep.Ticks(read_csv(prefix + "_ticks.csv"))
    rows = rep.enrich(frames, ticks)
    phases = {r.get("phase") for r in rows}
    assert {"静止", "加速", "等速", "減速"} <= phases
    res = [abs(r["res_forward_mm"]) for r in rows if r.get("res_forward_mm") is not None]
    assert max(res) < 0.5


def test_tau_sweep_recovers_camera_delay(tmp_path):
    prefix = make_logs(tmp_path, delay_ms=40.0)
    frames = read_csv(prefix + "_frames.csv")
    ticks = rep.Ticks(read_csv(prefix + "_ticks.csv"))
    tau, sd, sd0 = rep.tau_sweep(frames, ticks)["res_forward_mm"]
    # カメラが 40ms 古い絵を返す = 撮影時刻の 40ms 前の推定と合う
    assert tau == pytest.approx(-40.0, abs=5.0)
    assert sd < sd0


def test_report_runs_end_to_end(tmp_path):
    prefix = make_logs(tmp_path, sway_mm=1.0)
    text = rep.analyze(prefix, plot_path=str(tmp_path / "p.png"))
    assert "実効" in text and "場面ごとの読み" in text and "止まった後の収束" in text
    assert (tmp_path / "p.png").exists()
