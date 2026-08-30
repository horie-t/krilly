"""カメラの露出ロックのテスト (issue #78)。

**実機なしで固定したいのは「ロックした値を覚えているか」と「余力の数え方」。**
照明が変わっても自動露出が明るさを打ち消すので、フレームを見ても照明の違いは
分からない (実測: 照度 20 倍の差で画面の明るさ V は 157 対 156、赤割合は 0.401 対
0.401)。しかも**動くのはゲインだけ**で、露出は 30fps 固定のため 32.7ms から動かない。
記録しなければ、暗さに対する余力がどれだけ残っているかは誰にも分からない。
"""

from krilly.hal.camera import Camera


class FakePicam2:
    """``capture_metadata`` / ``set_controls`` だけを持つ最小のカメラ。"""

    def __init__(self, exposure_us=8000, gain=1.0, max_gain=16.0):
        self.meta = {"ExposureTime": exposure_us, "AnalogueGain": gain}
        self.controls = None
        self.camera_controls = {"AnalogueGain": (1.12, max_gain, 1.0)}

    def capture_metadata(self):
        return self.meta

    def set_controls(self, controls):
        self.controls = controls


def test_lock_records_what_it_locked():
    cam = Camera(picam2=FakePicam2(12345, 2.5))
    cam.lock_exposure(cam._picam2)
    assert cam.exposure_time_us == 12345
    assert cam.analogue_gain == 2.5
    # 実際に固定していること (AE/AWB を切り、読んだ値をそのまま書き戻す)
    assert cam._picam2.controls == {
        "AeEnable": False, "AwbEnable": False,
        "ExposureTime": 12345, "AnalogueGain": 2.5,
    }


def test_metadata_without_exposure_falls_back_instead_of_crashing():
    """メタデータが揃わないカメラでも走行を止めない (既定値でロックする)。"""
    fake = FakePicam2()
    fake.meta = {}
    cam = Camera(picam2=fake)
    cam.lock_exposure(fake)
    assert cam.exposure_time_us == 8000 and cam.analogue_gain == 1.0


def test_headroom_is_counted_in_stops_of_gain_not_in_exposure():
    """**暗さへの余力はゲインでしか測れない。** 露出は 30fps 固定 (33.3ms) で
    頭打ちになるので、暗くなっても伸びない — 実測でも 3 条件すべて 32.7ms だった。
    実測のゲインで段数を確かめる。"""
    for gain, expected in ((2.16, 2.89), (6.21, 1.37), (10.67, 0.58)):
        cam = Camera(picam2=FakePicam2(32700, gain))
        cam.lock_exposure(cam._picam2)
        assert abs(cam.headroom_stops() - expected) < 0.01


def test_running_out_of_gain_warns():
    """残り 1 段を切ったら警告する (そこから先は画面が暗くなって一斉に崩れる)。"""
    ok = Camera(picam2=FakePicam2(32700, 6.21))       # 調光 30% 相当、残り 1.4 段
    ok.lock_exposure(ok._picam2)
    assert ok.exposure_warning() is None

    tight = Camera(picam2=FakePicam2(32700, 10.67))   # 調光 5% 相当、残り 0.6 段
    tight.lock_exposure(tight._picam2)
    warning = tight.exposure_warning()
    assert warning and "0.6 段" in warning


def test_the_gain_ceiling_comes_from_the_camera_not_a_guess():
    """上限はセンサーに聞く (IMX708 は 16.0 だが、機種が変われば変わる)。"""
    cam = Camera(picam2=FakePicam2(32700, 4.0, max_gain=8.0))
    cam.lock_exposure(cam._picam2)
    assert cam.max_analogue_gain == 8.0
    assert abs(cam.headroom_stops() - 1.0) < 1e-9


def test_nothing_is_recorded_until_a_lock_happens():
    cam = Camera(picam2=FakePicam2())
    assert cam.exposure_time_us is None and cam.analogue_gain is None
    assert cam.exposure_warning() is None and cam.headroom_stops() == 0.0


# --- 手動で露出を固定する (#87、黒い床の白飛び対策) ------------------------

def test_a_forced_exposure_overrides_what_ae_chose():
    """**黒い床では AE を信じない選択肢が要る。**

    視野の大半が黒い床だと AE が開き、明るい壁上面が白飛びして彩度が落ちる
    (#56 は S=46-54 まで落ちて壁を見落とし、機体が衝突した)。露出を下げる手段が
    無いと打つ手が無くなる。
    """
    fake = FakePicam2(50_000, 12.0)             # AE は開ききっている
    cam = Camera(picam2=fake, exposure_us=8_000, gain=2.0)
    cam.lock_exposure(fake)
    assert cam.exposure_time_us == 8_000 and cam.analogue_gain == 2.0
    assert fake.controls["ExposureTime"] == 8_000
    assert fake.controls["AnalogueGain"] == 2.0


def test_forcing_only_one_of_them_leaves_the_other_to_ae():
    cam = Camera(picam2=FakePicam2(50_000, 12.0), exposure_us=8_000)
    cam.lock_exposure(cam._picam2)
    assert cam.exposure_time_us == 8_000        # 指定した方
    assert cam.analogue_gain == 12.0            # AE の判断のまま


def test_camera_args_round_trip():
    """引数ヘルパーが :class:`Camera` の引数へそのまま渡せること。"""
    import argparse

    from krilly.hal.camera import add_camera_args, camera_kwargs

    p = argparse.ArgumentParser()
    add_camera_args(p)
    assert camera_kwargs(p.parse_args([])) == {
        "max_frame_duration_us": 33_300, "ae_constraint": None,
        "exposure_value": None, "exposure_us": None, "gain": None,
    }
    got = camera_kwargs(p.parse_args(
        ["--max-frame-duration", "100", "--ae-constraint", "Highlight",
         "--exposure", "8", "--gain", "2.5"]))
    assert got == {"max_frame_duration_us": 100_000, "ae_constraint": "Highlight",
                   "exposure_value": None, "exposure_us": 8_000, "gain": 2.5}
    Camera(picam2=FakePicam2(), **got)          # 受け取れること
