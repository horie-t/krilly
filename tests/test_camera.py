"""カメラの露出ロックのテスト (issue #78)。

**実機なしで固定したいのは「ロックした値を覚えているか」だけ。** 照明が変わっても
自動露出が明るさを打ち消すので、フレームを見ても照明の違いは分からない (実測: 照度
20 倍の差で画面の明るさ V は 157 対 156、赤割合は 0.401 対 0.401)。動いていたのは
露出時間で、それを記録しない限り AE の余力がどれだけ残っているかは誰にも分からない。
"""

from krilly.hal.camera import Camera


class FakePicam2:
    """``capture_metadata`` / ``set_controls`` だけを持つ最小のカメラ。"""

    def __init__(self, exposure_us=8000, gain=1.0):
        self.meta = {"ExposureTime": exposure_us, "AnalogueGain": gain}
        self.controls = None

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


def test_a_long_exposure_warns_because_it_costs_time_at_every_stop():
    """**警告の理由はブレではなく時間予算。** 壁の撮影は必ず停止中に行われる
    (移動 -> 惰行 -> 撮影) ので、走行中のブレは判定に入らない。効くのは 1 停止
    あたりの所要時間で、探索ランは 20 回止まる。長い露出は AE の余力が尽きかけて
    いる印でもある。"""
    quick = Camera(picam2=FakePicam2(8000))
    quick.lock_exposure(quick._picam2)
    assert quick.exposure_warning() is None

    slow = Camera(picam2=FakePicam2(120000))
    slow.lock_exposure(slow._picam2)
    warning = slow.exposure_warning()
    assert warning and "120ms" in warning and "2.4s" in warning


def test_nothing_is_recorded_until_a_lock_happens():
    cam = Camera(picam2=FakePicam2())
    assert cam.exposure_time_us is None and cam.analogue_gain is None
    assert cam.exposure_warning() is None
