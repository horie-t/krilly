"""Raspberry Pi Camera Module V3 を picamera2 経由で扱う (issue #7)。

Pi 5 では ``cv2.VideoCapture`` が libcamera スタックで動作しないため、
**picamera2** を使い、フレームを OpenCV 向けの NumPy 配列として取得する。
下向きの壁検出では、低解像度・高 fps で、さらに **露出 / AWB をロック** したい。
そうすることで、ロボットの移動中も赤の HSV しきい値が安定して保たれる。

チャンネル順の落とし穴: picamera2 の ``"RGB888"`` フォーマットは、バイト列が
**B, G, R** の順に並んだ配列を返す。つまり OpenCV から見ればすでに BGR なので、
``capture()`` は変換なしで BGR フレームをそのまま返す。特定の環境で色が入れ替わって
見える場合は、呼び出し側で R/B を入れ替えること。
"""

from __future__ import annotations


class Camera:
    """OpenCV 向けに BGR フレームを返す Pi カメラのラッパー。

    ``picam2`` はテスト用に注入できる (``capture_array`` / ``stop`` /
    ``close`` を持つオブジェクト)。注入しない場合は ``Picamera2`` を開いて開始する。
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        lock_awb_exposure: bool = True,
        picam2=None,
    ) -> None:
        if picam2 is None:
            import time

            from picamera2 import Picamera2

            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            if lock_awb_exposure:
                time.sleep(0.5)  # 自動露出 / ホワイトバランスが落ち着くのを待つ
                meta = picam2.capture_metadata()
                picam2.set_controls({
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ExposureTime": int(meta.get("ExposureTime", 8000)),
                    "AnalogueGain": float(meta.get("AnalogueGain", 1.0)),
                })
        self._picam2 = picam2

    def capture(self):
        """最新のフレームを BGR の NumPy 配列として返す (チャンネルに関する注意を参照)。"""
        return self._picam2.capture_array()

    def close(self) -> None:
        try:
            self._picam2.stop()
        finally:
            self._picam2.close()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
