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

from krilly.logging_config import get_logger

log = get_logger("krilly.camera")


class Camera:
    """OpenCV 向けに BGR フレームを返す Pi カメラのラッパー。

    ``picam2`` はテスト用に注入できる (``capture_array`` / ``stop`` /
    ``close`` を持つオブジェクト)。注入しない場合は ``Picamera2`` を開いて開始する。
    """

    #: 全画素を読めるセンサーモード。IMX708 の 1536x864 モードは ``crop_limits`` が
    #: (768, 432, 3072, 1728) で**それ自体が中央 67% の切り出し**なので、これを選ばないと
    #: 画角が狭いままになる (2304x1296 の ``crop_limits`` は全画素 4608x2592)。
    FULL_FOV_SENSOR = (2304, 1296)

    #: 既定の撮影サイズと全画素モード (#88)。**壁判定の ROI はこのサイズで校正されて
    #: いる** (``perception.wall_detect.DEFAULT_FRAME_SIZE``)。片方だけ変えると ROI が
    #: 帯から外れて壁を見落とすので、必ず対で変えること
    #: (``WallDetector.measure`` が実フレームと突き合わせて検算する)。
    #:
    #: 960x720 + 全画素モードにすると、640x480 に対して**画角が 1.5 倍・分解能は据え置き**
    #: (px/mm 1.70)。画像処理は 3.4ms -> 6.0ms しか増えず、1 セルの停止 0.44s に対して
    #: 無視できる。
    DEFAULT_SIZE = (960, 720)

    def __init__(
        self,
        width: int = DEFAULT_SIZE[0],
        height: int = DEFAULT_SIZE[1],
        lock_awb_exposure: bool = True,
        full_fov: bool = True,
        picam2=None,
    ) -> None:
        #: ロックした露出時間 [us] とアナログゲイン (ロックしなければ None)。
        #: **照明が変わったときに実際に動くのはここ** (#78)。自動露出は明るさを
        #: 打ち消してしまうので、フレームを見ても照明が変わったことは分からない —
        #: 実測では照度 20 倍の差 (調光 100% と 5%) で画面の明るさ V は 157 と 156、
        #: 壁の赤割合は 0.401 と 0.401 でまったく動かず、露出時間だけが伸びていた。
        #: **AE の余力がどれだけ残っているかは、この 2 つを見ないと分からない。**
        self.exposure_time_us: int | None = None
        self.analogue_gain: float | None = None
        if picam2 is None:
            import time

            from picamera2 import Picamera2

            picam2 = Picamera2()
            # picamera2 は要求サイズから勝手にセンサーモードを選ぶ。640x480 を頼むと
            # 1536x864 モード + 4:3 への切り出しになり、**センサー面積の 33% しか
            # 使わない** (横 50% x 縦 67%)。full_fov で全画素モードを明示すると
            # 縦横とも 1.5 倍の画角になる。分解能を保つには出力も 1.5 倍にすること
            # (960x720 なら px/mm は据え置きで画角だけ広がる)。
            sensor = ({"sensor": {"output_size": self.FULL_FOV_SENSOR, "bit_depth": 10}}
                      if full_fov else {})
            config = picam2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}, **sensor
            )
            picam2.configure(config)
            picam2.start()
            if lock_awb_exposure:
                time.sleep(0.5)  # 自動露出 / ホワイトバランスが落ち着くのを待つ
                self.lock_exposure(picam2)
        self._picam2 = picam2

    def lock_exposure(self, picam2) -> None:
        """いまの露出 / AWB を固定し、その値を記録して表示する。

        **1 回の走行は「起動した瞬間の光」に焼き付けられる。** 照明が変わったら
        撮り直しではなくスクリプトの再起動が要る (#78)。会場では、スタートセルに
        置いて会場の照明の下で起動すること — 廊下で起動して運び込むのは別の光での
        ロックになる。
        """
        meta = picam2.capture_metadata()
        self.exposure_time_us = int(meta.get("ExposureTime", 8000))
        self.analogue_gain = float(meta.get("AnalogueGain", 1.0))
        picam2.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": self.exposure_time_us,
            "AnalogueGain": self.analogue_gain,
        })
        log.info("露出をロック: %.1fms / ゲイン %.2f%s",
                 self.exposure_time_us / 1000.0, self.analogue_gain,
                 self.exposure_warning() or "")

    #: 露出時間がこれを超えたら警告する [us]。根拠は**時間予算**であって像の乱れでは
    #: ない — 壁の撮影は必ず停止中に行われる (``search_run`` は移動→惰行→撮影の順)
    #: ので、走行中のブレは判定に入らない。効くのは 1 停止あたりの所要時間で、
    #: 探索ランは 20 回止まるから、露出 100ms は 7 分の持ち時間から 2 秒を削る。
    #: それ以上に効くのは**この値が AE の余力の目安になる**こと: カメラの上限に
    #: 張り付けば、そこから先は暗くなるだけで、赤の彩度も色相も一気に崩れる。
    SLOW_EXPOSURE_US = 50_000

    def exposure_warning(self) -> str | None:
        """露出が長すぎるときの注意書き (問題なければ None)。"""
        if self.exposure_time_us is None or self.exposure_time_us < self.SLOW_EXPOSURE_US:
            return None
        return (" ** 露出が長い。会場が暗いか AE が上限に近い。1 停止ごとに %.0fms を"
                "払う (探索 20 停止で %.1fs) **"
                % (self.exposure_time_us / 1000.0,
                   20 * self.exposure_time_us / 1e6))

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
