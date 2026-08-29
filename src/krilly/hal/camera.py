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

    #: フレーム間隔の上限 [us]。**露出時間はこれを超えられない。**
    #:
    #: picamera2 の既定は 30fps 固定 (33333us) で、暗いところでは AE がここで頭打ちに
    #: なり、あとはゲインだけで凌ぐことになる (#78 実測: 調光 100/30/5% で露出は
    #: 32.7ms のまま、ゲインが 2.16 → 6.21 → 10.67 と上がり、上限 16.0 まで残り
    #: 0.6 段だった)。伸ばせば露出で稼げるが、**買えるのは 0.6 段だけ**である。
    #:
    #: 実測 (同一照明、この上限だけを変えた): 33.3ms → 露出 32.7ms / ゲイン 3.51、
    #: **100ms → 露出 50.0ms / ゲイン 2.29**、200ms と 400ms も 50.0ms / 2.29 で
    #: 変わらない。libcamera の AE が 50ms で露出を打ち切り、その先はゲインに回すため。
    #: ``AeExposureMode=Long`` にしても 57.4ms (さらに 0.2 段) にしかならない。
    #: **したがって 100 より大きい値に意味は無く、それ以上暗ければ照明を足すしかない。**
    #:
    #: 代償は 1 停止あたりの待ち時間だけである。壁の撮影は必ず**停止中**に行われる
    #: (移動 → 惰行 → 撮影) ので、走行中のブレは判定に入らない。しかもこれは上限
    #: なので、明るいところでは AE が短い露出を選び、フレーム間隔は最短のまま
    #: — 暗いときにしか払わない。
    #:
    #: **既定は据え置き**にしてある。伸ばすと暗所でのみ挙動が変わるので、会場が
    #: 暗くて警告が出たときに明示的に上げること (上げるなら 100)。
    DEFAULT_MAX_FRAME_DURATION_US = 33_333

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
        max_frame_duration_us: int = DEFAULT_MAX_FRAME_DURATION_US,
        picam2=None,
    ) -> None:
        #: ロックした露出時間 [us] とアナログゲイン、およびゲインの上限 (#78)。
        #:
        #: **照明が変わったときに実際に動くのはゲインだけである。** 自動露出は明るさを
        #: 打ち消すので、フレームを見ても照明の違いは分からない — 実測 (調光 100/30/5%、
        #: 照度にして 20 倍): 画面の明るさ V は 157/156/156、壁の赤割合は
        #: 0.401/0.402/0.401 と動かず、**露出は 32.7ms で 3 条件とも同じ**、
        #: ゲインだけが **2.16 → 6.21 → 10.67** と上がっていた。
        #:
        #: 露出が動かないのは ``create_video_configuration`` の既定が 30fps 固定
        #: (``FrameDurationLimits`` = 33333us) だから。つまりこのカメラの暗さへの
        #: 対抗手段は**ゲインの 1.12〜16.0 倍、14 倍ぶんしかない**。
        #: :meth:`headroom_stops` がその残りを段数で返す。
        self.exposure_time_us: int | None = None
        self.analogue_gain: float | None = None
        self.max_analogue_gain: float = 16.0
        self.max_frame_duration_us: int = max(max_frame_duration_us,
                                              self.DEFAULT_MAX_FRAME_DURATION_US)
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
            # FrameDurationLimits は (最短, 最長)。最短を既定のままにするので、
            # 明るいところでの挙動は変わらない (AE が短い露出を選べば 30fps)。
            config = picam2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls={"FrameDurationLimits":
                          (self.DEFAULT_MAX_FRAME_DURATION_US,
                           max(max_frame_duration_us,
                               self.DEFAULT_MAX_FRAME_DURATION_US))},
                **sensor
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
        controls = getattr(picam2, "camera_controls", None) or {}
        if "AnalogueGain" in controls:
            self.max_analogue_gain = float(controls["AnalogueGain"][1])
        picam2.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": self.exposure_time_us,
            "AnalogueGain": self.analogue_gain,
        })
        stops = self.headroom_stops()
        log.info("露出をロック: %.1fms%s / ゲイン %.2f (上限 %.1f、残り %.1f 段)%s",
                 self.exposure_time_us / 1000.0,
                 " (フレーム間隔の上限)" if self.exposure_is_capped() else "",
                 self.analogue_gain, self.max_analogue_gain, stops,
                 self.exposure_warning() or "")

    def exposure_is_capped(self) -> bool:
        """露出がフレーム間隔の上限に張り付いているか。

        張り付いていれば、そこから先の暗さはすべてゲインが背負う。
        ``max_frame_duration_us`` を伸ばせば露出で稼げる余地がまだあるという印。
        """
        if self.exposure_time_us is None:
            return False
        return self.exposure_time_us >= 0.95 * self.max_frame_duration_us

    #: ゲインの残りがこれを下回ったら警告する [段] (1 段 = 明るさ半分)。
    #:
    #: **暗さに対する余力はゲインでしか測れない** — 露出は 30fps 固定で頭打ちなので
    #: 動かない。残り 1 段とは「いまの半分の明るさで上限に達する」という意味で、
    #: そこから先は画面が暗くなり、赤の彩度も色相もしきい値も一斉に崩れる。
    #:
    #: 実測の位置 (この部屋、6500K): 調光 100% でゲイン 2.16 = 残り 2.9 段、
    #: 30% で 6.21 = 1.4 段、5% で 10.67 = **0.6 段**。5% でも赤割合は 0.401 と
    #: 0.453 で健全だったので、**ゲイン 10.7 のノイズはマスクを壊さない**。
    #: 壊れるのは余力を使い切った先である。
    LOW_HEADROOM_STOPS = 1.0

    def headroom_stops(self) -> float:
        """あと何段暗くなってもフレームの明るさを保てるか (ロック前は 0.0)。"""
        if not self.analogue_gain or self.analogue_gain <= 0:
            return 0.0
        import math

        return max(0.0, math.log2(self.max_analogue_gain / self.analogue_gain))

    def exposure_warning(self) -> str | None:
        """暗さへの余力が乏しいときの注意書き (問題なければ None)。"""
        if self.analogue_gain is None:
            return None
        stops = self.headroom_stops()
        if stops >= self.LOW_HEADROOM_STOPS:
            return None
        lever = ("--max-frame-duration を伸ばして露出で稼ぐこと (いま %.0fms)"
                 % (self.max_frame_duration_us / 1000.0)
                 if self.exposure_is_capped() else "照明を足すこと")
        return (" ** 暗すぎる。ゲインの余力があと %.1f 段しかない。これ以上暗いと"
                "画面が暗くなり、赤の判定が一斉に崩れる。%s **" % (stops, lever))

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
