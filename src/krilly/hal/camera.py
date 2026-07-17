"""Raspberry Pi Camera Module V3 via picamera2 (issue #7).

On the Pi 5, ``cv2.VideoCapture`` does not work with the libcamera stack — we use
**picamera2** and pull frames as NumPy arrays for OpenCV. For a downward
wall-detector we want low resolution, high fps, and **locked exposure / AWB** so
the red HSV thresholds stay stable while the robot moves.

Channel order gotcha: picamera2's ``"RGB888"`` format yields an array whose bytes
are in **B, G, R** order, i.e. it is already BGR for OpenCV — so ``capture()``
returns a BGR frame directly (no conversion). If colors look swapped on a given
setup, swap R/B at the call site.
"""

from __future__ import annotations


class Camera:
    """Pi camera wrapper returning BGR frames for OpenCV.

    ``picam2`` may be injected (any object with ``capture_array`` / ``stop`` /
    ``close``) for testing; otherwise a ``Picamera2`` is opened and started.
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
                time.sleep(0.5)  # let auto exposure / white balance settle
                meta = picam2.capture_metadata()
                picam2.set_controls({
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ExposureTime": int(meta.get("ExposureTime", 8000)),
                    "AnalogueGain": float(meta.get("AnalogueGain", 1.0)),
                })
        self._picam2 = picam2

    def capture(self):
        """Return the latest frame as a BGR NumPy array (see channel note)."""
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
