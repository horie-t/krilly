"""ハードウェア抽象化レイヤ。

各ペリフェラルは、小さく個別にテスト可能なクラスでラップする:

    l6470  — SPI0 デイジーチェーン (単一 CS)、mode 3 で接続した 3 個のステッパドライバ
    imu    — UART モード (/dev/serial0) の BNO055 9軸 IMU
    camera — picamera2 経由の Pi Camera Module V3 -> OpenCV 用の NumPy 配列

マイルストーン M1 (issues #5-#7) で段階的に実装する。
"""
