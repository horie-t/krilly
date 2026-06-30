"""Krilly — holonomic (3 omni-wheel) Micromouse for Raspberry Pi 5.

Layered architecture (low → high level), see the project plan:

    hal          ハードウェア抽象化 (L6470 / BNO055 / camera)
    kinematics   kiwi-drive 正逆運動学・輪速⇄ステップ変換
    motion       運動制御 (速度指令 → 3輪, 加減速ランプ)
    localization 自己位置推定 ([X, Y, phi])
    perception   カメラ赤壁検出
    solver       迷路モデル(16x16) + flood-fill + 最短経路
    strategy     探索ラン / 最速ランの切替
    app          メインアプリ・状態機械
    config       車体・迷路寸法 / チューニング定数
"""

__version__ = "0.0.1"
