# krilly

Krilly はオムニホイールを備えたマイクロマウスです。Raspberry Pi 5 上で動作する
**ホロノミック（3輪オムニ・kiwi ドライブ）** マイクロマウスで、クラシック 16×16
マイクロマウス競技への出場を目標としています。

## ハードウェア

- Raspberry Pi 5
- オムニホイール 3個（Ø48 mm・幅 25.5 mm）
- ステッピングモーター 3個（ステップ角 1.8°）＋ L6470 ドライバ 3個（SPI デイジーチェーン）
- BNO055 9軸センサー（UART モード）
- Raspberry Pi カメラモジュール V3（広角）— 下向きに取り付け、迷路の**赤い壁上面**を検知

## アーキテクチャ

下位 → 上位のレイヤ構成（`src/krilly/` を参照）:

| レイヤ | 役割 |
|-------|------|
| `hal` | L6470（SPI）/ BNO055（UART）/ カメラ（picamera2） |
| `kinematics` | kiwi ドライブの正逆運動学、輪速 ⇄ ステップレート変換 |
| `motion` | ボディ速度 (vx, vy, ω) → 3輪、加減速ランプ |
| `localization` | デッドレコニング + ジャイロ姿勢 + カメラ格子補正 `[X, Y, φ]` |
| `perception` | 赤い壁上面の検出（HSV） |
| `solver` | 迷路モデル（16×16・共有エッジ）+ flood-fill + 最短経路 |
| `strategy` / `app` | 探索ラン / 最速ランの状態機械 |
| `config` | 車体・迷路の寸法（YAML） |

車体座標系・ホイール(モーター)番号・BNO055 取り付け向きの定義は
[`docs/coordinate-frames.md`](docs/coordinate-frames.md) を参照。

## 開発

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Raspberry Pi 5 の実機セットアップは [`docs/setup-pi5.md`](docs/setup-pi5.md) を参照してください。

進捗は [GitHub Project ボード](https://github.com/users/horie-t/projects/4) で
Milestone（M0〜M6）と Issue により管理し、コードはレビュー済みの PR を通じてマージします。

## ライセンス

MIT — [LICENSE](LICENSE) を参照。
