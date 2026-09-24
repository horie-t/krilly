# Raspberry Pi 5 セットアップ手順

Krilly を実機(Raspberry Pi 5 / Raspberry Pi OS)で動かすための初期設定。
Pi 5 は GPIO/SPI/I2C/カメラが新しい **RP1 I/O チップ**経由になっている点に注意。

## 1. OS と基本パッケージ

Raspberry Pi OS (Bookworm 以降) を前提とする。

```bash
sudo apt update && sudo apt full-upgrade
# picamera2 / lgpio は OS に同梱・apt 提供
sudo apt install -y python3-picamera2 python3-lgpio python3-opencv i2c-tools
```

> **GPIO ライブラリ**: Pi 5 では `RPi.GPIO` は動作しない(SoC レジスタ直叩きのため)。
> `lgpio` / `gpiozero` を使う。SPI(`spidev`)・I2C のバス自体はカーネルドライバ経由で正常。

## 2. SPI を有効化 (L6470 ×3 デイジーチェーン)

```bash
sudo raspi-config   # Interface Options -> SPI -> Enable
# もしくは /boot/firmware/config.txt に:
#   dtparam=spi=on
```

- 接続: SPI0、**CS 1本**でデイジーチェーン。SPI mode 3 (CPOL=1/CPHA=1)、MSB-first、~5MHz。
- 初期化時に各ドライバの STATUS を読み、UVLO/OCD 等の電源投入フラグをクリアすること。

## 3. I2C を有効化 (BNO055)

BNO055 モジュール(AE-BNO055-BO)は出荷時 **I2C モード(アドレス 0x28)** で、
ジャンパ変更なしに SDA/SCL 配線だけで使える。

```bash
sudo raspi-config   # Interface Options -> I2C -> Enable
# /boot/firmware/config.txt:
#   dtparam=i2c_arm=on
```

- 配線: VIN->3.3V(pin1), GND, SDA->GPIO2(pin3), SCL->GPIO3(pin5)。**電源は3.3V**
  (基板のレベル変換が VIN 電位になるため、5V 給電すると信号も5Vになり Pi を痛める)。
- 確認: `i2cdetect -y 1` で `0x28` が見えること。
- **クロックストレッチ**: 旧 Pi(1〜4) の Broadcom BSC はクロックストレッチのバグが
  あり BNO055 と相性が悪かったが、**Pi 5 は RP1(DesignWare)の I2C に変わり正しく扱える**。
  実機で **100kHz(既定)のまま BNO055 が安定動作することを確認済み**。低速化は不要。
  万一不安定な場合のみ `dtparam=i2c_arm_baudrate=10000` を追加する。
- Python からは `smbus2` で読む (`hal/imu.py`)。ドライバ側でも転送をリトライする。

## 4. カメラ (Camera Module V3 wide)

```bash
# Bookworm では libcamera/picamera2 が標準。接続後に確認:
rpicam-hello --list-cameras
```

- 取得は **picamera2** の `capture_array()` → NumPy → OpenCV。
- `cv2.VideoCapture` は libcamera スタックでは使えない。
- 下向き運用: 低解像度・露出/AWB をロックして赤壁上面を安定検出。

## 5. gpiochip 番号の確認

カーネル更新で RP1 の gpiochip 番号が変わることがある(`gpiochip4` ↔ `gpiochip0`)。

```bash
pinctl 2>/dev/null || gpioinfo | head
```

ライブラリが `gpiochip` を開けないエラーを出す場合は番号のズレを疑う。

## 6. Krilly のインストール

```bash
cd ~/repos/krilly
python3 -m venv --system-site-packages .venv   # picamera2/lgpio を OS から流用
source .venv/bin/activate
pip install -e ".[dev]"
pytest          # M0 スモークテスト
```

> 非Pi の開発マシンでは hardware-only 依存(`spidev`/`lgpio`/`picamera2`)は
> `platform_machine == 'aarch64'` 条件でスキップされ、ロジックの単体テストは実行可能。

## 7. ボタンとブザーで起動する (#79)

会場ではネットワーク越しに操作できないので、電源を入れたらボタン待ちになるようにする。

**配線** (BCM 番号。変えるなら `config/run.yaml` の `button_gpio` / `buzzer_gpio`):

| 部品 | 配線 | 備考 |
|---|---|---|
| タクトスイッチ | GPIO17 (物理ピン 11) と GND の間 | 内部プルアップで読むので抵抗は要らない |
| 圧電ブザー | GPIO18 (物理ピン 12) と GND の間 | 発振回路内蔵 (active) なら `buzzer_passive: false` |
| 非常停止トグル | **VS (モータ電源) の線を直接切る** | ソフトを通らない。SIGKILL で L6470 が回り続けたときの最後の手段 |

SPI0 (GPIO 7-11) と I2C1 (GPIO 2, 3) は使用中なので避ける。

**GND は物理ピン 6 / 9 / 14 / 20 / 25 / 30 / 34 / 39。隣の 22 は GPIO25 で GND ではない。**
共通の GND 線を 20 のつもりで 22 に挿すと、ボタンは押しても反応せず (GPIO17 が Low に
落ちない)、ブザー側の GPIO18 は Low を出しているのに High と読める。
`python -m scripts.launcher --beep-test` がボタンの状態も表示するので、配線を触ったら通すこと。

**入れ方**:

```bash
sudo cp deploy/krilly-launcher.service /etc/systemd/system/   # パスとユーザは環境に合わせる
sudo systemctl daemon-reload
sudo systemctl enable --now krilly-launcher
journalctl -u krilly-launcher -f
```

電源断 (5 秒長押し) は `sudo -n systemctl poweroff` を呼ぶ。Raspberry Pi OS の既定ユーザは
パスワード無しの sudo が通るが、そうでなければ sudoers に
`<ユーザ> ALL=(root) NOPASSWD: /usr/bin/systemctl poweroff` を足す。

**開発中は止めておく**: SSH から `speed_run` などを走らせるとき、ランチャが動いていると
ボタンの GPIO を取り合い、ボタンに触れると走行が起動しうる
(`sudo systemctl stop krilly-launcher`)。
