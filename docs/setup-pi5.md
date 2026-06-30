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

## 3. UART を有効化 (BNO055)

BNO055 は I2C クロックストレッチで Pi と相性が悪いため **UART モード**を使う。

```bash
sudo raspi-config   # Interface Options -> Serial Port
#   ログインシェル: No / シリアルHW: Yes
# /boot/firmware/config.txt:
#   enable_uart=1
```

- センサ側 PS1 を High にして UART を選択し、`/dev/serial0` に接続。
- Python からは `adafruit-circuitpython-bno055` + `pyserial` で読む。

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
