# hardware/cad/

機体全体の **FreeCAD** モデル置き場。`krilly-assembly.FCStd` がシャーシ・モータ・ホイール・
X-UPS1・ドライバ基板・Raspberry Pi 5・HAT・ルーフ・カメラを組んだアセンブリ。

## まず読むもの

- [`export/assembly.md`](export/assembly.md) — アセンブリの要約（各部品の位置、取付穴の一致、干渉、
  余裕の小さいすき間、主な寸法）。`make_assembly.py` が書き出す
- `export/krilly-{chassis,middle-plate,roof}-describe.md` — GUI で作った板 3 枚の寸法。
  `describe_fcstd.py` が書き出す
- [`docs/claude/hardware-cad.md`](../../docs/claude/hardware-cad.md) — 経緯・注意点・未解決の問題

`.FCStd` はバイナリ（zip）なので差分も中身も見えない。**FCStd を直したら、上のテキストも作り直して
一緒にコミットする。**

## ファイル

- GUI で作った部品: `krilly-chassis` / `krilly-middle-plate` / `krilly-roof`（と `krilly-roller-pin`）
- スクリプトで作った部品とアセンブリ: [`scripts/`](scripts/) の `make_*.py` が作る。生成順と役割は
  [`scripts/README.md`](scripts/README.md)
- [`export/`](export/) — STEP / STL / 三面図と、上のテキストの要約
- `vendor/` — 再配布できないメーカーのデータ（`.gitignore` 済み。取得手順は `scripts/README.md`）

## 規約

- 単位は **mm**。
- 座標系は [`docs/coordinate-frames.md`](../../docs/coordinate-frames.md) に準拠
  （**+x 前 / +y 左 / +z 上**、原点は車体中心）。ホイール位置・モーター番号
  （M0 前方 / M1 後左 / M2 後右）もこの図に合わせる。
- 主要寸法（ホイール Ø48mm・幅25.5mm、中心↔接地点 `L` など）は
  `src/krilly/config/robot.yaml` と整合させる。
- バックアップ・一時ファイル（`*.FCStd1`, `*.FCBak`, ロックファイル）は `.gitignore` 済み。
