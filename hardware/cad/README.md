# hardware/cad/

筐体（シャーシ）の **FreeCAD** 設計ファイル置き場。

## 方針
- 設計本体の **`*.FCStd`** をコミットする（例: `krilly-chassis.FCStd`）。
- バックアップ・一時ファイル（`*.FCStd1`, `*.FCBak`, ロックファイル）は
  `.gitignore` 済み。
- 書き出した中間データ（STL/STEP 等）は [`export/`](export/) に置く。

## 規約
- 単位は **mm**。
- 座標系は [`docs/coordinate-frames.md`](../../docs/coordinate-frames.md) に準拠
  （**+x 前 / +y 左 / +z 上**、原点は車体中心）。ホイール位置・モーター番号
  （M0 前方 / M1 後左 / M2 後右）もこの図に合わせる。
- 主要寸法（ホイール Ø48mm・幅25.5mm、中心↔接地点 `L` など）は
  `src/krilly/config/robot.yaml` と整合させる。

## 注意
- `.FCStd` はバイナリ（zip 形式）なので Git 上で差分は取りにくい。ファイルが
  大きくなる/版が増える場合は Git LFS の導入を検討する。
