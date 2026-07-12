# hardware/cad/export/

`hardware/cad/*.FCStd` から書き出した **STL / STEP** 等の共有データ置き場。

## 方針
- ファイル名は元の部品名に合わせる（例: `krilly-chassis.stl`, `krilly-chassis.step`）。
- STL は 3D プリント用。単位は **mm**、座標系は元の FreeCAD ファイルに準拠
  （[`docs/coordinate-frames.md`](../../../docs/coordinate-frames.md): +x前 / +y左 / +z上）。
- STEP は他 CAD との受け渡し用。
- これらは `.FCStd` から**再生成できる派生物**なので、更新時は元ファイルも
  必ず一緒にコミットすること。

## 3Dプリンタ設定

対象機は ELEGOO Mars 2 での印刷の設定。ELEGOO 光造形3Dプリンター用 水洗い樹脂 UVレジンを使用。

* レイヤーの高さ: 0.05mm
* 初期層の数: 10
* 露光時間: 5s
* 初期層の露光時間: 150s
* 消灯遅延: 7s
* 初期層消灯遅延: 8s
* 初期層リフト距離: 5mm
* リフト高さ: 5mm
* 初期層上昇速度: 30mm/s
* 上昇速度: 50mm/s
* リトラクト速度: 150mm/s

## 注意
- STL/STEP はバイナリで大きくなりやすい。版が増える場合は Git LFS を検討する。
