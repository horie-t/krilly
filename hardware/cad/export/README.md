# hardware/cad/export/

`hardware/cad/*.FCStd` から書き出した **STL / STEP** 等の共有データ置き場。

## 方針
- ファイル名は元の部品名に合わせる（例: `krilly-chassis.stl`, `krilly-chassis.step`）。
- STL は 3D プリント用。単位は **mm**、座標系は元の FreeCAD ファイルに準拠
  （[`docs/coordinate-frames.md`](../../../docs/coordinate-frames.md): +x前 / +y左 / +z上）。
- STEP は他 CAD との受け渡し用。
- これらは `.FCStd` から**再生成できる派生物**なので、更新時は元ファイルも
  必ず一緒にコミットすること。

## 注意
- STL/STEP はバイナリで大きくなりやすい。版が増える場合は Git LFS を検討する。
