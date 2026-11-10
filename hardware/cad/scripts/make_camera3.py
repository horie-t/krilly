"""Raspberry Pi Camera Module 3 Wide の簡易外形モデルを作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_camera3.py </dev/null

出力: hardware/cad/camera-module-3-wide.FCStd、hardware/cad/export/camera-module-3-wide.step

寸法は公式の機械図面 camera-module-3-wide-mechanical-drawing.pdf による。干渉チェック用に
次だけを作る:
    - Board     : 25 × 23.862 × 1.12、角 R2、取付穴 ø2.2 × 4（21 × 12.5）
    - Housing   : レンズの角形ホルダー 10.8 × 10.8（基板の表から 4.07）
    - Lens      : レンズの鏡筒 ø6.95（先端は基板の表から 8.3）
    - Connector : 裏面の FPC コネクタ 19.61 × 5.71（上辺に寄せる）
画角は水平 102° / 垂直 67°（図面の値。モデルには入れない）。

図面の側面図は数値どうしが少し合わない（全長 12 と 8.3 + 1.12 + 2.75 の差 0.17）。
コネクタの厚さは干渉チェックで安全側になるよう大きいほうの 2.75 を採った。

座標系: レンズ側から見た図面の向き。原点 = 基板の左下の角、+X 右、+Y 上
（コネクタのある辺が上）、+Z = 光軸の向き（レンズ側）。基板の表（レンズ側）が Z=0、
裏が Z=−1.12。光軸の中心は (12.5, 14.4)。
"""

import os

import FreeCAD as App
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.dirname(HERE)
NAME = "camera-module-3-wide"

BOARD_W, BOARD_H, BOARD_T = 25.0, 23.862, 1.12
CORNER_R = 2.0                     # 図面に数値なし（図からの読み取り）
HOLE_DIA = 2.2
HOLES = [(2.0, 2.0), (23.0, 2.0), (2.0, 14.5), (23.0, 14.5)]
AXIS_XY = (12.5, 14.4)             # 光軸の中心（左右の中央、下辺から 14.4）
HOUSING = 10.8                     # 角形ホルダーの一辺
HOUSING_H = 4.07                   # 基板の表から
LENS_DIA = 6.95
LENS_TIP = 8.3                     # レンズの先端（基板の表から）
CONN_W, CONN_H = 19.61, 5.71       # 裏面の FPC コネクタ（幅・高さ）
CONN_T = 2.75                      # 裏面からの厚さ


def rounded_rect(w, h, r):
    v = App.Vector
    z = v(0, 0, 1)
    edges = [
        Part.LineSegment(v(r, 0, 0), v(w - r, 0, 0)).toShape(),
        Part.makeCircle(r, v(w - r, r, 0), z, 270, 360),
        Part.LineSegment(v(w, r, 0), v(w, h - r, 0)).toShape(),
        Part.makeCircle(r, v(w - r, h - r, 0), z, 0, 90),
        Part.LineSegment(v(w - r, h, 0), v(r, h, 0)).toShape(),
        Part.makeCircle(r, v(r, h - r, 0), z, 90, 180),
        Part.LineSegment(v(0, h - r, 0), v(0, r, 0)).toShape(),
        Part.makeCircle(r, v(r, r, 0), z, 180, 270),
    ]
    return Part.Face(Part.Wire(edges))


def main():
    board = rounded_rect(BOARD_W, BOARD_H, CORNER_R).extrude(App.Vector(0, 0, BOARD_T))
    board.translate(App.Vector(0, 0, -BOARD_T))
    for x, y in HOLES:
        board = board.cut(Part.makeCylinder(HOLE_DIA / 2, BOARD_T + 2, App.Vector(x, y, -BOARD_T - 1)))

    ax, ay = AXIS_XY
    housing = Part.makeBox(HOUSING, HOUSING, HOUSING_H,
                           App.Vector(ax - HOUSING / 2, ay - HOUSING / 2, 0))
    lens = Part.makeCylinder(LENS_DIA / 2, LENS_TIP - HOUSING_H, App.Vector(ax, ay, HOUSING_H))
    conn = Part.makeBox(CONN_W, CONN_H, CONN_T,
                        App.Vector((BOARD_W - CONN_W) / 2, BOARD_H - CONN_H, -BOARD_T - CONN_T))

    parts = [("Board", board), ("Housing", housing), ("Lens", lens), ("Connector", conn)]
    doc = App.newDocument(NAME.replace("-", "_"))
    part = doc.addObject("App::Part", "Part")
    part.Label = "CameraModule3Wide"
    objs = []
    for label, shp in parts:
        o = doc.addObject("Part::Feature", label)
        o.Shape = shp
        part.addObject(o)
        objs.append(o)
    doc.recompute()
    path = os.path.join(CAD_DIR, f"{NAME}.FCStd")
    doc.saveAs(path)
    Part.export(objs, os.path.join(CAD_DIR, "export", f"{NAME}.step"))

    bb = Part.makeCompound([s for _, s in parts]).BoundBox
    print(f"saved {path}")
    print(f"bbox X {bb.XMin:.2f}..{bb.XMax:.2f}  Y {bb.YMin:.2f}..{bb.YMax:.2f}  "
          f"Z {bb.ZMin:.2f}..{bb.ZMax:.2f}")
    worst = max(parts[i][1].common(parts[j][1]).Volume
                for i in range(len(parts)) for j in range(i + 1, len(parts)))
    print(f"max overlap between parts {worst:.4f}")


main()
