"""M3 六角スペーサー（長さ 10）の外形モデルを作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_spacer.py </dev/null

出力: hardware/cad/m3-hex-spacer-10.FCStd、hardware/cad/export/m3-hex-spacer-10.step

シャーシ天板と X-UPS1 の間に 4 本使う。二面幅は M3 用の一般的な値で、実物とは未照合。
ねじ山はモデル化せず、両端からの雌ねじを貫通した下穴で表す。

座標系: 軸 = +Z、Z=0 = 下端、Z=LENGTH = 上端。
"""

import math
import os

import FreeCAD as App
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.dirname(HERE)
NAME = "m3-hex-spacer-10"

LENGTH = 10.0       # 高さ 10（実物）
HEX_S = 5.5         # 二面幅（M3 スペーサーの一般的な値。仮）
TAP_DIA = 2.5       # M3 の下穴


def main():
    r = HEX_S / math.sqrt(3)
    pts = [App.Vector(r * math.cos(math.radians(60 * i)), r * math.sin(math.radians(60 * i)), 0)
           for i in range(6)]
    solid = Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(App.Vector(0, 0, LENGTH))
    solid = solid.cut(Part.makeCylinder(TAP_DIA / 2, LENGTH))

    doc = App.newDocument(NAME.replace("-", "_"))
    part = doc.addObject("App::Part", "Part")
    part.Label = "M3_hex_spacer_10"
    o = doc.addObject("Part::Feature", "Spacer")
    o.Shape = solid
    part.addObject(o)
    doc.recompute()
    path = os.path.join(CAD_DIR, f"{NAME}.FCStd")
    doc.saveAs(path)
    Part.export([o], os.path.join(CAD_DIR, "export", f"{NAME}.step"))
    print(f"saved {path}  vol {solid.Volume:.1f}")


main()
