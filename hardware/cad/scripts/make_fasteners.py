"""M3 の締結部品（六角スペーサー・ナット・全ネジ）の外形モデルを作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_fasteners.py </dev/null

出力（hardware/cad/ と export/ に .FCStd / .step）:
    m3-hex-spacer-10 … シャーシ天板 ↔ X-UPS1（4 本）
    m3-hex-spacer-15 … ミドルプレートの裏 ↔ L6470 ドライバ基板（2 本 × 3 枚）
    m3-nut           … 全ネジをシャーシ天板とミドルプレートの上下から留める（8 個 × 2）
    m3-rod-350       … シャーシ ↔ ミドルプレート（から上の段）をつなぐ全ネジ（4 本）

寸法は実物で確認済み（スペーサーの高さと二面幅 5.5、全ネジの長さ 350）。ナットは
JIS の M3（二面幅 5.5、厚さ 2.4）。ねじ山はモデル化せず、雌ねじは下穴、全ネジは
呼び径の丸棒で表す（ナットの穴も呼び径にして、全ネジと重ならないようにする）。

座標系（どれも）: 軸 = +Z、Z=0 = 下端。
"""

import math
import os

import FreeCAD as App
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.dirname(HERE)

HEX_S = 5.5         # スペーサー・ナットの二面幅（実物 / JIS）
TAP_DIA = 2.5       # M3 の下穴
NUT_T = 2.4         # M3 ナットの厚さ（JIS）
ROD_DIA = 3.0       # 全ネジの呼び径


def hex_with_hole(length, hole=TAP_DIA):
    r = HEX_S / math.sqrt(3)
    pts = [App.Vector(r * math.cos(math.radians(60 * i)), r * math.sin(math.radians(60 * i)), 0)
           for i in range(6)]
    solid = Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(App.Vector(0, 0, length))
    return solid.cut(Part.makeCylinder(hole / 2, length))


PARTS = {
    # ファイル名: (App::Part のラベル, 中のオブジェクト名, 形状を作る関数)
    "m3-hex-spacer-10": ("M3_hex_spacer_10", "Spacer", lambda: hex_with_hole(10.0)),
    "m3-hex-spacer-15": ("M3_hex_spacer_15", "Spacer", lambda: hex_with_hole(15.0)),
    # ナットの穴は全ネジ（呼び径の丸棒）と同じ径にする。下穴径だと干渉チェックで食い合う
    "m3-nut": ("M3_nut", "Nut", lambda: hex_with_hole(NUT_T, ROD_DIA)),
    "m3-rod-350": ("M3_rod_350", "Rod", lambda: Part.makeCylinder(ROD_DIA / 2, 350.0)),
}


def main():
    for name, (label, obj_name, make) in PARTS.items():
        solid = make()
        doc = App.newDocument(name.replace("-", "_"))
        part = doc.addObject("App::Part", "Part")
        part.Label = label
        o = doc.addObject("Part::Feature", obj_name)
        o.Shape = solid
        part.addObject(o)
        doc.recompute()
        path = os.path.join(CAD_DIR, f"{name}.FCStd")
        doc.saveAs(path)
        Part.export([o], os.path.join(CAD_DIR, "export", f"{name}.step"))
        print(f"saved {path}  vol {solid.Volume:.1f}")
        App.closeDocument(doc.Name)


main()
