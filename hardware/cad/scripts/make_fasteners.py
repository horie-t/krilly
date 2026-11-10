"""M3 / M2.5 の締結部品（六角スペーサー・ナット・全ネジ）の外形モデルを作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_fasteners.py </dev/null

出力（hardware/cad/ と export/ に .FCStd / .step）:
    m3-hex-spacer-10 … シャーシ天板 ↔ X-UPS1（4 本）
    m3-hex-spacer-15 … ミドルプレートの裏 ↔ L6470 ドライバ基板（2 本 × 3 枚）
    m3-nut           … 全ネジをシャーシ天板とミドルプレートの上下から、ルーフの下から
                       ダブルナットで留める
    m3-nylon-nut     … 全ネジをルーフの上から留める（4 個）。厚さ 4.0（実物と合っている）
    m3-rod-350       … シャーシ ↔ ミドルプレート（から上の段）をつなぐ全ネジ（4 本）
    m25-hex-spacer-5 … ミドルプレート ↔ Raspberry Pi 5（4 本）。二面幅 4.5（実物）
    m25-hex-spacer-11 … Raspberry Pi 5 ↔ 中継基板 HAT（3 本）。二面幅 4.5

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
NYLON_NUT_T = 4.0   # M3 ナイロンナットの厚さ（一般的な値。実物と合っている）
ROD_DIA = 3.0       # 全ネジの呼び径
M25_HEX_S = 4.5     # M2.5 スペーサーの二面幅（実物）
M25_TAP_DIA = 2.05  # M2.5 の下穴


def hex_with_hole(length, hole=TAP_DIA, s=HEX_S):
    r = s / math.sqrt(3)
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
    "m3-nylon-nut": ("M3_nylon_nut", "Nut", lambda: hex_with_hole(NYLON_NUT_T, ROD_DIA)),
    "m3-rod-350": ("M3_rod_350", "Rod", lambda: Part.makeCylinder(ROD_DIA / 2, 350.0)),
    "m25-hex-spacer-5": ("M2.5_hex_spacer_5", "Spacer",
                         lambda: hex_with_hole(5.0, M25_TAP_DIA, M25_HEX_S)),
    "m25-hex-spacer-11": ("M2.5_hex_spacer_11", "Spacer",
                          lambda: hex_with_hole(11.0, M25_TAP_DIA, M25_HEX_S)),
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
