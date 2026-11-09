"""SS2421-5041（□42mm 薄型ステッピングモータ、片軸）の外形モデルを作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_ss2421.py

出力:
    hardware/cad/ss2421-5041.FCStd
    hardware/cad/export/ss2421-5041.step

寸法はメーカーカタログ p.44「外形図」による（単位 mm）。公差は中央値を採る。
干渉チェック用の外形モデルなので、内部構造（ステータ・ロータ）は作らない。

モータ自身の座標系:
    +Z = 出力軸の向き、Z=0 = 取付面（前面プレートの表面）
    本体は Z=-L..0、ボス・軸は Z>0
    +Y = リード線の出口（図の上方）

取付用の M3 タップ穴は 2 か所だけで、正面（軸側）から見て右上と左下（非貫通）。
残りの対角 2 か所はモータ自体の組立てねじなので、取付には使えない（モデル化しない）。
リード線そのものはモデルに含めない（出口の出っ張りだけ作る）。
"""

import os

import FreeCAD as App
import Part

# ---- カタログ寸法 ------------------------------------------------------
SQUARE = 42.0          # □42±0.5
L = 11.6               # モータ長（SS2421 = 11.6、SS2422 = 18.6）
CHAMFER = 7.0          # 四隅の面取り（図に寸法なし。実物で確認した値）
HOLE_PITCH = 31.0      # 2-31±0.3
TAP_DIA = 2.5          # M3×0.5 の下穴径（ねじ山はモデル化しない）
TAP_DEPTH = 2.9        # タップ深さ 2.9 以下
BOSS_DIA = 22.0        # ø22 0/-0.05
BOSS_H = 1.7           # 1.7±0.3
SHAFT_DIA = 5.0        # ø5 0/-0.013
SHAFT_LEN = 7.0        # 取付面からの突出 7±0.5
LEAD_EXIT_W = 24.0     # リード線出口の幅 24 以下
LEAD_EXIT_H = 5.0      # 本体上面からの高さ 5 以下

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.dirname(HERE)
NAME = "ss2421-5041"


def body_shape():
    """面取りした □42 × L の本体と、取付用 M3 タップ穴 2 か所。"""
    h = SQUARE / 2
    c = CHAMFER
    pts = [
        (h - c, h), (-h + c, h), (-h, h - c), (-h, -h + c),
        (-h + c, -h), (h - c, -h), (h, -h + c), (h, h - c),
    ]
    vecs = [App.Vector(x, y, -L) for x, y in pts]
    wire = Part.makePolygon(vecs + [vecs[0]])
    solid = Part.Face(wire).extrude(App.Vector(0, 0, L))
    # 正面（+Z 側）から見ると +X が右なので、右上 = (+, +)、左下 = (−, −)
    for s in (1, -1):
        hole = Part.makeCylinder(
            TAP_DIA / 2, TAP_DEPTH,
            App.Vector(s * HOLE_PITCH / 2, s * HOLE_PITCH / 2, -TAP_DEPTH),
        )
        solid = solid.cut(hole)
    return solid


def lead_exit_shape():
    """上面のリード線出口。正面から見て弓形（弦 24、矢高 5）を全長に押し出す。"""
    h = SQUARE / 2
    half = LEAD_EXIT_W / 2
    # 円弧は両端と頂点の 3 点で決める
    p1 = App.Vector(-half, h, -L)
    p2 = App.Vector(half, h, -L)
    top = App.Vector(0, h + LEAD_EXIT_H, -L)
    arc = Part.Arc(p1, top, p2).toShape()
    chord = Part.LineSegment(p2, p1).toShape()
    face = Part.Face(Part.Wire([arc, chord]))
    return face.extrude(App.Vector(0, 0, L))


def main():
    doc = App.newDocument(NAME.replace("-", "_"))

    parts = [
        ("Body", body_shape(), (0.75, 0.75, 0.78)),
        ("LeadExit", lead_exit_shape(), (0.15, 0.15, 0.15)),
        ("Boss", Part.makeCylinder(BOSS_DIA / 2, BOSS_H), (0.80, 0.80, 0.82)),
        ("Shaft", Part.makeCylinder(SHAFT_DIA / 2, SHAFT_LEN), (0.90, 0.90, 0.92)),
    ]

    # アセンブリから App::Link 1 本で参照できるよう App::Part にまとめる
    motor = doc.addObject("App::Part", "Motor")
    motor.Label = "SS2421-5041"
    objs = []
    for label, shp, rgb in parts:
        o = doc.addObject("Part::Feature", label)
        o.Shape = shp
        if o.ViewObject:  # コンソール版では ViewObject が無い
            o.ViewObject.ShapeColor = rgb
        motor.addObject(o)
        objs.append(o)

    # 干渉チェック・他アセンブリへの取り込み用に一体形状も置く
    solid = objs[0].Shape.fuse([o.Shape for o in objs[1:]]).removeSplitter()
    env = doc.addObject("Part::Feature", "Envelope")
    env.Shape = solid
    env.Visibility = False

    doc.recompute()
    fcstd = os.path.join(CAD_DIR, f"{NAME}.FCStd")
    doc.saveAs(fcstd)

    step = os.path.join(CAD_DIR, "export", f"{NAME}.step")
    Part.export(objs, step)

    bb = solid.BoundBox
    print(f"saved {fcstd}")
    print(f"saved {step}")
    print(f"envelope bbox X {bb.XMin:.2f}..{bb.XMax:.2f}  "
          f"Y {bb.YMin:.2f}..{bb.YMax:.2f}  Z {bb.ZMin:.2f}..{bb.ZMax:.2f}")
    print(f"volume {solid.Volume:.0f} mm^3")


main()
