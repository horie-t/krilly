"""φ48 オムニホイールと六角ハブ（ヴイストン販売品）の外形モデルを作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_omni.py </dev/null

出力:
    hardware/cad/omni-wheel-48.FCStd   … オムニホイール φ48 × 25.5（JAN 4571398312991）
    hardware/cad/omni-hub-hex12.FCStd  … φ5 軸用 六角ハブ S12（JAN 4571398313028）
    hardware/cad/export/omni-wheel-48.step / omni-hub-hex12.step

寸法は販売ページの寸法図と実物の実測による（単位 mm）。プレートの腕と中心の輪の
寸法は外観写真からの読み取りで、干渉チェック用の外形として作る。

取付け方（外観写真 _2 の中心から）:
    ホイールの片面に六角穴（深さ 10.5）が開いていて、その底 = 幅の中央に、4 本の
    リブで支えられた輪がある。ハブの六角部をこの面から差し込んで輪に突き当て、
    反対側の面から M4 ボルトを輪の穴に通してハブ端面の M4 タップに締める。
    ハブ円筒部の M4 止めねじ（半径方向）はモータ軸を掴む。
    → **六角穴の面がモータ側**を向く。

座標系（どちらも回転軸 = +Z）:
    ハブ    : Z=0 = 円筒側の端面（モータ側）、Z=18 = 六角側の端面
    ホイール: Z=0 = 六角穴の開いた面（モータ側）、Z=WHEEL_W = 反対の面（ボルト頭側）
"""

import math
import os

import FreeCAD as App
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.dirname(HERE)

# ---- ハブ（寸法図どおり） ---------------------------------------------
HUB_LEN = 18.0          # 全長
HUB_CYL_DIA = 11.5      # 円筒部 ø11.5 0/-0.02
HUB_HEX_S = 12.0        # 六角部 二面幅 S=12
HUB_HEX_LEN = 6.0       # 六角部の長さ
HUB_BORE_DIA = 5.0      # 軸穴 ø5 +0.02/0
HUB_BORE_DEPTH = 12.0   # 軸穴の深さ（円筒部の長さぶん。図からの読み取り）
HUB_TAP_DIA = 3.3       # 端面 M4 の下穴（六角部を貫通して軸穴につながる）
HUB_SET_Z = 5.0         # 止めねじ M4 の位置（円筒側の端面から 5）
HUB_SET_DIA = 3.3       # 止めねじ M4 の下穴（断面図では両側に抜けている）
HUB_C_CYL = 0.5         # 円筒側の端の面取り C0.5
HUB_C_HEX = 1.0         # 六角部の両端の面取り C1

# ---- オムニホイール ---------------------------------------------------
WHEEL_DIA = 48.0        # ø48
WHEEL_W = 25.5          # 幅 25.5
WHEEL_BORE = 14.0       # 中心穴 ø14
WHEEL_HEX_S = 12.0      # 六角穴 S12（ハブの六角部が入る）
WHEEL_HEX_DEPTH = 10.5  # 六角穴の深さ（Z=0 の面から。実測）
HUB_RING_Z1 = WHEEL_W - WHEEL_HEX_DEPTH  # 輪は幅の中央に対称: Z 10.5..15
HUB_RING_ID = 4.5       # 輪の穴（M4 ボルトが通る。写真からの読み取り）
HUB_RING_OD = 9.0       # 輪の外径（同上）
HUB_RIB_W = 1.5         # 輪を支える 4 本のリブの幅（同上）
ROLLERS_PER_ROW = 4     # 2 列 × 4 個、列どうしは 45° ずらし
ROLLER_DIA = 12.6       # ローラーの最大径（実測）
ROLLER_HALF_LEN = 9.25  # ローラー長さ 18.5 の半分（実測）
CORE_R = 10.5           # 中心の筒の外半径（ローラーの内側 = 半径 11.4 に収まる）
ARM_W = 5.0             # 各列のプレートの腕（ローラーの間。写真からの読み取り）
ARM_R = 23.5            # 腕の先（写真ではほぼ外周まで届いている）


def hex_prism(s, z0, length):
    r = s / math.sqrt(3)  # 二面幅 → 外接円の半径
    pts = [App.Vector(r * math.cos(math.radians(30 + 60 * i)),
                      r * math.sin(math.radians(30 + 60 * i)), z0) for i in range(6)]
    face = Part.Face(Part.makePolygon(pts + [pts[0]]))
    return face.extrude(App.Vector(0, 0, length))


def revolve(profile):
    """(r, z) の折れ線（r=0 の軸上で閉じる）を Z 軸まわりに回した立体。"""
    pts = [App.Vector(r, 0, z) for r, z in profile]
    face = Part.Face(Part.makePolygon(pts + [pts[0]]))
    return face.revolve(App.Vector(), App.Vector(0, 0, 1), 360)


# ---- ハブ -------------------------------------------------------------
def hub_shape():
    rc = HUB_CYL_DIA / 2
    z_hex = HUB_LEN - HUB_HEX_LEN
    cyl = revolve([(0, 0), (rc - HUB_C_CYL, 0), (rc, HUB_C_CYL), (rc, z_hex + 0.01),
                   (0, z_hex + 0.01)])
    # 六角の面取り C1: 外接円の円柱を両端で 1mm ずつ円錐に落とし、六角柱と交差させる
    rh = HUB_HEX_S / math.sqrt(3)
    c = HUB_C_HEX
    cone = revolve([(0, z_hex), (rh - c, z_hex), (rh, z_hex + c), (rh, HUB_LEN - c),
                    (rh - c, HUB_LEN), (0, HUB_LEN)])
    hexa = hex_prism(HUB_HEX_S, z_hex, HUB_HEX_LEN).common(cone)
    solid = cyl.fuse(hexa)
    solid = solid.cut(Part.makeCylinder(HUB_BORE_DIA / 2, HUB_BORE_DEPTH))
    solid = solid.cut(Part.makeCylinder(HUB_TAP_DIA / 2, HUB_LEN))
    solid = solid.cut(Part.makeCylinder(
        HUB_SET_DIA / 2, HUB_CYL_DIA + 2,
        App.Vector(-(HUB_CYL_DIA / 2 + 1), 0, HUB_SET_Z), App.Vector(1, 0, 0)))
    return solid.removeSplitter()


# ---- オムニホイール ---------------------------------------------------
def roller_shape():
    """原点中心・+X 軸のローラー。外形がホイールの外周円 R=24 に沿う樽形。"""
    R = WHEEL_DIA / 2
    axis_r = R - ROLLER_DIA / 2          # ローラー軸の中心からの距離
    n = 12
    prof = []
    for i in range(n + 1):
        t = -ROLLER_HALF_LEN + 2 * ROLLER_HALF_LEN * i / n
        prof.append((t, math.sqrt(R * R - t * t) - axis_r))
    pts = [App.Vector(t, rho, 0) for t, rho in prof]
    spline = Part.BSplineCurve()
    spline.interpolate(pts)
    a, b = pts[0], pts[-1]
    edges = [spline.toShape(),
             Part.LineSegment(b, App.Vector(b.x, 0, 0)).toShape(),
             Part.LineSegment(App.Vector(b.x, 0, 0), App.Vector(a.x, 0, 0)).toShape(),
             Part.LineSegment(App.Vector(a.x, 0, 0), a).toShape()]
    face = Part.Face(Part.Wire(edges))
    return face.revolve(App.Vector(), App.Vector(1, 0, 0), 360), axis_r


def wheel_shapes():
    row_w = WHEEL_W / 2
    rows = [(row_w / 2, 0.0), (WHEEL_W - row_w / 2, 180.0 / ROLLERS_PER_ROW)]
    base, axis_r = roller_shape()

    rollers = []
    for zc, phase in rows:
        for k in range(ROLLERS_PER_ROW):
            a = phase + 360.0 * k / ROLLERS_PER_ROW
            r = base.copy()
            # ローラー軸を接線方向（+Y）に向け、半径 axis_r の位置へ
            r.rotate(App.Vector(), App.Vector(0, 0, 1), 90)
            r.translate(App.Vector(axis_r, 0, zc))
            r.rotate(App.Vector(), App.Vector(0, 0, 1), a)
            rollers.append(r)

    # 中心の筒 + 各列の腕（腕はその列のローラーの間 = 位相 +45°）
    frame = Part.makeCylinder(CORE_R, WHEEL_W)
    roller_r = ROLLER_DIA / 2
    for zc, phase in rows:
        for k in range(ROLLERS_PER_ROW):
            a = phase + 180.0 / ROLLERS_PER_ROW + 360.0 * k / ROLLERS_PER_ROW
            z0 = max(0.0, zc - roller_r)
            z1 = min(WHEEL_W, zc + roller_r)
            arm = Part.makeBox(ARM_R - CORE_R + 1, ARM_W, z1 - z0,
                               App.Vector(CORE_R - 1, -ARM_W / 2, z0))
            arm.rotate(App.Vector(), App.Vector(0, 0, 1), a)
            frame = frame.fuse(arm)
    # 腕の先を外周に沿って丸める
    frame = frame.common(Part.makeCylinder(ARM_R, WHEEL_W))

    # 中心穴: Z=0 側に六角穴 S12 × 10.5、反対側は ø14（ボルト頭が入る）
    frame = frame.cut(hex_prism(WHEEL_HEX_S, -0.01, WHEEL_HEX_DEPTH + 0.01))
    frame = frame.cut(Part.makeCylinder(WHEEL_BORE / 2, WHEEL_W - HUB_RING_Z1 + 0.01,
                                        App.Vector(0, 0, HUB_RING_Z1)))
    # 底の輪とリブ: 六角穴の底 Z=10.5 から幅の中央に対称に Z=15 まで。
    # リブの間は抜けている（写真 _2 で向こうが白く見える）
    web_t = HUB_RING_Z1 - WHEEL_HEX_DEPTH
    web = Part.makeCylinder(WHEEL_BORE / 2, web_t, App.Vector(0, 0, WHEEL_HEX_DEPTH))
    keep = Part.makeCylinder(HUB_RING_OD / 2, web_t, App.Vector(0, 0, WHEEL_HEX_DEPTH))
    for k in range(4):
        rib = Part.makeBox(WHEEL_BORE, HUB_RIB_W, web_t,
                           App.Vector(0, -HUB_RIB_W / 2, WHEEL_HEX_DEPTH))
        rib.rotate(App.Vector(), App.Vector(0, 0, 1), 45 + 90 * k)
        keep = keep.fuse(rib)
    frame = frame.cut(web.cut(keep))
    frame = frame.cut(Part.makeCylinder(HUB_RING_ID / 2, WHEEL_W))
    return frame.removeSplitter(), rollers


def save(doc_name, part_label, objs_spec, envelope_shapes, fname):
    doc = App.newDocument(doc_name)
    part = doc.addObject("App::Part", "Part")
    part.Label = part_label
    objs = []
    for label, shp in objs_spec:
        o = doc.addObject("Part::Feature", label)
        o.Shape = shp
        part.addObject(o)
        objs.append(o)
    env = doc.addObject("Part::Feature", "Envelope")
    env.Shape = envelope_shapes
    env.Visibility = False
    doc.recompute()
    path = os.path.join(CAD_DIR, f"{fname}.FCStd")
    doc.saveAs(path)
    Part.export(objs, os.path.join(CAD_DIR, "export", f"{fname}.step"))
    bb = envelope_shapes.BoundBox
    print(f"saved {path}  bbox X {bb.XMin:.2f}..{bb.XMax:.2f} "
          f"Y {bb.YMin:.2f}..{bb.YMax:.2f} Z {bb.ZMin:.2f}..{bb.ZMax:.2f}  "
          f"vol {envelope_shapes.Volume:.0f}")


def main():
    hub = hub_shape()
    save("omni_hub_hex12", "Hub_hex12", [("Hub", hub)], hub, "omni-hub-hex12")

    frame, rollers = wheel_shapes()
    roller_comp = Part.makeCompound(rollers)
    env = frame.fuse(rollers)
    save("omni_wheel_48", "OmniWheel_48",
         [("Frame", frame), ("Rollers", roller_comp)], env, "omni-wheel-48")


main()
