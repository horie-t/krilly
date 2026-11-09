"""シャーシに SS2421-5041・六角ハブ・φ48 オムニホイールを 3 組取り付け、天板の上に
UPS 基板 X-UPS1 を載せ、全ネジでミドルプレートと L6470 ドライバ基板 3 枚を吊った
アセンブリを作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_ss2421.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_omni.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_fasteners.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_xups1.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_l6470.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_assembly.py </dev/null

出力: hardware/cad/krilly-assembly.FCStd

部品は App::Link で参照するだけなので、krilly-chassis.FCStd / ss2421-5041.FCStd /
omni-hub-hex12.FCStd / omni-wheel-48.FCStd / m3-hex-spacer-10.FCStd / x-ups1.FCStd を直せばアセンブリにもそのまま反映される
（同じフォルダに置いておくこと）。

取付位置はシャーシの M0 取付部（Sketch001 / Pocket）から決めている:
    - 取付壁の外面は x = WALL_OUTER_X、壁厚 WALL_T → 内面 x = 25.2
    - モータは**壁の内側**に置き、取付面を内面に当てる。φ22 ボスは壁の切り欠き
      （幅 23.5）に入り、軸は外向き（+x）に出る
    - 軸心は z = AXIS_Z。シャーシの φ3.5 穴 (y, z) = (15.5, 36) / (−15.5, 5) が
      モータのタップ穴（軸側から見て右上・左下）と一致する
    - リード線出口は上（+z）
ハブは軸に差し、円筒側の端面を取付壁の外面から HUB_GAP 離す。ホイールは六角穴の面を
モータ側に向けてハブの六角部にはめ、六角穴の底の輪にハブ端面を突き当てる（反対側から
M4 ボルトで締める）。
M1 / M2 は M0 を z 軸まわりに 120° / 240° 回したもの（docs/coordinate-frames.md の
スポーク角どおり。シャーシの Pocket001 / Pocket002 もこの位置にある）。

X-UPS1 は天板（Pad001、z 47..49）の上に高さ 10 の六角スペーサー 4 本で載せる。天板の
ø3.2 穴 (−41.75, ±35.05) / (47.75, 35.05) / (47.75, −19.95) が X-UPS1 の取付穴と一致し、
基板の長辺（基板座標 +Y）が車体 +x、基板の外形が天板の外形 ±53.75 × ±38.55 に重なる。
基板は表（電池側）が上。
天板中央の出っ張り（Pad002、z 49..52）の上は基板の裏面まで 7 しかない。そこにある裏面の
部品は 7 以下であることを実物で確認済み（裏面で一番高い GPIO ヘッダー 9 は出っ張りの外）。

ミドルプレートは M3 × 350 の全ネジ 4 本（シャーシ天板の ø3.2 穴 (±42, ±42)）で支え、
シャーシの上端（z 52）からミドルプレートの下面までを MID_GAP にする。全ネジは天板の
下に 4 突き出し、天板とミドルプレートをそれぞれ上下からナットで留める（ミドルプレートの
全ネジの穴のまわりは厚さ 2 で、リブには ø7 の逃げがある）。プレートはリブが上、x の
向きはシャーシと同じ。
L6470 ドライバ基板（AE-L6470DRV）3 枚はミドルプレートの裏に高さ 15 のスペーサーで吊る。
プレートの ø3.4 穴は 42 間隔のペアが 3 組 = 各基板の片側の 2 穴（基板座標 x=4 の列）で、
もう片側（x=36 の列、ピンヘッダー CN4 の側）がプレートの外側を向く。基板の**部品面は
プレート側（上）**: 下に向けると C5 の下端（z ≈ 79）が X-UPS1 の電池の上端（z 80.6）に
ぶつかるため。部品面を上にすると基板の向きが一意に決まり、L と R は鏡像にならない
（CN1 が L は後ろ、R は前を向く）。
"""

import math
import os

import FreeCAD as App
import Part

WALL_OUTER_X = 27.20170592217176   # シャーシ Sketch001 の取付面（外面）
WALL_T = 2.0                       # 取付壁の厚さ（Pocket の深さ）
AXIS_Z = 20.5                      # 穴 (z=5, 36) の中点 = 軸心の高さ
HUB_GAP = 0.5                      # 取付壁の外面 ↔ ハブ端面のすき間
HUB_LEN = 18.0                     # ハブ全長（六角側の端面 = 輪に突き当たる面）
WHEEL_HEX_DEPTH = 10.5             # ホイールの六角穴の深さ
WHEEL_W = 25.5
SPOKES_DEG = {"M0": 0.0, "M1": 120.0, "M2": 240.0}

TOP_PLATE_Z = 49.0                 # 天板（Pad001）の上面
SPACER_LEN = 10.0                  # 六角スペーサーの高さ
UPS_Z = TOP_PLATE_Z + SPACER_LEN   # X-UPS1 の裏面
# X-UPS1 の取付穴（基板座標）。make_xups1.py の出力と同じ
UPS_HOLES = [(3.5, 12.0), (73.6, 12.0), (3.5, 101.5), (58.5, 101.5)]
UPS_PARTS = ("Board", "Holders", "Cells", "GpioHeader")

CHASSIS_TOP_Z = 52.0               # シャーシの上端（Pad002 の上面）
MID_GAP = 56.0                     # シャーシの上端 ↔ ミドルプレートの下面
MID_Z = CHASSIS_TOP_Z + MID_GAP    # ミドルプレートの下面
TOP_PLATE_BOTTOM_Z = 47.0          # 天板（Pad001）の下面
ROD_XY = [(sx * 42.0, sy * 42.0) for sx in (1, -1) for sy in (1, -1)]
ROD_BELOW = 4.0                    # 全ネジが天板の下に突き出す長さ（実物）
MID_T_AT_ROD = 2.0                 # ミドルプレートの全ネジの穴のまわりの厚さ
NUT_T = 2.4
DRV_SPACER = 15.0                  # ドライバ基板のスペーサーの高さ
DRV_BOARD_T = 1.6
DRV_PARTS = ("Board", "C5", "CN1", "CN2", "CN3", "CN4", "JP1", "U1", "PinTails")
# ドライバ基板の基板座標 (bx, by) → プレート座標: z 軸まわりに rot 回して (tx, ty) 足す。
# 基板の x=4 の列の 2 穴 (4, 4) / (4, 46) がプレートの ø3.4 穴に来る
DRIVERS = {
    "L": (90.0, 10.5, 5.5),      # +y 側。穴 (6.5, 9.5) / (−35.5, 9.5)
    "R": (-90.0, -39.5, -5.5),   # −y 側。穴 (−35.5, −9.5) / (6.5, −9.5)
    "F": (0.0, 12.5, -25.0),     # +x 側。穴 (16.5, −21) / (16.5, 21)。プレートから 7 はみ出す
}
DRV_HOLES = [(4.0, 4.0), (4.0, 46.0)]

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.dirname(HERE)


def motor_placement(spoke_deg):
    """モータ座標（+Z 軸方向・+Y リード線・Z=0 取付面）→ 車体座標。"""
    # M0: モータ +X → 車体 +y、+Y → +z、+Z → +x
    m = App.Matrix(
        0, 0, 1, WALL_OUTER_X - WALL_T,
        1, 0, 0, 0,
        0, 1, 0, AXIS_Z,
        0, 0, 0, 1,
    )
    m0 = App.Placement(m)
    spin = App.Placement(App.Vector(), App.Rotation(App.Vector(0, 0, 1), spoke_deg))
    return spin.multiply(m0)


def along_shaft(spoke_deg, z):
    """モータ座標で軸方向に z 進めた位置（向きはモータと同じ）。"""
    return motor_placement(spoke_deg).multiply(App.Placement(App.Vector(0, 0, z), App.Rotation()))


def ups_placement():
    """X-UPS1 の基板座標（左下の角が原点、裏面 Z=0）→ 車体座標。

    基板 +Y（長辺）→ 車体 +x、基板 +X → 車体 −y（z 軸まわりに −90°。裏返しではない）。
    """
    return App.Placement(App.Matrix(
        0, 1, 0, -53.75,
        -1, 0, 0, 38.55,
        0, 0, 1, UPS_Z,
        0, 0, 0, 1,
    ))


def driver_placement(rot, tx, ty):
    """ドライバ基板の座標（左下の角が原点、裏面 Z=0）→ 車体座標。部品面が上。"""
    z = MID_Z - DRV_SPACER - DRV_BOARD_T
    return App.Placement(App.Vector(tx, ty, z), App.Rotation(App.Vector(0, 0, 1), rot))


HUB_Z = WALL_T + HUB_GAP                          # モータ座標でのハブ端面
WHEEL_Z = HUB_Z + HUB_LEN - WHEEL_HEX_DEPTH       # モータ座標でのホイール内側面


def _r(v):
    return math.hypot(v.x, v.y)


def main():
    chassis_doc = App.openDocument(os.path.join(CAD_DIR, "krilly-chassis.FCStd"))
    motor_doc = App.openDocument(os.path.join(CAD_DIR, "ss2421-5041.FCStd"))
    chassis = chassis_doc.getObject("Body")
    motor = motor_doc.getObject("Motor")
    envelope = motor_doc.getObject("Envelope")
    hub_doc = App.openDocument(os.path.join(CAD_DIR, "omni-hub-hex12.FCStd"))
    wheel_doc = App.openDocument(os.path.join(CAD_DIR, "omni-wheel-48.FCStd"))
    hub, hub_env = hub_doc.getObject("Part"), hub_doc.getObject("Envelope")
    wheel, wheel_env = wheel_doc.getObject("Part"), wheel_doc.getObject("Envelope")
    spacer_doc = App.openDocument(os.path.join(CAD_DIR, "m3-hex-spacer-10.FCStd"))
    ups_doc = App.openDocument(os.path.join(CAD_DIR, "x-ups1.FCStd"))
    spacer = spacer_doc.getObject("Part")
    ups = ups_doc.getObject("Part")
    mid_doc = App.openDocument(os.path.join(CAD_DIR, "krilly-middle-plate.FCStd"))
    sp15_doc = App.openDocument(os.path.join(CAD_DIR, "m3-hex-spacer-15.FCStd"))
    nut_doc = App.openDocument(os.path.join(CAD_DIR, "m3-nut.FCStd"))
    rod_doc = App.openDocument(os.path.join(CAD_DIR, "m3-rod-350.FCStd"))
    drv_doc = App.openDocument(os.path.join(CAD_DIR, "ae-l6470.FCStd"))

    # 別ファイルへの App::Link は相対パスで保存されるので、先に保存先を決めておく
    out = os.path.join(CAD_DIR, "krilly-assembly.FCStd")
    doc = App.newDocument("krilly_assembly")
    doc.saveAs(out)
    asm = doc.addObject("Assembly::AssemblyObject", "Assembly")
    asm.Label = "krilly"

    c = doc.addObject("App::Link", "Chassis")
    c.LinkedObject = chassis
    asm.addObject(c)

    links = {}
    extra = {}
    for name, spoke in SPOKES_DEG.items():
        lk = doc.addObject("App::Link", name)
        lk.Label = f"{name}_SS2421-5041"
        lk.LinkedObject = motor
        lk.Placement = motor_placement(spoke)
        asm.addObject(lk)
        links[name] = lk

        for kind, obj, z in (("Hub", hub, HUB_Z), ("Wheel", wheel, WHEEL_Z)):
            w = doc.addObject("App::Link", f"{kind}{name[1]}")
            w.Label = f"{name}_{kind}"
            w.LinkedObject = obj
            w.Placement = along_shaft(spoke, z)
            asm.addObject(w)
            extra[f"{kind}{name[1]}"] = w

    upl = ups_placement()
    u = doc.addObject("App::Link", "UPS")
    u.Label = "X-UPS1"
    u.LinkedObject = ups
    u.Placement = upl
    asm.addObject(u)
    spacers = []
    for i, (hx, hy) in enumerate(UPS_HOLES):
        p = upl.multVec(App.Vector(hx, hy, 0))
        sp = doc.addObject("App::Link", f"Spacer{i}")
        sp.Label = f"Spacer{i}_M3x10"
        sp.LinkedObject = spacer
        sp.Placement = App.Placement(App.Vector(p.x, p.y, TOP_PLATE_Z), App.Rotation())
        asm.addObject(sp)
        spacers.append(sp)

    # 上の段: 全ネジ・ナット・ミドルプレート・ドライバ基板。干渉チェック用に形状も控える
    upper = {}

    def add(name, label, part_doc, part_obj, placement, shape_objs):
        lk = doc.addObject("App::Link", name)
        lk.Label = label
        lk.LinkedObject = part_doc.getObject(part_obj)
        lk.Placement = placement
        asm.addObject(lk)
        upper[name] = Part.makeCompound([
            part_doc.getObject(k).Shape.transformGeometry(placement.toMatrix())
            for k in shape_objs])
        return lk

    def at(x, y, z, rot=0.0):
        return App.Placement(App.Vector(x, y, z), App.Rotation(App.Vector(0, 0, 1), rot))

    for i, (x, y) in enumerate(ROD_XY):
        add(f"Rod{i}", f"Rod{i}_M3x350", rod_doc, "Part",
            at(x, y, TOP_PLATE_BOTTOM_Z - ROD_BELOW), ["Rod"])
        add(f"NutLo{i}", f"Rod{i}_nut_below", nut_doc, "Part",
            at(x, y, TOP_PLATE_BOTTOM_Z - NUT_T), ["Nut"])
        add(f"NutHi{i}", f"Rod{i}_nut_above", nut_doc, "Part",
            at(x, y, TOP_PLATE_Z), ["Nut"])
        add(f"MidNutLo{i}", f"Rod{i}_mid_nut_below", nut_doc, "Part",
            at(x, y, MID_Z - NUT_T), ["Nut"])
        add(f"MidNutHi{i}", f"Rod{i}_mid_nut_above", nut_doc, "Part",
            at(x, y, MID_Z + MID_T_AT_ROD), ["Nut"])
    mid_body = mid_doc.getObject("Body")
    add("MiddlePlate", "MiddlePlate", mid_doc, "Body", at(0, 0, MID_Z), ["Body"])
    drv_holes = []
    for key, (rot, tx, ty) in DRIVERS.items():
        pl = driver_placement(rot, tx, ty)
        add(f"Driver{key}", f"Driver{key}_AE-L6470", drv_doc, "Part", pl, DRV_PARTS)
        for j, (hx, hy) in enumerate(DRV_HOLES):
            p = pl.multVec(App.Vector(hx, hy, 0))
            add(f"DrvSpacer{key}{j}", f"Driver{key}_spacer{j}_M3x15", sp15_doc, "Part",
                at(p.x, p.y, MID_Z - DRV_SPACER), ["Spacer"])
            drv_holes.append((key, p))

    # シャーシを固定（Assembly ワークベンチの Grounded joint）
    try:
        import JointObject  # noqa: F401  (FreeCAD 1.x の Assembly モジュール)
        joints = doc.addObject("Assembly::JointGroup", "Joints")
        asm.addObject(joints)
        g = joints.newObject("App::FeaturePython", "GroundedJoint")
        JointObject.GroundedJoint(g, c)
    except Exception as e:  # 版によって API が違うので、無くても配置自体は成立する
        print(f"grounded joint skipped: {e}")

    doc.recompute()
    doc.save()
    print(f"saved {out}")

    # ---- 検証: 穴位置の一致と干渉 -------------------------------------
    body = chassis.Shape
    for name, lk in links.items():
        pl = lk.Placement
        holes = [pl.multVec(App.Vector(s * 15.5, s * 15.5, 0)) for s in (1, -1)]
        hs = "  ".join(f"({h.x:.2f},{h.y:.2f},{h.z:.2f})" for h in holes)
        env = envelope.Shape.copy()
        env.Placement = pl
        inter = body.common(env).Volume
        bb = env.BoundBox
        print(f"{name}: tap holes {hs}  overlap {inter:.3f} mm^3  "
              f"z {bb.ZMin:.2f}..{bb.ZMax:.2f}  "
              f"shaft tip r={_r(pl.multVec(App.Vector(0, 0, 7))):.2f}")

    # 全部品の組み合わせで干渉体積を見る（はめあい面の接触は 0 になる）
    shapes = {"Chassis": body}
    for name, lk in links.items():
        shapes[name] = envelope.Shape.transformGeometry(lk.Placement.toMatrix())
    for name, lk in extra.items():
        env = hub_env if name.startswith("Hub") else wheel_env
        shapes[name] = env.Shape.transformGeometry(lk.Placement.toMatrix())
    for sp in spacers:
        shapes[sp.Name] = spacer_doc.getObject("Spacer").Shape.transformGeometry(
            sp.Placement.toMatrix())
    shapes["UPS"] = Part.makeCompound([
        ups_doc.getObject(k).Shape.transformGeometry(upl.toMatrix()) for k in UPS_PARTS])
    shapes.update(upper)
    names = list(shapes)
    worst = 0.0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not shapes[a].BoundBox.intersect(shapes[b].BoundBox):
                continue
            v = shapes[a].common(shapes[b]).Volume
            worst = max(worst, v)
            if v > 1e-3:
                print(f"  INTERFERENCE {a} x {b}: {v:.3f} mm^3")
    print(f"max overlap {worst:.4f} mm^3")

    for n in ("0",):
        w, h = extra[f"Wheel{n}"].Placement, extra[f"Hub{n}"].Placement
        print(f"M{n}: hub end r={_r(h.Base):.2f}  set screw r={_r(h.multVec(App.Vector(0, 0, 5))):.2f}  "
              f"wheel r={_r(w.Base):.2f}..{_r(w.multVec(App.Vector(0, 0, WHEEL_W))):.2f}  "
              f"mid r={_r(w.multVec(App.Vector(0, 0, WHEEL_W / 2))):.2f}  "
              f"floor z={w.Base.z - 24:.2f}")

    hs = "  ".join(f"({sp.Placement.Base.x:.2f},{sp.Placement.Base.y:.2f})" for sp in spacers)
    print(f"UPS spacers at {hs}")
    ub = shapes["UPS"].BoundBox
    print(f"UPS x {ub.XMin:.2f}..{ub.XMax:.2f} y {ub.YMin:.2f}..{ub.YMax:.2f} "
          f"z {ub.ZMin:.2f}..{ub.ZMax:.2f}")
    # 裏面の部品が占めうる範囲（外形 × 9）とシャーシの重なり。出っ張りの上の部品は 7 以下と
    # 実物で確認済みなので、ここが 0 でなくても干渉ではない（天板やシャーシを変えたときの目安）
    benv = ups_doc.getObject("BottomEnvelope").Shape.transformGeometry(upl.toMatrix())
    # ドライバ基板の穴がミドルプレートの ø3.4 穴の中心に来ているか
    mid_holes = [(f.Surface.Center.x, f.Surface.Center.y) for f in mid_body.Shape.Faces
                 if type(f.Surface).__name__ == "Cylinder" and abs(f.Surface.Radius - 1.7) < 1e-3]
    for key, p in drv_holes:
        d = min(math.hypot(p.x - x, p.y - y) for x, y in mid_holes)
        print(f"Driver{key} hole ({p.x:.2f},{p.y:.2f}) -> nearest plate ø3.4 hole {d:.3f} mm")
    for key in DRIVERS:
        db = upper[f"Driver{key}"].optimalBoundingBox()  # BoundBox は曲面の変換後に緩くなる
        gap = upper[f"Driver{key}"].distToShape(shapes["UPS"])[0]
        print(f"Driver{key} x {db.XMin:.2f}..{db.XMax:.2f} y {db.YMin:.2f}..{db.YMax:.2f} "
              f"z {db.ZMin:.2f}..{db.ZMax:.2f}  gap to UPS {gap:.2f}")
    mb = upper["MiddlePlate"].BoundBox
    print(f"MiddlePlate z {mb.ZMin:.2f}..{mb.ZMax:.2f}  rods z "
          f"{TOP_PLATE_BOTTOM_Z - ROD_BELOW:.1f}..{TOP_PLATE_BOTTOM_Z - ROD_BELOW + 350:.1f}")
    # 全ネジの下端と天板の下のナットは、後ろの 2 輪（M1 / M2）の上をかすめる
    for i in range(len(ROD_XY)):
        for part in ("Rod", "NutLo"):
            for n in ("0", "1", "2"):
                g = upper[f"{part}{i}"].distToShape(shapes[f"Wheel{n}"])[0]
                if g < 5:
                    print(f"  {part}{i} {ROD_XY[i]} - Wheel{n} gap {g:.2f} mm")
    print(f"UPS bottom envelope x chassis: {benv.common(body).Volume:.1f} mm^3  "
          f"(chassis top z={body.BoundBox.ZMax:.2f}, envelope bottom z={benv.BoundBox.ZMin:.2f})")


main()
