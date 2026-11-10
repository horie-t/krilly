"""シャーシに SS2421-5041・六角ハブ・φ48 オムニホイールを 3 組取り付け、天板の上に
UPS 基板 X-UPS1 を載せ、全ネジでミドルプレートと L6470 ドライバ基板 3 枚を吊り、
ミドルプレートの上に Raspberry Pi 5 と中継基板 HAT を載せ、全ネジの上端にルーフとカメラを付けた
アセンブリを作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_ss2421.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_omni.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_fasteners.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_xups1.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_l6470.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_rpi5.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_camera3.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_hat.py </dev/null
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

Raspberry Pi 5 はミドルプレートの上に高さ 5 の M2.5 六角スペーサー 4 本で載せる。プレートの
ø2.7 穴 (−36, ±24.5) / (22, ±24.5) が Pi 5 の取付穴（58 × 49）で、穴のまわりのプレートは
厚さ 2。Pi 5 の長辺を車体 x にし、USB / Ethernet の端（基板 X=85）を前（+x）に向けると、
基板の端がプレートの前の縁 x = 45.5 にちょうど揃う（この向きは実物で確認済み）。

ルーフは全ネジの上端に付け、全ネジがルーフの上面から ROOF_ROD_ABOVE（平均）突き出す。
上からナイロンナット、下からダブルナットで留める。ロールとピッチはこの 4 本のナットの
高さでルーフごと傾けて調整するが、モデルは水平（平均の高さ）にしておく。ルーフは
全ネジの穴のまわりが厚さ 2 で、リブが上、向きはシャーシと同じ。
カメラ（Camera Module 3 Wide）はスペーサー無しでルーフの下面に直接ねじ留めする。
基板の裏（FPC コネクタの面）をルーフの下面に当て、レンズは真下。ルーフの ø2.3 穴
(−0.1, ±10.5) / (12.4, ±10.5) がカメラの 21 × 12.5 の穴で、レンズ側の穴の列が x = −0.1、
コネクタの辺が後ろ（−x）を向き、コネクタはルーフ中央の開口に入る。光軸は車体の中心
(0, 0) に来る（docs/claude/camera-geometry.md）。
"""

import math
import os
import sys

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

PI_SPACER = 5.0                    # Raspberry Pi 5 のスペーサーの高さ
MID_T_AT_PI = 2.0                  # ミドルプレートの Pi 用の穴のまわりの厚さ
PI_Z = MID_Z + MID_T_AT_PI + PI_SPACER   # Pi 5 の基板の裏面
PI_HOLES = [(3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)]
ROOF_ROD_ABOVE = 5.0               # 全ネジがルーフの上面から突き出す長さ（平均、実物）
ROOF_T_AT_ROD = 2.0                # ルーフの全ネジの穴のまわりの厚さ
NYLON_NUT_T = 4.0
PI_BOARD_T = 1.3                   # Raspberry Pi 5 の基板の厚さ（make_rpi5.py）
HAT_SPACER = 11.0                  # Pi ↔ HAT の M2.5 スペーサー（実物）
# HAT の取付穴（HAT 座標 = Pi 座標）。Pi の右下の穴は HAT の切り欠きの中なので使わない
HAT_HOLES = [(3.5, 52.5), (61.5, 52.5), (3.5, 3.5)]
# HAT の部品（make_hat.py の NOT_FITTED = J13 は入っていない）
HAT_PARTS = ("Board", "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11",
             "J12", "JP1", "JP2", "JP3", "SW1", "SW2", "BZ1", "Q1", "R1", "R2", "R3",
             "D1", "BNO055")
PI_OFFSET = (-39.5, -28.0)         # Pi 5 の基板座標の原点 → プレート座標（穴 (3.5, 3.5) → (−36, −24.5)）

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


class Report:
    """確認結果を画面に出しつつ、Markdown として貯めて最後にファイルに書く。"""

    HEADER = """# krilly アセンブリの要約

`hardware/cad/scripts/make_assembly.py` が `krilly-assembly.FCStd` と一緒に書き出す。
**手で編集しない**（作り直すと上書きされる）。座標は車体座標（mm）: 原点 = 車体の中心、
+x 前 / +y 左 / +z 上（docs/coordinate-frames.md）。部品ごとの寸法と出典は
`hardware/cad/scripts/make_*.py`、GUI で作った板（シャーシ・ミドルプレート・ルーフ）は
`hardware/cad/export/*-describe.md` を見る。
"""

    def __init__(self):
        self.lines = [self.HEADER]

    def section(self, title):
        self.lines += ["", f"## {title}", ""]
        print(f"== {title}")

    def line(self, text):
        self.lines.append(text)
        if text:
            print(text)

    def write(self, path):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(self.lines) + "\n")


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
    pi_doc = App.openDocument(os.path.join(CAD_DIR, "rpi5.FCStd"))
    sp5_doc = App.openDocument(os.path.join(CAD_DIR, "m25-hex-spacer-5.FCStd"))
    roof_doc = App.openDocument(os.path.join(CAD_DIR, "krilly-roof.FCStd"))
    nylon_doc = App.openDocument(os.path.join(CAD_DIR, "m3-nylon-nut.FCStd"))
    cam_doc = App.openDocument(os.path.join(CAD_DIR, "camera-module-3-wide.FCStd"))
    hat_doc = App.openDocument(os.path.join(CAD_DIR, "krilly-hat.FCStd"))
    sp11_doc = App.openDocument(os.path.join(CAD_DIR, "m25-hex-spacer-11.FCStd"))

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

    pi_pl = at(PI_OFFSET[0], PI_OFFSET[1], PI_Z)
    add("Pi5", "RaspberryPi5", pi_doc, "Part", pi_pl,
        ["Board", "Usb", "Ethernet", "Gpio", "Camera", "SdSocket"])
    pi_holes = []
    for j, (hx, hy) in enumerate(PI_HOLES):
        p = pi_pl.multVec(App.Vector(hx, hy, 0))
        add(f"PiSpacer{j}", f"Pi5_spacer{j}_M2.5x5", sp5_doc, "Part",
            at(p.x, p.y, MID_Z + MID_T_AT_PI), ["Spacer"])
        pi_holes.append(p)

    rod_top = TOP_PLATE_BOTTOM_Z - ROD_BELOW + 350.0
    roof_z = rod_top - ROOF_ROD_ABOVE - ROOF_T_AT_ROD      # ルーフの下面
    roof_body = [o for o in roof_doc.Objects if o.TypeId == "PartDesign::Body"][0]
    add("Roof", "Roof", roof_doc, roof_body.Name, at(0, 0, roof_z), [roof_body.Name])
    for i, (x, y) in enumerate(ROD_XY):
        add(f"RoofNylon{i}", f"Rod{i}_roof_nylon_nut", nylon_doc, "Part",
            at(x, y, roof_z + ROOF_T_AT_ROD), ["Nut"])
        for k in (1, 2):  # ダブルナット
            add(f"RoofNutLo{i}_{k}", f"Rod{i}_roof_nut_below{k}", nut_doc, "Part",
                at(x, y, roof_z - k * NUT_T), ["Nut"])
    # カメラの座標（+X 右・+Y コネクタ側・+Z 光軸、表が Z=0、裏が Z=−1.12）→ 車体座標:
    # カメラ +X → 車体 −y、+Y → −x、+Z → −z（レンズが真下）。光軸 (12.5, 14.4) → (0, 0)
    cam_pl = App.Placement(App.Matrix(
        0, -1, 0, 14.4,
        -1, 0, 0, 12.5,
        0, 0, -1, roof_z - 1.12,
        0, 0, 0, 1,
    ))
    add("Camera", "CameraModule3Wide", cam_doc, "Part", cam_pl,
        ["Board", "Housing", "Lens", "Connector"])

    # 中継基板 HAT: Pi 5 の上に M2.5 スペーサー 3 本。HAT の座標は Pi の座標と同じ向き・原点
    hat_z = PI_Z + PI_BOARD_T + HAT_SPACER
    hat_pl = at(PI_OFFSET[0], PI_OFFSET[1], hat_z)
    add("Hat", "krilly_HAT", hat_doc, "Part", hat_pl,
        [k for k in HAT_PARTS if k != "J1"])
    # J1（裏面のメスヘッダー）は Pi の GPIO（オス）と差し込み合って重なるのが正しいので、
    # 全体の干渉チェックからは外し、下で J1 と GPIO の位置だけ確かめる
    hat_j1 = hat_doc.getObject("J1").Shape.transformGeometry(hat_pl.toMatrix())
    lk = doc.addObject("App::Link", "HatJ1")  # 表示用（形状は Hat のリンクに含まれない）
    lk.Label = "krilly_HAT_J1"
    lk.LinkedObject = hat_doc.getObject("J1")
    lk.Placement = hat_pl
    asm.addObject(lk)
    for j, (hx, hy) in enumerate(HAT_HOLES):
        add(f"HatSpacer{j}", f"HAT_spacer{j}_M2.5x11", sp11_doc, "Part",
            at(PI_OFFSET[0] + hx, PI_OFFSET[1] + hy, PI_Z + PI_BOARD_T), ["Spacer"])

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

    # ---- 検証と要約 ----------------------------------------------------
    # 結果は画面に出すと同時に export/assembly.md に書く。FCStd はバイナリで読めないので、
    # 他のセッション（や人）はこの要約で機体の状態を知る。差分を見れば何がどれだけ動いたか分かる
    report = Report()
    body = chassis.Shape

    # 干渉チェックとすき間に使う、車体座標に置いた形状
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
    # BoundBox は曲面を変換すると緩くなる（ホイールで 6 mm 低く出た）ので optimalBoundingBox を使う
    floor_z = min(shapes[f"Wheel{n}"].optimalBoundingBox().ZMin for n in "012")

    report.section("部品の範囲")
    report.line(f"床（ホイールの接地点）は z = {floor_z:.2f}。範囲は各部品の境界箱（mm、車体座標）。")
    report.line("")
    report.line("| 名前 | ラベル | ファイル | x | y | z |")
    report.line("|---|---|---|---|---|---|")
    bbox_only = {"HatJ1": hat_j1}   # 干渉チェックには入れないが範囲は載せる
    for o in doc.Objects:
        if o.TypeId != "App::Link":
            continue
        shp = shapes.get(o.Name, bbox_only.get(o.Name))
        if shp is None:
            continue
        b = shp.optimalBoundingBox()  # BoundBox は曲面の変換後に緩くなる
        src = os.path.basename(o.LinkedObject.Document.FileName)
        report.line(f"| {o.Name} | {o.Label} | {src} | {b.XMin:.2f}〜{b.XMax:.2f} | "
                    f"{b.YMin:.2f}〜{b.YMax:.2f} | {b.ZMin:.2f}〜{b.ZMax:.2f} |")

    report.section("取付穴の一致")
    report.line("相手の穴の中心からのずれ（0 なら一致）。")
    report.line("")
    for name, lk in links.items():
        pl = lk.Placement
        for s in (1, -1):
            h = pl.multVec(App.Vector(s * 15.5, s * 15.5, 0))
            report.line(f"- {name} のタップ穴 ({h.x:.2f}, {h.y:.2f}, {h.z:.2f})"
                        "（シャーシの ø3.5 穴と同じ軸上）")

    def hole_check(label, points, plate_shape, radius, plate_name):
        holes = [(f.Surface.Center.x, f.Surface.Center.y) for f in plate_shape.Faces
                 if type(f.Surface).__name__ == "Cylinder"
                 and abs(f.Surface.Radius - radius) < 1e-3]
        for p in points:
            d = min(math.hypot(p.x - x, p.y - y) for x, y in holes)
            report.line(f"- {label} ({p.x:.2f}, {p.y:.2f}) → {plate_name}の "
                        f"ø{2 * radius:.1f} 穴まで {d:.3f}")

    hole_check("ドライバ基板の穴", [p for _, p in drv_holes], mid_body.Shape, 1.7, "ミドルプレート")
    hole_check("Pi 5 の穴", pi_holes, mid_body.Shape, 1.35, "ミドルプレート")
    hole_check("カメラの穴", [cam_pl.multVec(App.Vector(x, y, 0))
                         for x, y in [(2.0, 2.0), (23.0, 2.0), (2.0, 14.5), (23.0, 14.5)]],
               roof_body.Shape, 1.15, "ルーフ")
    for sp in spacers:
        report.line(f"- X-UPS1 のスペーサー ({sp.Placement.Base.x:.2f}, {sp.Placement.Base.y:.2f})"
                    "（天板の ø3.2 穴）")

    report.section("干渉")
    names = list(shapes)
    worst = 0.0
    hits = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not shapes[a].BoundBox.intersect(shapes[b].BoundBox):
                continue
            v = shapes[a].common(shapes[b]).Volume
            worst = max(worst, v)
            if v > 1e-3:
                hits.append(f"- **{a} × {b}: {v:.3f} mm³**")
    report.line(f"全 {len(names)} 部品の組み合わせで、重なりの最大は {worst:.4f} mm³"
                "（はめあい面の接触は 0 になる）。")
    for h in hits:
        report.line(h)
    report.line("")
    report.line("チェックから外しているもの:")
    gpio = pi_doc.getObject("Gpio").Shape.transformGeometry(pi_pl.toMatrix())
    gb, jb = gpio.Solids[0].BoundBox, hat_j1.BoundBox   # Solids[0] = GPIO の樹脂部
    report.line(f"- HAT の J1（メス、x {jb.XMin:.2f}〜{jb.XMax:.2f}, y {jb.YMin:.2f}〜{jb.YMax:.2f}, "
                f"下端 z {jb.ZMin:.2f}）と Pi の GPIO（オス、x {gb.XMin:.2f}〜{gb.XMax:.2f}, "
                f"y {gb.YMin:.2f}〜{gb.YMax:.2f}, 樹脂部の上面 z {gb.ZMax:.2f}）。差し込み合うのが正しい")
    report.line("- HAT の中の BZ1 × SW1（0.13 mm³）。実物でも接している（make_hat.py）")

    report.section("すき間（余裕の小さい所）")
    report.line("最短距離（mm）。機構を変えるときは、ここが詰まらないか確かめる。")
    report.line("")

    def gap(a, b, note=""):
        d = shapes[a].distToShape(shapes[b])[0]
        report.line(f"- {a} ↔ {b}: {d:.2f}" + (f"（{note}）" if note else ""))

    for i in range(len(ROD_XY)):
        for part in ("Rod", "NutLo"):
            for n in ("0", "1", "2"):
                d = upper[f"{part}{i}"].distToShape(shapes[f"Wheel{n}"])[0]
                if d < 5:
                    report.line(f"- {part}{i} {ROD_XY[i]} ↔ Wheel{n}: {d:.2f}"
                                "（全ネジの下端 / 天板の下のナットが後輪の上をかすめる）")
    for key in DRIVERS:
        gap(f"Driver{key}", "UPS", "部品面を下にすると C5 が電池に当たる")
        gap(f"Driver{key}", "MiddlePlate", "部品面はプレート側")
    gap("Pi5", "MiddlePlate", "裏の GPIO のピンの先と SD ソケット ↔ リブ")
    gap("Hat", "Pi5", "J1 を除く")
    benv = ups_doc.getObject("BottomEnvelope").Shape.transformGeometry(upl.toMatrix())
    report.line(f"- X-UPS1 の裏面の部品の範囲（外形 × 9）× シャーシ: {benv.common(body).Volume:.1f} mm³"
                "（天板中央の出っ張りの上はすき間 7。そこにある裏面の部品は 7 以下と実物で確認済み"
                "なので干渉ではない）")

    report.section("主な寸法")
    for n in ("0", "1", "2"):
        w, h = extra[f"Wheel{n}"].Placement, extra[f"Hub{n}"].Placement
        tip = links[f"M{n}"].Placement.multVec(App.Vector(0, 0, 7))
        report.line(f"- M{n}: 軸の先端 r = {_r(tip):.2f}、ハブの端面 r = {_r(h.Base):.2f}、"
                    f"イモネジ r = {_r(h.multVec(App.Vector(0, 0, 5))):.2f}、"
                    f"ホイール r = {_r(w.Base):.2f}〜{_r(w.multVec(App.Vector(0, 0, WHEEL_W))):.2f}"
                    f"（幅の中央 {_r(w.multVec(App.Vector(0, 0, WHEEL_W / 2))):.2f}）")
    report.line("  - イモネジが軸の先端より外にある = 軸に半分ほどしか掛からない（#173）。"
                "幅の中央は robot.yaml の center_to_wheel_m と比べる")
    axis = cam_pl.multVec(App.Vector(12.5, 14.4, 8.3))
    report.line(f"- カメラのレンズの先端: ({axis.x:.2f}, {axis.y:.2f}, {axis.z:.2f})、"
                f"床から {axis.z - floor_z:.1f}")
    rod_z0 = TOP_PLATE_BOTTOM_Z - ROD_BELOW
    report.line(f"- 全ネジ: z = {rod_z0:.1f}〜{rod_z0 + 350:.1f}")

    report.write(os.path.join(CAD_DIR, "export", "assembly.md"))
    print(f"saved {os.path.join(CAD_DIR, 'export', 'assembly.md')}")


try:
    main()
except Exception:  # FreeCAD のコンソール版は例外を表示せずに終わるので、自分で出す
    import traceback
    traceback.print_exc()
    raise
finally:
    # flush しないと、終了時に標準出力の最後の部分が捨てられる
    sys.stdout.flush()
    sys.stderr.flush()
