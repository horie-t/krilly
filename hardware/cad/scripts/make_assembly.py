"""シャーシに SS2421-5041・六角ハブ・φ48 オムニホイールを 3 組取り付けたアセンブリを作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_ss2421.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_omni.py </dev/null
    hardware/cad/scripts/fcrun hardware/cad/scripts/make_assembly.py </dev/null

出力: hardware/cad/krilly-assembly.FCStd

部品は App::Link で参照するだけなので、krilly-chassis.FCStd / ss2421-5041.FCStd /
omni-hub-hex12.FCStd / omni-wheel-48.FCStd を直せばアセンブリにもそのまま反映される
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
"""

import math
import os

import FreeCAD as App

WALL_OUTER_X = 27.20170592217176   # シャーシ Sketch001 の取付面（外面）
WALL_T = 2.0                       # 取付壁の厚さ（Pocket の深さ）
AXIS_Z = 20.5                      # 穴 (z=5, 36) の中点 = 軸心の高さ
HUB_GAP = 0.5                      # 取付壁の外面 ↔ ハブ端面のすき間
HUB_LEN = 18.0                     # ハブ全長（六角側の端面 = 輪に突き当たる面）
WHEEL_HEX_DEPTH = 10.5             # ホイールの六角穴の深さ
WHEEL_W = 25.5
SPOKES_DEG = {"M0": 0.0, "M1": 120.0, "M2": 240.0}

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


main()
