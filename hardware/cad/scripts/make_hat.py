"""中継基板 krilly HAT（#140）の簡易外形モデルを、EasyEDA のプロジェクトから作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_hat.py </dev/null

入力: hardware/krilly-hat/krilly-hat.epro2（部品の位置・向き・フットプリントの外形）
出力: hardware/cad/krilly-hat.FCStd、hardware/cad/export/krilly-hat.step

基板の外形と穴は Gerber（fab/krilly-hat_Gerber.zip）と同じ値を使う: 65 × 56 × 1.6、右下
（x ≥ 44, 下辺から 25 以下）を切り落とした L 字、角 R3（切り欠きの角は R1）、取付穴 ø2.75 × 3。
部品は、フットプリントの外形（パッド + シルク / 部品外形レイヤー）を部品の向きに回した箱を、
HEIGHTS の高さで立てる。高さは EasyEDA の部品データ（ブザー 5.5、タクト 5、ヘッダー類）と
一般的な値で、HEIGHT_NOTES に出典を書く。

座標系: Raspberry Pi 5 のモデル（make_rpi5.py）と同じ。原点 = 基板の左下の角（Pi の左下と
一致）、+X 右、+Y 上（40 ピンの辺が上）、基板の裏面が Z=0、表面が Z=BOARD_T。
EasyEDA の座標（原点は左上、y は上向きで基板内は負）からは y_here = 56 + y_easyeda。
J1（2×20 のメスヘッダー）は裏面に実装するので、下（−Z）に伸ばす。
"""

import json
import math
import os
import zipfile

import FreeCAD as App
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(CAD_DIR))
EPRO = os.path.join(REPO, "hardware", "krilly-hat", "krilly-hat.epro2")
NAME = "krilly-hat"

MIL = 0.0254
BOARD_W, BOARD_H, BOARD_T = 65.0, 56.0, 1.6
CUT_X, CUT_Y = 44.0, 25.0          # 右下の切り欠き（x ≥ 44 かつ y ≤ 25）
HOLES = [(3.5, 52.5), (61.5, 52.5), (3.5, 3.5)]
HOLE_DIA = 2.75

# 部品の高さ（基板の表面から。J1 だけ裏面から下へ）
HEIGHTS = {
    "J1": 8.5, "J2": 9.0, "J3": 9.0, "J4": 9.0, "J5": 16.5,
    "J6": 7.0, "J7": 7.0, "J8": 7.0, "J9": 7.0,
    "J10": 8.5, "J11": 8.5, "J12": 8.5, "J13": 8.5,
    "JP1": 8.5, "JP2": 8.5, "JP3": 8.5,
    "SW1": 6.35, "SW2": 5.0, "BZ1": 5.5,
    "Q1": 1.1, "R1": 0.6, "R2": 0.6, "R3": 0.6, "D1": 1.1,
}
HEIGHT_NOTES = """
    J1 / J10 / J11 : メスヘッダー 8.5（EasyEDA の Insulation Height）
    J12 / J13 / JP : ピンヘッダー 樹脂部 2.5（同上）+ ピン 6 = 8.5
    J2〜J4         : 2×5 ボックスヘッダー 9.0（一般的な値）
    J5             : JST VH B2P-VH 16.5（一般的な値。安全側）
    J6〜J9         : JST XH B2B-XH-A 7.0（一般的な値）
    SW1            : 縦型スライドスイッチ（下記）。本体 6.35 + つまみ SLIDE_KNOB_H
    SW2            : タクト 5（EasyEDA の Switch Height）
    BZ1            : 電磁ブザー ø9.5 × 5.5 の円筒（径は実物、高さは EasyEDA の Height）
"""
# 非常停止の SW1 は、設計の横向きトグル（CAX-MTS102C3）がカメラの邪魔になるので、実物は
# 縦型のスライドスイッチ（本体 12.7 × 6.6 × 6.35、つまみ 3.8 角、ピン 3 本）に替えてある。
# ピンはトグルの 3 本並びのパッド（局所座標 y = 6.45、x = −4.7 / 0 / 4.7）に差し、本体の
# 長辺はその並びの向き。つまみは本体の上で長辺方向に動くので、動く範囲 6.6 を箱にする。
SLIDE_BODY = (12.7, 6.6)           # 長辺（パッドの並び）× 短辺
SLIDE_PAD_Y = 6.45                 # 3 本並びのパッドの局所 y
SLIDE_KNOB = (6.6, 3.8)            # つまみが動く範囲 × つまみの幅
SLIDE_KNOB_H = 5.0                 # つまみの高さ（図の X。実物）
BZ1_DIA = 9.5                      # ブザーの径（実物。シルクの円の中心 = 部品の原点）
# BZ1 と SW1（スライドスイッチの本体の角）は実物でも接している（はんだ付けは問題なし）。
# モデルでは 0.13 mm³ 重なるが、干渉ではない
# 実物で付けていない部品。J13（UART の 1×3 ピン）はスライドスイッチの本体と重なるので外した
NOT_FITTED = {"J13"}
# BNO055 モジュール（秋月 AE-BNO055-BO）: J10 / J11 のメスヘッダーにオスピン（樹脂部 2.5）で載る
BNO_SIZE = (27.0, 20.0)            # 仮の外形（x × y）。2 列のピンの中央に置く
BNO_T = 1.6
BNO_PARTS_H = 1.5                  # モジュールの部品の高さ（仮）
BNO_PIN_BASE = 2.5


def read_easyeda(path):
    """epro2 から PCB の部品と、使っているフットプリントの外形を読む。"""
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith(".epru")][0]
        text = z.read(name).decode("utf-8")
    docs, cur = [], None
    for line in text.splitlines():
        if "||" not in line:
            continue
        h, b = line.split("||", 1)
        b = b.rstrip("|")
        try:
            h = json.loads(h)
            b = json.loads(b) if b else {}
        except ValueError:
            continue
        if h.get("type") == "DOCHEAD":
            cur = {"head": b, "recs": []}
            docs.append(cur)
        elif cur is not None:
            cur["recs"].append((h, b))
    pcb = [d for d in docs if d["head"].get("docType") == "PCB"][0]
    fps = {d["head"]["uuid"]: d for d in docs if d["head"].get("docType") == "FOOTPRINT"}
    attrs = {}
    for h, b in pcb["recs"]:
        if h.get("type") == "ATTR" and b.get("parentId"):
            attrs.setdefault(b["parentId"], {})[b["key"]] = b["value"]
    comps = []
    for h, b in pcb["recs"]:
        if h.get("type") != "COMPONENT":
            continue
        a = attrs.get(h["id"], {})
        comps.append({"des": a.get("Designator"), "x": b["x"] * MIL, "y": b["y"] * MIL,
                      "angle": b["angle"], "fp": fps.get(a.get("Footprint"))})
    return comps


def numbers(path):
    out = []
    if isinstance(path, list):
        for v in path:
            if isinstance(v, (int, float)):
                out.append(v)
            elif isinstance(v, list):
                out += numbers(v)
    return out


def fp_rects(fp):
    """フットプリントの外形（局所座標の長方形）。"""
    xs, ys = [], []
    for h, b in fp["recs"]:
        t = h.get("type")
        if t == "PAD" and b.get("centerX") is not None:
            pad = b.get("defaultPad") or {}
            r = max(pad.get("width", 0), pad.get("height", 0)) / 2
            xs += [b["centerX"] - r, b["centerX"] + r]
            ys += [b["centerY"] - r, b["centerY"] + r]
        elif t in ("POLY", "FILL") and b.get("layerId") in (3, 48):  # シルク / 部品外形
            n = numbers(b.get("path"))
            xs += n[0::2]
            ys += n[1::2]
    return [(min(xs) * MIL, min(ys) * MIL, max(xs) * MIL, max(ys) * MIL)]


def placed_box(rect, cx, cy, angle, z0, z1):
    x0, y0, x1, y1 = rect
    box = Part.makeBox(x1 - x0, y1 - y0, z1 - z0, App.Vector(x0, y0, z0))
    box.rotate(App.Vector(), App.Vector(0, 0, 1), angle)
    box.translate(App.Vector(cx, cy, 0))
    return box


def board_shape():
    v = App.Vector
    z = v(0, 0, 1)
    r, rc = 3.0, 1.0
    W, H = BOARD_W, BOARD_H
    edges = [
        Part.LineSegment(v(r, 0, 0), v(CUT_X - r, 0, 0)).toShape(),
        Part.makeCircle(r, v(CUT_X - r, r, 0), z, 270, 360),
        Part.LineSegment(v(CUT_X, r, 0), v(CUT_X, CUT_Y, 0)).toShape(),
        Part.LineSegment(v(CUT_X, CUT_Y, 0), v(W - rc, CUT_Y, 0)).toShape(),
        Part.makeCircle(rc, v(W - rc, CUT_Y + rc, 0), z, 270, 360),
        Part.LineSegment(v(W, CUT_Y + rc, 0), v(W, H - r, 0)).toShape(),
        Part.makeCircle(r, v(W - r, H - r, 0), z, 0, 90),
        Part.LineSegment(v(W - r, H, 0), v(r, H, 0)).toShape(),
        Part.makeCircle(r, v(r, H - r, 0), z, 90, 180),
        Part.LineSegment(v(0, H - r, 0), v(0, r, 0)).toShape(),
        Part.makeCircle(r, v(r, r, 0), z, 180, 270),
    ]
    solid = Part.Face(Part.Wire(edges)).extrude(v(0, 0, BOARD_T))
    for x, y in HOLES:
        solid = solid.cut(Part.makeCylinder(HOLE_DIA / 2, BOARD_T + 2, v(x, y, -1)))
    return solid


def main():
    comps = read_easyeda(EPRO)
    parts = [("Board", board_shape())]
    pos = {}
    for c in sorted(comps, key=lambda c: c["des"]):
        des = c["des"]
        cx, cy = c["x"], BOARD_H + c["y"]
        pos[des] = (cx, cy)
        if des in NOT_FITTED:
            continue
        h = HEIGHTS[des]
        top = BOARD_T + h
        if des == "J1":
            shp = placed_box(fp_rects(c["fp"])[0], cx, cy, c["angle"], -h, 0.0)
        elif des == "SW1":
            bl, bw = SLIDE_BODY
            kl, kw = SLIDE_KNOB
            y = SLIDE_PAD_Y
            shp = Part.makeCompound([
                placed_box((-bl / 2, y - bw / 2, bl / 2, y + bw / 2), cx, cy, c["angle"],
                           BOARD_T, top),
                placed_box((-kl / 2, y - kw / 2, kl / 2, y + kw / 2), cx, cy, c["angle"],
                           top, top + SLIDE_KNOB_H)])
        elif des == "BZ1":
            shp = Part.makeCylinder(BZ1_DIA / 2, h, App.Vector(cx, cy, BOARD_T))
        else:
            shp = placed_box(fp_rects(c["fp"])[0], cx, cy, c["angle"], BOARD_T, top)
        parts.append((des, shp))

    # BNO055 モジュール: 2 列（J10 / J11）の中央
    (x10, y10), (x11, y11) = pos["J10"], pos["J11"]
    bx, by = (x10 + x11) / 2, (y10 + y11) / 2
    z0 = BOARD_T + HEIGHTS["J10"] + BNO_PIN_BASE
    w, d = BNO_SIZE
    bno = Part.makeBox(w, d, BNO_T + BNO_PARTS_H, App.Vector(bx - w / 2, by - d / 2, z0))
    parts.append(("BNO055", bno))

    doc = App.newDocument(NAME.replace("-", "_"))
    part = doc.addObject("App::Part", "Part")
    part.Label = "krilly_HAT"
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

    env = Part.makeCompound([s for _, s in parts])
    bb = env.BoundBox
    print(f"saved {path}")
    print(f"bbox X {bb.XMin:.2f}..{bb.XMax:.2f}  Y {bb.YMin:.2f}..{bb.YMax:.2f}  "
          f"Z {bb.ZMin:.2f}..{bb.ZMax:.2f}")
    for label, shp in parts[1:]:
        b = shp.BoundBox
        print(f"  {label:7s} x {b.XMin:6.2f}..{b.XMax:6.2f}  y {b.YMin:6.2f}..{b.YMax:6.2f}  "
              f"z {b.ZMin:5.2f}..{b.ZMax:5.2f}")
    # 部品どうし（基板以外）の重なり。フットプリントの外形は実物より大きめなので目安。
    # BZ1 x SW1 の 0.13 は実物でも接している所
    worst = []
    for i in range(1, len(parts)):
        for j in range(i + 1, len(parts)):
            v = parts[i][1].common(parts[j][1]).Volume
            if v > 1e-3:
                worst.append(f"{parts[i][0]} x {parts[j][0]}: {v:.2f}")
    print("overlaps: " + (", ".join(worst) if worst else "none"))


try:
    main()
finally:
    import sys
    sys.stdout.flush()  # flush しないと、終了時に標準出力の最後の部分が捨てられる
