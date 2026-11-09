"""秋月 AE-L6470DRV（L6470 ステッピングモータドライバ）の外形モデルを作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_l6470.py </dev/null

出力: hardware/cad/ae-l6470.FCStd、hardware/cad/export/ae-l6470.step

寸法は秋月の基板寸法図（AE-L6470DRV_board.pdf）による。図に数値の無い部品の位置と
大きさは、図を 600 dpi で読み取った値。高さ・厚さ・穴径は実物で確認した値（電解
コンデンサ C5 ≈ 12.5、端子台 CN1〜3 ≈ 11、ピンヘッダー 8.5、基板 1.6、取付穴 ø3.2、
裏のピンの先 2）。IC の厚さだけ一般的な値。

    - Board     : 40 × 50 × 1.6、四隅の取付穴（中心は各辺から 4）
    - C5        : 電解コンデンサ 100µF/63V（ø10、高さ 12.5）。最も背が高い
    - CN1       : 電源の端子台 2P（VS+ / GND）。電線は基板の上辺（+Y）側から入る
    - CN2 / CN3 : モータの端子台 2P（A 相 / B 相）。電線は左辺（−X）側から入る
    - CN4       : マイコン接続の 2×5 ピンヘッダー、JP1: 2×2 ピンヘッダー
    - U1        : L6470（HTSSOP28）
    - PinTails  : 裏面に出る端子台・ヘッダーのピンの先（長さ 2、実物）

座標系: 原点 = 基板の左下の角（寸法図の向き。CN1 が上、CN2/3 が左、CN4 が右）、
裏面が Z=0、表面が Z=BOARD_T。
"""

import os

import FreeCAD as App
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.dirname(HERE)
NAME = "ae-l6470"

BOARD_W, BOARD_H = 40.0, 50.0
BOARD_T = 1.6               # 基板厚（実物）
HOLE_DIA = 3.2              # 取付穴（実物。寸法図の灰色の円 ø4.2 はランド込み）
HOLE_INSET = 4.0            # 穴の中心 ↔ 各辺（寸法図の 4）
PITCH = 2.54
TB_PITCH = 5.08             # 端子台のピッチ

C5_XY = (14.15, 42.5)       # 電解コンデンサの中心（図からの読み取り）
C5_DIA = 10.0
C5_H = 12.5                 # 実測（基板表面から）
C5_BASE = 10.3              # 面実装の台座（一般的な ø10 品）
C5_BASE_H = 1.5

TB_H = 11.0                 # 端子台の高さ（実測）
TB_DEPTH = 6.8              # 端子台の奥行き（図の外形）
# 端子台: (名前, ピン 1 の XY, ピン 2 の XY, 本体の外形 x0, y0, x1, y1)
TERMINALS = [
    ("CN1", (23.5, 46.35), (28.58, 46.35), (20.96, 42.1, 31.12, 48.9)),
    ("CN2", (3.49, 32.70), (3.49, 27.62), (0.85, 25.08, 7.65, 35.24)),
    ("CN3", (3.49, 22.54), (3.49, 17.46), (0.85, 14.92, 7.65, 25.08)),
]

HDR_X = (35.24, 37.78)      # CN4 / JP1 の 2 列（右列は右辺から 2.22）
CN4_Y0 = 20.0               # CN4 の 1・2 番ピンの Y（寸法図の 20）
JP1_Y = (12.38, 14.92)      # JP1 の 2 行（CN4 の 3 ピッチ・2 ピッチ下）
HDR_BASE_H = 2.5            # ピンヘッダーの樹脂部（全高 8.5 は実物どおり）
HDR_PIN_UP = 6.0            # 樹脂部から上のピン長
HDR_PIN = 0.64              # ピンの角

U1_XY = (19.9, 25.2)        # L6470 の中心（図からの読み取り）
U1_SIZE = (4.4, 9.7, 1.2)   # HTSSOP28（長辺が Y）

TAIL_LEN = 2.0              # 裏面に出るピンの長さ（はんだ付け後に切った長さ。実物）
TB_PIN_DIA = 1.0


def box(x0, y0, z0, x1, y1, z1):
    return Part.makeBox(x1 - x0, y1 - y0, z1 - z0, App.Vector(x0, y0, z0))


def header(xs, ys):
    """樹脂部 + 上に出るピン。ピンの XY の一覧を返す（裏のピンの先に使う）。"""
    half = PITCH / 2
    base = box(min(xs) - half, min(ys) - half, BOARD_T,
               max(xs) + half, max(ys) + half, BOARD_T + HDR_BASE_H)
    pins, xy = [], []
    for x in xs:
        for y in ys:
            p = HDR_PIN / 2
            pins.append(box(x - p, y - p, BOARD_T + HDR_BASE_H,
                            x + p, y + p, BOARD_T + HDR_BASE_H + HDR_PIN_UP))
            xy.append((x, y))
    return base.fuse(pins), xy


def main():
    board = box(0, 0, 0, BOARD_W, BOARD_H, BOARD_T)
    for x in (HOLE_INSET, BOARD_W - HOLE_INSET):
        for y in (HOLE_INSET, BOARD_H - HOLE_INSET):
            board = board.cut(Part.makeCylinder(HOLE_DIA / 2, BOARD_T + 2, App.Vector(x, y, -1)))

    cx, cy = C5_XY
    hb = C5_BASE / 2
    c5 = box(cx - hb, cy - hb, BOARD_T, cx + hb, cy + hb, BOARD_T + C5_BASE_H).fuse(
        Part.makeCylinder(C5_DIA / 2, C5_H - C5_BASE_H, App.Vector(cx, cy, BOARD_T + C5_BASE_H)))

    parts = [("Board", board), ("C5", c5)]
    tails = []
    for name, p1, p2, (x0, y0, x1, y1) in TERMINALS:
        parts.append((name, box(x0, y0, BOARD_T, x1, y1, BOARD_T + TB_H)))
        for x, y in (p1, p2):
            tails.append(Part.makeCylinder(TB_PIN_DIA / 2, TAIL_LEN, App.Vector(x, y, -TAIL_LEN)))

    cn4, cn4_xy = header(HDR_X, [CN4_Y0 + PITCH * k for k in range(5)])
    jp1, jp1_xy = header(HDR_X, JP1_Y)
    parts += [("CN4", cn4), ("JP1", jp1)]
    for x, y in cn4_xy + jp1_xy:
        p = HDR_PIN / 2
        tails.append(box(x - p, y - p, -TAIL_LEN, x + p, y + p, 0))

    ux, uy = U1_XY
    w, h, t = U1_SIZE
    parts.append(("U1", box(ux - w / 2, uy - h / 2, BOARD_T, ux + w / 2, uy + h / 2, BOARD_T + t)))
    parts.append(("PinTails", Part.makeCompound(tails)))

    doc = App.newDocument(NAME.replace("-", "_"))
    part = doc.addObject("App::Part", "Part")
    part.Label = "AE-L6470DRV"
    objs = []
    for label, shp in parts:
        o = doc.addObject("Part::Feature", label)
        o.Shape = shp
        part.addObject(o)
        objs.append(o)
    env = Part.makeCompound([s for _, s in parts])
    doc.recompute()
    path = os.path.join(CAD_DIR, f"{NAME}.FCStd")
    doc.saveAs(path)
    Part.export(objs, os.path.join(CAD_DIR, "export", f"{NAME}.step"))

    bb = env.BoundBox
    print(f"saved {path}")
    print(f"bbox X {bb.XMin:.2f}..{bb.XMax:.2f}  Y {bb.YMin:.2f}..{bb.YMax:.2f}  "
          f"Z {bb.ZMin:.2f}..{bb.ZMax:.2f}")
    # 部品どうし・部品と基板の重なり（0 であること）
    worst = 0.0
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            v = parts[i][1].common(parts[j][1]).Volume
            if v > 1e-6:
                print(f"  overlap {parts[i][0]} x {parts[j][0]}: {v:.3f}")
            worst = max(worst, v)
    print(f"max overlap {worst:.4f}")


main()
