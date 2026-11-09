"""Raspberry Pi 5 の簡易外形モデルを作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_rpi5.py </dev/null

出力: hardware/cad/rpi5.FCStd、hardware/cad/export/rpi5.step

干渉チェック用に、基板と背の高い / 外に出る部品だけを箱で作る:
    - Board      : 85 × 56 × 1.3、角 R3、取付穴 ø2.7 × 4（58 × 49）
    - Usb        : USB-A の 2 段コネクタ × 2（基板の端から 3 はみ出す）
    - Ethernet   : RJ45（同上）
    - Gpio       : 40 ピンのピンヘッダー（樹脂部 + ピン）と、裏に出るピンの先
    - Camera     : カメラ / ディスプレイ用の FPC コネクタ × 2
    - SdSocket   : 裏面の microSD ソケットと、基板の端から出るカード
USB-C・micro HDMI・CPU などの低い部品は作らない。

寸法の出典: 公式の機械図面 raspberry-pi-5-mechanical-drawing.pdf（外形・穴・コネクタの
位置）と、公式 STEP（MIT）から測った部品の範囲・高さ。どちらも目安で、実物との照合は
していない。

座標系（公式 STEP・機械図面と同じ）: 原点 = 基板の左下の角（USB-C / micro HDMI の辺 =
下辺、GPIO の辺 = 上辺、USB / Ethernet の辺 = 右辺 X=85）、基板の裏面が Z=0、部品面が +Z。
"""

import os

import FreeCAD as App
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.dirname(HERE)
NAME = "rpi5"

BOARD_W, BOARD_H, BOARD_T = 85.0, 56.0, 1.3
CORNER_R = 3.0
HOLE_DIA = 2.7
HOLES = [(3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)]

# 部品の箱: (x0, y0, x1, y1, 上端の Z)。下端は基板の表面
USB = [(70.4, 21.75, 88.0, 36.25, 17.5), (70.4, 39.75, 88.0, 54.25, 17.5)]
ETHERNET = (66.5, 2.3, 88.05, 18.2, 14.6)
CAMERA = [(47.4, 0.4, 50.2, 16.0, BOARD_T + 4.0), (53.6, 0.4, 56.4, 16.0, BOARD_T + 4.0)]
GPIO_BASE = (7.1, 49.96, 57.9, 55.04, BOARD_T + 2.54)
GPIO_PIN_TOP = BOARD_T + 8.5      # ピンの先（一般的な 2.54 mm ヘッダー）
GPIO_PIN = 0.64
GPIO_TAIL = 1.45                  # 裏に出るピンの先（公式 STEP の裏面の最下点）
GPIO_ORIGIN = (8.37, 51.23)       # 1 番ピン（図の左下のピン）。ピッチ 2.54 で 20 × 2
SD_SOCKET = (0.0, 21.9, 14.0, 33.9, -1.45)   # 裏面: (x0, y0, x1, y1, 下端の Z)
SD_CARD = (-2.0, 22.4, 0.0, 33.4, -1.15, -0.45)  # 基板の左の端から出るカード


def box(x0, y0, z0, x1, y1, z1):
    return Part.makeBox(x1 - x0, y1 - y0, z1 - z0, App.Vector(x0, y0, z0))


def board_shape():
    r = CORNER_R
    face = Part.Face(Part.Wire(Part.makePolygon([
        App.Vector(r, 0, 0), App.Vector(BOARD_W - r, 0, 0)]).Edges + [
        Part.makeCircle(r, App.Vector(BOARD_W - r, r, 0), App.Vector(0, 0, 1), 270, 360),
        Part.LineSegment(App.Vector(BOARD_W, r, 0), App.Vector(BOARD_W, BOARD_H - r, 0)).toShape(),
        Part.makeCircle(r, App.Vector(BOARD_W - r, BOARD_H - r, 0), App.Vector(0, 0, 1), 0, 90),
        Part.LineSegment(App.Vector(BOARD_W - r, BOARD_H, 0), App.Vector(r, BOARD_H, 0)).toShape(),
        Part.makeCircle(r, App.Vector(r, BOARD_H - r, 0), App.Vector(0, 0, 1), 90, 180),
        Part.LineSegment(App.Vector(0, BOARD_H - r, 0), App.Vector(0, r, 0)).toShape(),
        Part.makeCircle(r, App.Vector(r, r, 0), App.Vector(0, 0, 1), 180, 270),
    ]))
    solid = face.extrude(App.Vector(0, 0, BOARD_T))
    for x, y in HOLES:
        solid = solid.cut(Part.makeCylinder(HOLE_DIA / 2, BOARD_T + 2, App.Vector(x, y, -1)))
    return solid


def gpio_shape():
    x0, y0, x1, y1, ztop = GPIO_BASE
    shapes = [box(x0, y0, BOARD_T, x1, y1, ztop)]
    p = GPIO_PIN / 2
    for i in range(20):
        for j in range(2):
            cx, cy = GPIO_ORIGIN[0] + 2.54 * i, GPIO_ORIGIN[1] + 2.54 * j
            shapes.append(box(cx - p, cy - p, ztop, cx + p, cy + p, GPIO_PIN_TOP))
            shapes.append(box(cx - p, cy - p, -GPIO_TAIL, cx + p, cy + p, 0))
    return Part.makeCompound(shapes)


def main():
    parts = [
        ("Board", board_shape()),
        ("Usb", Part.makeCompound([box(x0, y0, BOARD_T, x1, y1, zt) for x0, y0, x1, y1, zt in USB])),
        ("Ethernet", box(*ETHERNET[:2], BOARD_T, *ETHERNET[2:4], ETHERNET[4])),
        ("Gpio", gpio_shape()),
        ("Camera", Part.makeCompound([box(x0, y0, BOARD_T, x1, y1, zt)
                                      for x0, y0, x1, y1, zt in CAMERA])),
        ("SdSocket", Part.makeCompound([
            box(*SD_SOCKET[:2], SD_SOCKET[4], *SD_SOCKET[2:4], 0),
            box(SD_CARD[0], SD_CARD[1], SD_CARD[4], SD_CARD[2], SD_CARD[3], SD_CARD[5])])),
    ]
    doc = App.newDocument(NAME)
    part = doc.addObject("App::Part", "Part")
    part.Label = "RaspberryPi5"
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
    worst = max(parts[i][1].common(parts[j][1]).Volume
                for i in range(len(parts)) for j in range(i + 1, len(parts)))
    print(f"max overlap between parts {worst:.4f}")


main()
