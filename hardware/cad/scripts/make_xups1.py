"""Geekworm X-UPS1（18650 × 4 の UPS 基板）のモデルをメーカー配布の DXF から作る。

    hardware/cad/scripts/fcrun hardware/cad/scripts/make_xups1.py </dev/null

入力: hardware/cad/vendor/X-UPS1-PCB.dxf（メーカー配布。PADS 出力の R12 形式）
      再配布の許諾が無いのでリポジトリには入れていない（.gitignore 済み）。
      DXF_URL から各自ダウンロードして上の場所に置く。
出力: hardware/cad/x-ups1.FCStd、hardware/cad/export/x-ups1.step

DXF にあるのは 2D の情報だけなので、高さは実物の実測で補う。作るのは次のとおり:
    - Board      : BOARD_OUTLINE の外形（77.1 × 107.5、角 R3）を BOARD_T で押し出し、
                   TOOL_TOP / TOOL_BOT の丸穴と DRILLHOLE の長穴を開けた基板
    - Holders    : 18650 × 2 本用の電池ホルダー × 2。外形は PLACEMENT_OUTLINE_TOP、高さは実測
    - Cells      : 18650 × 4（B1〜B4）。Geekworm の許容最大寸法（ø18.5 × 65.3）で作る。
                   中心はホルダーのピン穴から決めた推定。てっぺんは基板表面から 20（実測）
    - GpioHeader : GPIO ヘッダー（2×9）。重ねて使うための貫通型で、表・裏とも
                   基板面から 9 出っ張る（実測）
    - BottomEnvelope : 裏面の部品が占めうる範囲（外形 × 裏面最大高さ）。非表示。
                   裏側の干渉チェック用
    - OutlineTop / OutlineBottom : 部品外形（PLACEMENT_OUTLINE_*）の線。基板の表・裏の面に置く
上下の縁のコネクタ（DC ジャック・USB-A・XH など）は線だけ。どれも電池より低い（実物で
確認）ので、表側の高さは電池で決まる。

取付穴は ø3.0 × 4（DXF の TOOL_TOP で「FM」が付いたもの）。メーカーのプレート図
X-UPS1-Plate.pdf の寸法（端から 3.5 / 12、反対側は 6 と 18.6）と一致する。

座標系: 原点 = 基板外形の左下の角（DXF の座標から平行移動しただけ）、基板の裏面が Z=0、
表面が Z=BOARD_T。+X = 基板の短辺方向、+Y = 長辺方向（DXF のまま）。
表面 = 電池ホルダーのある面（DXF の *_TOP）。
"""

import math
import os
import sys

import FreeCAD as App
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.dirname(HERE)
DXF = os.path.join(CAD_DIR, "vendor", "X-UPS1-PCB.dxf")
DXF_URL = "https://wiki.geekworm.com/File:X-UPS1-PCB.dxf"
NAME = "x-ups1"

BOARD_T = 1.6            # 基板厚（実測）
HOLDER_H = 15.0          # 電池ホルダーの高さ（基板表面から。実測）
GPIO_H = 9.0             # GPIO ヘッダーの高さ（表・裏とも基板面から）。裏面で一番高い部品（実測）
CELL_DIA = 18.5          # 18650 の直径（Geekworm の許容最大）
CELL_LEN = 65.3          # 18650 の長さ（同上。フラットトップ・保護回路なし）
CELL_GAP = 1.5           # 基板表面 ↔ 電池の下端。てっぺんが表面から 20 になる値（実測）
# 電池ホルダーの外形（DXF 座標、PLACEMENT_OUTLINE_TOP の長方形）: x 全幅 × y の範囲
HOLDER_Y_DXF = [(43.07, 82.97), (3.17, 43.07)]
# GPIO ヘッダーの外形（DXF 座標、PLACEMENT_OUTLINE_BOTTOM の長方形）
GPIO_RECT_DXF = (-29.94, 86.17, -7.08, 91.17)
# 電池の中心線の Y（DXF 座標）。シルクの B1〜B4 と接点の円・ホルダーのピン穴の Y
CELL_Y_DXF = [72.57, 53.47, 32.67, 13.57]
# 電池の中心の X（DXF 座標）。ホルダーのピン穴 x=-59.835 / 12.975 の中点
CELL_X_DXF = (-59.835 + 12.975) / 2


# ---- DXF（R12）の読み込み ------------------------------------------------
def read_entities(path):
    """ENTITIES の POLYLINE（頂点と bulge）と、その画層を返す。"""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    pairs = [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines) - 1, 2)]
    polys = []
    in_ent = False
    cur = None      # 今読んでいる POLYLINE
    vtx = None      # 今読んでいる VERTEX
    for i, (code, val) in enumerate(pairs):
        if code == "0":
            if val == "SECTION" and i + 1 < len(pairs):
                in_ent = pairs[i + 1][1] == "ENTITIES"
            if not in_ent:
                continue
            if val == "POLYLINE":
                cur = {"layer": None, "closed": False, "pts": []}
                vtx = None
            elif val == "VERTEX" and cur is not None:
                vtx = [0.0, 0.0, 0.0]
                cur["pts"].append(vtx)
            elif val == "SEQEND" and cur is not None:
                polys.append(cur)
                cur = vtx = None
            else:
                vtx = None
            continue
        if cur is None:
            continue
        if vtx is None:
            if code == "8":
                cur["layer"] = val
            elif code == "70":
                cur["closed"] = bool(int(val) & 1)
        else:
            if code == "10":
                vtx[0] = float(val)
            elif code == "20":
                vtx[1] = float(val)
            elif code == "42":
                vtx[2] = float(val)
    return polys


def poly_edges(poly, z, offset):
    """POLYLINE を辺（直線 / bulge の円弧）に変える。長さ 0 の辺は捨てる。"""
    pts = poly["pts"]
    seq = pts + ([pts[0]] if poly["closed"] else [])
    edges = []
    for (x1, y1, b), (x2, y2, _) in zip(seq, seq[1:]):
        p1 = App.Vector(x1 - offset[0], y1 - offset[1], z)
        p2 = App.Vector(x2 - offset[0], y2 - offset[1], z)
        if (p2 - p1).Length < 1e-6:
            continue
        if abs(b) < 1e-9:
            edges.append(Part.LineSegment(p1, p2).toShape())
            continue
        # bulge = tan(中心角/4)。正なら CCW。弦の中点から垂直に矢高ぶん出した点を通る
        chord = p2 - p1
        mid = (p1 + p2) * 0.5
        sagitta = b * chord.Length / 2
        normal = App.Vector(-chord.y, chord.x, 0).normalize()
        through = mid - normal * sagitta
        edges.append(Part.Arc(p1, through, p2).toShape())
    return edges


def circle_of(poly):
    """bulge = ±1 の 2 区間でできた円なら (cx, cy, 直径) を返す。

    長穴の半円（bulge ±1 が 1 区間だけ）は円ではないので、2 区間とも ±1 を要求する。
    """
    pts = poly["pts"]
    if len(pts) < 2 or any(abs(abs(v[2]) - 1) > 1e-6 for v in pts[:2]):
        return None
    (x1, y1, _), (x2, y2, _) = pts[0], pts[1]
    d = math.hypot(x2 - x1, y2 - y1)
    if d < 1e-6:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, d)


def dedupe_edges(edges):
    seen, out = set(), []
    for e in edges:
        a, b = e.Vertexes[0].Point, e.Vertexes[-1].Point
        m = e.valueAt((e.FirstParameter + e.LastParameter) / 2)
        key = tuple(sorted([(round(a.x, 3), round(a.y, 3)), (round(b.x, 3), round(b.y, 3))])
                    ) + ((round(m.x, 3), round(m.y, 3)),)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


# ---- 形状 ----------------------------------------------------------------
def build(polys):
    by_layer = {}
    for p in polys:
        by_layer.setdefault(p["layer"], []).append(p)

    # 外形: 太さ 0.25 の線と、その端点に描かれた小さな円（描画上の飾り）が混ざっている。
    # 円を除いて重複を落とし、1 本の閉じた線にする
    outline_polys = [p for p in by_layer["BOARD_OUTLINE"] if circle_of(p) is None]
    raw = []
    for p in outline_polys:
        raw += poly_edges(p, 0, (0, 0))
    raw = dedupe_edges(raw)
    bb = Part.makeCompound(raw).BoundBox
    offset = (bb.XMin, bb.YMin)
    edges = []
    for p in outline_polys:
        edges += poly_edges(p, 0, offset)
    wire = Part.Wire(Part.__sortEdges__(dedupe_edges(edges)))
    board = Part.Face(wire).extrude(App.Vector(0, 0, BOARD_T))

    # 丸穴: 部品のリード穴・ホルダーのピン穴・取付穴
    holes = []
    for layer in ("TOOL_TOP", "TOOL_BOT"):
        for p in by_layer.get(layer, []):
            c = circle_of(p)
            if c:
                holes.append((c[0] - offset[0], c[1] - offset[1], c[2], layer))
    for x, y, d, _ in holes:
        board = board.cut(Part.makeCylinder(d / 2, BOARD_T + 2, App.Vector(x, y, -1)))

    # 長穴（DC ジャックのピン）: DRILLHOLE の線を閉じた輪にまとめて抜く
    slot_edges = []
    for p in by_layer.get("DRILLHOLE", []):
        if circle_of(p) is None:  # 端点の飾りの円（直径 0）は poly_edges で消える
            slot_edges += poly_edges(p, -1, offset)
    slot_edges = dedupe_edges(slot_edges)
    n_slots = 0
    for group in Part.sortEdges(slot_edges):
        w = Part.Wire(group)
        if w.isClosed():
            board = board.cut(Part.Face(w).extrude(App.Vector(0, 0, BOARD_T + 2)))
            n_slots += 1

    def outline(layer, z):
        es = []
        for p in by_layer.get(layer, []):
            es += poly_edges(p, z, offset)
        return Part.makeCompound(dedupe_edges(es))

    top = outline("PLACEMENT_OUTLINE_TOP", BOARD_T)
    bottom = outline("PLACEMENT_OUTLINE_BOTTOM", 0.0)

    cells = []
    zc = BOARD_T + CELL_GAP + CELL_DIA / 2
    for y in CELL_Y_DXF:
        base = App.Vector(CELL_X_DXF - offset[0] - CELL_LEN / 2, y - offset[1], zc)
        cells.append(Part.makeCylinder(CELL_DIA / 2, CELL_LEN, base, App.Vector(1, 0, 0)))

    holders = []
    for y0, y1 in HOLDER_Y_DXF:
        box = Part.makeBox(bb.XLength, y1 - y0, HOLDER_H,
                           App.Vector(0, y0 - offset[1], BOARD_T))
        holders.append(box.common(Part.Face(wire).extrude(App.Vector(0, 0, 50))))

    gx0, gy0, gx1, gy1 = GPIO_RECT_DXF
    # 表側（基板表面から上）と裏側（裏面から下）の 2 つの箱
    gpio = Part.makeCompound([
        Part.makeBox(gx1 - gx0, gy1 - gy0, GPIO_H,
                     App.Vector(gx0 - offset[0], gy0 - offset[1], z0))
        for z0 in (BOARD_T, -GPIO_H)
    ])
    env = Part.Face(wire).extrude(App.Vector(0, 0, -GPIO_H))

    return dict(board=board.removeSplitter(), holes=holes, n_slots=n_slots, top=top,
                bottom=bottom, cells=Part.makeCompound(cells),
                holders=Part.makeCompound(holders), gpio=gpio, env=env, offset=offset)


def main():
    if not os.path.exists(DXF):
        print(f"make_xups1: DXF がありません: {DXF}\n"
              f"  {DXF_URL} からダウンロードして、この場所に置いてください。",
              file=sys.stderr, flush=True)  # flush しないと終了時に捨てられる
        raise SystemExit(1)
    polys = read_entities(DXF)
    r = build(polys)

    doc = App.newDocument(NAME.replace("-", "_"))
    part = doc.addObject("App::Part", "Part")
    part.Label = "X-UPS1"
    objs = {}
    for label, key in (("Board", "board"), ("Holders", "holders"), ("Cells", "cells"),
                       ("GpioHeader", "gpio"), ("OutlineTop", "top"),
                       ("OutlineBottom", "bottom"), ("BottomEnvelope", "env")):
        o = doc.addObject("Part::Feature", label)
        o.Shape = r[key]
        part.addObject(o)
        objs[label] = o
    objs["BottomEnvelope"].Visibility = False
    doc.recompute()
    path = os.path.join(CAD_DIR, f"{NAME}.FCStd")
    doc.saveAs(path)
    Part.export([objs[k] for k in ("Board", "Holders", "Cells", "GpioHeader")],
                os.path.join(CAD_DIR, "export", f"{NAME}.step"))

    bb = r["board"].BoundBox
    offset = r["offset"]
    print(f"saved {path}")
    print(f"DXF offset ({offset[0]:.3f}, {offset[1]:.3f}) -> origin")
    print(f"board {bb.XLength:.2f} x {bb.YLength:.2f} x {bb.ZLength:.2f}  "
          f"holes {len(r['holes'])}  slots {r['n_slots']}")
    sizes = {}
    for x, y, d, layer in r["holes"]:
        sizes.setdefault(round(d, 2), []).append((round(x, 2), round(y, 2)))
    for d in sorted(sizes):
        print(f"  d{d}: {len(sizes[d])}  {sizes[d][:6]}")
    cb, hb, gb = r["cells"].BoundBox, r["holders"].BoundBox, r["gpio"].BoundBox
    print(f"holders z {hb.ZMin:.2f}..{hb.ZMax:.2f}  cells z {cb.ZMin:.2f}..{cb.ZMax:.2f} "
          f"x {cb.XMin:.2f}..{cb.XMax:.2f}")
    print(f"gpio x {gb.XMin:.2f}..{gb.XMax:.2f} y {gb.YMin:.2f}..{gb.YMax:.2f} "
          f"z {gb.ZMin:.2f}..{gb.ZMax:.2f}")
    print(f"overall z {min(gb.ZMin, bb.ZMin):.2f}..{max(cb.ZMax, hb.ZMax):.2f}")


main()
