#!/usr/bin/env freecadcmd
# -*- coding: utf-8 -*-
"""FCStd から三面図(SVG)を生成する。

使い方:
    hardware/cad/scripts/fcrun hardware/cad/scripts/three_view.py

出力: hardware/cad/export/ に PARTS × LAYOUTS の SVG。
    <部品名>-3view.svg       … 平面図の前方が紙面上（krilly-frame.svg と同じ向き）
    <部品名>-3view-side.svg  … 側面図を主投影図にした車両一般配置図ふうの構成

図は **車体座標系**(docs/coordinate-frames.md: +x 前 / +y 左 / +z 上、原点は車体
中心)で描く。FCStd がその座標系で保存されていれば `MODEL_TO_BODY_DEG = 0`。
Z 回りにずれた FCStd を描く場合はその角度を入れて投影前に回転させる。

## 投影法

どちらのレイアウトも第三角法。平面図と真下の立面図は紙面の横軸を共有し、その
立面図と右隣の図は縦軸を共有する。この整列規則が平面図の前後向きを一意に決める
ので、「平面図の前方をどちらに向けるか」は主投影図の選び方と同義になる。

    layout "b"                        layout "side"
    [ 平面図 上=+x 右=−y ]            [ 平面図 上=+y 右=+x ]
    [ 背面図 上=+z 右=−y ][側面図]     [ 側面図 上=+z 右=+x ][正面図]

第一角法にしても各図の像の向きは変わらない（配置だけが変わる）ので、b で正面図を
中央に置くことはできない。

## 注意

FCStd は FreeCAD 1.1.0 で作られている。**recompute / save は行わない**
（古い版で書き戻すとファイルが壊れる）。保存済みの BREP をそのまま投影する。

Windows 版 freecadcmd.exe から実行すると Python の既定が cp932 + CRLF になる。
ファイル出力は encoding/newline を明示している。
"""

import io
import os
import sys

import FreeCAD
import Part
import TechDraw

V = FreeCAD.Vector

# ---------------------------------------------------------------- 設定

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.normpath(os.path.join(HERE, ".."))
EXPORT = os.path.join(CAD_DIR, "export")
DATE = "2026-08-14"

PARTS = [
    ("krilly-chassis.FCStd", "Body", "krilly-chassis",
     "krilly-chassis / Body"),
    ("krilly-middle-plate.FCStd", "Body", "krilly-middle-plate",
     "krilly-middle-plate / Body"),
    ("krilly-roof.FCStd", "Body", "krilly-roof",
     "krilly-roof / Body"),
]

# 3部品とも 2026-07-25 に Body.Placement で車体座標系へ揃えたので 0。
MODEL_TO_BODY_DEG = 0.0

DRAW_HIDDEN = True      # かくれ線を描くか
DEFLECTION = 0.02       # 曲線の折れ線近似 [mm]

# 尺度 1:1 を保ったまま収まる最小の用紙を選ぶ。SVG の 1 ユーザー単位 = 1mm。
PAGES = [("A4横", 297.0, 210.0), ("A3横", 420.0, 297.0)]
MARGIN = 10.0
GAP = 22.0                      # 図同士の間隔
BLOCK_X, BLOCK_Y = 44.0, 24.0   # 図面ブロック左上

LW_OUTLINE = 0.35
LW_HIDDEN = 0.18
LW_CENTER = 0.13
LW_DIM = 0.13
LW_FRAME = 0.5

FONT = "'Noto Sans CJK JP','Noto Sans JP','IPAGothic',sans-serif"
DIM_COLOR = "#0b6"
CENTER_COLOR = "#c22"
CENTER_DASH = "5,1.2,0.9,1.2"

# 視点の定義: normal = 物体から視点へ向かう向き、up = 紙面上。right = up x normal。
# スロットは plan（左上）/ elev1（その下）/ elev2（elev1 の右）。
LAYOUTS = [
    {
        "key": "b", "suffix": "-3view",
        "desc": "平面図の前方を紙面上に固定（docs/images/krilly-frame.svg と同じ向き）",
        "slots": [
            ("plan",  V(0, 0, 1), V(1, 0, 0), "平面図  TOP",
             "視点 +z（上から見下ろす）／上が +x 前方・左が +y 左"),
            ("elev1", V(-1, 0, 0), V(0, 0, 1), "背面図  REAR",
             "視点 −x（車体後方から）／右が車体右 −y"),
            ("elev2", V(0, -1, 0), V(0, 0, 1), "側面図  SIDE",
             "視点 −y（車体右側から）／右が +x 前方"),
        ],
    },
    {
        "key": "side", "suffix": "-3view-side",
        "desc": "側面図を主投影図にした構成（車両の一般配置図の慣例）",
        "slots": [
            ("plan",  V(0, 0, 1), V(0, 1, 0), "平面図  TOP",
             "視点 +z（上から見下ろす）／右が +x 前方・上が +y 左"),
            ("elev1", V(0, -1, 0), V(0, 0, 1), "側面図  SIDE",
             "視点 −y（車体右側から）／右が +x 前方"),
            ("elev2", V(1, 0, 0), V(0, 0, 1), "正面図  FRONT",
             "視点 +x（車体前方から）／右が車体左 +y"),
        ],
    },
]

# ---------------------------------------------------------------- 投影


def view_matrix(right, up, normal):
    """点を (right, up, normal) 基底の座標に写す行列（行が基底ベクトル）。"""
    return FreeCAD.Matrix(
        right.x, right.y, right.z, 0,
        up.x, up.y, up.z, 0,
        normal.x, normal.y, normal.z, 0,
        0, 0, 0, 1,
    )


def edge_points(edge):
    curve = edge.Curve
    if isinstance(curve, Part.Line) and len(edge.Vertexes) == 2:
        a, b = edge.Vertexes
        return [(a.X, a.Y), (b.X, b.Y)]
    try:
        pts = edge.discretize(Deflection=DEFLECTION)
    except Exception:
        pts = edge.discretize(Number=48)
    return [(p.x, p.y) for p in pts]


def project(shape, right, up, normal):
    """HLR で投影し、(可視ポリライン, かくれポリライン) を視点座標で返す。"""
    local = shape.copy()
    local.transformShape(view_matrix(right, up, normal))
    # projectEx の方向 = 物体から視点へ向かうベクトル。基底変換後は +Z。
    res = TechDraw.projectEx(local, V(0, 0, 1))

    def collect(idx):                   # (V, V1, VN, VO, VI, H, H1, HN, HO, HI)
        out = []
        for i in idx:
            g = res[i]
            if g is None or g.isNull():
                continue
            for e in g.Edges:
                pts = edge_points(e)
                if len(pts) >= 2:
                    out.append(pts)
        return out
    return collect((0, 3)), collect((5, 8))     # 稜線+外形 / かくれ


# ---------------------------------------------------------------- SVG 部品

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def polyline(pts, width, color, dash=None, opacity=None):
    d = "M " + " L ".join("%.4f %.4f" % (x, y) for x, y in pts)
    a = ['d="%s"' % d, 'stroke="%s"' % color, 'stroke-width="%.3f"' % width,
         'fill="none"', 'stroke-linecap="round"']
    if dash:
        a.append('stroke-dasharray="%s"' % dash)
    if opacity is not None:
        a.append('stroke-opacity="%.2f"' % opacity)
    return "<path %s/>" % " ".join(a)


def line(x1, y1, x2, y2, width, color, dash=None):
    a = ['x1="%.4f"' % x1, 'y1="%.4f"' % y1, 'x2="%.4f"' % x2, 'y2="%.4f"' % y2,
         'stroke="%s"' % color, 'stroke-width="%.3f"' % width]
    if dash:
        a.append('stroke-dasharray="%s"' % dash)
    return "<line %s/>" % " ".join(a)


def text(x, y, s, size=2.6, anchor="middle", color="#111", weight="normal",
         rotate=None):
    tf = ' transform="rotate(%g %.3f %.3f)"' % (rotate, x, y) if rotate else ""
    return ('<text x="%.3f" y="%.3f" font-family="%s" font-size="%.2f" '
            'text-anchor="%s" fill="%s" font-weight="%s"%s>%s</text>'
            % (x, y, FONT, size, anchor, color, weight, tf, esc(s)))


def arrow(x, y, dx, dy, color=DIM_COLOR):
    ln, hw = 2.6, 0.75
    bx, by = x - dx * ln, y - dy * ln
    px, py = -dy * hw, dx * hw
    return ('<polygon points="%.3f,%.3f %.3f,%.3f %.3f,%.3f" fill="%s"/>'
            % (x, y, bx + px, by + py, bx - px, by - py, color))


def dim_h(x1, x2, y, label, tick_to):
    """水平寸法線。tick_to = 引出線を伸ばす y 座標（図形側）。"""
    o = [line(x1, y, x2, y, LW_DIM, DIM_COLOR),
         line(x1, min(y, tick_to), x1, max(y, tick_to), LW_DIM, DIM_COLOR),
         line(x2, min(y, tick_to), x2, max(y, tick_to), LW_DIM, DIM_COLOR)]
    if (x2 - x1) >= 8.0:
        o += [arrow(x1, y, 1, 0), arrow(x2, y, -1, 0)]
    else:                                   # 狭いので矢は外向き
        o += [line(x1 - 4, y, x2 + 4, y, LW_DIM, DIM_COLOR),
              arrow(x1, y, -1, 0), arrow(x2, y, 1, 0)]
    o.append(text((x1 + x2) / 2.0, y - 1.4, label, 2.7, color=DIM_COLOR))
    return o


def dim_v(y1, y2, x, label, tick_to):
    """垂直寸法線。tick_to = 引出線を伸ばす x 座標（図形側）。"""
    o = [line(x, y1, x, y2, LW_DIM, DIM_COLOR),
         line(min(x, tick_to), y1, max(x, tick_to), y1, LW_DIM, DIM_COLOR),
         line(min(x, tick_to), y2, max(x, tick_to), y2, LW_DIM, DIM_COLOR)]
    if (y2 - y1) >= 8.0:
        o += [arrow(x, y1, 0, 1), arrow(x, y2, 0, -1)]
    else:
        o += [line(x, y1 - 4, x, y2 + 4, LW_DIM, DIM_COLOR),
              arrow(x, y1, 0, -1), arrow(x, y2, 0, 1)]
    o.append(text(x - 1.4, (y1 + y2) / 2.0, label, 2.7, color=DIM_COLOR,
                  rotate=-90))
    return o


# ---------------------------------------------------------------- 1枚の作図

def axis_name(vec):
    """基底ベクトルがどの車体軸に沿うかを寸法の呼び名で返す。"""
    for comp, name in ((vec.x, "全長"), (vec.y, "全幅"), (vec.z, "全高")):
        if abs(abs(comp) - 1.0) < 1e-9:
            return name
    raise AssertionError("基底が車体軸に沿っていない: %s" % (vec,))


def build(layout, shape, placement, src, body_name, stem, title):
    bb = shape.BoundBox
    x0, x1 = bb.XMin, bb.XMax           # 前後（+x = 前）
    y0, y1 = bb.YMin, bb.YMax           # 左右（+y = 左）
    z0, z1 = bb.ZMin, bb.ZMax           # 上下（+z = 上）

    def span(v):
        """車体 bbox を基底ベクトル v に射影した範囲（図同士が正確に整列する）。"""
        lo = (min(v.x * x0, v.x * x1) + min(v.y * y0, v.y * y1)
              + min(v.z * z0, v.z * z1))
        hi = (max(v.x * x0, v.x * x1) + max(v.y * y0, v.y * y1)
              + max(v.z * z0, v.z * z1))
        return lo, hi

    # --- 投影と各図の寸法
    views = {}
    for slot, normal, up, name, note in layout["slots"]:
        right = up.cross(normal)
        assert abs(right.Length - 1.0) < 1e-9, slot
        (sx_lo, sx_hi), (sy_lo, sy_hi) = span(right), span(up)
        vis, hid = project(shape, right, up, normal)
        views[slot] = {
            "name": name, "note": note, "right": right, "up": up,
            "normal": normal, "vis": vis, "hid": hid,
            "sx_lo": sx_lo, "sy_hi": sy_hi,
            "w": sx_hi - sx_lo, "h": sy_hi - sy_lo,
        }

    plan, elev1, elev2 = views["plan"], views["elev1"], views["elev2"]

    # --- 投影の整合（第三角法）を検査
    assert (plan["right"] - elev1["right"]).Length < 1e-9, \
        "平面図と真下の立面図が横軸を共有していない"
    assert (elev1["up"] - elev2["up"]).Length < 1e-9, \
        "立面図どうしが縦軸を共有していない"
    assert elev1["up"].dot(plan["normal"]) > 0, \
        "第三角法: 平面図は elev1 の +up 側から見た図でなければならない"
    assert elev1["right"].dot(elev2["normal"]) > 0, \
        "第三角法: elev2 は elev1 の +right 側から見た図でなければならない"
    assert abs(plan["w"] - elev1["w"]) < 1e-9, "平面図と立面図の幅が不一致"
    assert abs(elev1["h"] - elev2["h"]) < 1e-9, "立面図どうしの高さが不一致"

    # --- 尺度 1:1 のまま収まる最小の用紙を選ぶ
    W1, W2, H1, H2 = plan["w"], elev2["w"], plan["h"], elev1["h"]
    n_notes = 10                            # 注記の行数（下の notes と一致させる）
    need_right = BLOCK_X + W1 + GAP + W2
    need_bottom = BLOCK_Y + H1 + GAP + H2 + 11.0        # 図名の分
    title_bottom = BLOCK_Y + 2.0 + (12.0 + 4.4 * n_notes) + 5.0 + 20.0
    page_name = PAGE_W = PAGE_H = None
    for nm, pw, ph in PAGES:
        if (need_right < pw - MARGIN and need_bottom < ph - MARGIN
                and pw - MARGIN - 1.0 - (BLOCK_X + W1 + GAP) > 90.0
                and title_bottom < BLOCK_Y + H1 + GAP):
            page_name, PAGE_W, PAGE_H = nm, pw, ph
            break
    assert page_name, ("%s/%s: どの用紙にも収まらない (要 %.1f x %.1f)"
                       % (stem, layout["key"], need_right, need_bottom))

    # --- 配置と視点座標 -> ページ座標
    origins = {
        "plan":  (BLOCK_X, BLOCK_Y),
        "elev1": (BLOCK_X, BLOCK_Y + H1 + GAP),
        "elev2": (BLOCK_X + W1 + GAP, BLOCK_Y + H1 + GAP),
    }
    for slot, v in views.items():
        rx, ry = origins[slot]
        v["box"] = (rx, ry, v["w"], v["h"])
        v["map"] = (lambda rx=rx, ry=ry, lo=v["sx_lo"], hi=v["sy_hi"]:
                    (lambda sx, sy: (rx + (sx - lo), ry + (hi - sy))))()

    # --- 各図の向きの自己検査: right/up が紙面の右/上に落ちているか
    for slot, v in views.items():
        def page(w):
            return v["map"](w.dot(v["right"]), w.dot(v["up"]))
        o = page(V(0, 0, 0))
        assert page(v["right"])[0] > o[0], "%s: right が紙面右でない" % slot
        assert page(v["up"])[1] < o[1], "%s: up が紙面上でない" % slot

    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" '
           'width="%gmm" height="%gmm" viewBox="0 0 %g %g">'
           % (PAGE_W, PAGE_H, PAGE_W, PAGE_H),
           '<rect width="%g" height="%g" fill="#fff"/>' % (PAGE_W, PAGE_H),
           '<rect x="%g" y="%g" width="%g" height="%g" fill="none" '
           'stroke="#111" stroke-width="%.2f"/>'
           % (MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN,
              LW_FRAME)]

    order = ("plan", "elev1", "elev2")

    out.append('<g id="centerlines">')
    for slot in order:
        v = views[slot]
        rx, ry, w, h = v["box"]
        cx, cy = v["map"](0.0, 0.0)
        ext = 4.0
        if rx - ext <= cx <= rx + w + ext:
            out.append(line(cx, ry - ext, cx, ry + h + ext,
                            LW_CENTER, CENTER_COLOR, CENTER_DASH))
        if ry - ext <= cy <= ry + h + ext:
            out.append(line(rx - ext, cy, rx + w + ext, cy,
                            LW_CENTER, CENTER_COLOR, CENTER_DASH))
    out.append('</g>')

    if DRAW_HIDDEN:
        out.append('<g id="hidden">')
        for slot in order:
            v = views[slot]
            for pts in v["hid"]:
                out.append(polyline([v["map"](a, b) for a, b in pts],
                                    LW_HIDDEN, "#6b7280", "1.6,1.1", 0.75))
        out.append('</g>')

    out.append('<g id="outline">')
    for slot in order:
        v = views[slot]
        for pts in v["vis"]:
            out.append(polyline([v["map"](a, b) for a, b in pts],
                                LW_OUTLINE, "#111"))
    out.append('</g>')

    out.append('<g id="labels">')
    for slot in order:
        v = views[slot]
        rx, ry, w, h = v["box"]
        out.append(text(rx + w / 2.0, ry + h + 6.0, v["name"], 3.4,
                        weight="bold"))
        out.append(text(rx + w / 2.0, ry + h + 9.8, v["note"], 2.3,
                        color="#555"))
    out.append('</g>')

    # --- 全体寸法（平面図の上と左、elev2 の右）
    out.append('<g id="dims">')
    px, py = origins["plan"]
    ex, ey = origins["elev2"]
    out += dim_h(px, px + W1, py - 10.0,
                 "%s %.2f" % (axis_name(plan["right"]), W1), py)
    out += dim_v(py, py + H1, px - 11.0,
                 "%s %.2f" % (axis_name(plan["up"]), H1), px)
    out += dim_v(ey, ey + H2, ex + W2 + 11.0,
                 "%s %.2f" % (axis_name(elev2["up"]), H2), ex + W2)
    out.append('</g>')

    # --- 注記と表題欄（平面図の右の空き領域）
    p = placement
    notes = [
        ("投影法", "第三角法 / 尺度 1:1 / 単位 mm"),
        ("構成", layout["desc"]),
        ("座標系", "車体座標系 +x 前・+y 左・+z 上、原点は車体中心"),
        ("", "(docs/coordinate-frames.md)"),
        ("全長", "%.3f  (前 %+.3f / 後 %+.3f)" % (x1 - x0, x1, x0)),
        ("全幅", "%.3f  (左 %+.3f / 右 %+.3f)" % (y1 - y0, y1, y0)),
        ("全高", "%.3f  (上 %+.3f / 下 %+.3f)" % (z1 - z0, z1, z0)),
        ("配置", "Position (%.3f, %.3f, %.3f)  Yaw %.1f°"
                 % (p.Base.x, p.Base.y, p.Base.z, p.Rotation.toEuler()[0])),
        ("線種", "実線 = 外形線、灰破線 = かくれ線"),
        ("", "赤一点鎖線 = 車体原点を通る中心線"),
    ]
    nx = BLOCK_X + W1 + GAP
    nw = PAGE_W - MARGIN - 1.0 - nx
    ny = BLOCK_Y + 2.0
    nh = 12.0 + 4.4 * len(notes)
    out.append('<g id="notes">')
    out.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="none" '
               'stroke="#bbb" stroke-width="0.2"/>' % (nx, ny, nw, nh))
    out.append(text(nx + 3.0, ny + 5.0, "注記", 3.0, anchor="start",
                    weight="bold"))
    for i, (k, val) in enumerate(notes):
        yy = ny + 10.6 + 4.4 * i
        if k:
            out.append(text(nx + 3.0, yy, k, 2.4, anchor="start", color="#555"))
        out.append(text(nx + 22.0, yy, val, 2.4, anchor="start"))
    out.append('</g>')

    bx, by, bh = nx, ny + nh + 5.0, 20.0
    out.append('<g id="titleblock">')
    out.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="none" '
               'stroke="#111" stroke-width="0.3"/>' % (bx, by, nw, bh))
    out.append(text(bx + 3.0, by + 6.0, title, 3.6, anchor="start",
                    weight="bold"))
    rows = ["出典 hardware/cad/%s (%s)" % (src, body_name),
            "尺度 1:1   単位 mm   投影 第三角法   用紙 %s" % page_name,
            "生成 FreeCAD %s + TechDraw HLR   %s"
            % (".".join(FreeCAD.Version()[0:3]), DATE)]
    for i, row in enumerate(rows):
        out.append(text(bx + 3.0, by + 10.6 + 4.0 * i, row, 2.3,
                        anchor="start", color="#333"))
    out.append('</g>')
    out.append('</svg>')

    # --- ページ内に収まっているか
    right_edge = ex + W2
    bottom_edge = ey + H2 + 11.0                # 図名の分
    assert right_edge < PAGE_W - MARGIN, \
        "%s/%s: 右にはみ出す (%.1f)" % (stem, layout["key"], right_edge)
    assert bottom_edge < PAGE_H - MARGIN, \
        "%s/%s: 下にはみ出す (%.1f)" % (stem, layout["key"], bottom_edge)
    assert by + bh < ey, \
        "%s/%s: 表題欄が立面図と重なる" % (stem, layout["key"])
    assert nw > 90.0, \
        "%s/%s: 注記欄が狭い (%.1f)" % (stem, layout["key"], nw)

    dst = os.path.join(EXPORT, stem + layout["suffix"] + ".svg")
    with io.open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    sys.__stdout__.write(
        "  %-5s %-5s -> %-38s  block %.1f x %.1f  edges %d/%d %d/%d %d/%d\n"
        % (layout["key"], page_name, os.path.basename(dst),
           right_edge, bottom_edge,
           len(plan["vis"]), len(plan["hid"]),
           len(elev1["vis"]), len(elev1["hid"]),
           len(elev2["vis"]), len(elev2["hid"])))


# 環境変数 THREE_VIEW_PARTS で対象を絞れる（カンマ区切りの部品名）。
_only = [s for s in os.environ.get("THREE_VIEW_PARTS", "").split(",") if s]

for _src, _body, _stem, _title in PARTS:
    if _only and _stem not in _only:
        continue
    _doc = FreeCAD.openDocument(os.path.join(CAD_DIR, _src))
    _obj = _doc.getObject(_body)
    if _obj is None:
        raise SystemExit("object %r not found in %s" % (_body, _src))
    _shape = _obj.Shape.copy()          # 保存済み BREP をそのまま使う
    if MODEL_TO_BODY_DEG:
        _shape.rotate(V(0, 0, 0), V(0, 0, 1), MODEL_TO_BODY_DEG)
    sys.__stdout__.write("%s\n" % _src)
    for _layout in LAYOUTS:
        build(_layout, _shape, _obj.Placement, _src, _body, _stem, _title)
