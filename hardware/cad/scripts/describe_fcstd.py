"""GUI で作った FCStd（スクリプトの無い部品）の寸法を Markdown に書き出す。

    hardware/cad/scripts/fcrun hardware/cad/scripts/describe_fcstd.py </dev/null

出力: hardware/cad/export/<名前>-describe.md（TARGETS の各ファイル）

FCStd はバイナリ（zip）なので、他のセッションや差分では中身が分からない。そこで、
シャーシ・ミドルプレート・ルーフについて次を書き出し、FCStd と一緒にコミットする:
    - 外形の範囲
    - PartDesign の特徴（Pad / Pocket の種類・量・向き、使っているスケッチ）
    - スケッチの形状（直線・円・円弧。部品の座標に変換した値）
    - 穴（円筒面）: 軸の向き、中心、径、範囲
    - 水平な面の高さ
GUI で FCStd を直したら、これを実行し直して describe.md も更新する。
"""

import collections
import os

import FreeCAD as App

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.dirname(HERE)
TARGETS = ["krilly-chassis", "krilly-middle-plate", "krilly-roof"]


def fmt_v(v):
    return f"({v.x:.2f}, {v.y:.2f}, {v.z:.2f})"


def axis_name(v):
    for name, a in (("+X", (1, 0, 0)), ("−X", (-1, 0, 0)), ("+Y", (0, 1, 0)),
                    ("−Y", (0, -1, 0)), ("+Z", (0, 0, 1)), ("−Z", (0, 0, -1))):
        if abs(v.x - a[0]) < 1e-6 and abs(v.y - a[1]) < 1e-6 and abs(v.z - a[2]) < 1e-6:
            return name
    return fmt_v(v)


def describe(name):
    path = os.path.join(CAD_DIR, f"{name}.FCStd")
    doc = App.openDocument(path)
    body = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"][0]
    shape = body.Shape
    bb = shape.BoundBox
    out = [
        f"# {name}.FCStd の寸法",
        "",
        "`hardware/cad/scripts/describe_fcstd.py` が FCStd から書き出す。**手で編集しない**。",
        "GUI で FCStd を直したら、スクリプトを実行し直してこのファイルも更新する。",
        "座標は部品の座標（mm）。アセンブリでの位置は `assembly.md` を見る。",
        "",
        "## 外形",
        "",
        f"- 範囲: x {bb.XMin:.2f}〜{bb.XMax:.2f}、y {bb.YMin:.2f}〜{bb.YMax:.2f}、"
        f"z {bb.ZMin:.2f}〜{bb.ZMax:.2f}",
        f"- 体積 {shape.Volume:.0f} mm³",
        "",
        "## 特徴（作った順）",
        "",
    ]
    for o in body.Group:
        if not o.TypeId.startswith(("PartDesign::Pad", "PartDesign::Pocket", "PartDesign::Hole",
                                    "PartDesign::Fillet", "PartDesign::Chamfer",
                                    "PartDesign::Revolution", "PartDesign::Groove")):
            continue
        props = []
        for k in ("Type", "Length", "Length2", "Reversed", "Midplane", "Offset", "Size",
                  "Radius", "Angle"):
            if hasattr(o, k):
                v = getattr(o, k)
                if k == "Offset" and getattr(v, "Value", v) == 0:
                    continue
                if k == "Length2" and getattr(o, "Type", "") != "TwoLengths":
                    continue  # 片側の押し出しでは使われない
                if k in ("Reversed", "Midplane") and not v:
                    continue
                props.append(f"{k}={v}")
        prof = getattr(o, "Profile", None)
        sk = prof[0].Name if isinstance(prof, tuple) and prof else (prof.Name if prof else "")
        dirn = getattr(o, "Direction", None)
        extra = f"、向き {axis_name(dirn)}" if dirn is not None and dirn.Length > 0 else ""
        out.append(f"- **{o.Name}**（{o.TypeId.split('::')[1]}、スケッチ {sk}）: "
                   + ", ".join(props) + extra)
    out += ["", "## スケッチ", ""]
    for o in body.Group:
        if o.TypeId != "Sketcher::SketchObject":
            continue
        pl = o.Placement
        normal = pl.Rotation.multVec(App.Vector(0, 0, 1))
        out.append(f"### {o.Name}")
        out.append("")
        out.append(f"原点 {fmt_v(pl.Base)}、法線 {axis_name(normal)}。形状は部品の座標:")
        out.append("")
        for g in o.Geometry:
            t = type(g).__name__
            if t == "LineSegment":
                out.append(f"- 直線 {fmt_v(pl.multVec(g.StartPoint))} → {fmt_v(pl.multVec(g.EndPoint))}")
            elif t == "Circle":
                out.append(f"- 円 中心 {fmt_v(pl.multVec(g.Center))} ø{2 * g.Radius:.2f}")
            elif t == "ArcOfCircle":
                out.append(f"- 円弧 中心 {fmt_v(pl.multVec(g.Center))} R{g.Radius:.2f}、"
                           f"{fmt_v(pl.multVec(g.StartPoint))} → {fmt_v(pl.multVec(g.EndPoint))}")
            elif t == "Point":
                continue
            else:
                out.append(f"- {t}")
        out.append("")

    out += ["## 穴（円筒面）", "", "| 軸 | 中心 | 径 | 範囲（軸方向） |", "|---|---|---|---|"]
    holes = collections.OrderedDict()
    for f in shape.Faces:
        s = f.Surface
        if type(s).__name__ != "Cylinder":
            continue
        a = s.Axis
        b = f.BoundBox
        if abs(abs(a.z) - 1) < 1e-6:
            key = ("Z", round(s.Center.x, 2), round(s.Center.y, 2), round(2 * s.Radius, 2),
                   f"z {b.ZMin:.2f}〜{b.ZMax:.2f}")
            center = f"({s.Center.x:.2f}, {s.Center.y:.2f})"
        else:
            # 水平な穴（モータの取付穴など）: 軸上の中心は面の中心で代表させる
            c = b.Center
            key = (axis_name(a) if abs(a.Length - 1) < 1e-6 else fmt_v(a),
                   round(c.x, 2), round(c.y, 2), round(2 * s.Radius, 2),
                   f"z {b.ZMin:.2f}〜{b.ZMax:.2f}")
            center = fmt_v(c)
        if key not in holes:
            holes[key] = center
    for key, center in sorted(holes.items(), key=lambda kv: (kv[0][0], kv[0][3], kv[0][1], kv[0][2])):
        out.append(f"| {key[0]} | {center} | ø{key[3]:.2f} | {key[4]} |")

    zs = collections.Counter()
    for f in shape.Faces:
        if type(f.Surface).__name__ == "Plane" and f.BoundBox.ZLength < 1e-6:
            zs[round(f.BoundBox.ZMin, 2)] += f.Area
    out += ["", "## 水平な面の高さ", "", "| z | 面積の合計（mm²） |", "|---|---|"]
    for z in sorted(zs):
        out.append(f"| {z:.2f} | {zs[z]:.0f} |")

    dst = os.path.join(CAD_DIR, "export", f"{name}-describe.md")
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    App.closeDocument(doc.Name)
    print(f"saved {dst}  ({len(holes)} holes)")


try:
    for t in TARGETS:
        describe(t)
except Exception:  # FreeCAD のコンソール版は例外を表示せずに終わるので、自分で出す
    import traceback
    traceback.print_exc()
    raise
finally:
    import sys
    sys.stdout.flush()  # flush しないと、終了時に標準出力の最後の部分が捨てられる
    sys.stderr.flush()
