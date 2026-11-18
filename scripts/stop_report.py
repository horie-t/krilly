#!/usr/bin/env python3
"""speed_run のログから、止まるたびの位置ずれを横ずれと進行方向に分けて集計する (#187、off-Pi)。

``speed_run`` は止まるたびに ``停止 (x, y): 軸角 … / 位置ずれ 実測 前後=… 左右=…`` を
1 行残す (2026-10-17 から)。その直前の ``… 完了`` 行の動作名 (``Nへ3セル`` /
``Wへ1->Nへ2``) の**最後の向き**が、止まる直前に走っていた向き。機体は北を向いた
まま走る (#76) ので:

- 南北に走った後: 横ずれ = 左右、進行方向 = 前後
- 東西に走った後: 横ずれ = 前後、進行方向 = 左右

走行中の融合 (``--fuse``) が効いていれば、止まったときの**横ずれ**が小さくなる。
進行方向は融合しない (既定) ので変わらないはず — 比べる対照になる。

例:
    python -m scripts.stop_report logs/run_20261120_*.log
    python -m scripts.stop_report --label 融合なし a.log --label 融合あり b.log
"""

from __future__ import annotations

import re
import statistics
import sys

DONE = re.compile(r"\s(?P<label>(?:[NESW]へ\d+(?:セル)?(?:->)?)+)\s完了\s")
STOP = re.compile(r"停止 \((?P<x>-?\d+), (?P<y>-?\d+)\): .*位置ずれ 実測 前後=(?P<f>[-+0-9.]+)(?:mm)? "
                  r"左右=(?P<l>[-+0-9.]+)(?:mm)?")
LAST_DIR = re.compile(r"([NESW])へ")


def _mm(text: str) -> float | None:
    return None if text in ("-", "") else float(text)


def parse(lines) -> list[dict]:
    """ログの行から、(直前の動作の最後の向き, 横ずれ, 進行方向のずれ) の並びを作る。"""
    out = []
    last = None
    for line in lines:
        m = DONE.search(line)
        if m:
            last = LAST_DIR.findall(m.group("label"))[-1]
            continue
        m = STOP.search(line)
        if m and last is not None:
            f, l = _mm(m.group("f")), _mm(m.group("l"))
            ns = last in "NS"
            out.append({"dir": last, "cell": (int(m.group("x")), int(m.group("y"))),
                        "cross": l if ns else f, "along": f if ns else l})
            last = None
    return out


def summarize(stops: list[dict], key: str) -> str:
    v = [abs(s[key]) for s in stops if s[key] is not None]
    if not v:
        return "—"
    v.sort()
    p90 = v[min(len(v) - 1, int(0.9 * len(v)))]
    return (f"中央値 {statistics.median(v):.1f} / 90% {p90:.1f} / 最大 {v[-1]:.1f} mm "
            f"({len(v)} 回)")


def group_args(items: list[str]) -> list[tuple[str, list[str]]]:
    """``[--label 名] ログ… [--label 名] ログ…`` をまとまりに分ける。"""
    groups: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(items):
        if items[i] == "--label":
            groups.append((items[i + 1], []))
            i += 2
            continue
        if not groups:
            groups.append(("", []))
        groups[-1][1].append(items[i])
        i += 1
    return groups


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    groups = group_args(argv)
    for name, paths in groups:
        stops = []
        for path in paths:
            with open(path, errors="replace") as f:
                stops += parse(f)
        print(f"## {name or ', '.join(paths)}")
        print(f"- 横ずれ (止まる直前の進行方向と直角): {summarize(stops, 'cross')}")
        print(f"- 進行方向のずれ (対照。融合しない):     {summarize(stops, 'along')}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
