#!/usr/bin/env python3
"""セル壁判定の可視化・調整スクリプト (issue #16).

各辺の ROI と、その中の赤割合・壁有無を画像に重ねて保存する。保存済み画像
(--image) に対してオフラインで ROI 位置や閾値を追い込める。実機ではライブ取得。

``--measure N`` は**走らせずに機体の姿勢を読む**モード (#21)。機体を止めたまま
N フレーム撮り、辺ごとの赤割合と帯のずれ、そこから出る前後・左右のずれ、車体の
軸角を並べる。静止しているのだから値は動いてはいけない。動くならその辺の帯探索が
別のもの (ケーブルなど) を掴んでいる。定規で機体を既知量ずらして前後で撮れば、
絶対量の確認にもなる。

**モータ・ホイールの遊びの測り方**: 別端末で ``python -m scripts.teleop`` を
起動して保持トルクを掛けたまま、車体を手で左いっぱい/右いっぱいに捻り、それぞれで
``--measure 1`` を撮る。軸角の差が遊びの総量。カメラが測るのは**車体**の向きで、
実際に進む方向を決めるのは**車輪**なので、この差はどんな方位測定にも乗ってくる
(1 回の測定では消せないので、校正は必ず複数回の平均で決めること)。

``--hue-split`` は赤マスクを**色相の 2 帯に塗り分ける**モード (#87)。赤は H の下端
(オレンジ寄り) と上端 (マゼンタ寄り) に割れるので、どちらで拾ったかが異物の切り分けの
決め手になる。**単色で重ねても分からない。**

例:
    # 保存画像に対して ROI と判定を可視化
    python -m scripts.wall_detect --image ~/Documents/krilly/red_detect_20260730.png --out /tmp/walls.png
    # 色相の 2 帯に塗り分け (シアン=オレンジ寄り / 緑=壁と同じ色相)
    python -m scripts.wall_detect --image shot.png --hue-split --out /tmp/split.png
    # 一部を拡大して元画像と並べる
    python -m scripts.wall_detect --image shot.png --hue-split --zoom 610,790,580,720 --out /tmp/split.png
    # ライブ (カメラ)
    python -m scripts.wall_detect --out walls.png --thickness 70 --span 0.5
    # 位置測定の再現性 (実機・静止のまま 10 フレーム)
    python -m scripts.wall_detect --measure 10
    # 左右の隣セルまで読めているか (#89)。紫の枠は「未確定」
    python -m scripts.wall_detect --image shot.png --neighbors --out /tmp/walls.png
    python -m scripts.wall_detect --measure 5 --neighbors

ROI (front/back/left/right) を自機・ケーブル・支柱の外へ、壁が写る辺に合わせて
--thickness / --span / --threshold で調整する。カメラ取付の回転に応じて画像の
どの辺が機体前後左右かを確認すること。
"""

from __future__ import annotations

import argparse
import dataclasses

import cv2
import numpy as np

from krilly.logging_config import get_logger, setup_logging
from krilly.perception.axis_yaw import axis_yaw, calibrated_axis_yaw_config
from krilly.perception.cell_pose import (
    OFFSET_MIN_FRACTION,
    PX_PER_MM_X,
    PX_PER_MM_Y,
    CellOffset,
    cell_offset,
)
from krilly.perception.red_wall import RedDetectorConfig, red_mask, red_mask_parts
from krilly.perception.wall_detect import (
    BACK,
    BODY_DIRS,
    CALIBRATED_RED,
    FRONT,
    LEFT,
    RIGHT,
    WallDetector,
    WallDetectorConfig,
    WallTarget,
    calibrated_config,
    calibrated_neighbor_rois,
    calibrated_rois,
    neighbor_targets,
)

log = get_logger("krilly.wall_detect")


def measure_repeatability(count: int, interval: float, save_prefix: str | None,
                          neighbors: bool = False) -> None:
    """静止したまま N フレーム撮り、位置測定のばらつきを表にする (#21)。

    帯探索は ROI を ±40px スライドして赤割合が最大の位置を採るので、本物の帯の
    近くに別の赤 (カメラのリボンケーブルなど) があると、フレームによって掴む対象が
    入れ替わり、測定値が跳ぶ。静止中のばらつきがそのまま「位置補正がどれだけ嘘を
    つきうるか」になる。
    """
    import time

    from krilly.hal.camera import Camera

    det = WallDetector(calibrated_config(neighbors=neighbors))
    yaw_cfg = calibrated_axis_yaw_config()
    px_per_mm = {FRONT: PX_PER_MM_Y, BACK: PX_PER_MM_Y,
                 LEFT: PX_PER_MM_X, RIGHT: PX_PER_MM_X}
    edges = (FRONT, BACK, LEFT, RIGHT)
    rows: list[tuple[dict[str, tuple[float, float, bool]], CellOffset]] = []
    yaws: list[float] = []
    with Camera() as cam:
        for i in range(count):
            if i:
                time.sleep(interval)
            frame = cam.capture()
            measured = det.measure(frame)
            off = cell_offset(frame, det)
            yaw = axis_yaw(frame, yaw_cfg)
            if yaw is not None:
                yaws.append(yaw.angle_deg)
            rows.append((
                {e: (measured[e][0], measured[e][1] / px_per_mm[e], measured[e][2])
                 for e in edges}, off))
            if save_prefix:
                cv2.imwrite(f"{save_prefix}_{i + 1:02d}.png", frame)
            log.info(
                "#%02d %s | 前後=%s 左右=%s", i + 1,
                " ".join("%s %.2f/%+.1fmm%s" % (e, rows[-1][0][e][0], rows[-1][0][e][1],
                                                "!" if rows[-1][0][e][2] else "")
                         for e in edges),
                "--" if off.forward_m is None else "%+.1fmm" % (off.forward_m * 1e3),
                "--" if off.left_m is None else "%+.1fmm" % (off.left_m * 1e3),
            )
            log.info("      軸角 %s (+ = 迷路の軸より CCW)",
                     "測定不能" if yaw is None else "%+.3f°" % yaw.angle_deg)
            if neighbors:
                nb = det.neighbor_walls(measured)
                log.info("      隣セル %s | 4 壁が確定 %s",
                         " ".join("%s %.2f%s" % (k, v[0], "!" if v[2] else "")
                                  for k, v in measured.items() if k not in edges),
                         ", ".join("%s(%d/4)" % (side, len(w))
                                   for side, w in sorted(nb.items())) or "なし")

    def spread(values: list[float]) -> str:
        if not values:
            return "測定なし"
        lo, hi = min(values), max(values)
        mean = sum(values) / len(values)
        return "平均 %+.1fmm 幅 %.1fmm (%.1f〜%.1f)" % (mean, hi - lo, lo, hi)

    log.info("--- 静止中のばらつき (動いていないので幅は 0 であるべき。! = フレーム端で飽和) ---")
    for e in edges:
        used = [row[0][e][1] for row in rows
                if not row[0][e][2]
                and row[0][e][0] >= max(det.cfg.threshold_for(e), OFFSET_MIN_FRACTION)]
        saturated = sum(1 for row in rows if row[0][e][2])
        log.info("%-6s 位置測定に使えたフレーム %d/%d%s  %s",
                 e, len(used), len(rows),
                 " (うち飽和で捨てた %d)" % saturated if saturated else "", spread(used))
    log.info("%-6s %s", "前後", spread([r[1].forward_m * 1e3 for r in rows
                                        if r[1].forward_m is not None]))
    log.info("%-6s %s", "左右", spread([r[1].left_m * 1e3 for r in rows
                                        if r[1].left_m is not None]))
    if yaws:
        # 車体の向き。モータ・ホイールの遊びを測るときは、モータに保持トルクを掛けた
        # まま (別端末で teleop を起動しておく) 車体を左右いっぱいに捻り、その差を読む。
        log.info("%-6s 平均 %+.3f° 幅 %.3f° (%.3f〜%.3f)  ※迷路の軸に対する車体の向き",
                 "軸角", sum(yaws) / len(yaws), max(yaws) - min(yaws), min(yaws), max(yaws))


def hue_split(frame, red: RedDetectorConfig, out_path: str, zoom: str | None) -> None:
    """赤マスクを色相の 2 帯に塗り分け、帯ごとの HSV を出す (#87)。

    **単色で重ねても異物の正体は分からない。** 赤は H の下端 (オレンジ寄り) と上端
    (マゼンタ寄り) に割れるので、どちらで拾ったかが決め手になる。実例: リボンケーブルに
    黒テープを貼った後も 10% ほど残った赤判定は、壁がすべて h2 (H 169) なのに
    ケーブル上は h1 (H 7) で、隣のテープが H 15 だったことから**テープ自身が h1_hi の
    境界をまたいでいる**と特定できた (ラベルの貼り残しでも壁の映り込みでもない)。

    重ねる色: **シアン = h1 (オレンジ寄り) / 緑 = h2 (壁と同じ色相)**。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h1, h2 = red_mask_parts(frame, red)
    m1, m2 = h1 > 0, h2 > 0
    log.info("赤とみなす色相: H %d-%d (h1) と H %d-%d (h2)  s_min=%d v_min=%d",
             red.h1_lo, red.h1_hi, red.h2_lo, red.h2_hi, red.s_min, red.v_min)
    for name, m in (("h1 (オレンジ寄り)", m1), ("h2 (壁と同じ色相)", m2)):
        if not m.any():
            log.info("  %-18s 0 画素", name)
            continue
        px = hsv[m]
        log.info("  %-18s %6d 画素 (%.2f%%)  H %3.0f  S %3.0f  V %3.0f",
                 name, int(m.sum()), 100.0 * m.mean(),
                 np.median(px[:, 0]), np.median(px[:, 1]), np.median(px[:, 2]))
    if m1.any() and m2.any():
        log.info("  ※ 壁は h2 に出る。h1 に固まりがあれば、それは壁ではない何か")

    vis = (frame * 0.45).astype(np.uint8)
    vis[m2] = (0, 255, 0)        # 緑 = h2
    vis[m1] = (255, 255, 0)      # シアン = h1
    if zoom:
        x0, x1, y0, y1 = (int(v) for v in zoom.split(","))
        pair = np.hstack([cv2.convertScaleAbs(frame[y0:y1, x0:x1], alpha=1.9, beta=5),
                          np.full((y1 - y0, 6, 3), 60, np.uint8), vis[y0:y1, x0:x1]])
        vis = cv2.resize(pair, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, vis)
    log.info("塗り分け画像を保存: %s (シアン=h1 / 緑=h2)", out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="セル壁判定の可視化")
    p.add_argument("--measure", type=int, default=0, metavar="N",
                   help="静止したまま N フレーム撮り、位置測定の再現性を表示する")
    p.add_argument("--interval", type=float, default=0.3, help="--measure のフレーム間隔 [s]")
    p.add_argument("--save-prefix", default=None, help="--measure のフレーム保存先プレフィクス")
    p.add_argument("--image", default=None, help="入力画像 (未指定ならカメラ取得)")
    p.add_argument("--out", default="walls.png", help="注釈画像の保存先")
    p.add_argument("--threshold", type=float, default=0.15, help="壁ありとみなす赤割合")
    p.add_argument("--s-min", type=int, default=CALIBRATED_RED.s_min, help="赤HSVのS下限")
    p.add_argument("--v-min", type=int, default=CALIBRATED_RED.v_min, help="赤HSVのV下限")
    p.add_argument("--h2-lo", type=int, default=CALIBRATED_RED.h2_lo,
                   help="赤とみなす色相の上側の帯の下限 (#65 で 160->140 に広げた)")
    p.add_argument("--neighbors", action="store_true",
                   help="左右の隣セルを読む ROI も重ねる (#89)")
    p.add_argument("--hue-split", action="store_true",
                   help="赤マスクを色相の 2 帯に分けて塗り分ける (異物の切り分け用)")
    p.add_argument("--zoom", default=None, metavar="X0,X1,Y0,Y1",
                   help="--hue-split の拡大範囲 (画素)")
    args = p.parse_args()

    setup_logging()
    if args.measure:
        measure_repeatability(args.measure, args.interval, args.save_prefix,
                              args.neighbors)
        return
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            log.error("画像を読み込めません: %s", args.image)
            return
    else:
        from krilly.hal.camera import Camera

        with Camera() as cam:
            frame = cam.capture()

    # **校正済みの設定から派生させる。** ここで RedDetectorConfig() を素で作ると
    # h2_lo が既定の 160 に戻り、#65 で 140 まで広げた意味が消える (右壁の上面が
    # 場所によって H=141-155 のマゼンタ側へ流れるため、160 では帯の上 2/3 を落とす)。
    # 調整スクリプトが実機と違うマスクを使っていては意味がない。
    red = dataclasses.replace(CALIBRATED_RED, s_min=args.s_min, v_min=args.v_min,
                              h2_lo=args.h2_lo)
    if args.hue_split:
        hue_split(frame, red, args.out, args.zoom)
        return
    rois = calibrated_rois()  # 実機校正済みの各辺 ROI
    slots = {e: WallTarget(e) for e in rois}
    if args.neighbors:
        rois = rois | calibrated_neighbor_rois()
        slots = slots | neighbor_targets()
    # frame_size は**読み込んだ画像そのもの**にする。ROI は 960x720 の校正から作るので
    # 違うサイズの画像では帯から外れるが、調整用に眺めること自体は許したい。ここで
    # 実サイズを入れておくと、遠い側の帯が枠内かの判定 (#89) だけは効く。
    det = WallDetector(WallDetectorConfig(rois=rois, slots=slots, red=red,
                                          threshold=args.threshold,
                                          frame_size=(frame.shape[1], frame.shape[0])))

    measured = det.measure(frame)
    neighbors = det.neighbor_walls(measured) if args.neighbors else {}

    def verdict(name: str) -> tuple[str, tuple[int, int, int]]:
        """そのスロットの判定と色。隣セルは 3 値なので「未確定」がある (#89)。"""
        fraction = measured[name][0]
        if name in BODY_DIRS:
            present = fraction >= det.cfg.threshold_for(name)
            return ("WALL", (0, 255, 0)) if present else ("-", (0, 200, 255))
        target = det.cfg.target(name)
        side = name.split(":")[0]
        walls = neighbors.get(side, {})
        if target.edge not in walls:                    # 未確定 (紫)
            return ("?", (255, 0, 255))
        return ("WALL", (0, 255, 0)) if walls[target.edge] else ("-", (0, 200, 255))

    # 可視化: 検出した赤を薄く重ね、ROI 矩形とラベルを描く
    out = frame.copy()
    mask = red_mask(frame, red)
    out[mask > 0] = (0.4 * out[mask > 0] + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
    for d, roi in rois.items():
        label, color = verdict(d)
        cv2.rectangle(out, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), color, 2)
        cv2.putText(out, f"{d} {measured[d][0]:.2f} {label}",
                    (roi.x + 2, max(roi.y + 16, 16)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(args.out, out)
    for d in rois:
        label, _color = verdict(d)
        log.info("%-12s red=%.3f ずれ=%+3dpx%s -> %s", d, measured[d][0], measured[d][1],
                 " 飽和" if measured[d][2] else "    ",
                 {"WALL": "壁あり", "-": "なし", "?": "未確定"}[label])
    if args.neighbors:
        log.info("隣セルの 4 壁が確定した側: %s",
                 ", ".join(f"{side}({len(w)}/4)" for side, w in sorted(neighbors.items()))
                 or "なし")
    log.info("注釈画像を保存: %s", args.out)


if __name__ == "__main__":
    main()
