#!/usr/bin/env python3
"""校正データ (labels.csv) の解析 (issue #78).

探索ラン (``search_run --save-frames`` / ``speed_run --save-frames``) や手動 survey
(``survey_shot``) が残した labels.csv を読み、**辺ごとの赤割合を wall=0/1 で層別**して
分布・分離のマージン・誤判定数を出す。

**読むべきは「全部当たったか」ではなく「一番弱い壁がしきい値の何倍だったか」。**
誤判定 0 は当たり前で、問題はそこにどれだけ余裕が残っているか — #23 の 8x8 では
誤判定 0 のまま、新ロットの壁が 0.11 (しきい値の 1.4 倍) まで落ちていた。古い壁は
5-6 倍なので、同じ「0 個」でも意味がまるで違う。

画像を再判定せずに**しきい値だけ**変えた評価もできる (赤割合は CSV にあるので、
再測定が要るのは HSV を変えるときだけ — そちらは ``wall_detect --batch``)。

例:
    # 1 回の走行を解析する
    python -m scripts.survey_report survey/run_labels.csv
    # しきい値を変えたら誤判定が何個になるか (画像不要)
    python -m scripts.survey_report --threshold 0.12 survey/run_labels.csv
    python -m scripts.survey_report --edge-threshold back=0.20 survey/run_labels.csv
    # 各辺で「誤判定 0 になるしきい値の範囲」を出す
    python -m scripts.survey_report --sweep survey/run_labels.csv
    # 照明を変えた 2 回を比べる (前 -> 後)
    python -m scripts.survey_report survey/bright_labels.csv survey/dim_labels.csv
"""

from __future__ import annotations

import argparse

from krilly.logging_config import get_logger, setup_logging
from krilly.perception.survey import (
    compare,
    format_confusion,
    format_report,
    read_rows,
    summarize,
)
from krilly.perception.wall_detect import calibrated_config

log = get_logger("krilly.survey_report")


def threshold_function(threshold: float | None, overrides: list[str]):
    """スロット名 -> しきい値。既定は**実機と同じ設定** (辺別の上書きを含む)。

    ここで素の共通しきい値を使うと、実機が back に別の値を使っている場合に評価が
    ずれる。走っているものと同じ判定基準で読むこと。
    """
    cfg = calibrated_config(neighbors=True)
    if threshold is not None:
        cfg.threshold = threshold
        cfg.thresholds = {}
    for item in overrides:
        edge, _, value = item.partition("=")
        cfg.thresholds[edge] = float(value)
    return cfg.threshold_for


def sweep(rows, threshold_for) -> list[str]:
    """辺ごとに「誤判定が 0 になるしきい値の範囲」を出す。

    範囲 = (壁なしの最大, 壁ありの最小]。**幅が狭い辺が次に run を終わらせる。**
    範囲が空 (負) なら、そのデータではどんなしきい値でも分けられない。
    """
    lines = ["誤判定 0 になるしきい値の範囲 (壁なしの最大 < しきい値 <= 壁ありの最小):"]
    for edge, s in summarize(rows, threshold_for).items():
        if not s.walls or not s.clears:
            lines.append("  %-13s 片側しか無いので範囲が決まらない (壁 %d / 開 %d)"
                         % (edge, len(s.walls), len(s.clears)))
            continue
        lo, hi = s.clears[-1].fraction, s.walls[0].fraction
        inside = "現行 %.3f は範囲内" % s.threshold if lo < s.threshold <= hi else \
                 "** 現行 %.3f は範囲外 **" % s.threshold
        lines.append("  %-13s %.3f 〜 %.3f (幅 %+.3f)  %s" % (edge, lo, hi, hi - lo, inside))
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="labels.csv の解析 (#78)")
    p.add_argument("csv", nargs="+", help="labels.csv (2 つ渡すと前->後で比較する)")
    p.add_argument("--threshold", type=float, default=None,
                   help="共通しきい値を上書きして評価する (既定: 実機と同じ設定)")
    p.add_argument("--edge-threshold", action="append", default=[], metavar="EDGE=値",
                   help="辺別しきい値の上書き (例: back=0.20)。複数指定可")
    p.add_argument("--sweep", action="store_true",
                   help="誤判定 0 になるしきい値の範囲を辺ごとに出す")
    args = p.parse_args()

    setup_logging()
    threshold_for = threshold_function(args.threshold, args.edge_threshold)
    datasets = [(path, read_rows(path)) for path in args.csv]
    for path, rows in datasets:
        if not rows:
            log.error("行がない: %s", path)
            return

    for path, rows in datasets:
        log.info("=== %s ===", path)
        stats = summarize(rows, threshold_for)
        for line in format_report(stats, sources=[path]) + format_confusion(stats):
            log.info("%s", line)
        if args.sweep:
            for line in sweep(rows, threshold_for):
                log.info("%s", line)

    if len(datasets) >= 2:
        (name_a, rows_a), (name_b, rows_b) = datasets[0], datasets[1]
        log.info("=== 比較: %s -> %s ===", name_a, name_b)
        for line in compare(summarize(rows_a, threshold_for),
                            summarize(rows_b, threshold_for)):
            log.info("%s", line)
        log.info("弱壁が下がっていれば、それだけ見落とし (=衝突) に近づいている。"
                 "強開が上がっていれば誤検出 (=回り道) に近づいている。")
        if len(datasets) > 2:
            log.warning("3 つ以上渡されたが、比較は最初の 2 つだけ行った。")


if __name__ == "__main__":
    main()
