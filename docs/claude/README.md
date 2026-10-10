# Krilly 知見ページ (docs/claude)

CLAUDE.md が大きくなりすぎた (150k 字の上限超え) ため、実機・シミュレーションで得た知見をテーマ別に分けたもの。
CLAUDE.md には概要・コマンド・アーキテクチャ・常時必要な規約と、このページ群の索引だけを残す。

運用:
- 該当サブシステムを触る前に、そのページを読む (Claude Code は自動では読み込まない)。
- 新しい知見はここに追記する。CLAUDE.md には足さない。新しいページを作ったら CLAUDE.md の索引表に 1 行足す。
- 書き方は既存に合わせる: 太字の一行の主張 → 条件 (速度・床・符号の基準・issue 番号) 付きの測定値。

| ページ | 内容 |
|---|---|
| [rules-and-budget](rules-and-budget.md) | classic ルール (7 分 / 5 走行、3-4・3-5 など)、`RunManager` の時間見積もりと `time_margin`、時間定数 (セル・モーション単位のコスト)。 |
| [maze-sim](maze-sim.md) | 生成迷路と実迷路の難しさの差、ゴール/スタートの形のチェック、迷路の切り出し (excerpt) と blind run の指標。 |
| [search-and-planning](search-and-planning.md) | 隣接セル読み取り (#89)、unknown の扱い、斜行せず横移動する方針 (#76)、経路チェック、壁を消さない規則。 |
| [motion-and-corners](motion-and-corners.md) | 停止しないコーナー (#80)、チェイン数・脚長の上限、ランプ (accel/decel) と動作点の決め方、旋回のオーバーシュート (#73)、異常停止の裾 (#107)。 |
| [recovery](recovery.md) | `--max-heading-residual` で止まった後のその場回復: verify_cell、ガードの逆転、壁への接触 (lattice phase)、mod 90° の罠。 |
| [odometry-and-gyro](odometry-and-gyro.md) | ジャイロのサンプリング (#81)、融合方針、heading の基準、crab、W0 の径、較正値、`gyro_scale_z`、`--align` なしの比較の罠、カメラ vs 定規。 |
| [mechanics-and-faults](mechanics-and-faults.md) | L6470 の状態ポーリング、ボタンランチャー、非常停止 (シグナル)、励磁の跳ね、緩んだ締結、ローラーの擦れ、異音 4 原因の対照表。 |
| [camera-geometry](camera-geometry.md) | カメラの傾き・取り付け、全 FOV で使う壁、axis_yaw による heading 真値、cell_offset による位置補正とその飽和・閾値、横ゼロ点。 |
| [wall-detection](wall-detection.md) | ROI の較正と探索、HSV マスク (`h2_lo` など)、リボンケーブル、閾値の根拠、木の床と自立壁、手動/走行中の較正データ収集 (#78)。 |
| [lighting-and-floors](lighting-and-floors.md) | AE/ゲインの余裕、色温度、黒い床での BACK の白飛びと `--ev -2`、始点/終点の白・黄の壁上面 (#125/#126)。 |
| [hardware-sessions](hardware-sessions.md) | 5×5・8×8 の通し走行、試走会 (2026-09-26) と本番 (2026-09-27) で何が起き、何を学んだか。 |
