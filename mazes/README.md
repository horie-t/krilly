# 迷路データ

`Maze.from_ascii` / `Maze.to_ascii` の形式 (issue #77)。北が上、行数は 2N+1。

```
+---+---+---+
|     G     |     G = ゴール (複数書ける。外接矩形がゴール領域になる)
+   +---+   +
| S |       |     S = スタート
+---+---+---+
```

- セル幅は任意 (`+-+` でも `+---+` でもよい)。`to_ascii` は 3 文字幅で書く。
- `S` / `G` を省くと スタート (0,0) ・ゴール中央 2x2 (本番設定) になる。
- ゴールは**矩形**でなければならない (`G` の外接矩形が `G` で埋まっていること)。

## 収録

| ファイル | 用途 |
|---|---|
| `practice3.txt` | 3x3 の練習迷路。実機の `search_run --size 3` と同じ形 |
| `practice8.txt` | 8x8。**壁ちょうど 70 枚**なので手持ちで組める (#23) |
| `practice16.txt` | 16x16。解 67 セル・内壁 210 枚で、実際の大会迷路に近い密度 |

いずれも `krilly.sim.generate.random_maze` で作った合成迷路で、
`krilly.sim.check.check_maze` を通してある。

**実際の大会迷路は `contest/` にある** (2010-2024 年、31 面、#84)。
合成迷路との難易度の差もそこに測ってある — **壁の枚数を合わせても解の長さは 4 倍違う**ので、
合成迷路は本物の代わりにならない。

## 使い方

```bash
python -m scripts.maze_sim --maze mazes/practice16.txt --verbose
python -m scripts.maze_sim --maze mazes/practice8.txt --check-only --wall-budget 70
```

## 大会迷路を書き起こすとき

NTF の記録集は PDF で機械可読な形が無いので、図の画像から `scripts/maze_from_image.py`
で自動抽出する。**16x16 は壁の置き場所が 544 箇所あるので、目で写すと必ず読み違える。**
目視は「機械が出した結果が元図と合っているか」の確認だけに使う。

```bash
python -m scripts.maze_from_image <画像>... --out-dir mazes/contest
python -m scripts.maze_from_image <画像> --overlay-dir /tmp/ov   # 元図に重ねて確認
```

抽出の仕組みと検証の記録は `contest/README.md` を見ること。要点だけ:

- 壁と点線グリッドは**線に沿った平均太さ** (太さ x 連続率) で分ける。図によって
  効く手がかりが違う (太さだけ / 途切れだけ) のを 1 つの数で吸収する
- しきい値は**迷路として成立するかで自動選択**する (外周が閉じている / ゴールへ到達可能 /
  内壁が連結の上限以下)。**実際の大会迷路は必ず解ける**ので、これが強い判定器になる
- 疑わしい箇所は該当セルを拡大して目視で確認する
