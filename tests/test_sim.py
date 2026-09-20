"""迷路シミュレーション基盤 (issue #77) のユニットテスト。

離散層はハードウェアに触らないので、真の迷路を用意すればセッション全体が回る。
ここで**実機なしに担保する**のは 探索の完走 / 地図の一致 / 7 分 5 走の予算 / 経路計画。
カメラの見落としや姿勢の誤差は模擬しない (実機 #23 の担当)。
"""

from pathlib import Path

import pytest

from krilly.app.run_manager import RunPhase
from krilly.perception.wall_detect import (
    BACK,
    FRONT,
    LEFT,
    RIGHT,
    body_walls_to_maze,
)
from krilly.sim.sense import sense_neighbors
from krilly.sim import (
    check_maze,
    comb_maze,
    diagonal_goal,
    map_agrees,
    open_maze,
    random_maze,
    seal_goal,
    sense,
    serpentine_maze,
    simulate_session,
    wall_counts,
)
from krilly.sim.check import (
    goal_center_post,
    goal_entrances,
    goal_interior_walls,
    posts_without_wall,
    reachable_cells,
)
from krilly.sim.generate import walled_maze
from krilly.sim.session import fit_search_overhead
from krilly.solver.maze import Direction, Maze
from krilly.strategy.shortest_path import LEGACY_COST, MoveCost, shortest_path

MAZE_DIR = Path(__file__).resolve().parents[1] / "mazes"


# --- カメラの代役 -----------------------------------------------------------
def test_sense_is_the_exact_inverse_of_body_walls_to_maze():
    """観測の写像が往復すること。ここがずれると探索が静かに壊れる。"""
    truth = random_maze(6, seed=3)
    for facing in Direction:
        for cell in ((0, 0), (2, 3), (5, 5)):
            body = sense(truth, cell, facing)
            back = body_walls_to_maze(body, facing)
            for d, present in back.items():
                assert present == truth.has_wall(*cell, d), (cell, facing, d)


def test_sense_degenerates_to_a_constant_map_when_facing_north():
    """旋回レス走行 (#76) では向きが北で固定なので FRONT=N / BACK=S / LEFT=W / RIGHT=E。"""
    truth = random_maze(5, seed=1)
    got = sense(truth, (2, 2), Direction.N)
    assert got[FRONT] == truth.has_wall(2, 2, Direction.N)
    assert got[BACK] == truth.has_wall(2, 2, Direction.S)
    assert got[LEFT] == truth.has_wall(2, 2, Direction.W)
    assert got[RIGHT] == truth.has_wall(2, 2, Direction.E)


# --- ASCII のゴール表記 -----------------------------------------------------
def test_ascii_round_trips_start_and_goal():
    m = random_maze(6, seed=2)
    m.start = (0, 0)
    diagonal_goal(m)
    back = Maze.from_ascii(m.to_ascii())
    assert back.start == m.start
    assert back.goal_min == m.goal_min and back.goal_max == m.goal_max
    assert back.to_ascii() == m.to_ascii()


def test_ascii_without_markers_keeps_the_default_goal():
    m = Maze.from_ascii(open_maze(4).to_ascii(markers=False))
    assert m.goal_cells() == [(1, 1), (1, 2), (2, 1), (2, 2)]   # 中央 2x2


def test_ascii_rejects_a_non_rectangular_goal():
    text = "+-+-+-+\n|G|G|G|\n+-+-+-+\n| | | |\n+-+-+-+\n| |G| |\n+-+-+-+\n"
    with pytest.raises(ValueError, match="矩形"):
        Maze.from_ascii(text)


def test_set_goal_validates_the_rectangle():
    m = Maze(5)
    with pytest.raises(ValueError, match="下限が上限"):
        m.set_goal((3, 3), (1, 1))
    with pytest.raises(IndexError):
        m.set_goal((0, 0), (5, 5))


# --- 壁の枚数 ---------------------------------------------------------------
@pytest.mark.parametrize("n", [3, 5, 8, 10, 16])
def test_wall_bounds_follow_the_grid_arithmetic(n):
    """外周 + 内壁の上限 = **格子点の数** (4N + (N-1)^2 = (N+1)^2)。

    実際に立てる柱はそこから 1 本少ない。**2x2 のゴールの中央には柱を置かない**
    (NTF クラシック競技規定 9:「迷路の終点となる4区画内には壁や柱は存在しない。」)。
    奇数サイズはゴールが 1 セルなので中央の柱が在り、格子点の数と一致する。
    """
    c = wall_counts(open_maze(n))
    corners = (n + 1) ** 2
    assert c.outer == 4 * n
    assert c.inner == 0
    assert c.posts == corners - (1 if n % 2 == 0 else 0)
    assert c.inner_slots == 2 * n * (n + 1) - 4 * n
    assert 4 * n + c.inner_max == corners


def test_a_perfect_maze_sits_just_under_the_interior_upper_bound():
    """全セル到達可能なら内壁は (N-1)^2 枚以下。完全迷路はそこから 1 枚だけ少ない。

    **1 枚少ないのはゴールのため。** 2x2 のゴールは 1 つの開いた区画なので内部の
    4 辺が全部開いていなければならないが、全域木は 4 セルを 3 辺で繋ぐので、
    残る 1 辺を開ける分だけ上限を下回る (#23)。
    """
    m = random_maze(10, seed=0, loop_ratio=0.0)
    c = wall_counts(m)
    assert c.inner_max == 81
    assert c.inner == 80
    assert len(reachable_cells(m)) == 100


def test_a_maze_with_loops_stays_under_the_bound():
    for seed in range(5):
        m = random_maze(10, seed=seed, loop_ratio=0.2)
        c = wall_counts(m)
        assert c.inner_min <= c.inner <= c.inner_max


#: 手持ちの壁と柱 (#23 で買い足し・追加製作した後)。
WALL_STOCK, POST_STOCK = 80, 85


def test_the_practice_mazes_are_buildable_and_legal():
    """実機で組む迷路は**手持ちで組めて、競技の形をしている**こと (#23)。

    ゴールの形の検証を入れるまで practice8 / practice16 はどちらもゴールの内側に
    壁を抱えていた (2x2 が 1 つの区画になっていなかった)。直したぶん practice8 は
    70 -> 71 枚になったが、手持ちは 80 枚あるので問題ない。
    """
    for name, size in (("practice5", 5), ("practice8", 8),
                       ("excerpt8", 8), ("excerpt8_2015", 8)):
        m = Maze.from_ascii((MAZE_DIR / f"{name}.txt").read_text(encoding="utf-8"))
        assert m.size == size, name
        assert wall_counts(m).total <= WALL_STOCK, name
        # 柱は格子点の数ではなく**実際に立てる本数**で見る (ゴール中央の 1 本は無い)。
        assert wall_counts(m).posts <= POST_STOCK, name
        assert check_maze(m, wall_budget=WALL_STOCK).ok, name


# --- 柱 ---------------------------------------------------------------------
def test_generated_mazes_keep_every_post_walled():
    """公式規則「柱には必ず 1 枚以上の壁が接する」を守って壁を抜いていること。"""
    for seed in range(5):
        assert posts_without_wall(random_maze(8, seed=seed, loop_ratio=0.5)) == []


def test_an_open_maze_leaves_every_interior_post_bare():
    """壁がまったく無ければ内側の柱は全部裸。**ゴール中央の 1 本は数えない。**

    ゴール 2x2 は開いた区画なので、その中心の柱に壁が付かないのが正しい
    (:func:`goal_center_post`)。実測でも大会迷路 31 面のうち 28 面が
    「裸の柱はここ 1 本だけ」だった。
    """
    assert len(posts_without_wall(open_maze(8))) == 49 - 1      # (8-1)^2 - ゴール中央


def test_keep_posts_off_lets_posts_go_bare():
    bare = posts_without_wall(random_maze(8, seed=0, loop_ratio=0.9, keep_posts=False))
    assert bare, "keep_posts=False なら裸の柱が出るはず (対照条件)"


# --- 健全性チェック ---------------------------------------------------------
def test_check_passes_a_well_formed_maze():
    assert check_maze(random_maze(10, seed=4)).ok


def test_check_catches_an_open_outer_wall():
    m = random_maze(6, seed=0)
    m.set_wall(0, 0, Direction.S, False)
    report = check_maze(m)
    assert not report.ok
    assert "外周" in report.errors[0]


def test_check_catches_a_sealed_goal():
    report = check_maze(seal_goal(random_maze(8, seed=1)))
    assert not report.ok
    assert any("到達できない" in e for e in report.errors)


def test_check_catches_a_transcription_error():
    """書き起こしの読み違いを 1 枚の壁で作り、検出できること (#77 の受け入れ基準)。

    実際の大会迷路は必ず解けるので、到達不能が出たら迷路ではなく書き起こしを疑う。
    """
    m = random_maze(8, seed=2, loop_ratio=0.0)         # 完全迷路 = 経路が 1 本しかない
    path = shortest_path(m)
    (x, y), (nx, ny) = path[0], path[1]
    d = next(d for d in Direction if m.neighbor(x, y, d) == (nx, ny))
    m.set_wall(x, y, d, True)                          # 壁を 1 枚読み違えた
    report = check_maze(m)
    assert not report.ok
    assert any("到達できない" in e for e in report.errors)


def test_check_warns_about_a_wall_budget():
    report = check_maze(random_maze(10, seed=0), wall_budget=70)
    assert report.ok                                   # 迷路としては正しい
    assert any("足りない" in w for w in report.warnings)


# --- 地図の一致 -------------------------------------------------------------
def test_map_agrees_over_the_visited_cells():
    truth = random_maze(8, seed=5)
    result = simulate_session(truth)
    assert result.reached_goal
    assert map_agrees(truth, result.explorer.maze, result.explorer.visited) == []


def test_map_agrees_flags_an_injected_difference():
    truth = random_maze(6, seed=1)
    result = simulate_session(truth)
    learned = result.explorer.maze
    cell = next(iter(sorted(result.explorer.visited)))
    learned.set_wall(*cell, Direction.N, not learned.has_wall(*cell, Direction.N))
    assert map_agrees(truth, learned, result.explorer.visited)


def test_map_agrees_needs_the_visited_set():
    """未訪問セルまで照合すると必ず食い違う (未知は「開いている」と楽観視されるため)。"""
    truth = random_maze(8, seed=5, loop_ratio=0.0)
    result = simulate_session(truth)
    assert map_agrees(truth, result.explorer.maze) != []      # 全セル -> 不一致あり
    assert map_agrees(truth, result.explorer.maze, result.explorer.visited) == []


# --- 統合シミュレータ -------------------------------------------------------
@pytest.mark.parametrize("seed", range(12))
def test_sessions_complete_on_random_16x16_mazes(seed):
    result = simulate_session(random_maze(16, seed=seed))
    assert result.reached_goal, result.describe()
    assert result.mismatches == [], result.describe()
    assert result.elapsed_s <= 420.0
    assert 1 <= result.runs_used <= 5


@pytest.mark.parametrize("make", [open_maze, serpentine_maze, comb_maze])
def test_sessions_complete_on_adversarial_patterns(make):
    result = simulate_session(make(8))
    assert result.ok, result.describe()


def test_diagonal_goal_is_reached_too():
    result = simulate_session(diagonal_goal(random_maze(12, seed=7)))
    assert result.ok, result.describe()
    assert result.explorer.cell == (11, 11)


def test_a_sealed_goal_aborts_instead_of_looping():
    result = simulate_session(seal_goal(random_maze(8, seed=1)))
    assert not result.reached_goal
    assert result.aborted and "到達不能" in result.aborted
    assert result.runs_used == 1


# #87 (安全率 1.5 -> 1.2) の判断は **v=0.24 m/s の機体**で測った。既定は #103 で
# 0.30 m/s 用に更新したので、当時の結論を再現するテストには当時の定数を明示的に渡す。
# 機体が速くなれば予算はどこも楽になる方へ動くだけで、1.2 という選択は変わらない。
TIMES_V024 = {"cell_time_s": 0.74, "lateral_cell_time_s": 0.76, "straight_time_s": 0.83}
COST_V024 = MoveCost(cell_ns=1.0, cell_ew=1.03, leg=1.10)


def test_the_manager_refuses_a_run_it_cannot_finish():
    """探索が長引いたら最速ランを始めない (中途半端に走って時間切れになるより良い)。

    DFS で掘った完全迷路は蛇行が長く、16x16 で解が 169 セルになることがある。
    そのとき 復帰 + 最速 の見積もりは残り時間に入らないので、走行 1 回で終わる。
    """
    truth = random_maze(16, seed=0, loop_ratio=0.0)
    result = simulate_session(truth)
    assert result.reached_goal
    assert result.runs_used == 1
    assert result.speed_runs == []
    assert result.elapsed_s <= 420.0


@pytest.mark.parametrize("scale", [1.0, 1.1, 1.2])
def test_the_budget_holds_when_reality_is_up_to_20_percent_slower(scale):
    """``time_margin`` (1.2) が実際の遅れを吸収できる範囲 (#87)。

    ``actual_scale`` を上げると見積もりより実際が遅くなる。**20% 遅くても持ち時間を
    超えない**のが今の設計の約束で、実測の見積もり誤差は 8x8 で 0.5% なので 40 倍の余裕。
    超えそうなら走行を減らして対応する。
    """
    for seed in range(6):
        result = simulate_session(random_maze(16, seed=seed), actual_scale=scale)
        assert result.elapsed_s <= 420.0, result.describe()
        assert result.mismatches == []


def test_past_that_the_last_run_can_overrun_and_that_is_the_deal_we_took():
    """**40% 遅いと最後の 1 本がはみ出しうる。それを承知で 1.5 -> 1.2 にした** (#87)。

    安全率が守っているのは「始めた走行を終えられるか」だけ。守りすぎると走れたはずの
    最速ランを断る方に外れ、規定 3-1 では記録は**最速の 1 走行**なので、時間切れの
    最速ランは時間を失うだけで、それまでの記録は残る — **断る方が高くつく。**

    この seed がその取引そのもの: 1.2 は最速を 3 本走って 5.2s はみ出し、1.5 は
    2 本で収まる。大会迷路 31 面では、この取引で最速ランを 1 本も走れない面が
    7 面から 3 面へ、最速ランの総数が 34 本から 42 本へ増える。
    """
    truth = random_maze(16, seed=5)
    kw = dict(actual_scale=1.4, times=TIMES_V024, cost=COST_V024)
    bold = simulate_session(truth, time_margin=1.2, **kw)
    safe = simulate_session(truth, time_margin=1.5, **kw)

    assert bold.elapsed_s > 420.0 >= safe.elapsed_s     # 大胆な方ははみ出す
    assert len(bold.speed_runs) > len(safe.speed_runs)  # が、走った本数は多い
    # はみ出すのは最後の 1 本ぶんまで (青天井に伸びるわけではない)
    assert bold.elapsed_s <= 420.0 + max(r.duration_s for r in bold.speed_runs)


def test_slower_reality_costs_runs_not_the_time_limit():
    fast = simulate_session(random_maze(16, seed=3, loop_ratio=0.0), actual_scale=1.0)
    slow = simulate_session(random_maze(16, seed=3, loop_ratio=0.0), actual_scale=1.4)
    assert slow.runs_used <= fast.runs_used
    assert slow.elapsed_s <= 420.0


def test_turning_mode_is_slower_than_the_turn_free_one():
    """旋回レス化 (#76) の効果がシミュレーションでも出ること (実機は 2.10x)。"""
    truth = random_maze(16, seed=6)
    free = simulate_session(truth)
    turning = simulate_session(truth, holonomic=False, cost=LEGACY_COST)
    assert free.best_speed_s is not None and turning.best_speed_s is not None
    assert turning.best_speed_s > free.best_speed_s * 1.3


def test_records_line_up_with_the_phases():
    result = simulate_session(random_maze(10, seed=2))
    phases = [r.phase for r in result.records]
    assert phases[0] is RunPhase.SEARCH
    assert phases[1::2] == [RunPhase.RETURN_HOME] * (len(phases) // 2)
    assert phases[2::2] == [RunPhase.SPEED] * ((len(phases) - 1) // 2)
    assert result.runs_used == 1 + len(result.speed_runs)


def test_fit_search_overhead_recovers_what_it_is_given():
    truth = random_maze(10, seed=8)
    base = simulate_session(truth)
    stops = base.search.legs          # カメラを見るのは止まったときだけ (#89)
    measured = base.search.duration_s + 0.37 * stops
    assert fit_search_overhead(truth, measured) == pytest.approx(0.37)


def test_search_overhead_only_slows_the_search():
    truth = random_maze(10, seed=8)
    base = simulate_session(truth)
    slow = simulate_session(truth, search_step_overhead_s=0.5)
    assert slow.search.duration_s > base.search.duration_s
    assert slow.best_speed_s == pytest.approx(base.best_speed_s)


# --- 生成した迷路そのもの ---------------------------------------------------
def test_walled_maze_has_every_wall_up():
    c = wall_counts(walled_maze(6))
    assert c.inner == c.inner_slots and c.outer == 24


def test_serpentine_visits_every_cell_on_one_path():
    """分岐が無く全セルを 1 本で繋ぐこと (= 最長経路の迷路)。

    ゴールは中央 2x2 なので**経路の途中**で到達する。「解が 36 セル」ではなく
    「迷路そのものが 1 本道」を主張する方が意図に忠実。
    """
    m = serpentine_maze(6)
    assert len(reachable_cells(m)) == 36
    degrees = sorted(len(m.open_neighbors(x, y)) for x in range(6) for y in range(6))
    assert degrees[:2] == [1, 1]                # 端は 2 つだけ
    assert degrees[2:] == [2] * 34              # 残りはすべて通過点 = 分岐なし
    assert wall_counts(m).inner == wall_counts(m).inner_slots - 35   # 開通は N^2-1 本


def test_comb_maze_is_all_dead_ends():
    m = comb_maze(6)
    assert len(reachable_cells(m)) == 36
    # 幹線 (南端) 以外の各列は行き止まりの歯なので、北端のセルは南以外すべて壁
    for x in range(6):
        assert m.has_wall(x, 5, Direction.N)
        assert m.has_wall(x, 5, Direction.E) or x == 5


# --- 書き起こした大会迷路 (#84) ---------------------------------------------
CONTEST_DIR = MAZE_DIR / "contest"


def contest_mazes():
    return sorted(CONTEST_DIR.glob("*.txt"))


def test_the_contest_corpus_is_present():
    assert len(contest_mazes()) >= 30, "書き起こした大会迷路が見つからない"


#: 書き起こしが壊れていると分かっている迷路 (#84 / #23)。**元図で直すまで除外する。**
#: ゴール 2x2 が縦横とも壁で仕切られており、実在しない形になっている。
#: 消せば「通る」が、元図を確認せずに書き起こしデータを書き換えるのは筋が悪い。
KNOWN_BAD_TRANSCRIPTIONS = {"2016_japan_freash_q"}


@pytest.mark.parametrize("path", contest_mazes(), ids=lambda p: p.stem)
def test_every_contest_maze_is_solvable(path):
    """大会迷路は必ず解ける。解けないなら書き起こしの誤り (#84)。"""
    maze = Maze.from_ascii(path.read_text(encoding="utf-8"))
    assert maze.size == 16
    report = check_maze(maze)
    if path.stem in KNOWN_BAD_TRANSCRIPTIONS:
        assert not report.ok, f"{path.stem} が直ったなら除外リストから外すこと"
        return
    assert report.ok, f"{path.stem}: {report.errors}"


@pytest.mark.parametrize("path", contest_mazes(), ids=lambda p: p.stem)
def test_every_contest_maze_runs_a_session(path):
    """探索が完走し、訪問範囲で地図が一致すること。"""
    truth = Maze.from_ascii(path.read_text(encoding="utf-8"))
    result = simulate_session(truth)
    assert result.reached_goal, result.describe()
    assert result.mismatches == [], result.describe()
    assert result.elapsed_s <= 420.0


def test_the_official_answer_is_reproduced():
    """2013 年エキスパート予選の図に載っている公式解と一致すること。

    記載は「西回り 52歩29折、54歩29折 南回り 52歩25折、54歩23折」。抽出した迷路の
    最短経路が 52 歩 29 折、折れ最小の経路が 52 歩 25 折になる。**書き起こしが
    正しいことの、目視によらない唯一の裏付け**なので消さないこと。
    """
    from krilly.strategy.shortest_path import MoveCost, path_to_legs, turns_in
    maze = Maze.from_ascii(
        (CONTEST_DIR / "2013_japan_exp_q.txt").read_text(encoding="utf-8"))
    shortest = shortest_path(maze, cost=MoveCost(1, 1, 0, 0))
    assert len(shortest) - 1 == 52
    assert turns_in(path_to_legs(shortest), Direction.N) == 29
    fewest_turns = shortest_path(maze, cost=MoveCost(1, 1, 0, 0.01))
    assert len(fewest_turns) - 1 == 52
    assert turns_in(path_to_legs(fewest_turns), Direction.N) == 25


def test_contest_mazes_are_harder_than_generated_ones():
    """**生成した迷路は本物の代わりにならない** (#84 の根拠)。

    壁の枚数を合わせても解の長さが 3 倍違う。難しさは密度ではなく配置で決まる。

    比は #23 のゴール修正で 3.7 倍から 2.95 倍に縮んだ。生成迷路のゴールが
    「内側に壁があり入口も複数」だったのを競技の形 (開いた 2x2・入口 1 つ) に
    直した結果、生成側の解が 16 -> 20 セルに伸びたため。**結論は変わらない。**
    """
    import statistics
    real = [Maze.from_ascii(p.read_text(encoding="utf-8")) for p in contest_mazes()]
    fake = [random_maze(16, seed=s, loop_ratio=0.10) for s in range(len(real))]
    real_sol = statistics.median(len(shortest_path(m)) - 1 for m in real)
    fake_sol = statistics.median(len(shortest_path(m)) - 1 for m in fake)
    real_walls = statistics.median(wall_counts(m).inner for m in real)
    fake_walls = statistics.median(wall_counts(m).inner for m in fake)
    assert abs(real_walls - fake_walls) < 40      # 壁の枚数は近いのに
    assert real_sol > 2.5 * fake_sol             # 解の長さは 3 倍近く違う


# --- 左右の隣セルを読む (#89) -----------------------------------------------
def test_sense_neighbors_returns_the_side_cells_as_they_would_look():
    truth = open_maze(5)
    truth.set_wall(1, 0, Direction.N)
    truth.set_wall(2, 0, Direction.E)
    walls = sense_neighbors(truth, (1, 0), Direction.N)
    assert walls[LEFT] == sense(truth, (0, 0), Direction.N)
    assert walls[RIGHT] == sense(truth, (2, 0), Direction.N)
    assert walls[RIGHT][RIGHT] is True          # (2,0) の東壁


def test_sense_neighbors_skips_cells_outside_the_maze():
    truth = open_maze(5)
    walls = sense_neighbors(truth, (0, 2), Direction.N)
    assert set(walls) == {RIGHT}                # 西は迷路の外


def test_sense_neighbors_follows_the_facing():
    """旋回する走り方では「左右の隣」も機体の向きで変わる。"""
    truth = open_maze(5)
    walls = sense_neighbors(truth, (2, 2), Direction.E)
    assert set(walls) == {LEFT, RIGHT}          # 東を向くと左=北・右=南の隣


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_neighbour_sensing_cuts_stops_without_breaking_the_map(seed):
    """隣を読んで既知セルを通過しても、地図は真の迷路と一致したままであること。"""
    truth = random_maze(12, seed=seed)
    base = simulate_session(truth, neighbor_sensing=False, max_leg_cells=1)
    fast = simulate_session(truth)          # 既定 = 実機の既定 (隣を読む / 2 セル通過)
    assert fast.reached_goal and not fast.mismatches
    assert fast.search.legs < base.search.legs           # 停止回数が減る
    assert fast.search.legs <= fast.search.cells         # 1 停止で 1 セル以上進む


def test_neighbour_sensing_needs_the_pass_through_to_save_stops():
    """隣を読むだけでは停止は減らない (通過を許して初めて減る)。"""
    truth = random_maze(12, seed=3)
    looked = simulate_session(truth, neighbor_sensing=True, max_leg_cells=1)
    assert looked.search.legs == looked.search.cells


def test_pass_through_never_enters_a_cell_with_unobserved_walls():
    """止まらずに通過したセルも 4 壁が観測済みであること (見ていない壁へ突っ込まない)。"""
    truth = random_maze(12, seed=5)
    result = simulate_session(truth, neighbor_sensing=True, max_leg_cells=4)
    ex = result.explorer
    assert ex.visited <= ex.known
    assert not map_agrees(truth, ex.maze, ex.known)      # 確定した壁は真の迷路と一致


# --- ゴールの形 (#23) -------------------------------------------------------
def test_a_competition_goal_is_one_open_square_with_one_door():
    """切り出した迷路のゴールが競技の形になっていること。"""
    maze = Maze.from_ascii((MAZE_DIR / "excerpt8.txt").read_text(encoding="utf-8"))
    assert goal_interior_walls(maze) == 0
    assert len(goal_entrances(maze)) == 1


def test_the_goal_centre_post_is_bare_and_that_is_correct():
    """ゴール中央の柱に壁が付かないのは正しい。**数えてはいけない。**

    大会迷路 31 面のうち 28 面が「裸の柱はここ 1 本だけ」。例外を入れるまで、
    正しい迷路が軒並み「公式規則違反」と判定されていた。
    """
    maze = Maze.from_ascii((MAZE_DIR / "excerpt8.txt").read_text(encoding="utf-8"))
    centre = goal_center_post(maze)
    assert centre == (4, 4)
    assert centre not in posts_without_wall(maze)
    assert centre in posts_without_wall(maze, ignore_goal_center=False)
    assert goal_center_post(Maze(5)) is None        # 1x1 のゴールには中央の柱が無い


def test_check_catches_a_walled_goal():
    """ゴールの内側に壁があれば**誤り** (迷路として成立しない)。"""
    maze = open_maze(8)
    maze.set_wall(3, 3, Direction.E)               # ゴール 2x2 を割る
    report = check_maze(maze)
    assert not report.ok
    assert "ゴール区画の内側" in report.errors[0]


def test_open_goal_region_makes_a_generated_maze_legal():
    """生成器のゴールを競技の形に直す。**全セル到達可能なまま**であること。"""
    from krilly.sim import open_goal_region
    from krilly.sim.check import reachable_cells

    maze = walled_maze(6)
    for x in range(6):                              # 全部つながった素の迷路を作る
        for y in range(6):
            for d in (Direction.N, Direction.E):
                if maze.in_bounds(*maze.neighbor(x, y, d)):
                    maze.set_wall(x, y, d, False)
    open_goal_region(maze)
    assert goal_interior_walls(maze) == 0
    assert len(goal_entrances(maze)) == 1
    assert len(reachable_cells(maze)) == 36         # 孤立させていない


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_generated_mazes_now_have_a_competition_goal(seed):
    """生成器が出す迷路のゴールが競技の形であること (#23 で入れた)。"""
    maze = random_maze(12, seed=seed)
    assert goal_interior_walls(maze) == 0
    assert len(goal_entrances(maze)) >= 1


def test_lowering_the_margin_turns_a_shut_out_maze_into_a_racing_one():
    """**安全率 1.5 は崖の上に乗っていた** (#87)。

    2018 年全日本は、1.5 では復帰 + 最速を始める余裕が無いと判断されて**ゴールに
    着いたまま 1 本も走らずに終わる**が、1.2 なら最速を 1 本走って 7 分に収まる。
    大会迷路 30 面では、この変更で走れない面が 7 面から 3 面へ、最速ランの合計が
    30 本から 38 本へ増える (``maze_sim --budget-sweep`` で再現できる)。
    """
    maze = Maze.from_ascii(
        (CONTEST_DIR / "2018_japan.txt").read_text(encoding="utf-8"))
    safe = simulate_session(maze, time_margin=1.5, times=TIMES_V024, cost=COST_V024)
    bold = simulate_session(maze, time_margin=1.2, times=TIMES_V024, cost=COST_V024)

    assert safe.reached_goal and not safe.speed_runs      # 着いたが走れない
    assert len(bold.speed_runs) == 1                       # 走れる
    assert bold.elapsed_s <= 420.0                         # しかも時間内


def test_a_faster_machine_closes_the_mazes_the_margin_could_not():
    """**残る 3 面は速度でしか届かない** (#87 の梃子 3)。

    安全率は「始めた走行を終えられるか」しか動かせないので、探索そのものが長い面
    (2014/2015/2017 の exp 決勝) には効かない。0.24 -> 0.30 m/s にすると
    30 面すべてが最速ランを走れるようになる。

    速くしても**固定費は縮まない**ので、そこを一緒に割ってはいけない: 停止 + 撮影の
    0.55s はそのままで、ランプの超過 v/2*(1/accel + 1/decel) はむしろ増える。

    **#103 で両方の速度を実測したので、ここは推定式ではなく実測値で回す** (8x8 の
    対照セッション、同じ日・同じ電池)。区間のコスト比も速度で変わる (1.07 -> 1.48)
    ので、経路の選び方ごと切り替えて比べる。
    """
    mazes = [Maze.from_ascii(p.read_text(encoding="utf-8"))
             for p in contest_mazes() if p.stem not in KNOWN_BAD_TRANSCRIPTIONS]
    #: v -> (時間定数, コスト比) の実測 (#103)。
    MEASURED = {
        0.24: ({"cell_time_s": 0.77, "lateral_cell_time_s": 0.75,
                "straight_time_s": 0.825}, MoveCost(cell_ew=0.75 / 0.77, leg=0.825 / 0.77)),
        0.30: ({"cell_time_s": 0.61, "lateral_cell_time_s": 0.60,
                "straight_time_s": 0.900}, MoveCost(cell_ew=0.60 / 0.61, leg=0.900 / 0.61)),
    }

    def shut_out(v: float) -> int:
        times, cost = MEASURED[v]
        return sum(1 for m in mazes
                   if not simulate_session(m, times=times, cost=cost).speed_runs)

    assert shut_out(0.24) == 3
    assert shut_out(0.30) == 0
