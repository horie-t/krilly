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
from krilly.sim.check import posts_without_wall, reachable_cells
from krilly.sim.generate import walled_maze
from krilly.sim.session import fit_search_overhead
from krilly.solver.maze import Direction, Maze
from krilly.strategy.shortest_path import LEGACY_COST, shortest_path

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
    """外周 + 内壁の上限 = 柱の本数 (4N + (N-1)^2 = (N+1)^2)。"""
    c = wall_counts(open_maze(n))
    assert c.outer == 4 * n
    assert c.inner == 0
    assert c.posts == (n + 1) ** 2
    assert c.inner_slots == 2 * n * (n + 1) - 4 * n
    assert 4 * n + c.inner_max == c.posts


def test_a_perfect_maze_sits_exactly_at_the_interior_upper_bound():
    """全セル到達可能なら内壁は (N-1)^2 枚以下。完全迷路はちょうどその値になる。"""
    m = random_maze(10, seed=0, loop_ratio=0.0)
    c = wall_counts(m)
    assert c.inner == c.inner_max == 81
    assert len(reachable_cells(m)) == 100


def test_a_maze_with_loops_stays_under_the_bound():
    for seed in range(5):
        m = random_maze(10, seed=seed, loop_ratio=0.2)
        c = wall_counts(m)
        assert c.inner_min <= c.inner <= c.inner_max


def test_practice8_fits_the_70_wall_budget():
    """#23 の実機検証は手持ち 70 枚で組めなければ意味がない。"""
    m = Maze.from_ascii((MAZE_DIR / "practice8.txt").read_text(encoding="utf-8"))
    assert m.size == 8
    assert wall_counts(m).total <= 70
    assert check_maze(m, wall_budget=70).ok


# --- 柱 ---------------------------------------------------------------------
def test_generated_mazes_keep_every_post_walled():
    """公式規則「柱には必ず 1 枚以上の壁が接する」を守って壁を抜いていること。"""
    for seed in range(5):
        assert posts_without_wall(random_maze(8, seed=seed, loop_ratio=0.5)) == []


def test_an_open_maze_leaves_every_interior_post_bare():
    assert len(posts_without_wall(open_maze(8))) == 49          # (8-1)^2


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


@pytest.mark.parametrize("scale", [1.0, 1.2, 1.4])
def test_the_budget_holds_even_when_reality_is_slower_than_the_estimate(scale):
    """``time_margin`` (1.5) が実際の遅れを吸収できるか。

    ``actual_scale`` を上げると見積もりより実際が遅くなる。持ち時間を超えないことと、
    超えそうなら走行を減らして対応することを確かめる。
    """
    for seed in range(6):
        result = simulate_session(random_maze(16, seed=seed), actual_scale=scale)
        assert result.elapsed_s <= 420.0, result.describe()
        assert result.mismatches == []


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


@pytest.mark.parametrize("path", contest_mazes(), ids=lambda p: p.stem)
def test_every_contest_maze_is_solvable(path):
    """大会迷路は必ず解ける。解けないなら書き起こしの誤り (#84)。"""
    maze = Maze.from_ascii(path.read_text(encoding="utf-8"))
    assert maze.size == 16
    report = check_maze(maze)
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

    壁の枚数を合わせても解の長さが 4 倍違う。難しさは密度ではなく配置で決まる。
    """
    import statistics
    real = [Maze.from_ascii(p.read_text(encoding="utf-8")) for p in contest_mazes()]
    fake = [random_maze(16, seed=s, loop_ratio=0.10) for s in range(len(real))]
    real_sol = statistics.median(len(shortest_path(m)) - 1 for m in real)
    fake_sol = statistics.median(len(shortest_path(m)) - 1 for m in fake)
    real_walls = statistics.median(wall_counts(m).inner for m in real)
    fake_walls = statistics.median(wall_counts(m).inner for m in fake)
    assert abs(real_walls - fake_walls) < 40      # 壁の枚数は近いのに
    assert real_sol > 3 * fake_sol               # 解の長さは 3 倍以上違う


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
