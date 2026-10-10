# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Krilly is a **holonomic (3 omni-wheel, "kiwi drive") Micromouse** running on a Raspberry Pi 5, targeting the classic 16×16 Micromouse competition. Work is organized into milestones **M0–M6**: 足場(scaffold) → HAL bring-up → 運動学(kinematics) → 自己位置推定(localization) → 迷路+知覚(maze+perception) → 探索+最速(search+speed) → 統合(integration).

Code comments and docstrings are written in **Japanese** (technical terms, identifiers, register names, and formulas stay in English). Keep new comments consistent.

## Hardware / toolchain

- Raspberry Pi 5 (Raspberry Pi OS), Python ≥ 3.11.
- 3× stepper motors + 3× **L6470** (dSPIN) drivers over **SPI0** (mode 3), daisy-chained on one CS (CE0); single-driver control also supported.
- **BNO055** 9-axis IMU over **I2C** (address 0x28, bus 1). Pi 5's RP1/DesignWare I2C handles clock stretching correctly (unlike Pi 1–4), so 100 kHz works — no `i2c_arm_baudrate` reduction needed.
- Pi **Camera Module V3** (wide), mounted downward, via **picamera2** (`cv2.VideoCapture` does not work on the libcamera stack). Used to detect the **red wall-tops**.
- Hardware-only deps (`spidev`, `lgpio`, `picamera2`) are `aarch64`-gated in `pyproject.toml` and lazy-imported; `smbus2`/`numpy`/`opencv-python`/`pyyaml` are always installed so pure-logic code and tests run on any machine.

Raspberry Pi 5 enablement (SPI/I2C/camera): see `docs/setup-pi5.md`.
Competition-day procedure (what to check before a run, how to read the exposure line, what to do when the venue is dark): see `docs/venue-checklist.md`.

## Commands

```bash
# setup — --system-site-packages so picamera2/lgpio come from the OS apt packages
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e ".[dev]"

# tests (pure logic, no hardware required)
pytest                                          # all
pytest tests/test_kiwi.py                        # one file
pytest tests/test_kiwi.py::test_pure_forward     # one test
pytest -k red_mask                               # by name substring

# hardware bring-up CLIs (run on the Pi)
python -m scripts.motor_spin  --device 0 --speed 400 --duration 3   # one L6470 + motor
python -m scripts.motor_chain --devices 3 --speed 400               # 3 daisy-chained
python -m scripts.imu_stream  --calibrate-gyro                      # BNO055 orientation/gyro
python -m scripts.red_detect  --out red_detect.png                 # camera red-wall detection

# motion / localization (run on the Pi)
python -m scripts.drive_demo    --v 0.1 --omega 1.0   # straight → strafe → spin, ramped
python -m scripts.teleop                              # wasd + qe keyboard teleop
python -m scripts.calibrate      --straight 0.5        # → wheel_diameter_m
python -m scripts.calibrate      --rotate 2            # → center_to_wheel_m (L); do straight first
python -m scripts.odometry_demo  --seg 0.5             # L-path, prints estimated [X, Y, φ]
python -m scripts.heading_demo   --omega 1.0           # odometry-only φ vs gyro-fused φ
python -m scripts.cell_move_demo --seq F,A,B,D          # closed-loop 1-cell moves (WASD-style: F/B/A/D = 前/後/左/右)
python -m scripts.cell_move_demo --seq F,Q,F,E          # ... Q/E = in-place 90° turns (old H/G/L/R/U still parse)
python -m scripts.cell_move_demo --seq Q,Q,Q,Q --camera-yaw   # + camera ground-truth heading check
python -m scripts.cell_move_demo --seq F4 --camera-pose --camera-yaw  # speed tuning: 4 cells in one motion
python -m scripts.cell_move_demo --seq F,A,F,D --chain --camera-pose --camera-yaw  # corner without stopping (#80)
python -m scripts.cell_move_demo --seq F,A,F,D --chain --camera-yaw   # prints the gyro A/B in one run (#81)
python -m scripts.cell_move_demo --seq F,A,F,D --chain --gyro-sample 0  # ... back to the old point sampling

# stand-alone start/stop with one button + a buzzer (#79; systemd unit in deploy/)
python -m scripts.speed_run --size 16 --ev -2 --white-tops --save-run-config  # trial day: save the command that just completed
python -m scripts.launcher --check                 # show what the button will run (no GPIO)
python -m scripts.launcher                         # wait for the button (what systemd runs)

# STOP the machine (run on the Pi)
#   Ctrl-C stops any driving script and releases the coils. ESC does nothing (no key input).
python -m scripts.motor_stop                      # after a kill -9 left the motors running
python -m scripts.spi_watch --seconds 30          # STATUS at 20 Hz while you wiggle the SPI wires (--spin: wheels lifted)

# search run (run on the Pi; place the robot in the start cell facing maze north)
python -m scripts.search_run --size 3 --dry-run   # walls + next move only, no driving
python -m scripts.search_run --size 3             # flood-fill search on a 3×3 practice maze
python -m scripts.search_run                      # 16×16 (config/maze.yaml)
python -m scripts.search_run --size 5 --turn-in-place   # legacy: turn to face, then drive forward
python -m scripts.search_run --size 5 --no-neighbors     # legacy: don't read the E/W neighbours, stop every cell (#89)

# full competition cycle: search → return → speed ×N (run on the Pi)
python -m scripts.speed_run --size 5               # 10-min / 5-run budget, multi-cell straights
python -m scripts.speed_run --size 5 --max-runs 2  # search + one speed run only
python -m scripts.speed_run --size 5 --turn-in-place  # legacy turning mode (for comparison)
python -m scripts.speed_run --size 5 --no-neighbors   # search without the E/W neighbours (#89)
python -m scripts.speed_run --size 5 --chain-legs 1   # stop at every leg instead of rounding corners (#80)
python -m scripts.speed_run --size 5 --time-margin 1.5  # be conservative about starting a run (#87)
python -m scripts.speed_run --size 8 --max-leg-cells 0  # no cap on a leg's length (#85; default 4)
python -m scripts.speed_run --size 16 --ev -2 --white-tops  # competition floor: start/goal tops may be white/yellow (rule 2-1, #125); also seeds the start's 3 walls (#126)
python -m scripts.speed_run --size 3 --goal 2,2 --ev -2 --white-tops  # the 3×3 white-goal board (mazes/white_goal3.txt)
python -m scripts.speed_run --size 8                 # the 8×8 cut from a real maze (#23/#85; the board on the floor is mazes/excerpt8_2015.txt)

# maze simulation (no hardware; issue #77)
python -m scripts.maze_sim --generate 50 --size 16          # 50 random 16×16 sessions
python -m scripts.maze_sim --pattern all --size 8           # adversarial patterns
python -m scripts.maze_sim --maze mazes/practice16.txt --verbose
python -m scripts.maze_sim --maze mazes/practice8.txt --check-only --wall-budget 70
python -m scripts.maze_sim --generate 20 --goal diagonal --actual-scale 1.4
python -m scripts.maze_sim --maze mazes/contest/*.txt                # 31 real contest mazes
python -m scripts.maze_sim --maze mazes/contest/*.txt --budget-sweep  # margin × "reality is N× slower" (#87)
python -m scripts.maze_sim --maze mazes/contest/*.txt --time-margin 1.5  # ... one margin only
python -m scripts.maze_sim --maze mazes/contest/*.txt --no-neighbors  # ... without the E/W neighbours (#89)
python -m scripts.maze_sim --maze mazes/contest/*.txt --max-leg-cells 0  # ... what the leg cap costs (#85)
python -m scripts.maze_from_image shot.png --out-dir mazes/contest   # transcribe a contest maze
python -m scripts.maze_excerpt --size 8 --maze mazes/contest/*.txt --wall-budget 80 --min-search 18
                                                            # pick an 8×8 to build from the real mazes (#23/#85)
python -m scripts.maze_excerpt --size 8 --generate 2400 --wall-budget 80 --sort exposure --min-cells 24
                                                            # ... pick one for sample count instead: generated, twisty, no loops (#85)

# perception tuning
python -m scripts.wall_detect --image shot.png --out walls.png   # off-Pi: per-edge red fraction + verdict
python -m scripts.wall_detect --image shot.png --hue-split --zoom 610,790,580,720  # which hue band flagged it
python -m scripts.wall_detect --image shot.png --neighbors        # + the E/W neighbours' ROIs (purple = undecided)
python -m scripts.wall_detect --measure 5 --neighbors             # on the Pi: are the far bands readable?
python -m scripts.wall_detect --measure 5                        # on the Pi: pose readout without driving
python -m scripts.wall_detect --measure 5 --ev -2 --white-tops    # black floor: also read white/yellow tops (w = source, #125)
python -m scripts.wall_detect --image shot.png --white-tops       # off-Pi: white/yellow pixels overlaid in orange
python -m scripts.wall_survey --maze maze5.txt --out-dir survey  # on the Pi: tour a KNOWN maze, save labelled frames
python -m scripts.wall_survey --maze maze5.txt --dry-run          # off-Pi: just print the planned tour
python -m scripts.survey_shot --maze maze5.txt --out-dir survey   # on the Pi: HUMAN moves the robot, script shoots+labels

# lighting robustness: turn a search run into calibration data (#78)
python -m scripts.search_run --size 5 --save-frames survey/run   # on the Pi: frames + survey/run_labels.csv
python -m scripts.search_run --size 5 --save-frames survey/run --truth-maze mazes/practice5.txt
python -m scripts.survey_report survey/run_labels.csv             # off-Pi: distribution, margin, misses
python -m scripts.survey_report --sweep survey/run_labels.csv     # ... thresholds that give zero errors
python -m scripts.survey_report --threshold 0.12 survey/run_labels.csv   # ... what-if, no images needed
python -m scripts.survey_report bright_labels.csv dim_labels.csv  # ... two lighting conditions, side by side
python -m scripts.wall_detect --batch survey/ --h2-lo 150         # off-Pi: re-judge the saved frames, confusion matrix
python -m scripts.wall_detect --batch survey/ --weak 5            # ... why the weakest walls read weak (hue? saturation?)
```

## Architecture

Layered under `src/krilly/` (low → high level; each layer is independently testable):

- `hal/` — hardware abstraction: `l6470` (single driver) / `l6470_chain` (3× daisy-chain) over SPI; `imu` (BNO055/I2C); `gyro_sampler` (`GyroSampler`: samples gyro z on its own thread and integrates every sample, so the 50 Hz control loop stops throwing half the chip's output away, #81); `camera` (picamera2 → BGR frames).
- `kinematics/kiwi.py` — `KiwiKinematics`: forward/inverse kinematics + wheel-speed ⇄ stepper conversion.
- `motion/velocity_driver.py` — `VelocityDriver`: rate-limits body velocity (trapezoidal ramp), then commands all 3 wheels in one `run_all`. Ramping in *body* space keeps the wheel-speed ratio constant, so the path holds during accel (L6470's per-device ACC alone would skew it). `update(dt)` is pure computation — no sleeps; the caller drives the loop.
- `motion/corner.py` — the geometry of turning **without stopping** (#80): the blend distance `v²/2a`, the shape of the rounded corner, and how much room it leaves against the posts. Pure functions, no state; `tuning.check_limits` uses them to refuse a blend that would not fit.
- `motion/cell_motion.py` — `CellMotion`: closed-loop **1-cell translation (any of the 4 body directions) / 90° turn** primitives (M4 #17, M6 #76), plus **chained legs with rounded corners** (`queue_move` / `start_path`, #80). `start_move(distance, axis_quarter_turns)` picks the primary axis; `start_forward*` are thin wrappers with `axis_quarter_turns=0`. Holds an *ideal lattice* reference pose and drives the estimated pose to it, so per-move residuals don't accumulate across cells. Primary axis follows a `sqrt(2·a·remaining)` decel envelope (clamped to the `VelocityDriver` ramp so it stays followable) with a **minimum-speed floor** so the tail of the envelope doesn't stall on static friction; the other DOFs are P-held (translation: cross-track + heading; turn: X/Y). Terminates on *remaining*, then settles and re-tries once at creep speed if it coasted past. `update(dt, gyro_rate=…)` is pure computation and also integrates the estimator. On hardware the final residual sits at roughly the **termination tolerance**, so `pos_tol_m` / `angle_tol_rad` are the knobs that set accuracy (keep `floor · dt ≤ 2 · tol` so a floor-speed tick cannot jump the band).
- `localization/` — `estimator.py` (`DeadReckoning`: `[X, Y, φ]`, midpoint integration, pluggable input: commanded speeds / microsteps / distances) and `grid.py` (`GridCorrector`: snap X/Y to the 180 mm grid with a `max_error` guard; `apply_cell_offset`: pull the estimate onto a camera-measured intra-cell position).
- `perception/` — `red_wall.py` (red wall-top detection: two-range HSV mask → contour centroids), `wall_detect.py` (`WallDetector`: per-edge ROI red *fraction* → walls present front/back/left/right, then `body_walls_to_maze()` → `Maze`; with `calibrated_config(neighbors=True)` it also reads the **E/W neighbours' four walls** from six more ROIs placed by the same geometry, `neighbor_walls()` → tri-state per edge, #89), `survey.py` (the calibration-data layer, #78: one CSV row per frame-and-slot with the measured red fraction and the *ground-truth* label; `label_run` labels a whole run's frames **after** it finishes, from the map it ended with — `summarize`/`format_report`/`compare` then turn that into the per-edge distribution, the separation margin and the miss/false-alarm counts), `axis_yaw.py` (`axis_yaw`: yaw from the *orientation* of the red wall-top edges — gyro-independent heading truth, see below) and `cell_pose.py` (`cell_offset`: where the machine sits *inside* the cell, from how far the red bands moved — the absolute position fix odometry cannot provide).
- `solver/maze.py` — `Maze` / `Direction`: 16×16 grid with **shared-edge walls** (cell A's east wall *is* cell B's west wall, so they can never disagree), outer walls, centre-2×2 goal, `to_ascii()`.
- `strategy/` — `flood_fill.py` (`flood_fill`: 4-neighbour BFS distances from the goal cells, `UNREACHABLE` for walled-off cells; `next_direction`: descend the gradient, tie-broken by *unvisited → fewest quarter-turns → N/E/S/W* so it's deterministic), `explorer.py` (`Explorer`: observe walls → re-flood → next `Step(direction, to_cell)`, plus the maze↔world bridge `heading_rad` / `cell_center` / `quarter_turns`; `holonomic=True` keeps `facing` fixed and tracks `travel` = the last direction moved; `known` = cells whose **four walls were all observed** and `plan_leg(n)` = the steps that can run as one motion without stopping, #89) and `shortest_path.py` (`shortest_path`: Dijkstra over *(cell, previous direction)* restricted to `known` cells, weighted by a `MoveCost(cell_ns, cell_ew, leg, turn)`; `path_to_legs` run-length-encodes it into `Leg(direction, cells)` for the speed run. `DEFAULT_COST` is turn-free, `LEGACY_COST` reproduces the old turn-weighted search exactly). All discrete and hardware-free, so a whole search run can be simulated off-Pi against a ground-truth `Maze` (see `tests/test_explorer.py`).
- `app/` — `run_manager.py` (`RunManager` / `RunPhase`: the WAIT → SEARCH → RETURN_HOME → SPEED → … state machine under the classic **7-minute / 5-run** budget (rule 3-6; M5 #20). Counts a *run* as a departure from the start square toward the goal — the search is run 1, returns are free. It only *starts* a run when the remaining time covers the (return + speed) estimate × `time_margin`; the estimate counts N/S cells, E/W cells and legs separately (and turns too when `holonomic=False`), so re-measure those if speeds change. Pure logic — `now` is passed in, so the whole session simulates in tests. `scripts/speed_run.py` drives it: search like `search_run`, then executes `Leg(direction, cells)` lists with **multi-cell straights in one motion**, path-check + confidence-gated position correction at every leg boundary).
- `sim/` — **hardware-free simulation of a whole competition session** (#77): `excerpt.py` (cut an N×N out of a transcribed contest maze and score how much of it can only be tested on hardware — `longest_blind_cross_run` is the one that matters, #23/#85), `sense.py` (`sense`: the camera's stand-in — reads body-relative walls off a ground-truth `Maze`; the exact inverse of `body_walls_to_maze`; `sense_neighbors` does the same for the E/W neighbours), `generate.py` (`random_maze` = spanning tree + loops, plus `open_maze` / `serpentine_maze` / `comb_maze` / `seal_goal` / `diagonal_goal` as adversarial cases), `check.py` (`check_maze`: outer closed / goal reachable / bare posts / boxed cells, and `wall_counts` for "can I physically build this?"; `map_agrees` compares the learned map against truth **over the visited set**), `session.py` (`simulate_session`: SEARCH → RETURN → SPEED ×N driven by `RunManager` with `estimate_s` as the clock). `scripts/maze_sim.py` runs many mazes and exits non-zero on a problem. Maze data in `mazes/` (see its README for the ASCII format), including **31 real All-Japan mazes 2010–2024** in `mazes/contest/`, transcribed from the record-book figures by `scripts/maze_from_image.py` (#84).
- `config/` — `robot.yaml` / `maze.yaml` + typed loader (`RobotConfig`, `MazeConfig`).
- `logging_config.py` — `setup_logging` / `get_logger`.

`scripts/` = per-peripheral bring-up CLIs. `tests/` = pure-logic unit tests with faked transports.

### Conventions & gotchas (always-on)

- **Testable-HAL pattern**: every HAL class takes an injectable transport (`spi=` / `serial_obj=` / `bus=` / `picam2=`); tests pass a fake and assert the exact byte/register stream. The real transport is opened (and its hardware lib lazy-imported) only when the arg is omitted. Follow this for new HAL so it stays unit-testable off-Pi.
- **Coordinate frame** (`docs/coordinate-frames.md`, `config/robot.yaml`): body **+x forward, +y left, +z up** (right-handed), **+ω = CCW**. Wheels/motors: **M0 front, M1 rear-left, M2 rear-right**; L6470 device index *i* = motor M*i* = wheel W*i* (daisy-chain wired M0→M1→M2, index0 nearest MOSI).
- **`wheel_angles_deg` holds SPOKE angles `[0, 120, 240]`, not drive directions.** The IK formula `v_i = -sinθ·vx + cosθ·vy + L·ω` takes θ = spoke angle (wheel *position*); the drive direction is θ+90° = `[90, 210, 330]°`. Putting drive-direction angles in the config rotates all translation by 90° while rotation still looks correct — the exact symptom seen on hardware (#11). If forward/strafe are swapped but ω is fine, suspect this, not the wiring.
- **Maze grid vs body frame**: `solver/maze.py` cells use **east=+x, north=+y** with `Direction` N/E/S/W; that is separate from the body frame above. `perception.wall_detect.body_walls_to_maze()` maps body-relative walls to maze directions via the robot's facing. The metric bridge lives in `strategy/explorer.py`: cell (x, y) centre = `(x·pitch, y·pitch)` and **N=+90° / E=0° / S=−90° / W=180°**, so the start pose (cell (0,0) facing north) is `[0, 0, +π/2]` — not φ=0.
- **L6470 unit gotcha**: the Run/speed registers are in **full step/s** (microstepping does not scale speed); Move / positioning / odometry counts are in **microsteps**. `KiwiKinematics` exposes both (`wheel_speed_to_step_hz` vs `distance_to_microsteps`).
- **picamera2 channel order**: `RGB888` frames are byte-order **BGR** for OpenCV (used directly, no conversion); `red_detect --swap-rb` if colors look inverted.
- **Safety rules that apply everywhere** (details in the linked pages — read them before changing the code involved):
  - A missed wall is a collision, a false wall is a detour: wall-*detection* thresholds stay low, while anything that *moves* the machine (position fix, path check) needs strong evidence. **Never reuse a threshold under a different prior** ([wall-detection](docs/claude/wall-detection.md), [camera-geometry](docs/claude/camera-geometry.md)).
  - `Maze` has no tri-state: the `known` / `observed` sets carry "unknown", and every new consumer must thread them through ([search-and-planning](docs/claude/search-and-planning.md), [recovery](docs/claude/recovery.md)).
  - Odometry integrates *commanded* speeds, so slip, rubbing and a flat pack are invisible to it and to the L6470 flags ([mechanics-and-faults](docs/claude/mechanics-and-faults.md)).
  - Stop with Ctrl-C (ESC does nothing); after `kill -9` run `python -m scripts.motor_stop` or cut VS ([mechanics-and-faults](docs/claude/mechanics-and-faults.md)).
  - Calibrated config values are not nominal — don't "fix" them to spec values or hard-code them in tests ([odometry-and-gyro](docs/claude/odometry-and-gyro.md)).
  - Any exposure / lighting number must state the floor (wood vs black) and EV it was measured on ([lighting-and-floors](docs/claude/lighting-and-floors.md)).
  - Repeat a surprising measurement before theorising about it; a time constant fitted with `--save-frames` on is invalid ([hardware-sessions](docs/claude/hardware-sessions.md)).

## Knowledge pages (`docs/claude/`)

The measured findings, their numbers and the reasoning behind each design decision live in `docs/claude/`, one page per topic. **They are not loaded automatically**: before touching a subsystem, read its page (Read the file, or `grep -rn <keyword> docs/claude/`). Code comments and other docs that say "see CLAUDE.md" mean these pages too.

| ページ | 内容 / いつ読むか |
|---|---|
| [rules-and-budget](docs/claude/rules-and-budget.md) | classic ルール (7 分 / 5 走行、3-4・3-5 など)、`RunManager` の時間見積もりと `time_margin`、時間定数 (セル・モーション単位のコスト)。 |
| [maze-sim](docs/claude/maze-sim.md) | 生成迷路と実迷路の難しさの差、ゴール/スタートの形のチェック、迷路の切り出し (excerpt) と blind run の指標。 |
| [search-and-planning](docs/claude/search-and-planning.md) | 隣接セル読み取り (#89)、unknown の扱い、斜行せず横移動する方針 (#76)、経路チェック、壁を消さない規則。 |
| [motion-and-corners](docs/claude/motion-and-corners.md) | 停止しないコーナー (#80)、チェイン数・脚長の上限、ランプ (accel/decel) と動作点の決め方、旋回のオーバーシュート (#73)、異常停止の裾 (#107)。 |
| [recovery](docs/claude/recovery.md) | `--max-heading-residual` で止まった後のその場回復: verify_cell、ガードの逆転、壁への接触 (lattice phase)、mod 90° の罠。 |
| [odometry-and-gyro](docs/claude/odometry-and-gyro.md) | ジャイロのサンプリング (#81)、融合方針、heading の基準、crab、W0 の径、較正値、`gyro_scale_z`、`--align` なしの比較の罠、カメラ vs 定規。 |
| [mechanics-and-faults](docs/claude/mechanics-and-faults.md) | L6470 の状態ポーリング、ボタンランチャー、非常停止 (シグナル)、励磁の跳ね、緩んだ締結、ローラーの擦れ、異音 4 原因の対照表。 |
| [camera-geometry](docs/claude/camera-geometry.md) | カメラの傾き・取り付け、全 FOV で使う壁、axis_yaw による heading 真値、cell_offset による位置補正とその飽和・閾値、横ゼロ点。 |
| [wall-detection](docs/claude/wall-detection.md) | ROI の較正と探索、HSV マスク (`h2_lo` など)、リボンケーブル、閾値の根拠、木の床と自立壁、手動/走行中の較正データ収集 (#78)。 |
| [lighting-and-floors](docs/claude/lighting-and-floors.md) | AE/ゲインの余裕、色温度、黒い床での BACK の白飛びと `--ev -2`、始点/終点の白・黄の壁上面 (#125/#126)。 |
| [hardware-sessions](docs/claude/hardware-sessions.md) | 5×5・8×8 の通し走行、試走会 (2026-09-26) と本番 (2026-09-27) で何が起き、何を学んだか。 |

When a session produces a new finding, add it to the matching page (or create a new page and add a row here) — **do not grow CLAUDE.md itself**; it stays an overview + index so that it fits the context limit. Keep the existing style: a bold one-line claim, then the measurement with its conditions (speed, floor, sign convention, issue number).

## Contributing workflow

Progress is tracked on a GitHub Project with Milestones (M0–M6) and Issues. For each change: branch per issue → open a PR (`Closes #N`, assigned to the Milestone and Project) → **the repo owner reviews and merges; do not self-merge**. Keep `pytest` green and comments in Japanese.
