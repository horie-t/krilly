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
python -m scripts.cell_move_demo --seq F,L,F,R          # closed-loop 1-cell moves / 90° turns
python -m scripts.cell_move_demo --seq L,L,L,L --camera-yaw   # + camera ground-truth heading check

# STOP the machine (run on the Pi)
#   Ctrl-C stops any driving script and releases the coils. ESC does nothing (no key input).
python -m scripts.motor_stop                      # after a kill -9 left the motors running

# search run (run on the Pi; place the robot in the start cell facing maze north)
python -m scripts.search_run --size 3 --dry-run   # walls + next move only, no driving
python -m scripts.search_run --size 3             # flood-fill search on a 3×3 practice maze
python -m scripts.search_run                      # 16×16 (config/maze.yaml)

# perception tuning
python -m scripts.wall_detect --image shot.png --out walls.png   # off-Pi: per-edge red fraction + verdict
python -m scripts.wall_survey --maze maze3.txt --out-dir survey  # on the Pi: tour a KNOWN maze, save labelled frames
python -m scripts.wall_survey --maze maze3.txt --dry-run          # off-Pi: just print the planned tour
```

## Architecture

Layered under `src/krilly/` (low → high level; each layer is independently testable):

- `hal/` — hardware abstraction: `l6470` (single driver) / `l6470_chain` (3× daisy-chain) over SPI; `imu` (BNO055/I2C); `camera` (picamera2 → BGR frames).
- `kinematics/kiwi.py` — `KiwiKinematics`: forward/inverse kinematics + wheel-speed ⇄ stepper conversion.
- `motion/velocity_driver.py` — `VelocityDriver`: rate-limits body velocity (trapezoidal ramp), then commands all 3 wheels in one `run_all`. Ramping in *body* space keeps the wheel-speed ratio constant, so the path holds during accel (L6470's per-device ACC alone would skew it). `update(dt)` is pure computation — no sleeps; the caller drives the loop.
- `motion/cell_motion.py` — `CellMotion`: closed-loop **1-cell forward / 90° turn** primitives (M4 #17). Holds an *ideal lattice* reference pose and drives the estimated pose to it, so per-move residuals don't accumulate across cells. Primary axis follows a `sqrt(2·a·remaining)` decel envelope (clamped to the `VelocityDriver` ramp so it stays followable) with a **minimum-speed floor** so the tail of the envelope doesn't stall on static friction; the other DOFs are P-held (forward: cross-track + heading; turn: X/Y). Terminates on *remaining*, then settles and re-tries once at creep speed if it coasted past. `update(dt, gyro_rate=…)` is pure computation and also integrates the estimator. On hardware the final residual sits at roughly the **termination tolerance**, so `pos_tol_m` / `angle_tol_rad` are the knobs that set accuracy (keep `floor · dt ≤ 2 · tol` so a floor-speed tick cannot jump the band).
- `localization/` — `estimator.py` (`DeadReckoning`: `[X, Y, φ]`, midpoint integration, pluggable input: commanded speeds / microsteps / distances) and `grid.py` (`GridCorrector`: snap X/Y to the 180 mm grid with a `max_error` guard; `apply_cell_offset`: pull the estimate onto a camera-measured intra-cell position).
- `perception/` — `red_wall.py` (red wall-top detection: two-range HSV mask → contour centroids), `wall_detect.py` (`WallDetector`: per-edge ROI red *fraction* → walls present front/back/left/right, then `body_walls_to_maze()` → `Maze`), `axis_yaw.py` (`axis_yaw`: yaw from the *orientation* of the red wall-top edges — gyro-independent heading truth, see below) and `cell_pose.py` (`cell_offset`: where the machine sits *inside* the cell, from how far the red bands moved — the absolute position fix odometry cannot provide).
- `solver/maze.py` — `Maze` / `Direction`: 16×16 grid with **shared-edge walls** (cell A's east wall *is* cell B's west wall, so they can never disagree), outer walls, centre-2×2 goal, `to_ascii()`.
- `strategy/` — `flood_fill.py` (`flood_fill`: 4-neighbour BFS distances from the goal cells, `UNREACHABLE` for walled-off cells; `next_direction`: descend the gradient, tie-broken by *unvisited → fewest quarter-turns → N/E/S/W* so it's deterministic), `explorer.py` (`Explorer`: observe walls → re-flood → next `Step` (quarter-turns + direction), plus the maze↔world bridge `heading_rad` / `cell_center` / `quarter_turns`) and `shortest_path.py` (`shortest_path`: turn-weighted Dijkstra over *(cell, facing)* restricted to `known` cells; `path_to_legs` run-length-encodes it into `Leg(turn, cells)` for the speed run). All discrete and hardware-free, so a whole search run can be simulated off-Pi against a ground-truth `Maze` (see `tests/test_explorer.py`).
- `app/` — stub pending M6.
- `config/` — `robot.yaml` / `maze.yaml` + typed loader (`RobotConfig`, `MazeConfig`).
- `logging_config.py` — `setup_logging` / `get_logger`.

`scripts/` = per-peripheral bring-up CLIs. `tests/` = pure-logic unit tests with faked transports.

### Conventions & gotchas (read before touching HAL / kinematics)

- **Testable-HAL pattern**: every HAL class takes an injectable transport (`spi=` / `serial_obj=` / `bus=` / `picam2=`); tests pass a fake and assert the exact byte/register stream. The real transport is opened (and its hardware lib lazy-imported) only when the arg is omitted. Follow this for new HAL so it stays unit-testable off-Pi.
- **Coordinate frame** (`docs/coordinate-frames.md`, `config/robot.yaml`): body **+x forward, +y left, +z up** (right-handed), **+ω = CCW**. Wheels/motors: **M0 front, M1 rear-left, M2 rear-right**; L6470 device index *i* = motor M*i* = wheel W*i* (daisy-chain wired M0→M1→M2, index0 nearest MOSI).
- **`wheel_angles_deg` holds SPOKE angles `[0, 120, 240]`, not drive directions.** The IK formula `v_i = -sinθ·vx + cosθ·vy + L·ω` takes θ = spoke angle (wheel *position*); the drive direction is θ+90° = `[90, 210, 330]°`. Putting drive-direction angles in the config rotates all translation by 90° while rotation still looks correct — the exact symptom seen on hardware (#11). If forward/strafe are swapped but ω is fine, suspect this, not the wiring.
- **Maze grid vs body frame**: `solver/maze.py` cells use **east=+x, north=+y** with `Direction` N/E/S/W; that is separate from the body frame above. `perception.wall_detect.body_walls_to_maze()` maps body-relative walls to maze directions via the robot's facing. The metric bridge lives in `strategy/explorer.py`: cell (x, y) centre = `(x·pitch, y·pitch)` and **N=+90° / E=0° / S=−90° / W=180°**, so the start pose (cell (0,0) facing north) is `[0, 0, +π/2]` — not φ=0.
- **Search and speed runs treat unknown cells in *opposite* ways.** `flood_fill` (#18) is deliberately optimistic — an unseen wall is `False` in `Maze`, so unexplored cells read as open and the mouse goes and looks. `shortest_path` (#19) must be pessimistic: pass `known=explorer.visited` so the route only crosses cells whose walls were actually observed. Racing through an unobserved cell is how you hit a wall you never saw. `Maze` has no tri-state for walls; **the `known` set is what carries "unknown"**, so any new consumer has to thread it through.
- **Fewest cells ≠ fastest.** Measured on this machine (#17): 1 cell forward ≈ 1.75 s, a 90° turn ≈ 1.64 s, so a turn costs about one cell. `shortest_path`'s `turn_cost` defaults to `1.0` on that basis and it minimises `cells + turn_cost · quarter_turns` over *(cell, facing)* states — plain BFS on cells would happily pick a staircase of equal cell count. Re-measure `turn_cost` if the ramp limits or `CellMotionConfig` speeds change.
- **Stopping the machine is a signal problem, not a `finally` problem** (`motion/emergency_stop.py`). `with L6470Chain(...)` releases the coils on a clean exit, but `timeout`'s **SIGTERM** and some **SIGINT** paths skip it, and the L6470 then holds its last Run command — the machine spun until power was cut. Every driving script wraps the chain in `emergency_stop(chain)`, which releases (`soft_stop_all` → `hard_hiz_all`, in that order) *inside the signal handler* and then re-raises for the default action. **SIGKILL cannot be caught**: `python -m scripts.motor_stop` is the recovery, or cut VS. And **ESC does nothing** — these scripts read no keyboard; say Ctrl-C when telling someone how to stop.
- **Before driving forward, re-check that the front is clear** (`search_run --no-front-check` to disable). The planner only ever picks a direction the *map* says is open, so a wall appearing in the live frame means the **pose is wrong**, not the map. Aborting there is what keeps a pose error from becoming a series of collisions.
- **Never let a wall *observation* clear a wall the map already knows** (`strategy/explorer.py`). Two safety valves, both earned on hardware: (1) **outer walls are read-only** — the first real `search_run --dry-run` had the camera call the boundary behind the robot "no wall" and erase it from the map; (2) `sticky_walls` (default) keeps an already-seen wall when a later look says "absent", counting it in `Explorer.conflicts` instead. A false negative drives the machine into a wall; a false positive only costs a detour, so the asymmetry is deliberate. A rising `conflicts` means pose error or mis-detection, not a maze that changed.
- **Camera-measured position is the only absolute fix for translation** (#54, `perception/cell_pose.py`). Walls sit exactly 90 mm from the cell centre, so *how far a red band has moved from its calibrated place* is the machine's intra-cell offset: `offset_px / px_per_mm`, with `px_per_mm` self-calibrated from `CALIBRATED_BANDS` (opposite bands are one 180 mm pitch apart). `best_roi_red_fraction` already returns that offset as a by-product of the band search. Two rules: **only correct the axes you measured** (an axis with no wall has no band — treating it as 0 drags the machine toward the centre on evidence you don't have), and reject corrections beyond `max_error` (50 mm) as mis-measurements. On hardware the measured offsets stayed ≤18 mm over a 6-move run instead of growing; a **systematic ~6 mm body-fixed bias** remains (all 7 lateral readings negative, mixed signs in world terms → a camera-mount offset, or `CALIBRATED_BANDS` taken while the machine itself sat a few mm off). Null it by measuring with the machine ruler-centred, then shifting `CALIBRATED_BANDS`.
- **Fusion policy** (`localization/`): translation comes from wheel odometry, **heading from the gyro** (`update_with_gyro*`, since slip hits rotation hardest), and absolute drift is pulled in by `correct_heading` / `GridCorrector.apply_x|y` (snap to the known 180 mm grid). Prefer this split over trusting wheel-derived dφ.
- **Config values are calibrated, not nominal**: `wheel_diameter_m` (~0.0477, effective ≠ nominal 48 mm) and `center_to_wheel_m` (~0.0474) started from `scripts/calibrate.py` (#11) and were re-fitted **on the maze floor** in #17; `gyro_scale_z` (~0.984) came from the camera check below. Don't "fix" them back to spec values, and don't hard-code them in tests — read them off `RobotConfig` (see `tests/test_smoke.py`).
- **Re-fitting `wheel_diameter_m` must scale `center_to_wheel_m` by the same factor.** Only the ratio `L / d` sets the commanded ω, and `calibrate --rotate` derived `L` under the *then-current* `d`; changing `d` alone silently rescales every rotation command. (Turn *accuracy* doesn't depend on either, since heading closes on the gyro — but the commanded ω would no longer mean what it says.) Method used in #17, on the maze: run `--seq F,F`, measure the machine's offset from the end cell's centre as **(front wall gap − back wall gap) / 2** (no need to know the chassis dimensions), then `d ← d · actual / odometry`. Two passes took 2 cells from +7 mm to <1 mm (0.04629 → 0.04732 → 0.04771). Note the maze floor is interlocking panels, so the *true* pitch isn't exactly 180 mm (one cell here is ~1 mm oversized) — fold that into "actual" before dividing.
- **The BNO055 gyro z over-reports by ~1.6%** (#17). Measured against camera ground truth: +1.54 % over a 90° turn, +1.61 % over 360° — proportional to angle, so a *scale* error, not bias (bias is already removed by `measure_gyro_bias`). Uncorrected, a `CellMotion` turn stops ~1.4°/90° short and 4 turns lose ~6°; with `gyro_scale_z` applied the physical error over 360° is **0.08°**. Any consumer that integrates `imu.gyro[2]` must apply `RobotConfig.gyro_scale_z` (see `cell_move_demo`'s `gyro_rate()`), not just the bias.
- **Camera as heading ground truth** (`perception/axis_yaw.py`, `cell_move_demo --camera-yaw`): the red wall tops are long straight lines, so their *orientation* gives yaw without the gyro — repeatability σ≈0.05°, and **+angle = body CCW** (verified with a +30° turn). Only measurable **mod 90°** (the lattice is 90°-symmetric), which is exactly enough for 90°-multiple primitives: the folded before/after delta *is* the physical error. Two traps found the hard way:
  - Do **not** use `minAreaRect` on the red contours — a contour clipped by the image border gets a bounding box snapped to the image axes (reported exactly `0.000°`) and dragged the estimate 6° off. Fit **line segments** (`HoughLinesP`) instead and delete edges along the frame border / exclude-rect boundary, which are artefacts of clipping and masking.
  - Average in the **4θ domain** (circular mean of 4·angle), weighted by segment length; plain averaging breaks across the ±45° fold.
- **L6470 unit gotcha**: the Run/speed registers are in **full step/s** (microstepping does not scale speed); Move / positioning / odometry counts are in **microsteps**. `KiwiKinematics` exposes both (`wheel_speed_to_step_hz` vs `distance_to_microsteps`).
- **picamera2 channel order**: `RGB888` frames are byte-order **BGR** for OpenCV (used directly, no conversion); `red_detect --swap-rb` if colors look inverted.
- **Wall detection is ROI-based, and the ROIs are calibrated to this exact rig** (`perception/wall_detect.py`, `calibrated_rois()` / `CALIBRATED_RED`). Camera is centre-mounted ~39 cm up, one cell fills the 640×480 frame, and **image top = body forward (+x)**. Hard-won specifics — re-check all of these if the mount, height, or lighting changes:
  - Walls appear as bands **inside** the frame, *not* at its edges, so ROIs sit on the bands.
  - The four red **lattice posts are always visible at the corners** (even with no walls) → ROIs sit mid-edge to avoid them.
  - The **camera ribbon cable reads as red** in the lower part of the frame → the BACK ROI is offset left to dodge it (`WallDetectorConfig.exclude` can mask fixed self-occlusion too).
  - The right-hand wall is **washed out, not shadowed** (#56): its red-hue pixels measure S≈46–54 / V≈174–203 (the left wall is S≈179) — pale pink, not dark maroon. So `CALIBRATED_RED` loosens **saturation** to `s_min=50, v_min=40`; `red_wall`'s stricter defaults miss it entirely. Don't "fix" this by lowering `v_min` (the pixels are bright) or by switching to a redness metric like `R − max(G,B)` (washed-out red scores low there too — measured *worse* than the current mask). The root cause is exposure: the frame is mostly black floor, so AE opens up and blows out the bright wall tops (#21).
  - **ROIs must sit on the red bands, and the bands moved when closed-loop motion arrived** (#56). `CALIBRATED_BANDS` records the band positions measured from frames where `CellMotion` parked the robot (±1 mm): FRONT y=126–143, BACK y=425–445, LEFT x=176–194, RIGHT x=464–490. #16's ROIs came from hand-placed shots and were ~15–20 px off for RIGHT and BACK, so a real wall only filled half the ROI → red fraction 0.08–0.15 → **the machine drove into a wall it had judged absent**. To re-check, take the column (or row) profile of `red_mask` and see where the band actually is; `tests/test_wall_detect.py` asserts each ROI contains its band.
  - **A fixed ROI is not enough: the detector searches for the band** (`search_px=40`, `best_roi_red_fraction`). The dominant error source is *geometric, not chromatic* — the machine's real position inside a cell wanders up to ~25 mm (≈40 px), which slides a 26 px band clean out of a 46 px ROI. Sliding the ROI ±40 px and taking the max fixed 30 of 31 misses in the survey below. The returned offset (centre of the max plateau) doubles as an intra-cell position measurement — the seed for #54. The nearest *other* band is 292 px away, so ±40 px cannot grab the wrong wall.
  - **Odometry cannot see that drift.** Straight moves are good (≤1 mm over 2 cells, #17) but each in-place rotation slips a little, and `CellMotion` holds position during a turn using odometry, which is blind to slip. A survey with 4 rotations per cell accumulated 5–25 mm of real position error while the estimate still read "cell centre to 0.1 mm". Any perception that assumes the robot is centred must tolerate this until #54 lands.
  - **Calibrate against a full-maze survey, not a few photos** (`scripts/wall_survey.py`). It takes a *known* layout (`Maze.from_ascii`), plans a tour with `flood_fill` over the true map — so it drives without depending on the detector it is calibrating — and writes every frame plus ground-truth labels per edge. Room lighting changes per cell *and* per heading (glossy wall tops reflect the ceiling lamp), so one cell's thresholds do not generalise: the same wall read 0.00 facing east and 0.38 facing west. Measured over 9 cells × 4 headings = 144 labels: walls 0.102–0.55 (median 0.43), open edges 0.000–0.033 → **threshold 0.08, zero misses, zero false positives**. When in doubt lower the threshold: a missed wall is a collision, a false wall is a detour.
- **`l6470.decode_status()`**: all-`0x0000`/all-`0xFFFF` ⇒ no SPI communication (wiring/power/CS), not a real fault; fault bits are active-low; UVLO on the first read after power-up is the normal power-up latch.

## Contributing workflow

Progress is tracked on a GitHub Project with Milestones (M0–M6) and Issues. For each change: branch per issue → open a PR (`Closes #N`, assigned to the Milestone and Project) → **the repo owner reviews and merges; do not self-merge**. Keep `pytest` green and comments in Japanese.
