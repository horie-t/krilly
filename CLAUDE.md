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
```

## Architecture

Layered under `src/krilly/` (low → high level; each layer is independently testable):

- `hal/` — hardware abstraction: `l6470` (single driver) / `l6470_chain` (3× daisy-chain) over SPI; `imu` (BNO055/I2C); `camera` (picamera2 → BGR frames).
- `kinematics/kiwi.py` — `KiwiKinematics`: forward/inverse kinematics + wheel-speed ⇄ stepper conversion.
- `motion/velocity_driver.py` — `VelocityDriver`: rate-limits body velocity (trapezoidal ramp), then commands all 3 wheels in one `run_all`. Ramping in *body* space keeps the wheel-speed ratio constant, so the path holds during accel (L6470's per-device ACC alone would skew it). `update(dt)` is pure computation — no sleeps; the caller drives the loop.
- `localization/` — `estimator.py` (`DeadReckoning`: `[X, Y, φ]`, midpoint integration, pluggable input: commanded speeds / microsteps / distances) and `grid.py` (`GridCorrector`: snap X/Y to the 180 mm grid with a `max_error` guard).
- `perception/red_wall.py` — red wall-top detection (two-range HSV mask → contour centroids).
- `solver/maze.py` — `Maze` / `Direction`: 16×16 grid with **shared-edge walls** (cell A's east wall *is* cell B's west wall, so they can never disagree), outer walls, centre-2×2 goal, `to_ascii()`.
- `strategy/`, `app/` — stubs pending M5–M6.
- `config/` — `robot.yaml` / `maze.yaml` + typed loader (`RobotConfig`, `MazeConfig`).
- `logging_config.py` — `setup_logging` / `get_logger`.

`scripts/` = per-peripheral bring-up CLIs. `tests/` = pure-logic unit tests with faked transports.

### Conventions & gotchas (read before touching HAL / kinematics)

- **Testable-HAL pattern**: every HAL class takes an injectable transport (`spi=` / `serial_obj=` / `bus=` / `picam2=`); tests pass a fake and assert the exact byte/register stream. The real transport is opened (and its hardware lib lazy-imported) only when the arg is omitted. Follow this for new HAL so it stays unit-testable off-Pi.
- **Coordinate frame** (`docs/coordinate-frames.md`, `config/robot.yaml`): body **+x forward, +y left, +z up** (right-handed), **+ω = CCW**. Wheels/motors: **M0 front, M1 rear-left, M2 rear-right**; L6470 device index *i* = motor M*i* = wheel W*i* (daisy-chain wired M0→M1→M2, index0 nearest MOSI).
- **`wheel_angles_deg` holds SPOKE angles `[0, 120, 240]`, not drive directions.** The IK formula `v_i = -sinθ·vx + cosθ·vy + L·ω` takes θ = spoke angle (wheel *position*); the drive direction is θ+90° = `[90, 210, 330]°`. Putting drive-direction angles in the config rotates all translation by 90° while rotation still looks correct — the exact symptom seen on hardware (#11). If forward/strafe are swapped but ω is fine, suspect this, not the wiring.
- **Maze grid vs body frame**: `solver/maze.py` cells use **east=+x, north=+y** with `Direction` N/E/S/W; that is separate from the body frame above. `perception.wall_detect.body_walls_to_maze()` maps body-relative walls to maze directions via the robot's facing.
- **Fusion policy** (`localization/`): translation comes from wheel odometry, **heading from the gyro** (`update_with_gyro*`, since slip hits rotation hardest), and absolute drift is pulled in by `correct_heading` / `GridCorrector.apply_x|y` (snap to the known 180 mm grid). Prefer this split over trusting wheel-derived dφ.
- **Config values are calibrated, not nominal**: `wheel_diameter_m` (~0.0463, effective ≠ nominal 48 mm) and `center_to_wheel_m` (~0.0460) came from `scripts/calibrate.py`. Don't "fix" them back to spec values, and don't hard-code them in tests — read them off `RobotConfig` (see `tests/test_smoke.py`).
- **L6470 unit gotcha**: the Run/speed registers are in **full step/s** (microstepping does not scale speed); Move / positioning / odometry counts are in **microsteps**. `KiwiKinematics` exposes both (`wheel_speed_to_step_hz` vs `distance_to_microsteps`).
- **picamera2 channel order**: `RGB888` frames are byte-order **BGR** for OpenCV (used directly, no conversion); `red_detect --swap-rb` if colors look inverted.
- **`l6470.decode_status()`**: all-`0x0000`/all-`0xFFFF` ⇒ no SPI communication (wiring/power/CS), not a real fault; fault bits are active-low; UVLO on the first read after power-up is the normal power-up latch.

## Contributing workflow

Progress is tracked on a GitHub Project with Milestones (M0–M6) and Issues. For each change: branch per issue → open a PR (`Closes #N`, assigned to the Milestone and Project) → **the repo owner reviews and merges; do not self-merge**. Keep `pytest` green and comments in Japanese.
