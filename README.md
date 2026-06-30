# krilly

Krilly is a Micromouse equipped with omni-wheels — a **holonomic (3 omni-wheel,
kiwi-drive)** Micromouse running on a Raspberry Pi 5, targeting the classic
16×16 Micromouse competition.

## Hardware

- Raspberry Pi 5
- 3× omni-wheels (Ø48 mm, width 25.5 mm)
- 3× stepper motors (1.8°/step) + 3× L6470 drivers (SPI daisy-chain)
- BNO055 9-axis IMU (UART mode)
- Pi Camera Module V3 (wide), mounted downward to detect the **red wall-tops**
  of the maze

## Architecture

Layered, low → high level (see `src/krilly/`):

| Layer | Role |
|-------|------|
| `hal` | L6470 (SPI) / BNO055 (UART) / camera (picamera2) |
| `kinematics` | kiwi-drive forward/inverse kinematics, wheel-speed ⇄ step rate |
| `motion` | body velocity (vx, vy, ω) → 3 wheels, accel ramps |
| `localization` | dead-reckoning + gyro heading + camera grid correction `[X, Y, φ]` |
| `perception` | red wall-top detection (HSV) |
| `solver` | maze model (16×16, shared-edge) + flood-fill + fastest path |
| `strategy` / `app` | search-run / fastest-run state machine |
| `config` | robot & maze dimensions (YAML) |

## Development

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Raspberry Pi 5 hardware setup: see [`docs/setup-pi5.md`](docs/setup-pi5.md).

Progress is tracked on the [GitHub Project board](https://github.com/users/horie-t/projects/4)
via Milestones (M0–M6) and Issues; code lands through reviewed PRs.

## License

MIT — see [LICENSE](LICENSE).
