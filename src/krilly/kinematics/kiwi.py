"""kiwi-drive (3 omni-wheel) kinematics and wheel-speed <-> stepper conversion.

Conventions follow docs/coordinate-frames.md and config/robot.yaml:
- Body frame: +x forward, +y left, +z up (right-handed); +omega = CCW.
- ``wheel_angles_deg`` are the drive-direction angles theta_i (spoke + 90 deg),
  default [90, 210, 330] for wheels M0(front) / M1(rear-left) / M2(rear-right).

Inverse kinematics (body velocity -> wheel surface speed), for each wheel i:

    v_i = -sin(theta_i) * vx + cos(theta_i) * vy + L * omega

i.e. ``v = J @ [vx, vy, omega]`` with row_i = [-sin theta_i, cos theta_i, L].
Forward kinematics is ``J^-1 @ v`` (J is invertible for the symmetric layout).

Stepper conversion — note the two different units the L6470 uses:
- **Run (speed)**: the L6470 speed registers are in **full step/s** (micro-
  stepping is applied internally and does not scale the commanded speed). So
  ``wheel_speed_to_step_hz`` returns full step/s.
- **Move / odometry (position)**: distances are counted in **microsteps** (per
  the STEP_MODE), so use ``distance_to_microsteps`` for position/dead-reckoning.
"""

from __future__ import annotations

import math

import numpy as np

from krilly.config import RobotConfig, load_robot_config


class KiwiKinematics:
    """Kiwi-drive forward/inverse kinematics for a given robot geometry."""

    def __init__(self, config: RobotConfig | None = None) -> None:
        self.cfg = config or load_robot_config()
        L = self.cfg.center_to_wheel_m
        thetas = [math.radians(a) for a in self.cfg.wheel_angles_deg]
        self._J = np.array(
            [[-math.sin(t), math.cos(t), L] for t in thetas], dtype=float
        )
        self._J_inv = np.linalg.inv(self._J)
        self._m_per_fullstep = self.cfg.wheel_circumference_m / self.cfg.steps_per_rev
        self._m_per_microstep = self.cfg.metres_per_microstep

    # -- kinematics ---------------------------------------------------------
    def body_to_wheels(
        self, vx: float, vy: float, omega: float
    ) -> tuple[float, float, float]:
        """Body velocity (m/s, m/s, rad/s) -> wheel surface speeds (m/s)."""
        v = self._J @ np.array([vx, vy, omega], dtype=float)
        return (float(v[0]), float(v[1]), float(v[2]))

    def wheels_to_body(
        self, v0: float, v1: float, v2: float
    ) -> tuple[float, float, float]:
        """Wheel surface speeds (m/s) -> body velocity (vx, vy, omega)."""
        b = self._J_inv @ np.array([v0, v1, v2], dtype=float)
        return (float(b[0]), float(b[1]), float(b[2]))

    # -- stepper conversions ------------------------------------------------
    def wheel_speed_to_step_hz(self, v_mps: float) -> float:
        """Wheel surface speed (m/s) -> L6470 Run speed (full step/s)."""
        return v_mps / self._m_per_fullstep

    def step_hz_to_wheel_speed(self, step_hz: float) -> float:
        """L6470 Run speed (full step/s) -> wheel surface speed (m/s)."""
        return step_hz * self._m_per_fullstep

    def distance_to_microsteps(self, distance_m: float) -> float:
        """Wheel rolling distance (m) -> microsteps (for Move / odometry)."""
        return distance_m / self._m_per_microstep

    def microsteps_to_distance(self, microsteps: float) -> float:
        """Microsteps -> wheel rolling distance (m)."""
        return microsteps * self._m_per_microstep

    # -- convenience --------------------------------------------------------
    def body_to_wheel_step_hz(
        self, vx: float, vy: float, omega: float
    ) -> tuple[float, float, float]:
        """Body velocity -> each wheel's L6470 Run speed (full step/s)."""
        return tuple(  # type: ignore[return-value]
            self.wheel_speed_to_step_hz(v) for v in self.body_to_wheels(vx, vy, omega)
        )
