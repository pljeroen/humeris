# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Numerical orbit propagation with RK4 and pluggable force models.

4th-order Runge-Kutta integrator with composable perturbation forces.
Handles accumulated perturbation effects (drag, SRP, higher-order
gravitational harmonics) that analytical propagation cannot model.

No external dependencies — only stdlib math/dataclasses/typing/datetime
+ domain imports.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Protocol, runtime_checkable

from constellation_generator.domain.orbital_mechanics import (
    OrbitalConstants,
    kepler_to_cartesian,
)
from constellation_generator.domain.atmosphere import (
    DragConfig,
    atmospheric_density,
)
from constellation_generator.domain.solar import (
    sun_position_eci,
    AU_METERS,
)
from constellation_generator.domain.eclipse import is_eclipsed, EclipseType


# --- Types ---

@runtime_checkable
class ForceModel(Protocol):
    """Structural typing port for pluggable force models."""

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]: ...


@dataclass(frozen=True)
class PropagationStep:
    """Single step in a numerical propagation trajectory."""
    time: datetime
    position_eci: tuple[float, float, float]
    velocity_eci: tuple[float, float, float]


@dataclass(frozen=True)
class NumericalPropagationResult:
    """Complete result of a numerical propagation run."""
    steps: tuple[PropagationStep, ...]
    epoch: datetime
    duration_s: float
    force_model_names: tuple[str, ...]


# --- Force models ---

class TwoBodyGravity:
    """Central body gravitational acceleration: a = -mu * r / |r|^3."""

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        x, y, z = position
        r = math.sqrt(x * x + y * y + z * z)
        r3 = r * r * r
        coeff = -OrbitalConstants.MU_EARTH / r3
        return (coeff * x, coeff * y, coeff * z)


class J2Perturbation:
    """J2 zonal harmonic perturbation acceleration."""

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        x, y, z = position
        r2 = x * x + y * y + z * z
        r = math.sqrt(r2)
        r5 = r2 * r2 * r

        mu = OrbitalConstants.MU_EARTH
        j2 = OrbitalConstants.J2_EARTH
        re = OrbitalConstants.R_EARTH_EQUATORIAL

        coeff = -1.5 * j2 * mu * re * re / r5
        z2_r2 = z * z / r2

        ax = coeff * x * (1.0 - 5.0 * z2_r2)
        ay = coeff * y * (1.0 - 5.0 * z2_r2)
        az = coeff * z * (3.0 - 5.0 * z2_r2)
        return (ax, ay, az)


class J3Perturbation:
    """J3 zonal harmonic perturbation acceleration.

    Derived from geopotential gradient (Montenbruck & Gill).
    """

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        x, y, z = position
        r2 = x * x + y * y + z * z
        r = math.sqrt(r2)
        r7 = r2 * r2 * r2 * r

        mu = OrbitalConstants.MU_EARTH
        j3 = OrbitalConstants.J3_EARTH
        re = OrbitalConstants.R_EARTH_EQUATORIAL

        coeff = mu * j3 * re * re * re / 2.0

        z3_r2 = z * z * z / r2

        ax = coeff * (x / r7) * (35.0 * z3_r2 - 15.0 * z)
        ay = coeff * (y / r7) * (35.0 * z3_r2 - 15.0 * z)
        az = -coeff * (1.0 / r7) * (30.0 * z * z - 35.0 * z * z * z * z / r2 - 3.0 * r2)
        return (ax, ay, az)


class AtmosphericDragForce:
    """Atmospheric drag acceleration with co-rotating atmosphere.

    a = -0.5 * rho * Cd * (A/m) * |v_rel| * v_rel
    where v_rel accounts for atmosphere co-rotation.
    """

    def __init__(self, drag_config: DragConfig) -> None:
        self._config = drag_config

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        x, y, z = position
        vx, vy, vz = velocity

        # Altitude check
        r = math.sqrt(x * x + y * y + z * z)
        alt_km = (r - OrbitalConstants.R_EARTH_EQUATORIAL) / 1000.0

        # Return zero if outside atmosphere table range
        try:
            rho = atmospheric_density(alt_km)
        except ValueError:
            return (0.0, 0.0, 0.0)

        # Relative velocity (atmosphere co-rotates with Earth)
        omega_e = OrbitalConstants.EARTH_ROTATION_RATE
        vr_x = vx + omega_e * y
        vr_y = vy - omega_e * x
        vr_z = vz

        v_rel = math.sqrt(vr_x * vr_x + vr_y * vr_y + vr_z * vr_z)
        if v_rel < 1e-10:
            return (0.0, 0.0, 0.0)

        bc = self._config.ballistic_coefficient
        coeff = -0.5 * rho * bc * v_rel

        return (coeff * vr_x, coeff * vr_y, coeff * vr_z)


class SolarRadiationPressureForce:
    """Solar radiation pressure (cannonball model, no shadow).

    a = P_sr * Cr * (A/m) * (AU/|d|)^2 * d_hat
    where d = r_sat - r_sun.
    """

    _P_SR: float = 4.56e-6  # N/m² — solar radiation pressure at 1 AU

    def __init__(
        self,
        cr: float,
        area_m2: float,
        mass_kg: float,
        include_shadow: bool = False,
    ) -> None:
        self._cr = cr
        self._area_m2 = area_m2
        self._mass_kg = mass_kg
        self._include_shadow = include_shadow

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        sun = sun_position_eci(epoch)
        sx, sy, sz = sun.position_eci_m

        if self._include_shadow:
            eclipse = is_eclipsed(position, (sx, sy, sz))
            if eclipse == EclipseType.UMBRA:
                return (0.0, 0.0, 0.0)
            if eclipse == EclipseType.PENUMBRA:
                # Forward-compatible: current is_eclipsed never returns PENUMBRA
                # but apply 0.5 factor if it ever does
                shadow_factor = 0.5
            else:
                shadow_factor = 1.0
        else:
            shadow_factor = 1.0

        # d = r_sat - r_sun (from Sun toward satellite)
        dx = position[0] - sx
        dy = position[1] - sy
        dz = position[2] - sz

        d_mag = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d_mag < 1e-10:
            return (0.0, 0.0, 0.0)

        am_ratio = self._area_m2 / self._mass_kg
        au_ratio_sq = (AU_METERS / d_mag) ** 2
        coeff = self._P_SR * self._cr * am_ratio * au_ratio_sq / d_mag * shadow_factor

        return (coeff * dx, coeff * dy, coeff * dz)


# --- RK4 integrator ---

def rk4_step(
    t_s: float,
    state: tuple[float, ...],
    h: float,
    deriv_fn: Callable[[float, tuple[float, ...]], tuple[float, ...]],
) -> tuple[float, tuple[float, ...]]:
    """Single 4th-order Runge-Kutta integration step.

    Args:
        t_s: Current time (seconds).
        state: Current state vector.
        h: Step size (seconds).
        deriv_fn: Derivative function f(t, state) -> d(state)/dt.

    Returns:
        (t_new, state_new)
    """
    k1 = deriv_fn(t_s, state)
    s1 = tuple(s + 0.5 * h * k for s, k in zip(state, k1))

    k2 = deriv_fn(t_s + 0.5 * h, s1)
    s2 = tuple(s + 0.5 * h * k for s, k in zip(state, k2))

    k3 = deriv_fn(t_s + 0.5 * h, s2)
    s3 = tuple(s + h * k for s, k in zip(state, k3))

    k4 = deriv_fn(t_s + h, s3)

    state_new = tuple(
        s + (h / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        for s, a, b, c, d in zip(state, k1, k2, k3, k4)
    )
    return (t_s + h, state_new)


# --- Main propagation function ---

def propagate_numerical(
    initial_state: "OrbitalState",
    duration: timedelta,
    step: timedelta,
    force_models: list[ForceModel],
    epoch: datetime | None = None,
) -> NumericalPropagationResult:
    """RK4 integration with summed force model accelerations.

    1. Convert OrbitalState -> Cartesian ECI via kepler_to_cartesian
    2. Build derivative function: sums all force model accelerations
    3. Step through time, recording PropagationStep at each step
    4. Returns NumericalPropagationResult

    Args:
        initial_state: OrbitalState from derive_orbital_state.
        duration: Total propagation duration.
        step: Integration time step.
        force_models: List of force models to sum.
        epoch: Override epoch (defaults to initial_state.reference_epoch).
    """
    # Import here to avoid circular import at module level
    from constellation_generator.domain.propagation import OrbitalState as _OS

    ref_epoch = epoch if epoch is not None else initial_state.reference_epoch
    duration_s = duration.total_seconds()
    step_s = step.total_seconds()

    # Convert to Cartesian ECI
    pos_list, vel_list = kepler_to_cartesian(
        a=initial_state.semi_major_axis_m,
        e=initial_state.eccentricity,
        i_rad=initial_state.inclination_rad,
        omega_big_rad=initial_state.raan_rad,
        omega_small_rad=initial_state.arg_perigee_rad,
        nu_rad=initial_state.true_anomaly_rad,
    )
    pos = (pos_list[0], pos_list[1], pos_list[2])
    vel = (vel_list[0], vel_list[1], vel_list[2])

    # State vector: (x, y, z, vx, vy, vz)
    state_vec: tuple[float, ...] = pos + vel

    # Force model names for result
    model_names = tuple(type(fm).__name__ for fm in force_models)

    def deriv_fn(t_s: float, sv: tuple[float, ...]) -> tuple[float, ...]:
        current_epoch = ref_epoch + timedelta(seconds=t_s)
        p = (sv[0], sv[1], sv[2])
        v = (sv[3], sv[4], sv[5])

        ax_total, ay_total, az_total = 0.0, 0.0, 0.0
        for fm in force_models:
            ax, ay, az = fm.acceleration(current_epoch, p, v)
            ax_total += ax
            ay_total += ay
            az_total += az

        return (v[0], v[1], v[2], ax_total, ay_total, az_total)

    # Collect steps
    steps: list[PropagationStep] = []
    t_current = 0.0
    num_steps = int(duration_s / step_s) + 1

    for i in range(num_steps):
        current_time = ref_epoch + timedelta(seconds=t_current)
        p = (state_vec[0], state_vec[1], state_vec[2])
        v = (state_vec[3], state_vec[4], state_vec[5])
        steps.append(PropagationStep(time=current_time, position_eci=p, velocity_eci=v))

        if i < num_steps - 1:
            t_current, state_vec = rk4_step(t_current, state_vec, step_s, deriv_fn)

    return NumericalPropagationResult(
        steps=tuple(steps),
        epoch=ref_epoch,
        duration_s=duration_s,
        force_model_names=model_names,
    )
