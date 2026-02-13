# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Walker constellation shell generation.

Configurable satellite constellation generation with support for
arbitrary Walker shells and Sun-synchronous orbit bands.
No external dependencies — only stdlib math/dataclasses.
"""
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .orbital_mechanics import kepler_to_cartesian, sso_inclination_deg, OrbitalConstants


@dataclass(frozen=True)
class ShellConfig:
    """Immutable configuration for a constellation shell."""
    altitude_km: float
    inclination_deg: float
    num_planes: int
    sats_per_plane: int
    phase_factor: int
    raan_offset_deg: float
    shell_name: str


@dataclass(frozen=True)
class Satellite:
    """A satellite with ECI state vectors and orbital metadata."""
    name: str
    position_eci: tuple[float, float, float]
    velocity_eci: tuple[float, float, float]
    plane_index: int
    sat_index: int
    raan_deg: float
    true_anomaly_deg: float
    epoch: datetime | None = field(default=None)


def generate_walker_shell(config: ShellConfig) -> list[Satellite]:
    """
    Generate satellite positions/velocities for a Walker constellation shell.

    Args:
        config: Shell configuration parameters.

    Returns:
        List of Satellite objects with ECI positions/velocities.
    """
    r = OrbitalConstants.R_EARTH + config.altitude_km * 1000
    a = r
    e = 0.0
    i_rad = float(np.radians(config.inclination_deg))
    omega_small_rad = 0.0
    total_sats = config.num_planes * config.sats_per_plane

    satellites: list[Satellite] = []

    for plane_idx in range(config.num_planes):
        raan_deg = config.raan_offset_deg + (plane_idx * 360.0 / config.num_planes)
        omega_big_rad = float(np.radians(raan_deg))

        for sat_idx in range(config.sats_per_plane):
            nu_deg = (
                (sat_idx * 360.0 / config.sats_per_plane)
                + (plane_idx * config.phase_factor * 360.0 / total_sats)
            )
            nu_deg = nu_deg % 360.0
            nu_rad = float(np.radians(nu_deg))

            pos, vel = kepler_to_cartesian(
                a=a, e=e, i_rad=i_rad,
                omega_big_rad=omega_big_rad,
                omega_small_rad=omega_small_rad,
                nu_rad=nu_rad,
            )

            name = f"{config.shell_name}-Plane{plane_idx + 1}-Sat{sat_idx + 1}"

            satellites.append(Satellite(
                name=name,
                position_eci=(pos[0], pos[1], pos[2]),
                velocity_eci=(vel[0], vel[1], vel[2]),
                plane_index=plane_idx,
                sat_index=sat_idx,
                raan_deg=raan_deg,
                true_anomaly_deg=nu_deg,
            ))

    return satellites


def generate_sso_band_configs(
    start_alt_km: float,
    end_alt_km: float,
    step_km: float,
    sats_per_plane: int,
    shell_name_prefix: str = "SSO Band",
) -> list[ShellConfig]:
    """
    Generate ShellConfig objects for a Sun-synchronous orbit band.

    Creates single-plane shells at each altitude step with computed
    SSO inclinations.

    Args:
        start_alt_km: Starting altitude (km, inclusive).
        end_alt_km: Ending altitude (km, exclusive).
        step_km: Altitude step (km).
        sats_per_plane: Satellites per plane.
        shell_name_prefix: Prefix for shell names.

    Returns:
        List of ShellConfig objects.
    """
    configs: list[ShellConfig] = []
    alt = start_alt_km

    while alt < end_alt_km:
        inc_deg = sso_inclination_deg(alt)
        configs.append(ShellConfig(
            altitude_km=alt,
            inclination_deg=inc_deg,
            num_planes=1,
            sats_per_plane=sats_per_plane,
            phase_factor=0,
            raan_offset_deg=0.0,
            shell_name=f"{shell_name_prefix} {alt}",
        ))
        alt += step_km

    return configs
