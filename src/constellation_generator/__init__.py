"""
Constellation Generator

Generate Walker constellation satellite shells and fetch live orbital data
for orbit simulation tools. Includes J2-corrected propagation, topocentric
observation geometry, access window prediction, and coverage analysis.
"""

from constellation_generator.domain.orbital_mechanics import (
    OrbitalConstants,
    kepler_to_cartesian,
    sso_inclination_deg,
    j2_raan_rate,
    j2_arg_perigee_rate,
    j2_mean_motion_correction,
)
from constellation_generator.domain.constellation import (
    ShellConfig,
    Satellite,
    generate_walker_shell,
    generate_sso_band_configs,
)
from constellation_generator.domain.serialization import (
    format_position,
    format_velocity,
    build_satellite_entity,
)
from constellation_generator.domain.omm import (
    OrbitalElements,
    parse_omm_record,
)
from constellation_generator.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
    geodetic_to_ecef,
)
from constellation_generator.domain.ground_track import (
    GroundTrackPoint,
    compute_ground_track,
)
from constellation_generator.domain.propagation import (
    OrbitalState,
    derive_orbital_state,
    propagate_to,
    propagate_ecef_to,
)
from constellation_generator.domain.observation import (
    GroundStation,
    Observation,
    compute_observation,
)
from constellation_generator.domain.access_windows import (
    AccessWindow,
    compute_access_windows,
)
from constellation_generator.domain.coverage import (
    CoveragePoint,
    compute_coverage_snapshot,
)

__version__ = "1.4.0"

__all__ = [
    "OrbitalConstants",
    "kepler_to_cartesian",
    "sso_inclination_deg",
    "j2_raan_rate",
    "j2_arg_perigee_rate",
    "j2_mean_motion_correction",
    "ShellConfig",
    "Satellite",
    "generate_walker_shell",
    "generate_sso_band_configs",
    "format_position",
    "format_velocity",
    "build_satellite_entity",
    "OrbitalElements",
    "parse_omm_record",
    "gmst_rad",
    "eci_to_ecef",
    "ecef_to_geodetic",
    "geodetic_to_ecef",
    "GroundTrackPoint",
    "compute_ground_track",
    "OrbitalState",
    "derive_orbital_state",
    "propagate_to",
    "propagate_ecef_to",
    "GroundStation",
    "Observation",
    "compute_observation",
    "AccessWindow",
    "compute_access_windows",
    "CoveragePoint",
    "compute_coverage_snapshot",
]
