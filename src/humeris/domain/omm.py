# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
OMM (Orbit Mean-Elements Message) record parsing.

Converts CelesTrak JSON OMM records into domain objects and derives
semi-major axis from mean motion via Kepler's third law.
No external dependencies — only stdlib math/dataclasses.
"""
import math
from dataclasses import dataclass

import numpy as np

from .orbital_mechanics import OrbitalConstants


@dataclass(frozen=True)
class OrbitalElements:
    """Parsed orbital elements from an OMM record."""
    object_name: str
    object_id: str
    norad_cat_id: int
    epoch: str
    mean_motion_rev_per_day: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    mean_anomaly_deg: float
    bstar: float
    mean_motion_dot: float
    mean_motion_ddot: float
    classification_type: str
    element_set_no: int
    rev_at_epoch: int
    ephemeris_type: int
    semi_major_axis_m: float


def parse_omm_record(record: dict) -> OrbitalElements:
    """
    Parse a CelesTrak OMM JSON record into an OrbitalElements object.

    Derives semi-major axis from mean motion:
        n (rad/s) = mean_motion × 2π / 86400
        a = (μ / n²)^(1/3)

    Args:
        record: Dict from CelesTrak JSON API with OMM fields.

    Returns:
        OrbitalElements domain object.

    Raises:
        KeyError: If a required field is missing.
        ValueError: If mean motion is non-positive.
    """
    mean_motion_rpd = record["MEAN_MOTION"]
    if mean_motion_rpd <= 0:
        raise ValueError(f"Mean motion must be positive, got {mean_motion_rpd}")

    n_rad_s = mean_motion_rpd * 2.0 * np.pi / 86400.0
    a_m = (OrbitalConstants.MU_EARTH / (n_rad_s ** 2)) ** (1.0 / 3.0)

    return OrbitalElements(
        object_name=record["OBJECT_NAME"],
        object_id=record["OBJECT_ID"],
        norad_cat_id=record["NORAD_CAT_ID"],
        epoch=record["EPOCH"],
        mean_motion_rev_per_day=mean_motion_rpd,
        eccentricity=record["ECCENTRICITY"],
        inclination_deg=record["INCLINATION"],
        raan_deg=record["RA_OF_ASC_NODE"],
        arg_perigee_deg=record["ARG_OF_PERICENTER"],
        mean_anomaly_deg=record["MEAN_ANOMALY"],
        bstar=record["BSTAR"],
        mean_motion_dot=record["MEAN_MOTION_DOT"],
        mean_motion_ddot=record["MEAN_MOTION_DDOT"],
        classification_type=record.get("CLASSIFICATION_TYPE", "U"),
        element_set_no=record.get("ELEMENT_SET_NO", 0),
        rev_at_epoch=record.get("REV_AT_EPOCH", 0),
        ephemeris_type=record.get("EPHEMERIS_TYPE", 0),
        semi_major_axis_m=a_m,
    )
