# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
CCSDS OPM/OEM KVN parser.

Parses Orbit Parameter Message (OPM) and Orbit Ephemeris Message (OEM)
files in CCSDS Keyword-Value Notation (KVN) format into domain objects.

Reference: CCSDS 502.0-B-3 (Orbit Data Messages).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from humeris.domain.ccsds_contracts import CcsdsValidationError
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.propagation import OrbitalState


@dataclass(frozen=True)
class CcsdsOrbitData:
    """Parsed CCSDS orbit data with metadata."""
    object_name: str
    object_id: str
    center_name: str
    ref_frame: str
    time_system: str
    states: list[OrbitalState]


_STANDALONE_KEYWORDS = {"META_START", "META_STOP", "COVARIANCE_START", "COVARIANCE_STOP",
                        "DATA_START", "DATA_STOP"}


def _parse_kvn_pairs(text: str) -> list[tuple[str, str]]:
    """Parse KVN keyword = value pairs from text, preserving order."""
    pairs: list[tuple[str, str]] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("COMMENT"):
            continue
        if line in _STANDALONE_KEYWORDS:
            pairs.append((line, ""))
        elif "=" in line:
            key, _, val = line.partition("=")
            pairs.append((key.strip(), val.strip()))
        else:
            # Data line (ephemeris record) — store with special key
            pairs.append(("_DATA_LINE", line))
    return pairs


def _parse_epoch(epoch_str: str) -> datetime:
    """Parse CCSDS epoch string to datetime."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(epoch_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise CcsdsValidationError(f"Cannot parse epoch: {epoch_str}")


def _cartesian_to_orbital_state(
    x_km: float, y_km: float, z_km: float,
    vx_kms: float, vy_kms: float, vz_kms: float,
    epoch: datetime,
) -> OrbitalState:
    """Convert Cartesian state vector (km, km/s) to OrbitalState.

    Converts km -> m, km/s -> m/s, then derives Keplerian elements.
    """
    mu = OrbitalConstants.MU_EARTH

    # Convert to meters
    pos = np.array([x_km * 1000.0, y_km * 1000.0, z_km * 1000.0])
    vel = np.array([vx_kms * 1000.0, vy_kms * 1000.0, vz_kms * 1000.0])

    r_mag = float(np.linalg.norm(pos))
    v_mag = float(np.linalg.norm(vel))

    if r_mag < 1.0:
        raise CcsdsValidationError("Position vector magnitude too small")
    if v_mag < 1.0:
        raise CcsdsValidationError("Velocity vector magnitude too small")

    # Specific angular momentum
    h_vec = np.cross(pos, vel)
    h_mag = float(np.linalg.norm(h_vec))

    # Node vector
    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n_mag = float(np.linalg.norm(n_vec))

    # Eccentricity vector
    e_vec = ((v_mag**2 - mu / r_mag) * pos - np.dot(pos, vel) * vel) / mu
    ecc = float(np.linalg.norm(e_vec))

    # Semi-major axis
    energy = v_mag**2 / 2.0 - mu / r_mag
    if abs(energy) < 1e-10:
        # Parabolic — use r_mag as approximation
        a = r_mag
    else:
        a = -mu / (2.0 * energy)

    # Inclination
    inc = float(np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))) if h_mag > 0 else 0.0

    # RAAN
    if n_mag > 1e-10:
        raan = float(np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0)))
        if n_vec[1] < 0:
            raan = 2.0 * math.pi - raan
    else:
        raan = 0.0

    # Argument of perigee
    if n_mag > 1e-10 and ecc > 1e-10:
        arg_p = float(np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * ecc), -1.0, 1.0)))
        if e_vec[2] < 0:
            arg_p = 2.0 * math.pi - arg_p
    else:
        arg_p = 0.0

    # True anomaly
    if ecc > 1e-10:
        nu = float(np.arccos(np.clip(np.dot(e_vec, pos) / (ecc * r_mag), -1.0, 1.0)))
        if np.dot(pos, vel) < 0:
            nu = 2.0 * math.pi - nu
    else:
        # Circular orbit — use argument of latitude
        if n_mag > 1e-10:
            nu = float(np.arccos(np.clip(np.dot(n_vec, pos) / (n_mag * r_mag), -1.0, 1.0)))
            if pos[2] < 0:
                nu = 2.0 * math.pi - nu
        else:
            nu = 0.0

    # Mean motion
    if a > 0:
        n_mean = math.sqrt(mu / abs(a)**3)
    else:
        n_mean = math.sqrt(mu / r_mag**3)

    return OrbitalState(
        semi_major_axis_m=abs(a),
        eccentricity=ecc,
        inclination_rad=inc,
        raan_rad=raan,
        arg_perigee_rad=arg_p,
        true_anomaly_rad=nu,
        mean_motion_rad_s=n_mean,
        reference_epoch=epoch,
    )


def parse_opm(path: str) -> CcsdsOrbitData:
    """Parse a CCSDS OPM file in KVN format.

    Args:
        path: Path to .opm file.

    Returns:
        CcsdsOrbitData with a single state.

    Raises:
        CcsdsValidationError: If the file is malformed or missing required fields.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    pairs = _parse_kvn_pairs(text)

    # Check for duplicate required keys
    required = {
        "OBJECT_NAME", "OBJECT_ID", "CENTER_NAME", "REF_FRAME",
        "TIME_SYSTEM", "EPOCH", "X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT",
    }
    seen: dict[str, int] = {}
    for key, _ in pairs:
        if key in required:
            seen[key] = seen.get(key, 0) + 1
    duplicates = sorted(k for k, v in seen.items() if v > 1)
    if duplicates:
        raise CcsdsValidationError(
            f"Duplicate required keys in OPM: {', '.join(duplicates)}"
        )

    kvn = dict(pairs)

    # Validate required fields
    missing = sorted(f for f in required if f not in kvn)
    if missing:
        raise CcsdsValidationError(
            f"OPM missing required fields: {', '.join(missing)}"
        )

    epoch = _parse_epoch(kvn["EPOCH"])

    state = _cartesian_to_orbital_state(
        x_km=float(kvn["X"]),
        y_km=float(kvn["Y"]),
        z_km=float(kvn["Z"]),
        vx_kms=float(kvn["X_DOT"]),
        vy_kms=float(kvn["Y_DOT"]),
        vz_kms=float(kvn["Z_DOT"]),
        epoch=epoch,
    )

    return CcsdsOrbitData(
        object_name=kvn["OBJECT_NAME"],
        object_id=kvn["OBJECT_ID"],
        center_name=kvn.get("CENTER_NAME", "EARTH"),
        ref_frame=kvn.get("REF_FRAME", "EME2000"),
        time_system=kvn.get("TIME_SYSTEM", "UTC"),
        states=[state],
    )


def parse_oem(path: str) -> CcsdsOrbitData:
    """Parse a CCSDS OEM file in KVN format.

    Supports single and multi-segment files.

    Args:
        path: Path to .oem file.

    Returns:
        CcsdsOrbitData with states from all segments.

    Raises:
        CcsdsValidationError: If the file is malformed.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    pairs = _parse_kvn_pairs(text)

    # Collect segments
    segments: list[tuple[dict[str, str], list[str]]] = []
    meta = {}
    data_lines = []
    in_meta = False

    for key, val in pairs:
        if key == "META_START":
            # Save previous segment if it has data
            if meta and data_lines:
                segments.append((dict(meta), list(data_lines)))
            in_meta = True
            meta = {}
            data_lines = []
        elif key == "META_STOP":
            in_meta = False
        elif in_meta:
            meta[key] = val
        elif key == "_DATA_LINE":
            data_lines.append(val)

    # Save last segment
    if meta and data_lines:
        segments.append((dict(meta), list(data_lines)))

    if not segments:
        raise CcsdsValidationError("OEM file contains no data segments")

    # Use first segment for metadata
    first_meta = segments[0][0]
    object_name = first_meta.get("OBJECT_NAME", "UNKNOWN")
    object_id = first_meta.get("OBJECT_ID", "UNKNOWN")

    # Parse all data lines into states
    all_states: list[OrbitalState] = []
    for seg_meta, seg_data in segments:
        for line in seg_data:
            parts = line.split()
            if len(parts) < 7:
                continue
            epoch = _parse_epoch(parts[0])
            try:
                values = [float(parts[i]) for i in range(1, 7)]
            except ValueError as e:
                raise CcsdsValidationError(
                    f"Non-numeric value in OEM data line: {e}"
                ) from e
            for i, v in enumerate(values):
                if math.isnan(v) or math.isinf(v):
                    raise CcsdsValidationError(
                        f"NaN or Inf in OEM data line column {i + 1}"
                    )
            state = _cartesian_to_orbital_state(
                x_km=values[0], y_km=values[1], z_km=values[2],
                vx_kms=values[3], vy_kms=values[4], vz_kms=values[5],
                epoch=epoch,
            )
            all_states.append(state)

    if not all_states:
        raise CcsdsValidationError("OEM file contains no valid ephemeris data")

    return CcsdsOrbitData(
        object_name=object_name,
        object_id=object_id,
        center_name=first_meta.get("CENTER_NAME", "EARTH"),
        ref_frame=first_meta.get("REF_FRAME", "EME2000"),
        time_system=first_meta.get("TIME_SYSTEM", "UTC"),
        states=all_states,
    )
