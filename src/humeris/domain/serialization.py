# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Simulation-compatible serialization.

Pure formatting functions for converting ECI state vectors to the
semicolon-delimited string format with Y/Z axis swap.
No external dependencies.
"""

import numpy as np


def format_position(pos_eci: list[float] | tuple[float, float, float]) -> str:
    """
    Format ECI position to simulation string with Y/Z axis swap.

    ECI [x, y, z] → Simulation "{x:.3f};{z:.3f};{y:.3f}"

    Args:
        pos_eci: Position in ECI frame [x, y, z] (meters).

    Returns:
        Semicolon-delimited position string, 3 decimal places.
    """
    x, y, z = pos_eci[0], pos_eci[1], pos_eci[2]
    return f"{x:.3f};{z:.3f};{y:.3f}"


def format_velocity(vel_eci: list[float] | tuple[float, float, float]) -> str:
    """
    Format ECI velocity to simulation string with Y/Z axis swap.

    ECI [vx, vy, vz] → Simulation "{vx:.6f};{vz:.6f};{vy:.6f}"

    Args:
        vel_eci: Velocity in ECI frame [vx, vy, vz] (m/s).

    Returns:
        Semicolon-delimited velocity string, 6 decimal places.
    """
    vx, vy, vz = vel_eci[0], vel_eci[1], vel_eci[2]
    return f"{vx:.6f};{vz:.6f};{vy:.6f}"


def build_satellite_entity(
    satellite,
    template: dict,
    base_id: int,
) -> dict:
    """
    Build a simulation entity dict from a Satellite domain object.

    Deep-copies the template and fills in satellite-specific fields.
    Does NOT mutate the template.

    Args:
        satellite: Satellite with name, position_eci, velocity_eci.
        template: Template entity dict (not mutated).
        base_id: Entity ID to assign.

    Returns:
        New entity dict ready for simulation JSON.
    """
    entity = _deep_copy_dict(template)
    entity['Id'] = base_id
    entity['Name'] = satellite.name
    entity['Position'] = format_position(satellite.position_eci)
    entity['Velocity'] = format_velocity(satellite.velocity_eci)
    return entity


def _deep_copy_dict(d: dict) -> dict:
    """Deep copy a dict without json module (domain purity)."""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _deep_copy_dict(value)
        elif isinstance(value, list):
            result[key] = _deep_copy_list(value)
        else:
            result[key] = value
    return result


def _deep_copy_list(lst: list) -> list:
    """Deep copy a list without json module (domain purity)."""
    result = []
    for item in lst:
        if isinstance(item, dict):
            result.append(_deep_copy_dict(item))
        elif isinstance(item, list):
            result.append(_deep_copy_list(item))
        else:
            result.append(item)
    return result
