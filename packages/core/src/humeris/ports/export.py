# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Port interface for satellite data export.

Adapters implement this to export satellite positions in various formats
(CSV, GeoJSON, etc.).
"""
from datetime import datetime
from typing import Protocol, runtime_checkable

from humeris.domain.constellation import Satellite


@runtime_checkable
class SatelliteExporter(Protocol):
    """Port for exporting satellite data to file."""

    def export(
        self,
        satellites: list[Satellite],
        path: str,
        epoch: datetime | None = None,
    ) -> int:
        """
        Export satellite positions to a file.

        Converts ECI positions to geodetic coordinates for export.
        Uses each satellite's epoch for GMST computation; falls back
        to the epoch parameter, then to J2000.0.

        Args:
            satellites: List of Satellite domain objects.
            path: Output file path.
            epoch: Fallback epoch for GMST computation when
                satellite.epoch is None.

        Returns:
            Number of satellites exported.
        """
        ...
