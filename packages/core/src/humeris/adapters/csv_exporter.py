# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
CSV satellite exporter.

Exports satellite positions as CSV with geodetic coordinates.
External dependencies (csv, file I/O) are confined to this adapter.
"""
import csv
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from humeris.ports.export import SatelliteExporter
from humeris.domain.constellation import Satellite
from humeris.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
)
from humeris.adapters.enrichment import compute_satellite_enrichment


_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

_HEADER = [
    'name', 'lat_deg', 'lon_deg', 'alt_km', 'epoch',
    'plane_index', 'sat_index', 'raan_deg', 'true_anomaly_deg',
    'altitude_km', 'inclination_deg', 'orbital_period_min',
    'beta_angle_deg', 'atmospheric_density_kg_m3', 'l_shell',
]


class CsvSatelliteExporter(SatelliteExporter):
    """Exports satellite positions to CSV with geodetic coordinates."""

    def export(
        self,
        satellites: list[Satellite],
        path: str,
        epoch: datetime | None = None,
    ) -> int:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(_HEADER)

            _warned_j2000 = False
            for sat in satellites:
                sat_epoch = sat.epoch or epoch or _J2000
                if sat_epoch is _J2000 and not _warned_j2000:
                    logger.warning(
                        "No epoch provided, defaulting to J2000 (2000-01-01T12:00:00Z)"
                    )
                    _warned_j2000 = True
                gmst_angle = gmst_rad(sat_epoch)
                pos_ecef, _ = eci_to_ecef(
                    sat.position_eci, sat.velocity_eci, gmst_angle,
                )
                lat_deg, lon_deg, alt_m = ecef_to_geodetic(pos_ecef)

                epoch_str = sat_epoch.isoformat()

                enrich = compute_satellite_enrichment(sat, sat_epoch)

                writer.writerow([
                    sat.name,
                    f'{lat_deg:.6f}',
                    f'{lon_deg:.6f}',
                    f'{alt_m / 1000.0:.3f}',
                    epoch_str,
                    sat.plane_index,
                    sat.sat_index,
                    f'{sat.raan_deg:.6f}',
                    f'{sat.true_anomaly_deg:.6f}',
                    f'{enrich.altitude_km:.3f}',
                    f'{enrich.inclination_deg:.4f}',
                    f'{enrich.orbital_period_min:.4f}',
                    f'{enrich.beta_angle_deg:.4f}',
                    f'{enrich.atmospheric_density_kg_m3:.6e}',
                    f'{enrich.l_shell:.4f}',
                ])

        return len(satellites)
