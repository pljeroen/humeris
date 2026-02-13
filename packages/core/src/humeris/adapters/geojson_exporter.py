# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
GeoJSON satellite exporter.

Exports satellite positions as a GeoJSON FeatureCollection with
Point geometries. Coordinates follow the GeoJSON spec: [lon, lat, alt].
External dependencies (json, file I/O) are confined to this adapter.
"""
import json
from datetime import datetime, timezone

from humeris.ports.export import SatelliteExporter
from humeris.domain.constellation import Satellite
from humeris.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
)
from humeris.adapters.enrichment import compute_satellite_enrichment


_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class GeoJsonSatelliteExporter(SatelliteExporter):
    """Exports satellite positions as GeoJSON FeatureCollection."""

    def export(
        self,
        satellites: list[Satellite],
        path: str,
        epoch: datetime | None = None,
    ) -> int:
        features = []

        for sat in satellites:
            sat_epoch = sat.epoch or epoch or _J2000
            gmst_angle = gmst_rad(sat_epoch)
            pos_ecef, _ = eci_to_ecef(
                sat.position_eci, sat.velocity_eci, gmst_angle,
            )
            lat_deg, lon_deg, alt_m = ecef_to_geodetic(pos_ecef)

            enrich = compute_satellite_enrichment(sat, sat_epoch)

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [
                        round(lon_deg, 6),
                        round(lat_deg, 6),
                        round(alt_m / 1000.0, 3),
                    ],
                },
                'properties': {
                    'name': sat.name,
                    'epoch': sat_epoch.isoformat(),
                    'plane_index': sat.plane_index,
                    'sat_index': sat.sat_index,
                    'raan_deg': sat.raan_deg,
                    'true_anomaly_deg': sat.true_anomaly_deg,
                    'altitude_km': round(enrich.altitude_km, 3),
                    'inclination_deg': round(enrich.inclination_deg, 4),
                    'orbital_period_min': round(enrich.orbital_period_min, 4),
                    'beta_angle_deg': round(enrich.beta_angle_deg, 4),
                    'atmospheric_density_kg_m3': enrich.atmospheric_density_kg_m3,
                    'l_shell': round(enrich.l_shell, 4),
                },
            }
            features.append(feature)

        collection = {
            'type': 'FeatureCollection',
            'features': features,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(collection, f, indent=2, ensure_ascii=False)

        return len(satellites)
