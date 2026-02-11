"""
GeoJSON satellite exporter.

Exports satellite positions as a GeoJSON FeatureCollection with
Point geometries. Coordinates follow the GeoJSON spec: [lon, lat, alt].
External dependencies (json, file I/O) are confined to this adapter.
"""
import json
from datetime import datetime, timezone

from constellation_generator.ports.export import SatelliteExporter
from constellation_generator.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
)


_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class GeoJsonSatelliteExporter(SatelliteExporter):
    """Exports satellite positions as GeoJSON FeatureCollection."""

    def export(
        self,
        satellites: list,
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
