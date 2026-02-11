"""Tests for topocentric observation geometry (az/el/range)."""
import ast
import math

import pytest

from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.coordinate_frames import geodetic_to_ecef


# ── Helpers ──────────────────────────────────────────────────────────

_R_EQ = OrbitalConstants.R_EARTH_EQUATORIAL


def _overhead_ecef(lat_deg, lon_deg, altitude_m):
    """ECEF position directly above a ground point at given altitude."""
    return geodetic_to_ecef(lat_deg, lon_deg, altitude_m)


# ── Dataclass immutability ───────────────────────────────────────────

class TestGroundStation:

    def test_frozen(self):
        from constellation_generator.domain.observation import GroundStation

        station = GroundStation(name='Test', lat_deg=0.0, lon_deg=0.0)
        with pytest.raises(AttributeError):
            station.lat_deg = 10.0

    def test_alt_default_zero(self):
        from constellation_generator.domain.observation import GroundStation

        station = GroundStation(name='Test', lat_deg=52.0, lon_deg=4.4)
        assert station.alt_m == 0.0

    def test_fields(self):
        from constellation_generator.domain.observation import GroundStation

        station = GroundStation(name='Delft', lat_deg=52.0, lon_deg=4.4, alt_m=10.0)
        assert station.name == 'Delft'
        assert station.lat_deg == 52.0
        assert station.lon_deg == 4.4
        assert station.alt_m == 10.0


class TestObservation:

    def test_frozen(self):
        from constellation_generator.domain.observation import Observation

        obs = Observation(azimuth_deg=0.0, elevation_deg=45.0, slant_range_m=1000.0)
        with pytest.raises(AttributeError):
            obs.azimuth_deg = 90.0


# ── Observation geometry ─────────────────────────────────────────────

class TestComputeObservation:

    def test_satellite_directly_overhead(self):
        """Satellite directly above station → el≈90°, range≈altitude."""
        from constellation_generator.domain.observation import GroundStation, compute_observation

        station = GroundStation(name='Eq', lat_deg=0.0, lon_deg=0.0, alt_m=0.0)
        sat_alt = 500_000.0
        sat_ecef = geodetic_to_ecef(0.0, 0.0, sat_alt)
        obs = compute_observation(station, sat_ecef)
        assert abs(obs.elevation_deg - 90.0) < 1.0
        assert abs(obs.slant_range_m - sat_alt) < 100.0

    def test_cardinal_north(self):
        """Satellite to the north → azimuth ≈ 0° (or 360°)."""
        from constellation_generator.domain.observation import GroundStation, compute_observation

        station = GroundStation(name='Eq', lat_deg=0.0, lon_deg=0.0, alt_m=0.0)
        sat_ecef = geodetic_to_ecef(10.0, 0.0, 500_000.0)
        obs = compute_observation(station, sat_ecef)
        # azimuth near 0 or 360
        az = obs.azimuth_deg
        assert az < 10.0 or az > 350.0

    def test_cardinal_east(self):
        """Satellite to the east → azimuth ≈ 90°."""
        from constellation_generator.domain.observation import GroundStation, compute_observation

        station = GroundStation(name='Eq', lat_deg=0.0, lon_deg=0.0, alt_m=0.0)
        sat_ecef = geodetic_to_ecef(0.0, 10.0, 500_000.0)
        obs = compute_observation(station, sat_ecef)
        assert abs(obs.azimuth_deg - 90.0) < 10.0

    def test_cardinal_south(self):
        """Satellite to the south → azimuth ≈ 180°."""
        from constellation_generator.domain.observation import GroundStation, compute_observation

        station = GroundStation(name='Eq', lat_deg=0.0, lon_deg=0.0, alt_m=0.0)
        sat_ecef = geodetic_to_ecef(-10.0, 0.0, 500_000.0)
        obs = compute_observation(station, sat_ecef)
        assert abs(obs.azimuth_deg - 180.0) < 10.0

    def test_cardinal_west(self):
        """Satellite to the west → azimuth ≈ 270°."""
        from constellation_generator.domain.observation import GroundStation, compute_observation

        station = GroundStation(name='Eq', lat_deg=0.0, lon_deg=0.0, alt_m=0.0)
        sat_ecef = geodetic_to_ecef(0.0, -10.0, 500_000.0)
        obs = compute_observation(station, sat_ecef)
        assert abs(obs.azimuth_deg - 270.0) < 10.0

    def test_below_horizon_negative_elevation(self):
        """Satellite on opposite side of Earth → negative elevation."""
        from constellation_generator.domain.observation import GroundStation, compute_observation

        station = GroundStation(name='NP', lat_deg=89.0, lon_deg=0.0, alt_m=0.0)
        sat_ecef = geodetic_to_ecef(-89.0, 0.0, 500_000.0)
        obs = compute_observation(station, sat_ecef)
        assert obs.elevation_deg < 0.0

    def test_slant_range_plausible_leo(self):
        """For LEO satellite, slant range should be plausible (< 3000 km for visible pass)."""
        from constellation_generator.domain.observation import GroundStation, compute_observation

        station = GroundStation(name='Eq', lat_deg=0.0, lon_deg=0.0, alt_m=0.0)
        sat_ecef = geodetic_to_ecef(5.0, 5.0, 500_000.0)
        obs = compute_observation(station, sat_ecef)
        assert 400_000 < obs.slant_range_m < 3_000_000

    def test_azimuth_range(self):
        """Azimuth always in [0, 360)."""
        from constellation_generator.domain.observation import GroundStation, compute_observation

        station = GroundStation(name='Mid', lat_deg=45.0, lon_deg=10.0, alt_m=0.0)
        for lat, lon in [(50, 10), (45, 20), (40, 10), (45, 0), (50, 20), (40, 0)]:
            sat_ecef = geodetic_to_ecef(lat, lon, 500_000.0)
            obs = compute_observation(station, sat_ecef)
            assert 0.0 <= obs.azimuth_deg < 360.0

    def test_elevation_range(self):
        """Elevation always in [-90, 90]."""
        from constellation_generator.domain.observation import GroundStation, compute_observation

        station = GroundStation(name='Mid', lat_deg=45.0, lon_deg=10.0, alt_m=0.0)
        for lat, lon in [(45, 10), (90, 0), (-45, 180)]:
            sat_ecef = geodetic_to_ecef(lat, lon, 500_000.0)
            obs = compute_observation(station, sat_ecef)
            assert -90.0 <= obs.elevation_deg <= 90.0


# ── Domain purity ────────────────────────────────────────────────────

class TestObservationPurity:

    def test_observation_imports_only_stdlib_and_domain(self):
        import constellation_generator.domain.observation as mod

        allowed = {'math', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in allowed and not root.startswith('constellation_generator'):
                        assert False, f"Disallowed import '{alias.name}'"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    root = node.module.split('.')[0]
                    if root not in allowed and root != 'constellation_generator':
                        assert False, f"Disallowed import from '{node.module}'"
