# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Tests for CelesTrak live data fetching and SGP4 propagation."""
import math

import pytest

from constellation_generator.domain.omm import parse_omm_record, OrbitalElements
from constellation_generator.domain.orbital_mechanics import OrbitalConstants


sgp4 = pytest.importorskip("sgp4", reason="sgp4 not installed (pip install constellation-generator[live])")


SAMPLE_OMM = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "EPOCH": "2026-02-11T04:21:12.145248",
    "MEAN_MOTION": 15.4854011,
    "ECCENTRICITY": 0.00110736,
    "INCLINATION": 51.6314,
    "RA_OF_ASC_NODE": 203.3958,
    "ARG_OF_PERICENTER": 86.686,
    "MEAN_ANOMALY": 273.5395,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": 25544,
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 55222,
    "BSTAR": 0.00022024123,
    "MEAN_MOTION_DOT": 0.00011529,
    "MEAN_MOTION_DDOT": 0,
}


# ── OMM parsing (pure domain) ───────────────────────────────────────

class TestOMMParsing:

    def test_parse_fields(self):
        elements = parse_omm_record(SAMPLE_OMM)
        assert isinstance(elements, OrbitalElements)
        assert elements.object_name == "ISS (ZARYA)"
        assert elements.norad_cat_id == 25544
        assert abs(elements.inclination_deg - 51.6314) < 1e-4
        assert abs(elements.eccentricity - 0.00110736) < 1e-8
        assert abs(elements.raan_deg - 203.3958) < 1e-4
        assert abs(elements.mean_motion_rev_per_day - 15.4854011) < 1e-7

    def test_semi_major_axis_from_mean_motion(self):
        elements = parse_omm_record(SAMPLE_OMM)
        a_km = elements.semi_major_axis_m / 1000
        assert 6700 < a_km < 6900  # ISS ~6791 km

    def test_altitude_from_mean_motion(self):
        elements = parse_omm_record(SAMPLE_OMM)
        alt_km = (elements.semi_major_axis_m - OrbitalConstants.R_EARTH) / 1000
        assert 380 < alt_km < 450  # ISS ~420 km

    def test_missing_field_raises(self):
        with pytest.raises((ValueError, KeyError)):
            parse_omm_record({"OBJECT_NAME": "TEST"})

    def test_elements_immutable(self):
        elements = parse_omm_record(SAMPLE_OMM)
        with pytest.raises((AttributeError, TypeError)):
            elements.inclination_deg = 0


# ── SGP4 adapter ────────────────────────────────────────────────────

class TestSGP4Adapter:

    def test_produces_satellite(self):
        from constellation_generator.adapters.celestrak import SGP4Adapter
        sat = SGP4Adapter().omm_to_satellite(SAMPLE_OMM)
        assert sat.name == "ISS (ZARYA)"
        assert len(sat.position_eci) == 3
        assert len(sat.velocity_eci) == 3

    def test_position_plausible(self):
        from constellation_generator.adapters.celestrak import SGP4Adapter
        sat = SGP4Adapter().omm_to_satellite(SAMPLE_OMM)
        r_km = math.sqrt(sum(p**2 for p in sat.position_eci)) / 1000
        assert 6500 < r_km < 7100

    def test_velocity_plausible(self):
        from constellation_generator.adapters.celestrak import SGP4Adapter
        sat = SGP4Adapter().omm_to_satellite(SAMPLE_OMM)
        v_km_s = math.sqrt(sum(v**2 for v in sat.velocity_eci)) / 1000
        assert 7.0 < v_km_s < 8.5


# ── CelesTrak adapter (network) ─────────────────────────────────────

class TestCelesTrakAdapter:

    def test_implements_port(self):
        from constellation_generator.ports.orbital_data import OrbitalDataSource
        from constellation_generator.adapters.celestrak import CelesTrakAdapter
        assert isinstance(CelesTrakAdapter(), OrbitalDataSource)

    def test_fetch_group(self):
        from constellation_generator.adapters.celestrak import CelesTrakAdapter
        records = CelesTrakAdapter().fetch_group("GPS-OPS")
        assert isinstance(records, list)
        assert len(records) > 0
        assert "OBJECT_NAME" in records[0]

    def test_fetch_by_name(self):
        from constellation_generator.adapters.celestrak import CelesTrakAdapter
        records = CelesTrakAdapter().fetch_by_name("ISS (ZARYA)")
        assert len(records) >= 1
        assert records[0]["NORAD_CAT_ID"] == 25544

    def test_fetch_to_satellites(self):
        from constellation_generator.adapters.celestrak import CelesTrakAdapter
        satellites = CelesTrakAdapter().fetch_satellites(group="STATIONS")
        assert len(satellites) > 0
        for sat in satellites[:3]:
            r = math.sqrt(sum(p**2 for p in sat.position_eci))
            assert r > 6_000_000
