"""Tests for concurrent CelesTrak adapter."""
import math
from unittest.mock import patch, MagicMock

import pytest

sgp4 = pytest.importorskip("sgp4", reason="sgp4 not installed (pip install constellation-generator[live])")

from constellation_generator.domain.constellation import Satellite


# Use the SAMPLE_OMM from test_live_data for consistency
SAMPLE_OMM_RECORDS = [
    {
        "OBJECT_NAME": f"SAT-{i}",
        "OBJECT_ID": f"2026-001{chr(65+i)}",
        "EPOCH": "2026-02-11T04:21:12.145248",
        "MEAN_MOTION": 15.4854011,
        "ECCENTRICITY": 0.00110736,
        "INCLINATION": 51.6314,
        "RA_OF_ASC_NODE": 203.3958,
        "ARG_OF_PERICENTER": 86.686,
        "MEAN_ANOMALY": 273.5395 + i * 10,
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": "U",
        "NORAD_CAT_ID": 90000 + i,
        "ELEMENT_SET_NO": 999,
        "REV_AT_EPOCH": 100,
        "BSTAR": 0.00022024123,
        "MEAN_MOTION_DOT": 0.00011529,
        "MEAN_MOTION_DDOT": 0,
    }
    for i in range(20)
]


# ── ConcurrentCelesTrakAdapter ────────────────────────────────────

class TestConcurrentCelesTrakAdapter:

    def test_implements_orbital_data_source(self):
        """Must implement OrbitalDataSource port."""
        from constellation_generator.ports.orbital_data import OrbitalDataSource
        from constellation_generator.adapters.concurrent_celestrak import (
            ConcurrentCelesTrakAdapter,
        )
        adapter = ConcurrentCelesTrakAdapter()
        assert isinstance(adapter, OrbitalDataSource)

    def test_configurable_max_workers(self):
        """max_workers parameter is accepted and stored."""
        from constellation_generator.adapters.concurrent_celestrak import (
            ConcurrentCelesTrakAdapter,
        )
        adapter = ConcurrentCelesTrakAdapter(max_workers=8)
        assert adapter._max_workers == 8

    def test_default_max_workers(self):
        """Default max_workers is sensible (> 1)."""
        from constellation_generator.adapters.concurrent_celestrak import (
            ConcurrentCelesTrakAdapter,
        )
        adapter = ConcurrentCelesTrakAdapter()
        assert adapter._max_workers >= 2

    def test_fetch_satellites_returns_satellite_list(self):
        """fetch_satellites returns list of Satellite domain objects."""
        from constellation_generator.adapters.concurrent_celestrak import (
            ConcurrentCelesTrakAdapter,
        )
        adapter = ConcurrentCelesTrakAdapter()

        with patch.object(adapter, '_fetch_json', return_value=SAMPLE_OMM_RECORDS):
            satellites = adapter.fetch_satellites(group="TEST")

        assert isinstance(satellites, list)
        assert len(satellites) > 0
        assert all(isinstance(s, Satellite) for s in satellites)

    def test_concurrent_results_match_sync(self):
        """Concurrent adapter produces same satellites as sync adapter."""
        from constellation_generator.adapters.celestrak import CelesTrakAdapter
        from constellation_generator.adapters.concurrent_celestrak import (
            ConcurrentCelesTrakAdapter,
        )
        sync_adapter = CelesTrakAdapter()
        concurrent_adapter = ConcurrentCelesTrakAdapter()

        with patch.object(sync_adapter, '_fetch_json', return_value=SAMPLE_OMM_RECORDS), \
             patch.object(concurrent_adapter, '_fetch_json', return_value=SAMPLE_OMM_RECORDS):
            sync_sats = sync_adapter.fetch_satellites(group="TEST")
            concurrent_sats = concurrent_adapter.fetch_satellites(group="TEST")

        sync_names = sorted(s.name for s in sync_sats)
        concurrent_names = sorted(s.name for s in concurrent_sats)
        assert sync_names == concurrent_names

        # Position magnitude equivalence
        for s_sync in sync_sats:
            matching = [s for s in concurrent_sats if s.name == s_sync.name]
            assert len(matching) == 1
            r_sync = math.sqrt(sum(p**2 for p in s_sync.position_eci))
            r_conc = math.sqrt(sum(p**2 for p in matching[0].position_eci))
            assert abs(r_sync - r_conc) < 1.0

    def test_graceful_sgp4_failure(self):
        """Records that fail SGP4 are skipped, others still returned."""
        from constellation_generator.adapters.concurrent_celestrak import (
            ConcurrentCelesTrakAdapter,
        )
        bad_record = dict(SAMPLE_OMM_RECORDS[0])
        bad_record["MEAN_MOTION"] = -1  # will fail
        records = [bad_record] + SAMPLE_OMM_RECORDS[1:5]

        adapter = ConcurrentCelesTrakAdapter()
        with patch.object(adapter, '_fetch_json', return_value=records):
            satellites = adapter.fetch_satellites(group="TEST")

        # 4 valid records minus the bad one = at least 3
        assert len(satellites) >= 3

    def test_single_http_request_per_call(self):
        """Only 1 HTTP request per fetch_satellites call (concurrency = SGP4 only)."""
        from constellation_generator.adapters.concurrent_celestrak import (
            ConcurrentCelesTrakAdapter,
        )
        adapter = ConcurrentCelesTrakAdapter()

        with patch.object(adapter, '_fetch_json', return_value=SAMPLE_OMM_RECORDS) as mock_fetch:
            adapter.fetch_satellites(group="TEST")

        assert mock_fetch.call_count == 1

    def test_fetch_group_delegates_correctly(self):
        """fetch_group returns raw OMM records (same as sync)."""
        from constellation_generator.adapters.concurrent_celestrak import (
            ConcurrentCelesTrakAdapter,
        )
        adapter = ConcurrentCelesTrakAdapter()
        with patch.object(adapter, '_fetch_json', return_value=SAMPLE_OMM_RECORDS):
            records = adapter.fetch_group("TEST")
        assert records == SAMPLE_OMM_RECORDS

    def test_fetch_by_name_delegates(self):
        """fetch_by_name returns raw OMM records."""
        from constellation_generator.adapters.concurrent_celestrak import (
            ConcurrentCelesTrakAdapter,
        )
        adapter = ConcurrentCelesTrakAdapter()
        with patch.object(adapter, '_fetch_json', return_value=SAMPLE_OMM_RECORDS[:1]):
            records = adapter.fetch_by_name("SAT-0")
        assert len(records) == 1

    def test_fetch_by_catnr_delegates(self):
        """fetch_by_catnr returns raw OMM records."""
        from constellation_generator.adapters.concurrent_celestrak import (
            ConcurrentCelesTrakAdapter,
        )
        adapter = ConcurrentCelesTrakAdapter()
        with patch.object(adapter, '_fetch_json', return_value=SAMPLE_OMM_RECORDS[:1]):
            records = adapter.fetch_by_catnr(90000)
        assert len(records) == 1


# ── SGP4Adapter epoch population ──────────────────────────────────

class TestSGP4AdapterEpoch:

    def test_satellite_has_epoch_from_omm(self):
        """SGP4Adapter populates Satellite.epoch from OMM record."""
        from constellation_generator.adapters.celestrak import SGP4Adapter
        sat = SGP4Adapter().omm_to_satellite(SAMPLE_OMM_RECORDS[0])
        assert sat.epoch is not None

    def test_satellite_epoch_matches_omm_epoch(self):
        """Satellite.epoch matches the OMM EPOCH field."""
        from datetime import datetime
        from constellation_generator.adapters.celestrak import SGP4Adapter
        sat = SGP4Adapter().omm_to_satellite(SAMPLE_OMM_RECORDS[0])
        expected = datetime.fromisoformat(SAMPLE_OMM_RECORDS[0]["EPOCH"])
        assert sat.epoch == expected
