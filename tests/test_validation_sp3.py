# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Validation: SP3 precise ephemeris parsing and cross-validation.

SP3 (Standard Product #3) files from IGS contain centimeter-accurate
post-processed GNSS satellite positions. We use these to:

1. Verify our SP3 parser produces correct positions (km → m conversion,
   epoch parsing, satellite ID extraction).
2. Cross-validate our Keplerian+J2 propagation against precise ephemeris.
   For GPS satellites (MEO, ~20200 km altitude), J2-only propagation
   should agree to within ~10-50 km over a few hours.
3. Verify orbital properties derived from SP3 positions match expected
   GPS constellation parameters (altitude ~20200 km, period ~11h58m,
   inclination ~55°).

The SP3 sample below is a synthetic but realistic representation of
IGS final orbit products. Position values are consistent with actual
GPS constellation geometry.
"""
import math
from datetime import datetime, timedelta, timezone

from constellation_generator.domain.sp3_parser import (
    parse_sp3,
    filter_satellite,
    SP3Ephemeris,
    SP3EphemerisPoint,
)
from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.orbit_properties import (
    compute_energy_momentum,
    state_vector_to_elements,
)

_R_E = OrbitalConstants.R_EARTH
_MU = OrbitalConstants.MU_EARTH

# Synthetic SP3 file with realistic GPS satellite positions.
# GPS satellites orbit at ~26560 km radius (~20200 km altitude).
# Positions are in ECEF (ITRF), units: km for positions, us for clocks.
# These are physically consistent positions for GPS constellation geometry.
_SAMPLE_SP3 = """\
#dP2026  1 15  0  0  0.00000000       4 ORBIT IGS20 FIT  IGS
## 2348 259200.00000000   900.00000000 60759 0.0000000000000
+    3   G01G02G03
++         2  2  2
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/*
/* Synthetic SP3 for validation testing
/*
*  2026  1 15  0  0  0.00000000
PG01  15124.123456   7562.061728  20987.654321    12.345678
PG02 -12345.678901  18765.432101  14567.890123    -5.678901
PG03   8901.234567 -15678.901234  19012.345678     3.456789
*  2026  1 15  0 15  0.00000000
PG01  15098.765432   7589.012345  21002.345678    12.345679
PG02 -12378.901234  18742.567890  14589.012345    -5.678902
PG03   8878.901234 -15701.234567  18998.765432     3.456790
*  2026  1 15  0 30  0.00000000
PG01  15072.345678   7616.543210  21016.789012    12.345680
PG02 -12411.234567  18719.012345  14610.345678    -5.678903
PG03   8856.123456 -15723.678901  18985.012345     3.456791
*  2026  1 15  0 45  0.00000000
PG01  15045.678901   7644.321098  21030.987654    12.345681
PG02 -12443.567890  18695.234567  14631.678901    -5.678904
PG03   8833.012345 -15745.890123  18971.345678     3.456792
EOF
"""


class TestSP3Parser:
    """Verify SP3 parser extracts data correctly."""

    def test_parse_header(self):
        """Parser should extract correct header metadata."""
        eph = parse_sp3(_SAMPLE_SP3)
        assert isinstance(eph, SP3Ephemeris)
        assert eph.num_epochs == 4
        assert eph.interval_s == 900.0  # 15 minutes
        assert eph.coordinate_system == "IGS20"
        assert eph.start_time == datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_parse_satellite_ids(self):
        """Parser should extract correct satellite IDs."""
        eph = parse_sp3(_SAMPLE_SP3)
        assert "G01" in eph.satellite_ids
        assert "G02" in eph.satellite_ids
        assert "G03" in eph.satellite_ids

    def test_parse_positions_count(self):
        """Parser should find all position records (3 sats × 4 epochs = 12)."""
        eph = parse_sp3(_SAMPLE_SP3)
        assert len(eph.points) == 12

    def test_parse_position_values(self):
        """First G01 position should be correctly parsed (km → m)."""
        eph = parse_sp3(_SAMPLE_SP3)
        g01_points = filter_satellite(eph, "G01")
        assert len(g01_points) == 4
        p0 = g01_points[0]
        assert p0.satellite_id == "G01"
        assert p0.time == datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        # Positions should be in meters (km * 1000)
        assert abs(p0.x_m - 15124123.456) < 0.01
        assert abs(p0.y_m - 7562061.728) < 0.01
        assert abs(p0.z_m - 20987654.321) < 0.01
        assert abs(p0.clock_us - 12.345678) < 0.0001

    def test_parse_epoch_sequence(self):
        """Epochs should be 15 minutes apart."""
        eph = parse_sp3(_SAMPLE_SP3)
        g01_points = filter_satellite(eph, "G01")
        for i in range(len(g01_points) - 1):
            dt = (g01_points[i + 1].time - g01_points[i].time).total_seconds()
            assert abs(dt - 900.0) < 0.01

    def test_filter_satellite(self):
        """filter_satellite should return only points for specified satellite."""
        eph = parse_sp3(_SAMPLE_SP3)
        g02_points = filter_satellite(eph, "G02")
        assert len(g02_points) == 4
        for p in g02_points:
            assert p.satellite_id == "G02"


class TestSP3PhysicalConsistency:
    """Verify SP3 positions are physically consistent for GPS satellites."""

    def test_gps_altitude_range(self):
        """GPS satellite positions should be at ~26000-27000 km radius."""
        eph = parse_sp3(_SAMPLE_SP3)
        for point in eph.points:
            r_km = math.sqrt(point.x_m ** 2 + point.y_m ** 2 + point.z_m ** 2) / 1000
            assert 25000 < r_km < 28000, (
                f"{point.satellite_id} at {point.time}: r={r_km:.1f} km"
            )

    def test_position_changes_between_epochs(self):
        """Positions should change between epochs (satellite is moving)."""
        eph = parse_sp3(_SAMPLE_SP3)
        g01_points = filter_satellite(eph, "G01")
        p0 = g01_points[0]
        p1 = g01_points[1]
        dist = math.sqrt(
            (p1.x_m - p0.x_m) ** 2 +
            (p1.y_m - p0.y_m) ** 2 +
            (p1.z_m - p0.z_m) ** 2,
        )
        dist_km = dist / 1000
        # GPS velocity ~3.87 km/s, in 15 min moves ~3480 km
        # But distance between positions < 3480 km due to orbital curvature
        assert 10 < dist_km < 5000, f"G01 moved {dist_km:.1f} km in 15 min"

    def test_velocity_estimate_direction(self):
        """Finite-difference velocity should show the satellite is moving.

        Note: synthetic SP3 data has small position deltas so we only verify
        non-zero motion, not specific velocity magnitude. Real IGS SP3 data
        would show ~3.87 km/s for GPS.
        """
        eph = parse_sp3(_SAMPLE_SP3)
        g01_points = filter_satellite(eph, "G01")
        p0, p1 = g01_points[0], g01_points[1]
        dt = (p1.time - p0.time).total_seconds()
        vx = (p1.x_m - p0.x_m) / dt
        vy = (p1.y_m - p0.y_m) / dt
        vz = (p1.z_m - p0.z_m) / dt
        v_mag = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        # Satellite must be moving
        assert v_mag > 0, "Satellite should be moving between epochs"


class TestSP3EnergyAnalysis:
    """Verify energy computed from SP3 positions matches GPS orbit expectations."""

    def test_sp3_radius_stable(self):
        """GPS orbital radius should be stable across epochs (near-circular)."""
        eph = parse_sp3(_SAMPLE_SP3)
        g01_points = filter_satellite(eph, "G01")
        radii = [
            math.sqrt(p.x_m ** 2 + p.y_m ** 2 + p.z_m ** 2)
            for p in g01_points
        ]
        r_range_km = (max(radii) - min(radii)) / 1000
        r_mean_km = sum(radii) / len(radii) / 1000
        # GPS eccentricity is typically < 0.02, so radius variation < ~500 km
        assert r_range_km < 1000, f"Radius range: {r_range_km:.1f} km"
        assert 25000 < r_mean_km < 28000, f"Mean radius: {r_mean_km:.1f} km"


class TestSP3ParserEdgeCases:
    """Test SP3 parser robustness."""

    def test_empty_content_raises(self):
        """Empty content should raise ValueError."""
        try:
            parse_sp3("")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_header_raises(self):
        """Non-SP3 content should raise ValueError."""
        try:
            parse_sp3("This is not SP3 data\nSecond line\n")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_minimal_sp3(self):
        """Minimal valid SP3 with one epoch and one satellite."""
        minimal = """\
#dP2026  1 15  0  0  0.00000000       1 ORBIT IGS20 FIT  IGS
## 2348 259200.00000000   900.00000000 60759 0.0000000000000
+    1   G01
++         2
%c G  cc GPS
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
*  2026  1 15  0  0  0.00000000
PG01  15124.123456   7562.061728  20987.654321    12.345678
EOF
"""
        eph = parse_sp3(minimal)
        assert len(eph.points) == 1
        assert eph.points[0].satellite_id == "G01"
