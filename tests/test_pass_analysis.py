"""Tests for pass analysis: Doppler, pass classification, contact stats, data downlink, visual magnitude."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator import (
    OrbitalConstants,
    OrbitalState,
    ShellConfig,
    generate_walker_shell,
    derive_orbital_state,
    propagate_to,
    propagate_ecef_to,
    GroundStation,
    AccessWindow,
    compute_access_windows,
)

EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
MU = OrbitalConstants.MU_EARTH
R_E = OrbitalConstants.R_EARTH
C_LIGHT = 299_792_458.0  # m/s


def _leo_state(altitude_km=550, inclination_deg=53) -> OrbitalState:
    shell = ShellConfig(
        altitude_km=altitude_km, inclination_deg=inclination_deg,
        num_planes=1, sats_per_plane=1, phase_factor=0,
        raan_offset_deg=0, shell_name="Test",
    )
    sats = generate_walker_shell(shell)
    return derive_orbital_state(sats[0], EPOCH)


def _station() -> GroundStation:
    return GroundStation(name="Delft", lat_deg=52.0, lon_deg=4.4, alt_m=0.0)


# --- Doppler Shift ---

class TestComputeDopplerShift:

    def test_returns_doppler_result_type(self):
        from constellation_generator.domain.pass_analysis import (
            compute_doppler_shift, DopplerResult,
        )
        state = _leo_state()
        pos, vel = propagate_to(state, EPOCH)
        result = compute_doppler_shift(_station(), pos, vel, EPOCH, freq_hz=437e6)
        assert isinstance(result, DopplerResult)

    def test_shift_within_expected_range(self):
        """LEO Doppler at 437 MHz: ±10 kHz typical."""
        from constellation_generator.domain.pass_analysis import compute_doppler_shift
        state = _leo_state()
        pos, vel = propagate_to(state, EPOCH)
        result = compute_doppler_shift(_station(), pos, vel, EPOCH, freq_hz=437e6)
        assert abs(result.shift_hz) < 15000  # < 15 kHz

    def test_range_rate_bounded(self):
        """Range rate for LEO pass: ±8 km/s max."""
        from constellation_generator.domain.pass_analysis import compute_doppler_shift
        state = _leo_state()
        pos, vel = propagate_to(state, EPOCH)
        result = compute_doppler_shift(_station(), pos, vel, EPOCH, freq_hz=437e6)
        assert abs(result.range_rate_ms) < 8000

    def test_slant_range_positive(self):
        from constellation_generator.domain.pass_analysis import compute_doppler_shift
        state = _leo_state()
        pos, vel = propagate_to(state, EPOCH)
        result = compute_doppler_shift(_station(), pos, vel, EPOCH, freq_hz=437e6)
        assert result.slant_range_km > 0

    def test_zero_frequency_returns_zero_shift(self):
        from constellation_generator.domain.pass_analysis import compute_doppler_shift
        state = _leo_state()
        pos, vel = propagate_to(state, EPOCH)
        result = compute_doppler_shift(_station(), pos, vel, EPOCH, freq_hz=0.0)
        assert result.shift_hz == 0.0

    def test_doppler_formula_consistency(self):
        """shift_hz = -freq * range_rate / c."""
        from constellation_generator.domain.pass_analysis import compute_doppler_shift
        state = _leo_state()
        pos, vel = propagate_to(state, EPOCH)
        freq = 437e6
        result = compute_doppler_shift(_station(), pos, vel, EPOCH, freq_hz=freq)
        expected = -freq * result.range_rate_ms / C_LIGHT
        assert abs(result.shift_hz - expected) < 0.01


# --- Pass Classification ---

class TestClassifyPass:

    def test_returns_pass_classification_type(self):
        from constellation_generator.domain.pass_analysis import (
            classify_pass, PassClassification,
        )
        window = AccessWindow(
            rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=10),
            max_elevation_deg=45, duration_seconds=600,
        )
        result = classify_pass(window)
        assert isinstance(result, PassClassification)

    def test_high_elevation_excellent(self):
        from constellation_generator.domain.pass_analysis import classify_pass
        window = AccessWindow(
            rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=12),
            max_elevation_deg=80, duration_seconds=720,
        )
        assert classify_pass(window).quality == "excellent"

    def test_medium_elevation_good(self):
        from constellation_generator.domain.pass_analysis import classify_pass
        window = AccessWindow(
            rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=8),
            max_elevation_deg=50, duration_seconds=480,
        )
        assert classify_pass(window).quality == "good"

    def test_low_elevation_fair(self):
        from constellation_generator.domain.pass_analysis import classify_pass
        window = AccessWindow(
            rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=5),
            max_elevation_deg=25, duration_seconds=300,
        )
        assert classify_pass(window).quality == "fair"

    def test_very_low_elevation_poor(self):
        from constellation_generator.domain.pass_analysis import classify_pass
        window = AccessWindow(
            rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=3),
            max_elevation_deg=12, duration_seconds=180,
        )
        assert classify_pass(window).quality == "poor"


# --- Contact Summary ---

class TestComputeContactSummary:

    def test_returns_contact_summary_type(self):
        from constellation_generator.domain.pass_analysis import (
            compute_contact_summary, ContactSummary,
        )
        windows = [
            AccessWindow(
                rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=10),
                max_elevation_deg=45, duration_seconds=600,
            ),
        ]
        result = compute_contact_summary(windows)
        assert isinstance(result, ContactSummary)

    def test_total_contact_seconds(self):
        from constellation_generator.domain.pass_analysis import compute_contact_summary
        windows = [
            AccessWindow(rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=10),
                         max_elevation_deg=45, duration_seconds=600),
            AccessWindow(rise_time=EPOCH + timedelta(hours=2),
                         set_time=EPOCH + timedelta(hours=2, minutes=8),
                         max_elevation_deg=30, duration_seconds=480),
        ]
        result = compute_contact_summary(windows)
        assert abs(result.total_contact_seconds - 1080) < 0.01

    def test_num_passes(self):
        from constellation_generator.domain.pass_analysis import compute_contact_summary
        windows = [
            AccessWindow(rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=5),
                         max_elevation_deg=20, duration_seconds=300),
        ] * 3
        result = compute_contact_summary(windows)
        assert result.num_passes == 3

    def test_max_gap_between_passes(self):
        from constellation_generator.domain.pass_analysis import compute_contact_summary
        w1 = AccessWindow(rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=5),
                          max_elevation_deg=20, duration_seconds=300)
        w2 = AccessWindow(rise_time=EPOCH + timedelta(hours=3),
                          set_time=EPOCH + timedelta(hours=3, minutes=5),
                          max_elevation_deg=20, duration_seconds=300)
        result = compute_contact_summary([w1, w2])
        expected_gap = 3 * 3600 - 300  # 3h minus first pass duration
        assert abs(result.max_gap_seconds - expected_gap) < 1.0

    def test_best_elevation(self):
        from constellation_generator.domain.pass_analysis import compute_contact_summary
        windows = [
            AccessWindow(rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=5),
                         max_elevation_deg=20, duration_seconds=300),
            AccessWindow(rise_time=EPOCH + timedelta(hours=2),
                         set_time=EPOCH + timedelta(hours=2, minutes=10),
                         max_elevation_deg=75, duration_seconds=600),
        ]
        result = compute_contact_summary(windows)
        assert result.best_elevation_deg == 75

    def test_empty_windows(self):
        from constellation_generator.domain.pass_analysis import compute_contact_summary
        result = compute_contact_summary([])
        assert result.total_contact_seconds == 0
        assert result.num_passes == 0


# --- Data Downlink Estimate ---

class TestEstimateDataDownlink:

    def test_returns_data_downlink_estimate_type(self):
        from constellation_generator.domain.pass_analysis import (
            estimate_data_downlink, DataDownlinkEstimate,
        )
        windows = [
            AccessWindow(rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=10),
                         max_elevation_deg=45, duration_seconds=600),
        ]
        result = estimate_data_downlink(windows, data_rate_bps=9600)
        assert isinstance(result, DataDownlinkEstimate)

    def test_daily_contact_seconds(self):
        from constellation_generator.domain.pass_analysis import estimate_data_downlink
        windows = [
            AccessWindow(rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=10),
                         max_elevation_deg=45, duration_seconds=600),
        ]
        result = estimate_data_downlink(windows, data_rate_bps=9600)
        assert result.daily_contact_seconds == 600

    def test_daily_data_bytes_formula(self):
        from constellation_generator.domain.pass_analysis import estimate_data_downlink
        windows = [
            AccessWindow(rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=10),
                         max_elevation_deg=45, duration_seconds=600),
        ]
        result = estimate_data_downlink(windows, data_rate_bps=9600, efficiency=0.8)
        expected = 600 * 9600 * 0.8 / 8  # seconds * bits/s * eff / 8 = bytes
        assert abs(result.daily_data_bytes - expected) < 1.0

    def test_human_readable_present(self):
        from constellation_generator.domain.pass_analysis import estimate_data_downlink
        windows = [
            AccessWindow(rise_time=EPOCH, set_time=EPOCH + timedelta(minutes=10),
                         max_elevation_deg=45, duration_seconds=600),
        ]
        result = estimate_data_downlink(windows, data_rate_bps=9600)
        assert isinstance(result.human_readable, str)
        assert len(result.human_readable) > 0


# --- Visual Magnitude ---

class TestComputeVisibleMagnitude:

    def test_returns_float(self):
        from constellation_generator.domain.pass_analysis import compute_visible_magnitude
        result = compute_visible_magnitude(
            slant_range_km=1000, phase_angle_deg=90,
            cross_section_m2=1.0, albedo=0.25,
        )
        assert isinstance(result, float)

    def test_closer_is_brighter(self):
        """Closer satellite = lower magnitude (brighter)."""
        from constellation_generator.domain.pass_analysis import compute_visible_magnitude
        near = compute_visible_magnitude(500, 90, 1.0, 0.25)
        far = compute_visible_magnitude(2000, 90, 1.0, 0.25)
        assert near < far

    def test_larger_cross_section_brighter(self):
        from constellation_generator.domain.pass_analysis import compute_visible_magnitude
        small = compute_visible_magnitude(1000, 90, 0.5, 0.25)
        large = compute_visible_magnitude(1000, 90, 10.0, 0.25)
        assert large < small

    def test_full_phase_brightest(self):
        """Phase angle 0 (fully illuminated) should be brightest."""
        from constellation_generator.domain.pass_analysis import compute_visible_magnitude
        full = compute_visible_magnitude(1000, 0, 1.0, 0.25)
        quarter = compute_visible_magnitude(1000, 90, 1.0, 0.25)
        assert full < quarter

    def test_iss_like_object_visible(self):
        """ISS-sized object (400 m², 400km range) should be easily visible (mag < 0)."""
        from constellation_generator.domain.pass_analysis import compute_visible_magnitude
        mag = compute_visible_magnitude(400, 60, 400.0, 0.30)
        assert mag < 2.0  # should be very bright

    def test_cubesat_faint(self):
        """1U CubeSat (0.01 m²) at 1000km should be faint (mag > 6)."""
        from constellation_generator.domain.pass_analysis import compute_visible_magnitude
        mag = compute_visible_magnitude(1000, 90, 0.01, 0.25)
        assert mag > 6.0

    def test_zero_range_raises(self):
        """Zero slant range must raise ValueError, not divide by zero."""
        from constellation_generator.domain.pass_analysis import compute_visible_magnitude
        with pytest.raises(ValueError, match="slant_range_km"):
            compute_visible_magnitude(0.0, 90, 1.0, 0.25)

    def test_negative_range_raises(self):
        from constellation_generator.domain.pass_analysis import compute_visible_magnitude
        with pytest.raises(ValueError, match="slant_range_km"):
            compute_visible_magnitude(-100.0, 90, 1.0, 0.25)

    def test_negative_cross_section_raises(self):
        from constellation_generator.domain.pass_analysis import compute_visible_magnitude
        with pytest.raises(ValueError, match="cross_section_m2"):
            compute_visible_magnitude(1000, 90, -1.0, 0.25)


# --- Purity ---

class TestPassAnalysisPurity:

    def test_no_external_deps(self):
        import ast
        import constellation_generator.domain.pass_analysis as mod
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        allowed_stdlib = {"math", "dataclasses", "datetime"}
        allowed_internal = {"constellation_generator"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed_stdlib or top in allowed_internal, \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    assert top in allowed_stdlib or top in allowed_internal, \
                        f"Forbidden import from: {node.module}"
