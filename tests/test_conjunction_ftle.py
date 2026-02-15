# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for FTLE conjunction risk classification in domain/conjunction.py."""
import math
from datetime import datetime, timezone

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.propagation import OrbitalState
from humeris.domain.conjunction import (
    ConjunctionPredictability,
    compute_conjunction_ftle,
)


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _circular_state(altitude_km, inclination_deg, raan_deg=0.0, nu_deg=0.0):
    """Create a circular OrbitalState at given altitude."""
    a = OrbitalConstants.R_EARTH + altitude_km * 1000.0
    n = math.sqrt(OrbitalConstants.MU_EARTH / a ** 3)
    return OrbitalState(
        semi_major_axis_m=a,
        eccentricity=0.0,
        inclination_rad=math.radians(inclination_deg),
        raan_rad=math.radians(raan_deg),
        arg_perigee_rad=0.0,
        true_anomaly_rad=math.radians(nu_deg),
        mean_motion_rad_s=n,
        reference_epoch=EPOCH,
    )


class TestFTLECoplanarLowDivergence:
    """Co-planar near-circular orbits at similar altitudes have low FTLE."""

    def test_coplanar_same_altitude_low_ftle(self):
        """Two sats in same plane, same altitude, small phase offset => low FTLE.

        FTLE = ln(sigma_max) / window_s. A longer window dilutes the
        exponent, giving a more stable measurement. 2700 s (~45 min) is
        roughly half an orbital period at 550 km, a natural scale for
        co-planar relative dynamics.
        """
        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=5)

        result = compute_conjunction_ftle(s1, s2, EPOCH, window_s=2700.0)
        # Co-planar, same altitude => gentle relative dynamics
        # FTLE should be small (below typical chaos threshold)
        assert result.ftle < 1e-3

    def test_coplanar_returns_predictability(self):
        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=5)

        result = compute_conjunction_ftle(s1, s2, EPOCH)
        assert isinstance(result, ConjunctionPredictability)


class TestFTLECrossingOrbits:
    """Crossing orbits (different inclinations) have higher encounter risk.

    Under two-body Keplerian dynamics, FTLE depends only on the individual
    orbit shape (semi-major axis, eccentricity) — it cannot distinguish
    encounter geometries. The margin_multiplier captures encounter geometry
    via relative velocity at TCA: crossing orbits have higher relative
    velocity, shorter conjunction windows, and less predictable outcomes.
    """

    def test_crossing_orbits_higher_margin(self):
        """Different inclinations at same altitude => higher margin_multiplier."""
        # Co-planar reference: low relative velocity
        s1_co = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2_co = _circular_state(550, 53, raan_deg=0, nu_deg=5)
        result_co = compute_conjunction_ftle(s1_co, s2_co, EPOCH)

        # Crossing orbits: high relative velocity from inclination difference
        s1_cross = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2_cross = _circular_state(550, 97, raan_deg=30, nu_deg=0)
        result_cross = compute_conjunction_ftle(s1_cross, s2_cross, EPOCH)

        assert result_cross.margin_multiplier > result_co.margin_multiplier


class TestFTLEMarginMultiplier:
    """margin_multiplier >= 1.0 always."""

    def test_margin_at_least_one_coplanar(self):
        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=5)
        result = compute_conjunction_ftle(s1, s2, EPOCH)
        assert result.margin_multiplier >= 1.0

    def test_margin_at_least_one_crossing(self):
        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 97, raan_deg=30, nu_deg=0)
        result = compute_conjunction_ftle(s1, s2, EPOCH)
        assert result.margin_multiplier >= 1.0


class TestFTLEPredictabilityHorizon:
    """predictability_horizon_s > 0 always."""

    def test_horizon_positive_coplanar(self):
        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=5)
        result = compute_conjunction_ftle(s1, s2, EPOCH)
        assert result.predictability_horizon_s > 0.0

    def test_horizon_positive_crossing(self):
        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 97, raan_deg=30, nu_deg=0)
        result = compute_conjunction_ftle(s1, s2, EPOCH)
        assert result.predictability_horizon_s > 0.0


class TestFTLENonNegative:
    """FTLE is non-negative."""

    def test_ftle_nonnegative_coplanar(self):
        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=5)
        result = compute_conjunction_ftle(s1, s2, EPOCH)
        assert result.ftle >= 0.0

    def test_ftle_nonnegative_crossing(self):
        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 97, raan_deg=30, nu_deg=0)
        result = compute_conjunction_ftle(s1, s2, EPOCH)
        assert result.ftle >= 0.0

    def test_ftle_nonnegative_different_altitude(self):
        s1 = _circular_state(400, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(800, 53, raan_deg=0, nu_deg=0)
        result = compute_conjunction_ftle(s1, s2, EPOCH)
        assert result.ftle >= 0.0


class TestConjunctionPredictabilityDataclass:
    """Verify ConjunctionPredictability is a proper frozen dataclass."""

    def test_frozen(self):
        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=5)
        result = compute_conjunction_ftle(s1, s2, EPOCH)

        import pytest
        with pytest.raises(AttributeError):
            result.ftle = 99.0

    def test_is_chaotic_boolean(self):
        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=5)
        result = compute_conjunction_ftle(s1, s2, EPOCH)
        assert isinstance(result.is_chaotic, bool)

    def test_max_singular_value_positive(self):
        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=5)
        result = compute_conjunction_ftle(s1, s2, EPOCH)
        assert result.max_singular_value > 0.0
