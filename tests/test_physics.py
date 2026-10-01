"""Physics layer: every number here is checked against an external reference."""

from __future__ import annotations

import numpy as np
import pytest

from cne.physics import (
    AB_ZEROPOINT,
    SNANA_ZEROPOINT,
    ab_magnitude,
    comoving_distance_mpc,
    distance_modulus_from_z,
    extinction_ab,
    galactic_coordinates,
    luminosity_distance_mpc,
    mc_redshift_draws,
    reliability_from_redshift,
    resolve_distmod,
)


class TestCosmology:
    """Flat Lambda-CDM with H0=70, Omega_m=0.3, checked against published values."""

    @pytest.mark.parametrize("z,expected", [(0.1, 418.5), (0.5, 1888.6), (1.0, 3301.0)])
    def test_comoving_distance(self, z, expected):
        got = float(comoving_distance_mpc(np.array([z]))[0])
        assert got == pytest.approx(expected, rel=0.02), f"d_C({z}) = {got:.1f}, expected ~{expected}"

    @pytest.mark.parametrize("z,expected", [(0.1, 38.31), (0.5, 42.27), (1.0, 44.09)])
    def test_distance_modulus(self, z, expected):
        got = float(distance_modulus_from_z(np.array([z]))[0])
        assert got == pytest.approx(expected, abs=0.02)

    def test_luminosity_distance_is_comoving_times_one_plus_z(self):
        z = np.array([0.3, 0.8, 1.5])
        np.testing.assert_allclose(luminosity_distance_mpc(z), comoving_distance_mpc(z) * (1 + z), rtol=1e-9)

    def test_monotonic_in_redshift(self):
        z = np.linspace(0.01, 3.0, 40)
        d = comoving_distance_mpc(z)
        assert np.all(np.diff(d) > 0)

    def test_deterministic(self):
        z = np.array([0.42, 1.13])
        np.testing.assert_array_equal(comoving_distance_mpc(z), comoving_distance_mpc(z))


class TestMagnitudes:
    def test_zero_point_is_the_snana_convention(self):
        """PLAsTiCC ships SNANA fluxes; ZP=27.5 is 5 mag offset from nMgy/AB (22.5)."""
        assert SNANA_ZEROPOINT == 27.5
        assert AB_ZEROPOINT == 27.5

    def test_known_magnitudes(self):
        assert float(ab_magnitude(np.array([1.0]))[0]) == pytest.approx(27.5)
        assert float(ab_magnitude(np.array([10.0]))[0]) == pytest.approx(25.0)
        assert float(ab_magnitude(np.array([1000.0]))[0]) == pytest.approx(20.0)

    def test_non_positive_flux_is_nan_not_minus_inf(self):
        mag = ab_magnitude(np.array([-5.0, 0.0]))
        assert np.all(np.isnan(mag))

    def test_brighter_flux_gives_smaller_magnitude(self):
        mags = ab_magnitude(np.array([10.0, 100.0, 1000.0]))
        assert np.all(np.diff(mags) < 0)

    def test_error_propagation(self):
        mag, err = ab_magnitude(np.array([100.0]), np.array([10.0]))
        # 10% flux error -> 2.5/ln(10) * 0.1 = 0.1086 mag
        assert float(err[0]) == pytest.approx(0.1086, abs=1e-3)


class TestExtinction:
    def test_zero_extinction_when_no_dust(self):
        assert float(extinction_ab(np.array([0.0]), 2)) == 0.0

    def test_bluer_bands_are_extincted_more(self):
        mwebv = np.array([0.2])
        values = [float(extinction_ab(mwebv, band)) for band in range(6)]
        assert np.all(np.diff(values) < 0), "extinction must decrease from u to y"

    def test_scales_linearly_with_r_v(self):
        a1 = float(extinction_ab(np.array([0.1]), 2, r_v=3.1))
        a2 = float(extinction_ab(np.array([0.1]), 2, r_v=6.2))
        assert a2 == pytest.approx(2 * a1)


class TestGalacticCoordinates:
    def test_galactic_centre(self):
        lon, lat = galactic_coordinates(np.array([266.417]), np.array([-28.983]))
        # Longitude wraps at 360, so compare angular distance to zero.
        assert min(float(lon[0]), 360.0 - float(lon[0])) < 0.1
        assert float(lat[0]) == pytest.approx(0.0, abs=0.1)

    def test_andromeda(self):
        lon, lat = galactic_coordinates(np.array([10.684]), np.array([41.269]))
        assert float(lon[0]) == pytest.approx(121.2, abs=0.3)
        assert float(lat[0]) == pytest.approx(-21.6, abs=0.3)

    def test_latitude_is_bounded(self):
        rng = np.random.default_rng(0)
        ra, dec = rng.uniform(0, 360, 500), rng.uniform(-90, 90, 500)
        _, lat = galactic_coordinates(ra, dec)
        assert np.all(np.abs(lat) <= 90.0 + 1e-9)


class TestRedshiftHandling:
    def test_distmod_prefers_survey_value(self):
        dm, flag = resolve_distmod(np.array([41.0]), np.array([0.5]))
        assert float(dm[0]) == pytest.approx(41.0)
        assert float(flag[0]) == 1.0

    def test_distmod_falls_back_to_photoz(self):
        dm, flag = resolve_distmod(np.array([0.0]), np.array([0.5]))
        assert float(flag[0]) == 0.0
        assert float(dm[0]) > 40.0

    def test_no_distance_information_is_flagged(self):
        _dm, flag = resolve_distmod(np.array([0.0]), np.array([0.0]))
        assert float(flag[0]) == -1.0

    def test_galactic_sources_are_fully_reliable(self):
        """A galactic transient has no redshift by construction - that is not
        missing information, and abstaining on it discards a clean population."""
        rel = reliability_from_redshift(np.array([0.0]), np.array([0.0]), np.array([-1.0]))
        assert float(rel[0]) == 1.0

    def test_bad_photoz_is_unreliable(self):
        rel = reliability_from_redshift(np.array([1.0]), np.array([0.9]), np.array([0.0]))
        assert float(rel[0]) < 0.05

    def test_spectroscopic_redshift_is_reliable(self):
        rel = reliability_from_redshift(np.array([0.4]), np.array([0.2]), np.array([1.0]))
        assert float(rel[0]) >= 0.9

    def test_extragalactic_with_no_distance_is_unreliable(self):
        """z > 0 but no usable distance modulus: genuinely unknown distance."""
        rel = reliability_from_redshift(np.array([0.5]), np.array([0.2]), np.array([-1.0]))
        assert float(rel[0]) == 0.0

    def test_mc_draws_are_deterministic_and_centred(self):
        z = np.array([0.5, 1.0])
        err = np.array([0.05, 0.1])
        a = mc_redshift_draws(z, err, 32, seed=42)
        b = mc_redshift_draws(z, err, 32, seed=42)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (32, 2)
        np.testing.assert_allclose(a.mean(axis=0), z, atol=0.03)
        assert np.all(a > 0), "redshift draws must stay physical"
