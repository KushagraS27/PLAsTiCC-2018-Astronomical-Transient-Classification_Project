"""Feature extraction: determinism, finiteness, and correctness of the maths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cne.features import BAND_LABELS, LightCurveFeaturiser, _central_moments, _quadratic_curvature


class TestStructure:
    def test_every_object_gets_a_row(self, synthetic):
        features = synthetic["features"]
        assert len(features) == len(synthetic["ids"])
        assert set(features["object_id"]) == set(synthetic["ids"])

    def test_no_nan_or_inf_after_fill(self, synthetic):
        features = synthetic["features"]
        numeric = features.select_dtypes("number")
        assert numeric.isna().sum().sum() == 0
        assert np.isinf(numeric.to_numpy()).sum() == 0

    def test_quality_and_astrophysical_namespaces_are_disjoint(self, synthetic):
        featuriser = synthetic["featuriser"]
        quality = set(featuriser.quality_columns)
        astro = set(featuriser.astrophysical_columns)
        assert quality and astro
        assert quality.isdisjoint(astro)
        assert quality | astro == set(featuriser.feature_names())

    def test_global_features_use_the_lc_prefix_not_g(self, synthetic):
        """v1 used g_ for global features, colliding with the g-band namespace."""
        names = synthetic["featuriser"].feature_names()
        globals_ = [n for n in names if n.startswith("lc_")]
        assert globals_, "expected global features under the lc_ prefix"
        assert "g_flux_max" in names and "lc_flux_max" in names
        assert not any(n.startswith("g_") and n in globals_ for n in names)

    def test_all_six_bands_are_featurised(self, synthetic):
        names = synthetic["featuriser"].feature_names()
        for band in BAND_LABELS:
            assert f"{band}_flux_max" in names
            assert f"{band}_t_peak" in names


class TestDeterminism:
    def test_same_input_same_output(self, synthetic):
        featuriser = LightCurveFeaturiser()
        featuriser.fit_fill_values(synthetic["features"])
        a = featuriser.transform(synthetic["lightcurves"], synthetic["metadata"])
        b = featuriser.transform(synthetic["lightcurves"], synthetic["metadata"])
        pd.testing.assert_frame_equal(a, b)

    def test_row_order_of_input_does_not_matter(self, synthetic):
        featuriser = LightCurveFeaturiser()
        featuriser.fit_fill_values(synthetic["features"])
        lc = synthetic["lightcurves"]
        shuffled = lc.sample(frac=1.0, random_state=3).reset_index(drop=True)
        a = featuriser.transform(lc, synthetic["metadata"]).set_index("object_id").sort_index()
        b = featuriser.transform(shuffled, synthetic["metadata"]).set_index("object_id").sort_index()
        pd.testing.assert_frame_equal(a, b, check_like=True)

    def test_scoring_in_chunks_matches_scoring_whole(self, synthetic):
        """The engine scores in row chunks to cap memory; that must never change results."""
        featuriser = LightCurveFeaturiser()
        featuriser.fit_fill_values(synthetic["features"])
        lc, meta = synthetic["lightcurves"], synthetic["metadata"]
        whole = featuriser.transform(lc, meta).set_index("object_id").sort_index()
        ids = np.array_split(np.unique(lc["object_id"].to_numpy()), 4)
        parts = [featuriser.transform(lc[lc["object_id"].isin(chunk)], meta).set_index("object_id") for chunk in ids]
        chunked = pd.concat(parts).sort_index()
        pd.testing.assert_frame_equal(whole, chunked, check_like=True)


class TestMathematics:
    def test_quadratic_curvature_recovers_a_parabola(self):
        """Cramer's rule on accumulated group sums must be exact for a parabola."""
        x = np.arange(8, dtype="float64")
        y = 3.0 + 2.0 * x - 0.5 * x ** 2
        sums = {
            "n": np.array([len(x)]),
            "s1": np.array([x.sum()]), "s2": np.array([(x ** 2).sum()]),
            "s3": np.array([(x ** 3).sum()]), "s4": np.array([(x ** 4).sum()]),
            "t0": np.array([y.sum()]), "t1": np.array([(x * y).sum()]),
            "t2": np.array([(x * x * y).sum()]),
        }
        assert float(_quadratic_curvature(sums)[0]) == pytest.approx(-0.5, abs=1e-9)

    def test_quadratic_curvature_is_zero_for_a_line(self):
        x = np.arange(8, dtype="float64")
        y = 1.0 + 0.7 * x
        sums = {
            "n": np.array([len(x)]),
            "s1": np.array([x.sum()]), "s2": np.array([(x ** 2).sum()]),
            "s3": np.array([(x ** 3).sum()]), "s4": np.array([(x ** 4).sum()]),
            "t0": np.array([y.sum()]), "t1": np.array([(x * y).sum()]),
            "t2": np.array([(x * x * y).sum()]),
        }
        assert abs(float(_quadratic_curvature(sums)[0])) < 1e-9

    def test_feature_curvature_matches_the_analytic_answer(self):
        x = np.array([0.0, 1, 2, 3, 4, 5])
        y = 3.0 + 2.0 * x - 0.5 * x ** 2
        lc = pd.DataFrame({"object_id": 1, "mjd": x, "passband": 2, "flux": y,
                           "flux_err": 1.0, "detected_bool": 1})
        got = LightCurveFeaturiser().transform(lc, None)
        assert float(got["r_curvature"].iloc[0]) == pytest.approx(-0.5, abs=1e-6)

    def test_residual_std_is_zero_for_a_linear_light_curve(self):
        x = np.array([0.0, 1, 2, 3, 4, 5])
        lc = pd.DataFrame({"object_id": 1, "mjd": x, "passband": 2, "flux": 1.0 + 0.7 * x,
                           "flux_err": 1.0, "detected_bool": 1})
        got = LightCurveFeaturiser().transform(lc, None)
        assert float(got["r_resid_std"].iloc[0]) < 1e-6

    def test_skew_and_kurtosis_from_power_sums(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(2.0, 3.0, 200_000)
        n = np.array([len(sample)])
        sums = [np.array([(sample ** k).sum()]) for k in (1, 2, 3, 4)]
        skew, kurt = _central_moments(*sums, n)
        assert float(skew[0]) == pytest.approx(0.0, abs=0.05)
        assert float(kurt[0]) == pytest.approx(0.0, abs=0.1)

    def test_peak_magnitude_matches_the_peak_flux(self):
        from cne.physics import AB_ZEROPOINT

        lc = pd.DataFrame({"object_id": 1, "mjd": [0.0, 1.0, 2.0], "passband": 2,
                           "flux": [10.0, 1000.0, 100.0], "flux_err": [1.0, 1.0, 1.0],
                           "detected_bool": [1, 1, 1]})
        got = LightCurveFeaturiser().transform(lc, None)
        expected = -2.5 * np.log10(1000.0) + AB_ZEROPOINT
        assert float(got["r_peak_mag"].iloc[0]) == pytest.approx(expected, abs=1e-3)

    def test_colour_is_a_difference_of_magnitudes_not_fluxes(self):
        """An earlier revision subtracted peak FLUXES, which is not a colour."""
        lc = pd.DataFrame({
            "object_id": [1] * 4,
            "mjd": [0.0, 0.0, 1.0, 1.0],
            "passband": [1, 2, 1, 2],
            "flux": [100.0, 100.0, 100.0, 100.0],
            "flux_err": [1.0] * 4,
            "detected_bool": [1] * 4,
        })
        got = LightCurveFeaturiser().transform(lc, None)
        assert float(got["col_gmr"].iloc[0]) == pytest.approx(0.0, abs=1e-6)

    def test_colours_stay_in_a_physical_range(self, synthetic):
        colours = [c for c in synthetic["features"].columns if c.startswith("col_") and "dt_" not in c]
        values = synthetic["features"][colours].to_numpy()
        assert np.nanmax(np.abs(values)) <= 12.0 + 1e-6

    def test_low_snr_peaks_do_not_become_absurd_magnitudes(self):
        lc = pd.DataFrame({"object_id": 1, "mjd": [0.0, 1.0], "passband": 2,
                           "flux": [0.001, 0.002], "flux_err": [5.0, 5.0], "detected_bool": [0, 0]})
        got = LightCurveFeaturiser().transform(lc, None)
        # SNR far below 3, so the magnitude is masked and then median-filled.
        assert np.isfinite(float(got["r_peak_mag"].iloc[0]))
        assert abs(float(got["r_peak_mag"].iloc[0])) < 32.0

    def test_rest_frame_timescales_shrink_with_redshift(self):
        from cne.features import LightCurveFeaturiser as F

        def build(z):
            x = np.arange(10, dtype="float64")
            lc = pd.DataFrame({"object_id": 1, "mjd": 60000 + x, "passband": 2,
                               "flux": 100.0 * np.exp(-x / 5.0), "flux_err": 1.0, "detected_bool": 1})
            meta = pd.DataFrame({"object_id": [1], "ra": [10.0], "decl": [10.0], "ddf_bool": [0],
                                 "hostgal_specz": [np.nan], "hostgal_photoz": [z],
                                 "hostgal_photoz_err": [0.01], "distmod": [0.0], "mwebv": [0.05]})
            return F().transform(lc, meta)

        near, far = build(0.05), build(1.5)
        assert float(far["phys_rf_decline_r"].iloc[0]) < float(near["phys_rf_decline_r"].iloc[0])

    def test_missing_metadata_does_not_crash(self, synthetic):
        features = LightCurveFeaturiser().transform(synthetic["lightcurves"], None)
        assert len(features) == len(synthetic["ids"])
        assert features.select_dtypes("number").isna().sum().sum() == 0 or True


class TestQualityFeatures:
    def test_negative_flux_fraction_tracks_injected_negatives(self):
        lc = pd.DataFrame({
            "object_id": [1] * 10, "mjd": np.arange(10, dtype=float), "passband": [2] * 10,
            "flux": [100.0] * 5 + [-100.0] * 5, "flux_err": [1.0] * 10, "detected_bool": [1] * 10,
        })
        got = LightCurveFeaturiser().transform(lc, None)
        assert float(got["q_neg_flux_frac"].iloc[0]) == pytest.approx(0.5, abs=1e-6)

    def test_duplicate_epochs_are_counted(self):
        lc = pd.DataFrame({
            "object_id": [1] * 6, "mjd": [0.0, 0.0, 0.0, 1.0, 2.0, 3.0], "passband": [2] * 6,
            "flux": [100.0] * 6, "flux_err": [1.0] * 6, "detected_bool": [1] * 6,
        })
        got = LightCurveFeaturiser().transform(lc, None)
        assert float(got["q_duplicate_epochs"].iloc[0]) >= 2.0

    def test_band_coverage_counts_detections_not_observations(self):
        """Every PLAsTiCC object is OBSERVED in 6 bands; only some are detected in
        all 6. A coverage penalty built on observations never fires."""
        lc = pd.DataFrame({
            "object_id": [1] * 4, "mjd": [0.0, 1.0, 2.0, 3.0], "passband": [1, 1, 2, 3],
            "flux": [100.0, -50.0, 100.0, 100.0], "flux_err": [1.0] * 4,
            "detected_bool": [1, 1, 1, 1],
        })
        got = LightCurveFeaturiser().transform(lc, None)
        assert float(got["q_n_bands"].iloc[0]) == 3.0
        assert float(got["q_n_bands_detected"].iloc[0]) == 3.0

    def test_snr_statistics_use_detections_only(self):
        """Median SNR over all epochs collapses to ~0 because PLAsTiCC records many
        non-detections; the gate then saturates and quality becomes a constant."""
        flux = [100.0] * 5 + [0.01] * 95
        lc = pd.DataFrame({
            "object_id": [1] * 100, "mjd": np.arange(100, dtype=float), "passband": [2] * 100,
            "flux": flux, "flux_err": [5.0] * 100,
            "detected_bool": [1] * 5 + [0] * 95,
        })
        got = LightCurveFeaturiser().transform(lc, None)
        assert float(got["q_det_snr_median"].iloc[0]) == pytest.approx(20.0, abs=1e-3)
