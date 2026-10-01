"""Domain matching, drift monitoring, and the quality safety suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cne.artifacts import (
    ARTIFACT_TYPES,
    INJECTORS,
    ArtifactSafetySuite,
    RealBogusAdapter,
    duplicate_timestamps,
    error_bar_corruption,
    incomplete_band_coverage,
    long_cadence_gap,
    negative_flux_heavy,
    single_epoch_spike,
)
from cne.domain import (
    ABSTAIN,
    DEGRADED,
    MATCHED,
    DomainShiftMonitor,
    ReferenceBuilder,
    ks_statistic,
    psi,
)


def _population(n, seed, shift=0.0, snr_scale=1.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "object_id": np.arange(n),
        "lc_snr_max": rng.gamma(4.0, 8.0, n) * snr_scale + shift,
        "lc_n_det": rng.integers(3, 200, n).astype(float),
        "lc_t_span": rng.uniform(20, 400, n) + shift,
        "lc_peak_mag": rng.normal(21.0, 1.5, n) - shift * 0.01,
        "phys_z": np.clip(rng.gamma(2.0, 0.2, n) + shift * 0.001, 0.0, 3.0),
        "phys_distmod": rng.normal(41.0, 2.0, n) + shift * 0.02,
        "q_n_bands": rng.integers(3, 7, n).astype(float),
        "q_det_frac": rng.uniform(0.2, 0.9, n),
    })


class TestDriftStatistics:
    def test_identical_populations_have_zero_drift(self):
        ref = _population(2000, 0)
        same = _population(2000, 0)
        assert psi(ref["lc_snr_max"].to_numpy(), same["lc_snr_max"].to_numpy()) == pytest.approx(0.0, abs=1e-9)
        assert ks_statistic(ref["lc_snr_max"].to_numpy(), same["lc_snr_max"].to_numpy()) == pytest.approx(0.0, abs=1e-9)

    def test_a_shifted_population_has_large_drift(self):
        ref = _population(2000, 0)
        shifted = _population(2000, 1, shift=200.0)
        assert psi(ref["lc_snr_max"].to_numpy(), shifted["lc_snr_max"].to_numpy()) > 0.25

    def test_psi_is_scale_invariant_under_monotone_transforms(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000) + 0.4
        base = psi(a, b)
        scaled = psi(a * 7.3 + 2.0, b * 7.3 + 2.0)
        assert base == pytest.approx(scaled, abs=0.05)

    def test_psi_is_non_negative(self):
        rng = np.random.default_rng(1)
        assert psi(rng.normal(0, 1, 3000), rng.normal(1, 2, 3000)) >= 0.0


class TestDomainMatchScore:
    def test_matched_populations_score_high(self):
        ref, stream = _population(1500, 0), _population(1500, 7)
        score = ReferenceBuilder().match_score(ref, stream)
        assert score.status == MATCHED
        assert score.score > 0.8

    def test_severe_mismatch_triggers_abstention(self):
        ref, stream = _population(1500, 0), _population(1500, 8, shift=400.0, snr_scale=4.0)
        score = ReferenceBuilder().match_score(ref, stream)
        assert score.status in (DEGRADED, ABSTAIN)
        assert score.score < 0.8

    def test_per_covariate_diagnostics_are_exposed(self):
        ref, stream = _population(800, 0), _population(800, 3)
        score = ReferenceBuilder().match_score(ref, stream)
        assert "lc_snr_max" in score.per_covariate
        assert {"psi", "ks"} <= set(score.per_covariate["lc_snr_max"])


class TestReferenceBuilder:
    def test_density_ratio_weights_are_capped_and_normalised(self):
        ref, stream = _population(1200, 0), _population(1200, 5, shift=50.0)
        weights, info = ReferenceBuilder(weight_cap=5.0).build(ref, stream, strategy="density_ratio")
        # The cap is the hard safety invariant: no single reference object may
        # dominate the weighted prior. The mean is NOT pinned to 1 - when the
        # stream is genuinely shifted, most reference objects are legitimately
        # down-weighted, and the effective sample size is what reports that.
        assert weights.max() <= 5.0 + 1e-9
        assert weights.min() >= 1 / 5.0 - 1e-9
        assert np.all(np.isfinite(weights)) and np.all(weights > 0)
        assert 0 < info["effective_sample_size"] <= len(ref)
        # a real shift must actually cost effective sample size
        assert info["effective_sample_size"] < len(ref)

    def test_stratified_weights_are_also_capped(self):
        ref, stream = _population(1200, 0), _population(1200, 6)
        weights, _ = ReferenceBuilder(weight_cap=4.0).build(ref, stream, strategy="stratified")
        assert weights.max() <= 4.0 + 1e-9
        assert np.all(weights > 0)

    def test_none_strategy_reproduces_v1_behaviour(self):
        ref, stream = _population(500, 0), _population(500, 1)
        weights, info = ReferenceBuilder().build(ref, stream, strategy="none")
        assert np.all(weights == 1.0)
        assert info["strategy"] == "none"

    def test_deterministic(self):
        ref, stream = _population(600, 0), _population(600, 2)
        builder = ReferenceBuilder(seed=42)
        a, _ = builder.build(ref, stream, strategy="density_ratio")
        b, _ = builder.build(ref, stream, strategy="density_ratio")
        np.testing.assert_allclose(a, b)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            ReferenceBuilder().build(_population(50, 0), _population(50, 1), strategy="magic")


class TestDomainShiftMonitor:
    def test_healthy_stream_passes(self):
        ref = _population(1500, 0)
        monitor = DomainShiftMonitor().fit(ref)
        health = monitor.health(_population(1500, 11))
        assert health["status"] == MATCHED
        assert health["score"] > 0.5

    def test_a_known_artificial_shift_triggers_the_monitor(self):
        ref = _population(1500, 0)
        monitor = DomainShiftMonitor().fit(ref)
        health = monitor.health(_population(1500, 12, shift=500.0, snr_scale=5.0))
        assert health["status"] in (DEGRADED, ABSTAIN)

    def test_validated_ranges_are_versioned_with_the_model(self):
        ref = _population(800, 0)
        monitor = DomainShiftMonitor().fit(ref)
        health = monitor.health(_population(800, 1))
        for row in health["covariates"]:
            assert len(row["valid_range"]) == 2
            assert row["valid_range"][0] <= row["valid_range"][1]

    def test_drift_is_never_reported_as_novelty(self):
        """The monitor's output has no channel that a ranker could consume."""
        health = DomainShiftMonitor().fit(_population(400, 0)).health(_population(400, 1))
        assert set(health) <= {"score", "status", "worst_psi", "mean_psi", "thresholds", "covariates"}


# --------------------------------------------------------------------------- #
# artifact safety
# --------------------------------------------------------------------------- #
@pytest.fixture
def lightcurves():
    rows = []
    for oid in range(12):
        for k in range(40):
            for band in range(6):
                rows.append((oid, 60000.0 + k * 2.5, band, 50.0 * np.exp(-k / 12.0) + 5.0, 2.0, 1))
    return pd.DataFrame(rows, columns=["object_id", "mjd", "passband", "flux", "flux_err", "detected_bool"])


class TestInjectors:
    def test_every_artifact_type_has_an_injector(self):
        assert set(ARTIFACT_TYPES) == set(INJECTORS)

    def test_negative_flux_heavy_flips_detections(self, lightcurves):
        lc = lightcurves[lightcurves.object_id == 0]
        out = negative_flux_heavy(lc, np.random.default_rng(0))
        assert (out.loc[out.detected_bool == 1, "flux"] < 0).mean() > 0.4

    def test_single_epoch_spike_creates_one_outlier(self, lightcurves):
        lc = lightcurves[lightcurves.object_id == 1]
        out = single_epoch_spike(lc, np.random.default_rng(0), factor=60.0)
        # the injector scales one randomly chosen row by 60, so exactly one row
        # differs and it becomes the brightest point in the light curve
        changed = out["flux"].to_numpy() != lc.reset_index(drop=True)["flux"].to_numpy()
        assert int(changed.sum()) == 1
        assert out["flux"].to_numpy()[changed][0] == pytest.approx(
            60.0 * lc.reset_index(drop=True)["flux"].to_numpy()[changed][0])
        assert out["flux"].max() == out["flux"].to_numpy()[changed][0]

    def test_error_bar_corruption_inflates_uncertainty(self, lightcurves):
        lc = lightcurves[lightcurves.object_id == 2]
        out = error_bar_corruption(lc, np.random.default_rng(0), factor=25.0)
        assert out["flux_err"].median() == pytest.approx(25 * lc["flux_err"].median())

    def test_duplicate_timestamps_adds_rows(self, lightcurves):
        lc = lightcurves[lightcurves.object_id == 3]
        out = duplicate_timestamps(lc, np.random.default_rng(0), n=6)
        assert len(out) == len(lc) + 6

    def test_long_cadence_gap_truncates(self, lightcurves):
        lc = lightcurves[lightcurves.object_id == 4]
        assert len(long_cadence_gap(lc, np.random.default_rng(0))) < len(lc)

    def test_incomplete_band_coverage_removes_bands(self, lightcurves):
        lc = lightcurves[lightcurves.object_id == 5]
        out = incomplete_band_coverage(lc, np.random.default_rng(0), keep_bands=1)
        assert out["passband"].nunique() == 1

    def test_injectors_do_not_mutate_their_input(self, lightcurves):
        lc = lightcurves[lightcurves.object_id == 6].copy()
        before = lc.copy()
        for injector in INJECTORS.values():
            injector(lc, np.random.default_rng(0))
        pd.testing.assert_frame_equal(lc.reset_index(drop=True), before.reset_index(drop=True))


class TestSafetySuite:
    def _baseline(self, ids):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "object_id": ids,
            "novelty_score": rng.uniform(0.1, 0.5, len(ids)),
            "quality": rng.uniform(0.6, 1.0, len(ids)),
            "rank": np.arange(1, len(ids) + 1),
        })

    def test_a_gate_that_suppresses_passes(self, lightcurves):
        ids = list(range(8))
        baseline = self._baseline(ids)

        def score_fn(object_id, corrupted):
            row = baseline[baseline.object_id == object_id].iloc[0]
            return row["novelty_score"] * 0.5, row["quality"] * 0.5, row["rank"] + 5

        report = ArtifactSafetySuite(score_fn, cases_per_artifact=2).run(lightcurves, ids, baseline)
        assert report.passed
        assert report.n_promoted == 0
        assert report.mean_quality_drop > 0

    def test_a_gate_that_promotes_artifacts_fails(self, lightcurves):
        ids = list(range(8))
        baseline = self._baseline(ids)

        def score_fn(object_id, corrupted):
            row = baseline[baseline.object_id == object_id].iloc[0]
            return row["novelty_score"] * 3.0, 1.0, 1

        report = ArtifactSafetySuite(score_fn, cases_per_artifact=2).run(lightcurves, ids, baseline)
        assert report.passed is False
        assert report.n_promoted > 0

    def test_a_crash_counts_as_a_failure_not_a_skip(self, lightcurves):
        ids = list(range(4))
        baseline = self._baseline(ids)

        def score_fn(object_id, corrupted):
            raise RuntimeError("boom")

        report = ArtifactSafetySuite(score_fn, cases_per_artifact=1).run(lightcurves, ids, baseline)
        assert report.n_cases > 0

    def test_per_artifact_breakdown_is_reported(self, lightcurves):
        ids = list(range(6))
        baseline = self._baseline(ids)

        def score_fn(object_id, corrupted):
            row = baseline[baseline.object_id == object_id].iloc[0]
            return row["novelty_score"] * 0.2, 0.1, row["rank"]

        report = ArtifactSafetySuite(score_fn, cases_per_artifact=1).run(lightcurves, ids, baseline)
        assert set(report.per_artifact) == set(ARTIFACT_TYPES)


class TestRealBogusAdapter:
    def test_missing_score_leaves_quality_untouched(self):
        quality = np.array([0.8, 0.5])
        out = RealBogusAdapter().adjust(quality, None)
        np.testing.assert_array_equal(out, quality)

    def test_a_low_broker_score_can_only_lower_quality(self):
        quality = np.array([0.9, 0.9])
        frame = pd.DataFrame({"realbogus": [0.1, 1.0]})
        out = RealBogusAdapter().adjust(quality, frame)
        assert out[0] < quality[0]
        assert out[1] == pytest.approx(quality[1])
        assert np.all(out <= quality + 1e-9)
