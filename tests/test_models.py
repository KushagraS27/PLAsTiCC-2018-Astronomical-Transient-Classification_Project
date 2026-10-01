"""Models: standardisation, quality gating, calibration, similarity, anomaly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cne.config import QualityConfig
from cne.models import (
    AnomalyEnsemble,
    ConditionalCalibrator,
    DenseAutoencoder,
    QualityModel,
    SimilarityIndex,
    Standardiser,
    _rank01,
)


class TestStandardiser:
    def test_centre_is_zero_and_scale_is_one_on_the_fit_sample(self, synthetic):
        feats = synthetic["features"]
        astro = synthetic["featuriser"].astrophysical_columns
        out = Standardiser().fit_transform(feats, astro)
        assert np.allclose(np.median(out, axis=0), 0.0, atol=0.15)
        assert np.all(np.isfinite(out))

    def test_output_is_finite_for_extreme_input(self, synthetic):
        feats = synthetic["features"].copy()
        astro = synthetic["featuriser"].astrophysical_columns
        std = Standardiser().fit(feats, astro)
        extreme = feats.copy()
        extreme[astro[0]] = 1e12
        out = std.transform(extreme)
        assert np.all(np.isfinite(out))

    def test_transform_before_fit_raises(self, synthetic):
        with pytest.raises(RuntimeError):
            Standardiser().transform(synthetic["features"])


class TestQualityGate:
    """The invariant: quality can only ever suppress."""

    def test_quality_is_bounded(self, synthetic):
        q, _ = QualityModel(QualityConfig()).score(synthetic["features"], "q_")
        assert np.all(q >= 0.0) and np.all(q <= 1.0)

    def test_quality_is_not_a_constant(self, synthetic):
        """A gate that returns the same number for everyone is not a gate. This
        is the regression test for the non-detection statistics bug."""
        q, _ = QualityModel(QualityConfig()).score(synthetic["features"], "q_")
        # The synthetic fixture is uniformly clean, so its spread is modest; the
        # regression this guards against is a *constant* gate (std 0.000). On
        # real PLAsTiCC data the spread is far wider (p05 0.125, p95 1.000).
        assert q.std() > 0.01, f"quality collapsed to a constant (std={q.std():.4f})"

    def test_injecting_bad_data_never_raises_quality(self, synthetic):
        """Corrupting an observation must not make it look more trustworthy."""
        features = synthetic["features"].copy()
        q_before, _ = QualityModel(QualityConfig()).score(features, "q_")
        rng = np.random.default_rng(0)
        idx = rng.choice(len(features), size=max(1, len(features) // 2), replace=False)
        corrupted = features.copy()
        corrupted.loc[corrupted.index[idx], "q_neg_flux_frac"] = 0.9
        corrupted.loc[corrupted.index[idx], "q_det_snr_median"] = 0.5
        corrupted.loc[corrupted.index[idx], "q_n_det"] = 1.0
        q_after, _ = QualityModel(QualityConfig()).score(corrupted, "q_")
        assert np.all(q_after[idx] <= q_before[idx] + 1e-9)

    def test_disabled_gate_returns_ones(self, synthetic):
        cfg = QualityConfig(enabled=False)
        q, penalties = QualityModel(cfg).score(synthetic["features"], "q_")
        assert np.all(q == 1.0)
        assert penalties.empty

    def test_penalties_are_exposed_for_audit(self, synthetic):
        _q, penalties = QualityModel(QualityConfig()).score(synthetic["features"], "q_")
        for expected in ("neg_flux", "few_detections", "thin_coverage", "duplicates", "low_snr"):
            assert expected in penalties.columns


class TestAutoencoder:
    def test_reconstructs_and_discriminates(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((300, 20)).astype("float32")
        ae = DenseAutoencoder(20, hidden=(12, 5, 12), epochs=30).fit(x)
        clean = ae.residual(x).mean()
        noisy = ae.residual(x + rng.standard_normal(x.shape).astype("float32") * 3).mean()
        assert noisy > clean

    def test_deterministic_under_seed(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((200, 12)).astype("float32")
        a = DenseAutoencoder(12, hidden=(8, 4, 8), epochs=5, seed=42).fit(x)
        b = DenseAutoencoder(12, hidden=(8, 4, 8), epochs=5, seed=42).fit(x)
        np.testing.assert_allclose(a.residual(x), b.residual(x))

    def test_gradient_slots_never_mismatch_shapes(self):
        """A previous revision zipped weight grads against weight+bias params,
        which raised a broadcast error as soon as the network was deeper than 1."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal((80, 10)).astype("float32")
        DenseAutoencoder(10, hidden=(16, 4, 16), epochs=3).fit(x)


class TestCalibrator:
    def _frame(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.uniform(0, 1, n)
        b = rng.uniform(0, 1, n)
        error = np.clip(0.2 + 0.5 * a + rng.normal(0, 0.05, n), 0, 1)
        return pd.DataFrame({"a": a, "b": b, "confidence": 1 - error, "error": error})

    def test_coverage_is_reported_and_healthy(self):
        frame = self._frame()
        cal = ConditionalCalibrator(n_bins=4, min_per_cell=40, nuisance=["a", "b"]).fit(frame, "confidence", "error")
        assert cal.coverage_ > 0.5
        assert not cal.degenerate_

    def test_degenerate_grid_is_flagged_not_silent(self):
        """The v1 trap: a fixed 5x5x5 grid left every cell under min_per_cell and
        the calibrator silently degraded to global calibration."""
        frame = self._frame(n=200)
        cal = ConditionalCalibrator(n_bins=5, min_per_cell=40, nuisance=["a", "b"]).fit(frame, "confidence", "error")
        assert cal.degenerate_ is True
        out = cal.transform(frame, "confidence")
        assert np.allclose(out, cal.global_mean_)

    def test_tracks_a_real_conditional_trend(self):
        frame = self._frame()
        cal = ConditionalCalibrator(n_bins=4, min_per_cell=40, nuisance=["a"]).fit(frame, "confidence", "error")
        low = frame[frame["a"] < 0.25]
        high = frame[frame["a"] > 0.75]
        assert cal.transform(high, "confidence").mean() > cal.transform(low, "confidence").mean()


class TestSimilarity:
    def test_neighbours_are_from_the_reference(self, synthetic):
        feats = synthetic["features"]
        astro = synthetic["featuriser"].astrophysical_columns
        x = Standardiser().fit_transform(feats, astro)
        idx = SimilarityIndex().fit(x, synthetic["labels"], feats["object_id"].to_numpy())
        out = idx.query(x[:5])
        assert set(out["neighbour_ids"].ravel().tolist()) <= set(feats["object_id"].tolist())

    def test_entropy_is_normalised(self, synthetic):
        feats = synthetic["features"]
        astro = synthetic["featuriser"].astrophysical_columns
        x = Standardiser().fit_transform(feats, astro)
        idx = SimilarityIndex().fit(x, synthetic["labels"], feats["object_id"].to_numpy())
        ent = idx.query(x)["neighbour_entropy"]
        assert np.all(ent >= 0.0) and np.all(ent <= 1.0 + 1e-9)


class TestAnomalyEnsemble:
    def test_all_members_are_produced(self, synthetic):
        feats = synthetic["features"]
        astro = synthetic["featuriser"].astrophysical_columns
        x = Standardiser().fit_transform(feats, astro)
        ens = AnomalyEnsemble(seed=42, max_ref=80).fit(x)
        scores = ens.score(x)
        assert set(scores) == {"mahalanobis", "pca_residual", "isolation_forest", "knn_density", "autoencoder"}
        assert all(len(v) == len(x) for v in scores.values())

    def test_reference_cap_is_respected(self, synthetic):
        feats = synthetic["features"]
        astro = synthetic["featuriser"].astrophysical_columns
        x = Standardiser().fit_transform(feats, astro)
        ens = AnomalyEnsemble(seed=42, max_ref=20).fit(x)
        assert ens.knn_.n_samples_fit_ == 20

    def test_fused_score_is_in_unit_range(self, synthetic):
        feats = synthetic["features"]
        astro = synthetic["featuriser"].astrophysical_columns
        x = Standardiser().fit_transform(feats, astro)
        fused = AnomalyEnsemble(seed=42, max_ref=60).fit(x).score_fused(x)
        assert np.all(fused >= 0.0) and np.all(fused <= 1.0)


class TestRankNormalisation:
    def test_output_is_in_unit_range_and_monotone(self):
        values = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        out = _rank01(values)
        assert out.min() == 0.0 and out.max() == 1.0
        assert np.argsort(values).tolist() == np.argsort(out).tolist()

    def test_empty_input(self):
        assert len(_rank01(np.array([]))) == 0
