"""Evidence channels and the engine: definitions, bounds, chunk bit-identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cne.novelty import ALL_CHANNELS, CosmicNoveltyEngine, KnownPhysicsPrior, NoveltyEvidence
from cne.taxonomy import KNOWN_CLASS_CODES


@pytest.fixture(scope="module")
def engine(synthetic):
    """An engine fitted on the synthetic population (fast, no LightGBM waits)."""
    feats, labels = synthetic["features"], synthetic["labels"]
    known = np.isin(labels, sorted(KNOWN_CLASS_CODES))
    eng = CosmicNoveltyEngine(seed=42)
    eng.fit(feats[known].reset_index(drop=True), labels[known])
    return eng


class TestChannelDefinitions:
    def test_taxonomy_gap_is_one_minus_top_probability(self):
        prior = KnownPhysicsPrior()
        prior.centroids_ = np.eye(3)
        ev = NoveltyEvidence(prior)
        proba = np.array([[0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3]])
        out = ev.from_probabilities(proba)
        assert out["taxonomy_gap"][0] == pytest.approx(0.3)
        assert out["taxonomy_gap"][1] == pytest.approx(2 / 3)

    def test_entropy_is_zero_for_a_certain_prediction(self):
        prior = KnownPhysicsPrior()
        prior.centroids_ = np.eye(3)
        ev = NoveltyEvidence(prior)
        out = ev.from_probabilities(np.array([[1.0, 0.0, 0.0]]))
        assert out["prior_entropy"][0] == pytest.approx(0.0, abs=1e-9)

    def test_entropy_is_one_for_a_uniform_prediction(self):
        prior = KnownPhysicsPrior()
        prior.centroids_ = np.eye(4)
        ev = NoveltyEvidence(prior)
        out = ev.from_probabilities(np.full((1, 4), 0.25))
        assert out["prior_entropy"][0] == pytest.approx(1.0, abs=1e-9)

    def test_novelty_gap_is_one_minus_the_margin(self):
        prior = KnownPhysicsPrior()
        prior.centroids_ = np.eye(3)
        ev = NoveltyEvidence(prior)
        out = ev.from_probabilities(np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]]))
        assert out["novelty_gap"][0] == pytest.approx(1 - 0.85)
        assert out["novelty_gap"][1] == pytest.approx(1 - 0.05)

    def test_simplex_novelty_is_zero_at_a_known_centroid(self):
        prior = KnownPhysicsPrior()
        prior.centroids_ = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ev = NoveltyEvidence(prior)
        out = ev.from_probabilities(np.array([[1.0, 0.0, 0.0]]))
        assert out["simplex_novelty"][0] == pytest.approx(0.0, abs=1e-9)

    def test_probabilities_are_renormalised(self):
        prior = KnownPhysicsPrior()
        prior.centroids_ = np.eye(2)
        ev = NoveltyEvidence(prior)
        out = ev.from_probabilities(np.array([[2.0, 2.0]]))  # unnormalised on purpose
        assert np.isfinite(out["taxonomy_gap"][0])
        assert 0.0 <= out["taxonomy_gap"][0] <= 1.0

    def test_all_channels_are_bounded(self, engine, synthetic):
        evidence = engine.score(synthetic["features"].head(60))
        for channel in ALL_CHANNELS:
            values = evidence[channel].to_numpy(dtype="float64")
            assert np.all(np.isfinite(values)), f"{channel} produced non-finite values"
            assert values.min() >= -1e-9 and values.max() <= 1.0 + 1e-9, f"{channel} outside [0,1]"


class TestEngine:
    def test_quality_features_are_excluded_from_the_model_input(self, engine, synthetic):
        quality = set(synthetic["featuriser"].quality_columns)
        assert quality.isdisjoint(set(engine.feature_names_))

    def test_scoring_in_chunks_matches_unchunked_scoring(self, engine, synthetic):
        """Chunking may change peak memory; it must not change the answer.

        Exact bit-identity is NOT achievable and is not claimed: the autoencoder
        residual differs between batch sizes at the 3e-14 level (floating-point
        reduction order), and when such a value lands on a reference grid boundary
        the quantile lookup moves by half a grid step. The bound below is one grid
        step, 1/N_ref. What this test really pins is the *algorithmic* dependence:
        before the reference grids were frozen at fit time, ``physics_gap``,
        ``cc_weighted`` and ``anomaly_score`` were rank-normalised inside the
        batch, so a chunk of 30 and a chunk of 400 disagreed by up to 0.26 - two
        orders of magnitude larger than the tolerance here.
        """
        features = synthetic["features"]
        full = engine.score(features, chunk=10_000)
        parts = [engine.score(features.iloc[i:i + 30].reset_index(drop=True), chunk=10_000)
                 for i in range(0, len(features), 30)]
        chunked = pd.concat(parts, ignore_index=True)
        tol = 1.0 / len(features)  # one quantile grid step
        numeric = [c for c in full.columns if full[c].dtype.kind in "fc"]
        worst = max(((c, float(np.nanmax(np.abs(full[c].to_numpy() - chunked[c].to_numpy()))))
                     for c in numeric), key=lambda t: t[1])
        assert worst[1] <= tol, f"{worst[0]} differs by {worst[1]:.3e} > {tol:.3e}"

    def test_weighted_reference_produces_a_valid_prior(self, synthetic):
        """Density-ratio weighting (domain matching) must not break the prior."""
        feats, labels = synthetic["features"], synthetic["labels"]
        known = np.isin(labels, sorted(KNOWN_CLASS_CODES))
        reference = feats[known].reset_index(drop=True)
        weights = np.linspace(0.5, 2.0, len(reference))
        prior = KnownPhysicsPrior(seed=42).fit(reference, labels[known],
                                               list(feats.columns[1:]), sample_weight=weights)
        proba = prior.probabilities(reference.head(10))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert 0.0 < prior.summary().accuracy <= 1.0
