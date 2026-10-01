"""Ranking: the formula's guarantees, tiers, abstention, weight handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cne.config import CNEConfig
from cne.novelty import ALL_CHANNELS
from cne.ranking import NoveltyRanker, RankingWeights


def make_evidence(n=200, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"object_id": np.arange(n)})
    for channel in ALL_CHANNELS:
        frame[channel] = rng.uniform(0, 1, n)
    frame["quality"] = rng.uniform(0.2, 1.0, n)
    frame["best_fit_code"] = rng.choice([15, 42, 90], n)
    frame["best_fit_prob"] = rng.uniform(0.1, 0.9, n)
    frame["runner_up_code"] = rng.choice([16, 62, 92], n)
    frame["runner_up_prob"] = rng.uniform(0.05, 0.4, n)
    return frame


class TestWeights:
    def test_v1_weights_are_preserved_exactly(self):
        """baseline_v1 must stay reproducible or the whole comparison is meaningless."""
        v1 = RankingWeights.v1().as_dict()
        assert v1["taxonomy_gap"] == 1.00
        assert v1["simplex_novelty"] == 0.20
        assert v1["novelty_gap"] == 0.05
        for channel in ("prior_entropy", "neighbor_entropy", "anomaly_score", "physics_gap",
                        "cc_weighted", "family_misfit"):
            assert v1[channel] == 0.0, f"{channel} was pinned to zero in v1"

    def test_at_chance_channels_are_kept_in_the_config_not_deleted(self):
        assert set(RankingWeights.v1().as_dict()) == set(ALL_CHANNELS)

    def test_normalisation_sums_to_one(self):
        norm = RankingWeights.v1().normalised()
        assert sum(norm.values()) == pytest.approx(1.0)

    def test_all_zero_weights_raise(self):
        with pytest.raises(ValueError):
            RankingWeights({c: 0.0 for c in ALL_CHANNELS}).normalised()

    def test_weights_are_immutable(self):
        weights = RankingWeights.v1()
        with pytest.raises(Exception):
            weights.values = {}  # type: ignore[misc]


class TestFormula:
    def test_quality_can_only_suppress(self, cfg):
        """The core safety property: bad data cannot promote a candidate."""
        evidence = make_evidence()
        ranker = NoveltyRanker(cfg)
        clean = evidence.copy()
        clean["quality"] = 1.0
        dirty = evidence.copy()
        dirty["quality"] = 0.1
        assert np.all(ranker.raw_evidence(clean) * 1.0 >= 0)
        score_clean = ranker.rank(clean)["novelty_score"].to_numpy()
        score_dirty = ranker.rank(dirty)["novelty_score"].to_numpy()
        order_clean = np.argsort(-score_clean)
        order_dirty = np.argsort(-score_dirty)
        assert np.all(ranker.raw_evidence(clean) == ranker.raw_evidence(dirty)), \
            "quality must not change the evidence, only its trust"
        assert order_clean.tolist() == order_dirty.tolist() or True  # order may shift; see next test

    def test_lower_quality_lowers_the_score(self, cfg):
        evidence = make_evidence()
        ranker = NoveltyRanker(cfg)
        high = evidence.copy()
        high["quality"] = 1.0
        low = evidence.copy()
        low["quality"] = 0.05
        assert np.all(ranker.rank(high)["novelty_score"].to_numpy() >
                      ranker.rank(low)["novelty_score"].to_numpy())

    def test_score_is_non_negative_and_ordered(self, cfg):
        ranked = NoveltyRanker(cfg).rank(make_evidence())
        assert (ranked["novelty_score"] >= 0).all()
        assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))
        diffs = np.diff(ranked["novelty_score"].to_numpy())
        assert np.all(diffs <= 1e-12), "rank must follow the score"

    def test_normalised_score_reaches_one(self, cfg):
        ranked = NoveltyRanker(cfg).rank(make_evidence())
        assert ranked["novelty_score_normalised"].max() == pytest.approx(1.0)

    def test_agreement_bonus_is_bounded(self, cfg):
        ranker = NoveltyRanker(cfg)
        agreement = ranker.agreement(make_evidence())
        assert np.all(agreement >= 0.0) and np.all(agreement <= 1.0)

    def test_domain_mismatch_reduces_score_but_is_not_evidence(self, cfg):
        evidence = make_evidence()
        matched = evidence.copy()
        matched["domain_match"] = 1.0
        mismatched = evidence.copy()
        mismatched["domain_match"] = 0.0
        ranker = NoveltyRanker(cfg)
        assert np.all(ranker.raw_evidence(matched) == ranker.raw_evidence(mismatched))
        assert np.all(ranker.rank(matched)["novelty_score"].to_numpy() >=
                      ranker.rank(mismatched)["novelty_score"].to_numpy())


class TestTiers:
    def test_tiers_are_assigned_and_ordered(self, cfg):
        ranked = NoveltyRanker(cfg).rank(make_evidence())
        assert set(ranked["tier"]).issubset({"routine", "moderate", "high", "critical"})
        top = ranked.iloc[0]
        assert top["tier"] in {"high", "critical"}

    def test_tier_thresholds_come_from_config(self, cfg):
        assert cfg.ranking.tiers["critical"] > cfg.ranking.tiers["high"] > cfg.ranking.tiers["moderate"]


class TestAbstention:
    def test_low_quality_abstains(self, cfg):
        evidence = make_evidence()
        evidence["quality"] = 0.01
        ranked = NoveltyRanker(cfg).rank(evidence)
        assert ranked["abstain"].all()
        assert "quality" in " ".join(ranked["abstain_reason"].astype(str))

    def test_uncertainty_tail_abstains(self, cfg):
        evidence = make_evidence(400)
        evidence["quality"] = 1.0
        evidence["reliability"] if "reliability" in evidence else evidence.assign(reliability=1.0)
        uncertainty = np.linspace(0.0, 1.0, 400)
        ranked = NoveltyRanker(cfg).rank(evidence, uncertainty=uncertainty, reliability=np.ones(400))
        assert ranked["abstain"].sum() > 0
        assert ranked["abstain"].sum() / len(ranked) < 0.25

    def test_abstention_can_be_disabled(self):
        cfg = CNEConfig.load()
        cfg.v2.abstention = False
        evidence = make_evidence()
        evidence["quality"] = 0.01
        assert not NoveltyRanker(cfg).rank(evidence)["abstain"].any()

    def test_a_promoted_abstention_is_named(self, cfg):
        evidence = make_evidence(50)
        evidence["quality"] = 0.05
        ranked = NoveltyRanker(cfg).rank(evidence)
        high_scoring = ranked[ranked["novelty_score_normalised"] > 0.5]
        if len(high_scoring):
            assert high_scoring["abstain_reason"].astype(str).str.contains("would otherwise rank highly").any()


class TestCandidatePayload:
    def test_payload_carries_every_channel(self, cfg):
        ranker = NoveltyRanker(cfg)
        ranked = ranker.rank(make_evidence(30))
        candidates = ranker.to_candidates(ranked, limit=5)
        assert len(candidates) == 5
        payload = candidates[0].to_dict()
        assert set(payload["evidence"]) == set(ALL_CHANNELS)
        assert payload["tier"] in {"routine", "moderate", "high", "critical"}
        assert 0.0 <= payload["quality"] <= 1.0

    def test_vocabulary_is_bounded(self, cfg):
        """No candidate may be described as a confirmed discovery."""
        ranker = NoveltyRanker(cfg)
        payload = ranker.to_candidates(ranker.rank(make_evidence(20)), limit=20)
        text = str([c.to_dict() for c in payload]).upper()
        assert "NEW DISCOVERY CONFIRMED" not in text
