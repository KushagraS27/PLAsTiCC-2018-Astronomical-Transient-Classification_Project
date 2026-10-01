"""Evaluation metrics: hand-computed answers, bootstrap determinism, honesty."""

from __future__ import annotations

import numpy as np
import pytest

from cne.evaluation import (
    average_precision,
    bootstrap_ci,
    evaluate_ranking,
    lift_at_k,
    per_class_metrics,
    precision_at_k,
    recall_at_k,
    roc_auc,
)
from cne.stress import (
    negative_weight,
    review_budget_curve,
    sample_weights,
    stress_test,
    weighted_average_precision,
    weighted_precision_at_k,
    weighted_recall_at_k,
    weighted_roc_auc,
)

# A hand-built queue: ranks 1..8, positives at ranks 1, 3, 6.
Y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
S = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])


class TestPointMetrics:
    def test_precision_at_k(self):
        assert precision_at_k(Y, S, 1) == pytest.approx(1.0)
        assert precision_at_k(Y, S, 2) == pytest.approx(0.5)
        assert precision_at_k(Y, S, 3) == pytest.approx(2 / 3)
        assert precision_at_k(Y, S, 8) == pytest.approx(3 / 8)

    def test_recall_at_k(self):
        assert recall_at_k(Y, S, 3) == pytest.approx(2 / 3)
        assert recall_at_k(Y, S, 8) == pytest.approx(1.0)

    def test_lift_over_random(self):
        # base rate is 3/8, precision@3 is 2/3 -> lift 16/9
        assert lift_at_k(Y, S, 3) == pytest.approx((2 / 3) / (3 / 8))

    def test_average_precision_matches_the_textbook_value(self):
        # AP = (1/1 * 1/1 + 2/3 * 1/3 + 3/6 * 1/6) / 3
        expected = (1.0 + (2 / 3) + 0.5) / 3
        assert average_precision(Y, S) == pytest.approx(expected, abs=1e-9)

    def test_roc_auc_matches_mann_whitney(self):
        # positives at scores 8,6,3; negatives at 7,5,4,2,1
        # concordant pairs: 8 beats all 5 negatives, 6 beats 5,4,2,1 (4),
        # 3 beats 2,1 (2) => 11 of 15 pairs
        assert roc_auc(Y, S) == pytest.approx(11 / 15)

    def test_perfect_ranking(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([4.0, 3.0, 2.0, 1.0])
        assert roc_auc(y, s) == 1.0
        assert precision_at_k(y, s, 2) == 1.0

    def test_reversed_ranking(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([1.0, 2.0, 3.0, 4.0])
        assert roc_auc(y, s) == 0.0

    def test_degenerate_inputs_are_nan_not_a_crash(self):
        assert np.isnan(roc_auc(np.zeros(5), S[:5]))
        assert np.isnan(average_precision(np.zeros(5), S[:5]))
        assert np.isnan(recall_at_k(np.zeros(5), S[:5], 3))


class TestBootstrap:
    def test_point_estimate_is_returned_with_the_interval(self):
        y = np.array([1] * 40 + [0] * 160)
        s = np.concatenate([np.random.default_rng(0).uniform(0.5, 1.0, 40),
                            np.random.default_rng(1).uniform(0.0, 0.7, 160)])
        point, lo, hi = bootstrap_ci(y, s, roc_auc, draws=200, seed=42)
        assert lo <= point <= hi

    def test_deterministic_under_seed(self):
        y = np.array([1] * 30 + [0] * 70)
        s = np.random.default_rng(3).uniform(0, 1, 100)
        a = bootstrap_ci(y, s, roc_auc, draws=100, seed=7)
        b = bootstrap_ci(y, s, roc_auc, draws=100, seed=7)
        assert a == b

    def test_single_class_resample_is_skipped_not_fatal(self):
        y = np.array([1, 0, 0, 0])
        s = np.array([4.0, 3.0, 2.0, 1.0])
        point, lo, hi = bootstrap_ci(y, s, roc_auc, draws=50, seed=1)
        assert np.isfinite(point)


class TestPerClass:
    def test_small_n_is_flagged(self):
        codes = np.array([1] * 5 + [2] * 200)
        scores = np.arange(205, dtype=float)
        table = per_class_metrics(codes, scores, k=50, min_n=30)
        assert bool(table.loc[table["class_code"] == 1, "low_n_warning"].iloc[0]) is True
        assert bool(table.loc[table["class_code"] == 2, "low_n_warning"].iloc[0]) is False

    def test_enrichment_is_relative_to_the_population_share(self):
        codes = np.array([1] * 50 + [2] * 50)
        scores = np.concatenate([np.ones(50), np.zeros(50)])  # class 1 entirely on top
        table = per_class_metrics(codes, scores, k=50)
        row = table.loc[table["class_code"] == 1].iloc[0]
        assert row[f"recall@50"] == pytest.approx(1.0)
        assert row["enrichment"] == pytest.approx(2.0)

    def test_held_out_flag_is_set(self):
        from cne.taxonomy import WITHHELD_FROM_PRIOR

        novel = sorted(WITHHELD_FROM_PRIOR)[0]
        codes = np.array([novel] * 10 + [90] * 90)
        table = per_class_metrics(codes, np.arange(100, dtype=float), k=20, novel_codes=WITHHELD_FROM_PRIOR)
        assert bool(table.loc[table["class_code"] == novel, "held_out"].iloc[0]) is True


class TestBaseRateStress:
    def test_weighted_metrics_reduce_to_the_unweighted_ones_at_the_observed_prior(self):
        """The mathematical consistency check the upgrade document demands."""
        rng = np.random.default_rng(0)
        y = (rng.random(2000) < 0.1).astype(int)
        s = np.where(y == 1, rng.uniform(0.5, 1.0, 2000), rng.uniform(0.0, 0.8, 2000))
        pi0 = y.mean()
        w = sample_weights(y, pi0, pi0)
        assert np.allclose(w, 1.0)
        assert weighted_precision_at_k(y, s, w, 100) == pytest.approx(precision_at_k(y, s, 100), abs=1e-9)
        assert weighted_average_precision(y, s, w) == pytest.approx(average_precision(y, s), abs=1e-6)
        assert weighted_roc_auc(y, s, w) == pytest.approx(roc_auc(y, s), abs=1e-6)

    def test_recall_is_prior_invariant(self):
        rng = np.random.default_rng(1)
        y = (rng.random(500) < 0.2).astype(int)
        s = rng.uniform(0, 1, 500)
        w = sample_weights(y, y.mean(), 0.01)
        assert weighted_recall_at_k(y, s, w, 100) == pytest.approx(recall_at_k(y, s, 100))

    def test_precision_falls_as_the_prior_falls(self):
        """The whole point: enriched-stream precision is not a deployment number."""
        rng = np.random.default_rng(2)
        y = (rng.random(3000) < 0.2).astype(int)
        s = np.where(y == 1, rng.uniform(0.5, 1.0, 3000), rng.uniform(0.0, 0.9, 3000))
        precisions = []
        for target in (0.2, 0.05, 0.01, 0.002):
            w = sample_weights(y, y.mean(), target)
            precisions.append(weighted_precision_at_k(y, s, w, 100))
        assert precisions == sorted(precisions, reverse=True), precisions

    def test_negative_weight_formula(self):
        assert negative_weight(0.1, 0.01) == pytest.approx((0.01 / 0.99) * (0.9 / 0.1))
        with pytest.raises(ValueError):
            negative_weight(0.0, 0.5)

    def test_stress_table_covers_every_requested_prior(self):
        rng = np.random.default_rng(3)
        y = (rng.random(4000) < 0.15).astype(int)
        s = np.where(y == 1, rng.uniform(0.4, 1.0, 4000), rng.uniform(0.0, 0.9, 4000))
        result = stress_test(y, s, target_rates=(0.10, 0.01, 0.001), draws=20, seed=42)
        assert [r["target_rate"] for r in result.rows] == [0.10, 0.01, 0.001]
        for row in result.rows:
            assert row["lift@100"] == pytest.approx(row["P@100"] / row["target_rate"])

    def test_a_prior_above_the_observed_base_rate_is_skipped(self):
        y = np.array([1] * 10 + [0] * 90)
        s = np.arange(100, dtype=float)
        result = stress_test(y, s, target_rates=(0.5,), draws=10)
        assert result.rows == []

    def test_review_budget_curve_is_monotone_in_budget(self):
        rng = np.random.default_rng(4)
        y = (rng.random(2000) < 0.1).astype(int)
        s = np.where(y == 1, rng.uniform(0.5, 1.0, 2000), rng.uniform(0.0, 0.8, 2000))
        rows = review_budget_curve(y, s, target_rate=0.004, budgets=(10, 100, 1000))
        recovered = [r["expected_novel_recovered"] for r in rows]
        assert recovered == sorted(recovered)


class TestAggregate:
    def test_evaluate_ranking_reports_the_base_rate(self):
        metrics = evaluate_ranking(Y, S, draws=20).as_dict()
        assert metrics["base_rate"] == pytest.approx(3 / 8)
        assert metrics["n_novel"] == 3
        assert metrics["n_objects"] == 8

    def test_confidence_intervals_bracket_the_point_estimate(self):
        rng = np.random.default_rng(5)
        y = (rng.random(1000) < 0.2).astype(int)
        s = np.where(y == 1, rng.uniform(0.4, 1.0, 1000), rng.uniform(0.0, 0.9, 1000))
        metrics = evaluate_ranking(y, s, draws=150).as_dict()
        for key in ("roc_auc", "average_precision"):
            assert metrics["ci"][key]["low"] <= metrics[key] <= metrics["ci"][key]["high"]
