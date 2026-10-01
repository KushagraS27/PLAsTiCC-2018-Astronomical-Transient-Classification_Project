"""Base-rate stress testing (CNE v2, prompt 04).

The benchmark stream is enriched: ~11% of scored objects are novel, versus a
realistic survey rate nearer 0.4%. Absolute precision@K is a function of base
rate, so a precision number measured on an enriched stream cannot be quoted as a
deployment number. This module re-expresses the same ranking under arbitrary
priors.

Method: importance reweighting, not resampling. Keep every positive at weight 1
and give each negative the weight

    w = [pi / (1 - pi)] * [(1 - pi_0) / pi_0]

which makes the weighted population have exactly prior ``pi`` while preserving the
ranking. This is exact and smooth, so the curves do not carry Monte-Carlo noise,
and at ``pi = pi_0`` the weighted metrics reduce to the ordinary ones (a property
covered by a test).

What to quote: **lift over random** and **novel events recovered per review
budget**. Both are prior-portable in a way that precision is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .evaluation import bootstrap_ci
from .logging import get_logger

log = get_logger("stress")


def negative_weight(base_rate: float, target_rate: float) -> float:
    """Weight for negative objects that moves the prior from ``base_rate`` to ``target_rate``."""
    if not 0 < base_rate < 1 or not 0 < target_rate < 1:
        raise ValueError("base_rate and target_rate must both lie in (0, 1)")
    return (target_rate / (1.0 - target_rate)) * ((1.0 - base_rate) / base_rate)


def sample_weights(y_true: np.ndarray, base_rate: float, target_rate: float) -> np.ndarray:
    y = np.asarray(y_true).astype(int)
    w = np.ones(len(y), dtype="float64")
    w[y == 0] = negative_weight(base_rate, target_rate)
    return w


# --------------------------------------------------------------------------- #
# weighted metrics
# --------------------------------------------------------------------------- #
def weighted_precision_at_k(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray, k: int) -> float:
    y = np.asarray(y_true).astype(int)
    w = np.asarray(weights, dtype="float64")
    k = min(int(k), len(scores))
    if k <= 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    denom = w[order].sum()
    if denom <= 0:
        return float("nan")
    return float((w[order] * y[order]).sum() / denom)


def weighted_recall_at_k(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray, k: int) -> float:
    """Recall is prior-invariant: it only depends on how many positives rank high."""
    y = np.asarray(y_true).astype(int)
    total = int(y.sum())
    if total == 0:
        return float("nan")
    k = min(int(k), len(scores))
    order = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    return float(y[order].sum() / total)


def weighted_average_precision(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    """AP under the reweighted prior (step interpolation, matching sklearn's convention)."""
    y = np.asarray(y_true).astype(int)
    w = np.asarray(weights, dtype="float64")
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    y_s, w_s = y[order], w[order]
    total = (w_s * y_s).sum()
    if total <= 0:
        return float("nan")
    cum_tp = np.cumsum(w_s * y_s)
    cum_all = np.cumsum(w_s)
    precision = cum_tp / np.clip(cum_all, 1e-12, None)
    recall = cum_tp / total
    # Only steps that add a positive contribute, per the standard AP definition.
    mask = y_s == 1
    if not mask.any():
        return float("nan")
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall[mask] - recall_prev[mask]) * precision[mask]))


def weighted_roc_auc(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    """Weighted ROC-AUC via a single descending sweep over score-tied blocks.

    Implements ``AUC = Σ_pairs w_i⁺ w_j⁻ [s_i > s_j] / (Σw⁺ Σw⁻)`` with ties
    counting half, matching sklearn's convention. With uniform weights this
    reduces exactly to the unweighted Mann-Whitney statistic, which the test
    suite asserts against ``roc_auc``.
    """
    y = np.asarray(y_true).astype(int)
    w = np.asarray(weights, dtype="float64")
    s_ = np.asarray(scores, dtype="float64")
    pos = y == 1
    if pos.all() or (~pos).all() or len(y) < 2:
        return float("nan")
    total_pos = float(w[pos].sum())
    total_neg = float(w[~pos].sum())
    if total_pos <= 0.0 or total_neg <= 0.0:
        return float("nan")
    order = np.argsort(-s_, kind="mergesort")
    y_s, w_s, s_s = y[order], w[order], s_[order]
    cum_pos = 0.0
    numerator = 0.0
    i, n = 0, len(y_s)
    while i < n:
        j = i
        while j < n and s_s[j] == s_s[i]:
            j += 1
        block_pos = float(w_s[i:j][y_s[i:j] == 1].sum())
        block_neg = float(w_s[i:j][y_s[i:j] == 0].sum())
        if block_neg > 0.0:
            # every positive above this block beats all of it; the positives inside
            # the block tie with it and count half.
            numerator += block_neg * (cum_pos + 0.5 * block_pos)
        cum_pos += block_pos
        i = j
    return numerator / (total_pos * total_neg)


# --------------------------------------------------------------------------- #
# stress table
# --------------------------------------------------------------------------- #
@dataclass
class BaseRateStressResult:
    base_rate_observed: float
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"base_rate_observed": round(self.base_rate_observed, 6), "priors": self.rows}

    def summary_table(self) -> str:
        header = f"{'prior':>9} {'AUC':>7} {'AP':>7} {'P@50':>7} {'P@100':>7} {'lift@100':>9} {'R@200':>7}"
        lines = [header, "-" * len(header)]
        for row in self.rows:
            lines.append(
                f"{row['target_rate']:>9.4%} {row['roc_auc']:>7.3f} {row['average_precision']:>7.3f} "
                f"{row.get('P@50', float('nan')):>7.3f} {row.get('P@100', float('nan')):>7.3f} "
                f"{row.get('lift@100', float('nan')):>9.3f} {row.get('R@200', float('nan')):>7.3f}"
            )
        return "\n".join(lines)


def stress_test(y_true: np.ndarray, scores: np.ndarray, target_rates: Sequence[float] = (0.10, 0.01, 0.001, 0.0001),
                ks: Sequence[int] = (50, 100, 200), draws: int = 400, seed: int = 42) -> BaseRateStressResult:
    """Re-express one ranking under a ladder of deployment priors."""
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype="float64")
    pi0 = float(y.mean())
    result = BaseRateStressResult(base_rate_observed=pi0)
    for target in target_rates:
        target = float(target)
        if not 0 < target < 1 or target >= pi0:
            log.warning("skipping target prior %.4f (observed base rate is %.4f)", target, pi0)
            continue
        w = sample_weights(y, pi0, target)
        row: Dict[str, Any] = {
            "target_rate": target,
            "negative_weight": float(negative_weight(pi0, target)),
            "roc_auc": weighted_roc_auc(y, scores, w),
            "average_precision": weighted_average_precision(y, scores, w),
        }
        for k in ks:
            p = weighted_precision_at_k(y, scores, w, k)
            row[f"P@{k}"] = p
            row[f"lift@{k}"] = p / target if target > 0 else float("nan")
        for k in (200, 500):
            row[f"R@{k}"] = weighted_recall_at_k(y, scores, w, k)
        # Bootstrap the headline number at this prior.
        _, lo, hi = _bootstrap_weighted_ap(y, scores, pi0, target, draws=draws, seed=seed)
        row["average_precision_ci"] = {"low": lo, "high": hi}
        result.rows.append(row)
    return result


def _bootstrap_weighted_ap(y, scores, pi0, target, draws: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    point = weighted_average_precision(y, scores, sample_weights(y, pi0, target))
    values = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        val = weighted_average_precision(y[idx], scores[idx], sample_weights(y[idx], pi0, target))
        if np.isfinite(val):
            values.append(val)
    if not values:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def review_budget_curve(y_true: np.ndarray, scores: np.ndarray, target_rate: float,
                        budgets: Sequence[int] = (10, 50, 100, 500, 1000)) -> List[Dict[str, Any]]:
    """What an operator actually gets: novel events recovered per review budget.

    Reported per unit of stream, so it can be multiplied by a real nightly alert
    volume without re-running anything.
    """
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype="float64")
    pi0 = float(y.mean())
    w = sample_weights(y, pi0, target_rate)
    order = np.argsort(-scores, kind="mergesort")
    rows = []
    for budget in budgets:
        k = min(int(budget), len(scores))
        sel = order[:k]
        mass = w[sel].sum()
        pos_mass = (w[sel] * y[sel]).sum()
        rows.append({
            "budget": int(budget),
            "expected_reviews": float(mass),
            "expected_novel_recovered": float(pos_mass),
            "precision": float(pos_mass / mass) if mass > 0 else float("nan"),
            "lift": float((pos_mass / mass) / target_rate) if mass > 0 and target_rate > 0 else float("nan"),
        })
    return rows
