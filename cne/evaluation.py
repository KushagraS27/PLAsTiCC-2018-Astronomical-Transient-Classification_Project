"""Evaluation: the metrics that actually matter for a ranked discovery queue.

AI-05 is *unsupervised discovery*, so "accuracy" is undefined. The primary
metrics are precision at a review budget, held-out-population recall, ROC-AUC on
held-out rare classes, lift over random, and early-detection lead time.

Every metric here carries its base rate. Absolute precision is a function of base
rate, so a precision number without its prior is not comparable across runs - the
portable figure is lift over random.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .logging import get_logger
from .taxonomy import name_of

log = get_logger("evaluation")


# --------------------------------------------------------------------------- #
# point metrics
# --------------------------------------------------------------------------- #
def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    k = min(int(k), len(scores))
    if k <= 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    return float(y_true[order].mean())


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    total = int(y_true.sum())
    if total == 0:
        return float("nan")
    k = min(int(k), len(scores))
    order = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    return float(y_true[order].sum() / total)


def lift_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    base = float(np.mean(np.asarray(y_true)))
    if base <= 0:
        return float("nan")
    return precision_at_k(y_true, scores, k) / base


def top_k_ids(object_ids: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-np.asarray(scores), kind="mergesort")[: int(k)]
    return np.asarray(object_ids)[order]


# --------------------------------------------------------------------------- #
# uncertainty
# --------------------------------------------------------------------------- #
def bootstrap_ci(y_true: np.ndarray, scores: np.ndarray, metric, k: Optional[int] = None,
                 draws: int = 1000, seed: int = 42, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Percentile bootstrap CI over OBJECTS (not over metrics).

    Resampling objects is the correct unit here: the objects are the independent
    draws, and precision@K depends on the composition of the resampled queue.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype="float64")
    point = metric(y_true, scores, k) if k is not None else metric(y_true, scores)
    n = len(y_true)
    values: List[float] = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            val = metric(y_true[idx], scores[idx], k) if k is not None else metric(y_true[idx], scores[idx])
        except Exception:  # pragma: no cover - degenerate resample
            continue
        if np.isfinite(val):
            values.append(float(val))
    if not values:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
@dataclass
class RankingMetrics:
    n_objects: int
    n_novel: int
    base_rate: float
    roc_auc: float
    average_precision: float
    precision_at_k: Dict[str, float]
    recall_at_k: Dict[str, float]
    lift_at_k: Dict[str, float]
    ci: Dict[str, Dict[str, float]]

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["base_rate"] = round(self.base_rate, 4)
        return out


def evaluate_ranking(y_true: np.ndarray, scores: np.ndarray, precision_k: Sequence[int] = (10, 20, 50, 100),
                     recall_k: Sequence[int] = (200, 500), draws: int = 1000, seed: int = 42) -> RankingMetrics:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype="float64")
    pk = {f"P@{k}": precision_at_k(y_true, scores, k) for k in precision_k}
    rk = {f"R@{k}": recall_at_k(y_true, scores, k) for k in recall_k}
    lk = {f"lift@{k}": lift_at_k(y_true, scores, k) for k in precision_k}
    ci: Dict[str, Dict[str, float]] = {}
    _, auc_lo, auc_hi = bootstrap_ci(y_true, scores, roc_auc, draws=draws, seed=seed)
    ci["roc_auc"] = {"low": auc_lo, "high": auc_hi}
    _, ap_lo, ap_hi = bootstrap_ci(y_true, scores, average_precision, draws=draws, seed=seed)
    ci["average_precision"] = {"low": ap_lo, "high": ap_hi}
    for k in precision_k:
        _, lo, hi = bootstrap_ci(y_true, scores, precision_at_k, k=k, draws=draws, seed=seed)
        ci[f"P@{k}"] = {"low": lo, "high": hi}
    return RankingMetrics(
        n_objects=int(len(y_true)),
        n_novel=int(y_true.sum()),
        base_rate=float(y_true.mean()) if len(y_true) else 0.0,
        roc_auc=roc_auc(y_true, scores),
        average_precision=average_precision(y_true, scores),
        precision_at_k=pk,
        recall_at_k=rk,
        lift_at_k=lk,
        ci=ci,
    )


def per_class_metrics(codes: np.ndarray, scores: np.ndarray, k: int = 200,
                      min_n: int = 30, novel_codes: Sequence[int] = ()) -> pd.DataFrame:
    """Recall@K and enrichment per population, with an explicit small-n warning."""
    codes = np.asarray(codes)
    scores = np.asarray(scores, dtype="float64")
    order = np.argsort(-scores, kind="mergesort")[: int(k)]
    top_codes = codes[order]
    base_rate_per_class = pd.Series(codes).value_counts(normalize=True)
    rows = []
    for code in sorted(set(codes.tolist())):
        n = int((codes == code).sum())
        in_top = int((top_codes == code).sum())
        recall = in_top / n if n else float("nan")
        expected = float(base_rate_per_class.get(code, 0.0)) * k
        rows.append({
            "class_code": int(code),
            "class_name": name_of(int(code)),
            "n": n,
            f"in_top{k}": in_top,
            f"recall@{k}": recall,
            "enrichment": (in_top / expected) if expected > 0 else float("nan"),
            "held_out": int(code) in set(int(c) for c in novel_codes),
            "low_n_warning": n < min_n,
        })
    return pd.DataFrame(rows).sort_values(f"recall@{k}", ascending=False).reset_index(drop=True)


def false_positive_audit(ranked: pd.DataFrame, k: int = 100, quality_col: str = "quality") -> Dict[str, Any]:
    """Characterise what the queue gets wrong at the operating point that matters."""
    top = ranked.head(k)
    tiers = top["tier"].value_counts().to_dict() if "tier" in top.columns else {}
    return {
        "k": int(k),
        "n": int(len(top)),
        "mean_quality": float(top[quality_col].mean()) if quality_col in top.columns else None,
        "min_quality": float(top[quality_col].min()) if quality_col in top.columns else None,
        "n_low_quality": int((top[quality_col] < 0.5).sum()) if quality_col in top.columns else None,
        "mean_uncertainty": float(top["uncertainty"].mean()) if "uncertainty" in top.columns else None,
        "n_abstained": int(top["abstain"].sum()) if "abstain" in top.columns else None,
        "tiers": {str(a): int(b) for a, b in tiers.items()},
        "n_false_positives": int((~top["is_novel"].astype(bool)).sum()) if "is_novel" in top.columns else None,
    }


def channel_power(y_true: np.ndarray, evidence: pd.DataFrame, channels: Sequence[str]) -> pd.DataFrame:
    """AUC and AP of every channel in isolation - the evidence behind the weights."""
    rows = []
    for channel in channels:
        values = evidence[channel].to_numpy(dtype="float64")
        rows.append({
            "channel": channel,
            "roc_auc": roc_auc(y_true, values),
            "average_precision": average_precision(y_true, values),
        })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def weight_selection_optimism(evidence: pd.DataFrame, y_true: np.ndarray, grid: Dict[str, Sequence[float]],
                              pinned: Dict[str, float], seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> Dict[str, Any]:
    """Measure how much a weight search flatters itself.

    Splits the stream into disjoint SELECT / REPORT halves, searches on SELECT,
    scores on REPORT, and reports the optimism. This is the honest way to publish
    a grid-searched weight vector.
    """
    from .ranking import RankingWeights, NoveltyRanker

    ids = evidence["object_id"].to_numpy()
    report: List[Dict[str, float]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # rng.random(), NOT rng.permutation(): permutation returns a shuffled
        # integer array, so `< 0.5` is True only for the single element equal to
        # zero. That made SELECT one object with no positives, every AP NaN, and
        # the whole optimism analysis silently reported the -1.0 sentinel.
        mask = rng.random(len(ids)) < 0.5
        select, holdout = mask, ~mask
        if y_true[select].sum() == 0 or y_true[holdout].sum() == 0:
            continue  # a half with no positives carries no information
        best = (-1.0, dict(pinned))
        for combo in _grid_combinations(grid, pinned):
            weights = RankingWeights(combo)
            ranker = NoveltyRanker(weights=weights)
            sel_scores = ranker.raw_evidence(evidence.loc[select].reset_index(drop=True))
            ap = average_precision(y_true[select], sel_scores)
            if np.isfinite(ap) and ap > best[0]:
                best = (ap, combo)
        ranker = NoveltyRanker(weights=RankingWeights(best[1]))
        ap_select = best[0]
        ap_report = average_precision(y_true[holdout], ranker.raw_evidence(evidence.loc[holdout].reset_index(drop=True)))
        report.append({"seed": int(seed), "ap_select": float(ap_select), "ap_report": float(ap_report),
                       "optimism": float(ap_select - ap_report), "weights": best[1]})
    frame = pd.DataFrame(report)
    return {
        "seeds": [r["seed"] for r in report],
        "mean_ap_select": float(frame["ap_select"].mean()),
        "mean_ap_report": float(frame["ap_report"].mean()),
        "optimism_mean": float(frame["optimism"].mean()),
        "optimism_std": float(frame["optimism"].std(ddof=0)),
        "n_negative_optimism": int((frame["optimism"] < 0).sum()),
        "detail": report,
    }


def _grid_combinations(grid: Dict[str, Sequence[float]], pinned: Dict[str, float]):
    """Cartesian product over the searchable channels; pinned channels stay fixed."""
    import itertools

    keys = list(grid.keys())
    for combo in itertools.product(*(grid[k] for k in keys)):
        weights = dict(pinned)
        weights.update(dict(zip(keys, combo)))
        if sum(weights.values()) <= 0:
            continue
        yield weights


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def write_metrics(payload: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    log.info("wrote metrics -> %s", path)
    return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def nan_to_none(payload):
    """Recursively replace non-finite floats so the JSON stays valid and honest."""
    if isinstance(payload, dict):
        return {k: nan_to_none(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [nan_to_none(v) for v in payload]
    if isinstance(payload, float):
        return None if not np.isfinite(payload) else payload
    return payload
