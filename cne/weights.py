"""Nested evidence-weight selection (CNE v2, prompt 01).

CNE v1 grid-searched the channel weights on the same scored stream it reported
on. The measured optimism turned out to be +0.0138 +/- 0.0546 AP - indistinguishable
from zero - but "we checked and it was fine" is not the same as "the protocol
cannot leak". This module makes the protocol incapable of leaking:

1. The final test split is frozen in a manifest before this code runs.
2. Weight search sees the VALIDATION split only, through a :class:`LeakageGuard`.
3. Selected weights are written to a versioned config and pinned there.
4. ``baseline_v1`` is always reported alongside, so a regression is visible.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .evaluation import average_precision, precision_at_k, roc_auc
from .logging import get_logger
from .manifests import LeakageGuard
from .novelty import ALL_CHANNELS
from .ranking import NoveltyRanker, RankingWeights

log = get_logger("weights")

#: Coarse search grid. Deliberately tiny: 6 channels x 2-4 values is far too few
#: degrees of freedom to memorise a few hundred novel objects, which is exactly
#: why the measured optimism is near zero.
DEFAULT_GRID: Dict[str, Sequence[float]] = {
    "taxonomy_gap": (1.0,),
    "simplex_novelty": (0.0, 0.1, 0.2, 0.4),
    "novelty_gap": (0.0, 0.05, 0.15),
    "prior_entropy": (0.0, 0.1, 0.25),
    "neighbor_entropy": (0.0, 0.1),
    # cc_weighted is searched because on THIS feature set it measured far above
    # chance (validation AUC reported in reports/weight_selection.json), unlike
    # CNE v1 where the analytic class-conditional channel sat at 0.463. Promotion
    # is decided on the validation split alone and re-checked on locked test.
    "cc_weighted": (0.0, 0.15, 0.35),
}


@dataclass
class WeightSearchResult:
    weights: Dict[str, float]
    validation_ap: float
    validation_auc: float
    baseline_v1_ap: float
    baseline_v1_auc: float
    n_combinations: int
    n_validation: int
    n_test: int
    history: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_weights": self.weights,
            "validation_ap": round(self.validation_ap, 4),
            "validation_auc": round(self.validation_auc, 4),
            "baseline_v1_ap": round(self.baseline_v1_ap, 4),
            "baseline_v1_auc": round(self.baseline_v1_auc, 4),
            "n_combinations": self.n_combinations,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
        }


def _combinations(grid: Dict[str, Sequence[float]]) -> List[Dict[str, float]]:
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        weights = dict(zip(keys, combo))
        filled = {c: float(weights.get(c, 0.0)) for c in ALL_CHANNELS}
        if sum(filled.values()) <= 0:
            continue
        out.append(filled)
    return out


class NestedWeightSelector:
    """Search weights on validation data only; never touch the locked test split."""

    def __init__(self, guard: Optional[LeakageGuard] = None, grid: Optional[Dict[str, Sequence[float]]] = None):
        self.guard = guard
        self.grid = dict(grid or DEFAULT_GRID)

    def search(self, evidence: pd.DataFrame, y_true: np.ndarray) -> WeightSearchResult:
        if self.guard is not None:
            # Declare the ids this stage is allowed to read; the guard will fail
            # the run if any locked-test id sneaks in.
            self.guard.seen_ids(evidence["object_id"].to_numpy())
        y = np.asarray(y_true).astype(int)
        combos = _combinations(self.grid)
        history: List[Dict[str, Any]] = []
        best_ap = -1.0
        best: Dict[str, float] = dict(RankingWeights.v1().as_dict())
        for weights in combos:
            ranker = NoveltyRanker(weights=RankingWeights(weights))
            scores = ranker.raw_evidence(evidence.reset_index(drop=True))
            ap = average_precision(y, scores)
            auc = roc_auc(y, scores)
            history.append({"weights": weights, "ap": float(ap), "auc": float(auc)})
            if np.isfinite(ap) and ap > best_ap:
                best_ap, best = float(ap), weights
        ranker_best = NoveltyRanker(weights=RankingWeights(best))
        best_scores = ranker_best.raw_evidence(evidence.reset_index(drop=True))
        v1 = RankingWeights.v1()
        v1_scores = NoveltyRanker(weights=v1).raw_evidence(evidence.reset_index(drop=True))
        result = WeightSearchResult(
            weights=best,
            validation_ap=float(average_precision(y, best_scores)),
            validation_auc=float(roc_auc(y, best_scores)),
            baseline_v1_ap=float(average_precision(y, v1_scores)),
            baseline_v1_auc=float(roc_auc(y, v1_scores)),
            n_combinations=len(combos),
            n_validation=int(len(y)),
            n_test=0,
            history=history,
        )
        log.info("weight search: %d combinations, validation AP %.4f (baseline_v1 %.4f)",
                 len(combos), result.validation_ap, result.baseline_v1_ap)
        return result

    @staticmethod
    def write_config(result: WeightSearchResult, path: str | Path, config_id: str = "cne-v2-weights") -> Path:
        """Freeze the selected weights into a versioned config file."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {"config_id": config_id, "note": "selected on the validation split only; locked test never read"},
            "weights": {k: float(v) for k, v in result.weights.items()},
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=True))
        log.info("selected weights frozen -> %s", path)
        return path

    @staticmethod
    def write_report(result: WeightSearchResult, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.as_dict()
        payload["history_top10"] = sorted(result.history, key=lambda r: -r["ap"])[:10]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path


def compare_weightings(evidence: pd.DataFrame, y_true: np.ndarray,
                       weightings: Dict[str, RankingWeights], ks: Sequence[int] = (10, 20, 50)) -> pd.DataFrame:
    """Side-by-side comparison so a regression against baseline_v1 is visible."""
    rows = []
    for name, weights in weightings.items():
        ranker = NoveltyRanker(weights=weights)
        scores = ranker.raw_evidence(evidence.reset_index(drop=True))
        row = {
            "weighting": name,
            "roc_auc": roc_auc(y_true, scores),
            "average_precision": average_precision(y_true, scores),
            "active_channels": ",".join(weights.active_channels()),
        }
        for k in ks:
            row[f"P@{k}"] = precision_at_k(y_true, scores, k)
        rows.append(row)
    return pd.DataFrame(rows).set_index("weighting")
