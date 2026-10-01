"""Uncertainty propagation and abstention (CNE v2, prompt 06).

Three distinct uncertainties matter, and conflating them is how a system ends up
confidently ranking garbage:

1. **Predictive uncertainty** - the prior does not know what class this is.
   Estimated with a bootstrap ensemble of LightGBM models.
2. **Feature reliability** - the distance-dependent features may be meaningless
   because the photometric redshift is bad. Propagated by Monte Carlo in
   ``cne.features`` and summarised here.
3. **Domain confidence** - the reference population may not represent this
   stream. Handled by ``cne.domain`` and applied as a multiplier, never as
   evidence.

A candidate can be scored and still be *refused*. Abstention is a first-class
output, not an error state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .config import PriorConfig, UncertaintyConfig
from .logging import get_logger

log = get_logger("uncertainty")


class BootstrapEnsemble:
    """Predictive uncertainty from a bootstrap ensemble of LightGBM models.

    Uses the *spread of the ensemble's own belief* as the uncertainty: an object
    the models disagree about is genuinely harder, and the spread is calibrated
    against realised error by :class:`cne.models.ConditionalCalibrator`.
    """

    def __init__(self, config: Optional[PriorConfig] = None, n_models: int = 6, seed: int = 42,
                 n_estimators: int = 160):
        self.cfg = config or PriorConfig()
        self.n_models = int(n_models)
        self.seed = seed
        self.n_estimators = int(n_estimators)
        self.models_: List[lgb.LGBMClassifier] = []
        self.classes_: np.ndarray = np.array([])
        self.feature_names_: List[str] = []

    def fit(self, x: pd.DataFrame, y: np.ndarray, feature_names: Optional[Sequence[str]] = None) -> "BootstrapEnsemble":
        self.feature_names_ = list(feature_names) if feature_names is not None else list(x.columns)
        x_arr = x[self.feature_names_].to_numpy(dtype="float32")
        classes, y_enc = np.unique(np.asarray(y), return_inverse=True)
        y_enc = y_enc.ravel()
        self.classes_ = classes
        params = {
            "objective": "multiclass",
            "num_class": len(classes),
            "n_estimators": self.n_estimators,
            "learning_rate": self.cfg.learning_rate,
            "num_leaves": self.cfg.num_leaves,
            "min_child_samples": self.cfg.min_child_samples,
            "subsample": self.cfg.subsample,
            "subsample_freq": 1,
            "colsample_bytree": self.cfg.colsample_bytree,
            "random_state": self.seed,
            "n_jobs": 2,
            "verbosity": -1,
        }
        self.models_ = []
        sss = StratifiedShuffleSplit(n_splits=self.n_models, test_size=0.25, random_state=self.seed)
        for i, (train_idx, _) in enumerate(sss.split(x_arr, y_enc)):
            model = lgb.LGBMClassifier(**{**params, "random_state": self.seed + i})
            model.fit(x_arr[train_idx], y_enc[train_idx])
            self.models_.append(model)
        log.info("bootstrap ensemble fitted: %d models on %d objects", len(self.models_), len(x_arr))
        return self

    def predict_stack(self, x: pd.DataFrame) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("BootstrapEnsemble used before fit()")
        x_arr = x[self.feature_names_].to_numpy(dtype="float32")
        return np.stack([m.predict_proba(x_arr) for m in self.models_], axis=0)

    def uncertainty(self, x: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (uncertainty, mean_max_prob, ensemble_disagreement).

        Uncertainty blends two signals:
        * how unconfident the mean prediction is (1 - max prob), and
        * how much the ensemble members disagree about which class wins.
        """
        stack = self.predict_stack(x)
        mean_proba = stack.mean(axis=0)
        max_prob = mean_proba.max(axis=1)
        # Fraction of ensemble members that agree on the winning class, per object.
        winner = stack.argmax(axis=2)  # (n_models, n_objects)
        agreement = np.array([
            np.max(np.bincount(winner[:, i], minlength=len(self.classes_))) / stack.shape[0]
            for i in range(stack.shape[1])
        ])
        disagreement = 1.0 - agreement
        unc = np.clip(0.7 * (1.0 - max_prob) + 0.3 * disagreement, 0.0, 1.0)
        return unc, max_prob, disagreement


def feature_reliability(features: pd.DataFrame, prefix: str = "unc_") -> np.ndarray:
    """Per-object reliability of the distance-sensitive features, in [0, 1]."""
    col = f"{prefix}feature_reliability"
    if col in features.columns:
        return np.clip(np.nan_to_num(features[col].to_numpy(dtype="float64"), nan=0.0), 0.0, 1.0)
    return np.ones(len(features), dtype="float64")


@dataclass
class AbstentionDecision:
    abstain: bool
    reason: str
    confidence: float

    def as_dict(self) -> Dict[str, Any]:
        return {"abstain": self.abstain, "reason": self.reason, "confidence": round(float(self.confidence), 4)}


class AbstentionPolicy:
    """Turn uncertainty, quality and reliability into an explicit refusal."""

    def __init__(self, config: Optional[UncertaintyConfig] = None, enabled: bool = True):
        self.cfg = config or UncertaintyConfig()
        self.enabled = enabled

    def decide(self, quality: float, uncertainty: float, reliability: float,
               domain_status: str = "matched") -> AbstentionDecision:
        confidence = float(np.clip((1.0 - uncertainty) * quality, 0.0, 1.0))
        if not self.enabled:
            return AbstentionDecision(False, "", confidence)
        if quality < self.cfg.abstain_min_quality:
            return AbstentionDecision(True, "low data quality", confidence)
        if uncertainty > self.cfg.abstain_quantile:
            return AbstentionDecision(True, "high predictive uncertainty", confidence)
        if reliability < 0.05:
            return AbstentionDecision(True, "no reliable distance information", confidence)
        if domain_status == "abstain":
            return AbstentionDecision(True, "reference domain does not represent this stream", confidence)
        if domain_status == "degraded":
            return AbstentionDecision(False, "degraded domain match: treat ranking as indicative", confidence)
        return AbstentionDecision(False, "", confidence)

    def summarise(self, decisions: Sequence[AbstentionDecision]) -> Dict[str, Any]:
        n = len(decisions)
        abstained = [d for d in decisions if d.abstain]
        reasons: Dict[str, int] = {}
        for d in abstained:
            reasons[d.reason] = reasons.get(d.reason, 0) + 1
        return {
            "n": n,
            "n_abstained": len(abstained),
            "abstention_rate": len(abstained) / max(n, 1),
            "reasons": reasons,
            "mean_confidence_kept": float(np.mean([d.confidence for d in decisions if not d.abstain])) if n else None,
        }


def evaluate_uncertainty_value(y_true: np.ndarray, scores: np.ndarray, uncertainty: np.ndarray,
                               ks: Sequence[int] = (20, 50, 100)) -> Dict[str, Any]:
    """Does uncertainty help the QUEUE, or only global AUC?

    The honest test is queue purity: drop the most uncertain objects and see
    whether precision@K improves. A channel that improves AUC but dirties the top
    of the queue is not worth promoting.
    """
    from .evaluation import average_precision, precision_at_k, roc_auc

    y = np.asarray(y_true).astype(int)
    out: Dict[str, Any] = {
        "all_objects": {"roc_auc": roc_auc(y, scores), "average_precision": average_precision(y, scores)},
    }
    for k in ks:
        out["all_objects"][f"P@{k}"] = precision_at_k(y, scores, k)
    for keep_frac in (0.9, 0.75, 0.5):
        threshold = np.quantile(uncertainty, keep_frac)
        mask = uncertainty <= threshold
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            continue
        key = f"keep_most_confident_{int(keep_frac * 100)}pct"
        out[key] = {
            "n_kept": int(mask.sum()),
            "roc_auc": roc_auc(y[mask], scores[mask]),
            "average_precision": average_precision(y[mask], scores[mask]),
        }
        for k in ks:
            out[key][f"P@{k}"] = precision_at_k(y[mask], scores[mask], k)
    return out
