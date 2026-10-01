"""Adversarial domain validation - is the engine finding novelty, or just domain shift?

Adapted from the diagnostic that won SETI Breakthrough Listen (Team Watercooled,
1st place). Their models scored differently on the leaderboard despite similar CV,
and the cause was that the models had learned the *background* rather than the
signal. They found it by grouping predictions by how "train-like" each sample was
and observing that confidence tracked background familiarity.

The same failure mode is possible here, and it is the single most dangerous one
for a novelty detector: an object can look novel purely because it comes from a
different population than the reference, not because it is astrophysically
unusual. If that is happening, the review queue is a domain-shift detector with
an astronomy label on it.

The test is cheap and decisive. Train a classifier to tell reference objects from
target objects using the same features the engine uses:

* **AUC ~ 0.5** - the two populations are indistinguishable, so "novel" cannot be
  a proxy for "different domain". The engine is measuring what it claims to.
* **AUC ~ 1.0** - the populations are trivially separable, so novelty scores are
  contaminated. Every headline number needs that caveat.

Then, and this is the part that actually matters, correlate the adversarial
score against the novelty score. High AUC alone only says the populations
differ; the correlation says whether the *ranking* is driven by that difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .logging import get_logger

log = get_logger("adversarial")


@dataclass
class AdversarialResult:
    """Outcome of a reference-vs-target separability test."""

    roc_auc: float
    n_reference: int
    n_target: int
    n_features: int
    top_features: List[Dict[str, Any]] = field(default_factory=list)
    # Correlation between the adversarial "target-likeness" score and novelty.
    # This is the number that says whether the ranking is contaminated.
    novelty_correlation: Optional[float] = None
    novelty_correlation_spearman: Optional[float] = None
    verdict: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "roc_auc": self.roc_auc,
            "n_reference": self.n_reference,
            "n_target": self.n_target,
            "n_features": self.n_features,
            "top_features": self.top_features,
            "novelty_correlation": self.novelty_correlation,
            "novelty_correlation_spearman": self.novelty_correlation_spearman,
            "verdict": self.verdict,
        }


def _verdict(roc_auc: float, correlation: Optional[float]) -> str:
    """Plain-language reading of the two numbers."""
    if roc_auc < 0.6:
        sep = "populations are effectively indistinguishable"
    elif roc_auc < 0.75:
        sep = "populations are weakly separable"
    elif roc_auc < 0.9:
        sep = "populations are clearly separable"
    else:
        sep = "populations are trivially separable"

    if correlation is None:
        return f"{sep}; novelty correlation not measured"
    mag = abs(correlation)
    if mag < 0.1:
        drive = "the novelty ranking is essentially independent of domain-likeness"
    elif mag < 0.3:
        drive = "domain-likeness contributes weakly to the novelty ranking"
    else:
        drive = "the novelty ranking is substantially driven by domain-likeness"
    return f"{sep}; {drive} (r={correlation:+.3f})"


def adversarial_validation(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    feature_names: List[str],
    novelty: Optional[pd.Series] = None,
    seed: int = 42,
    n_folds: int = 5,
    top_k: int = 12,
) -> AdversarialResult:
    """Measure how separable the reference and target populations are.

    ``novelty``, when given, must be aligned row-for-row with ``target``. It is
    the engine's novelty score for each target object, used to test whether the
    ranking tracks domain-likeness rather than astrophysical unusualness.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import spearmanr

    cols = [c for c in feature_names if c in reference.columns and c in target.columns]
    if not cols:
        raise ValueError("no shared feature columns between reference and target")

    X = np.vstack([
        reference[cols].to_numpy(dtype="float32"),
        target[cols].to_numpy(dtype="float32"),
    ])
    y = np.concatenate([np.zeros(len(reference), dtype="int8"),
                        np.ones(len(target), dtype="int8")])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    params = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                  n_estimators=300, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, min_child_samples=40,
                  random_state=seed, n_jobs=-1, verbose=-1)

    oof = np.zeros(len(X), dtype="float64")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    importance = np.zeros(len(cols), dtype="float64")
    for tr, te in skf.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
        importance += model.feature_importances_
    importance /= n_folds

    auc = float(roc_auc_score(y, oof))

    order = np.argsort(importance)[::-1][:top_k]
    top = [{"feature": cols[i], "importance": float(importance[i])} for i in order]

    corr = sp = None
    if novelty is not None:
        nv = np.asarray(novelty, dtype="float64")
        if len(nv) != len(target):
            raise ValueError(
                f"novelty has {len(nv)} rows but target has {len(target)}; "
                "they must be aligned row-for-row"
            )
        target_likeness = oof[len(reference):]
        finite = np.isfinite(nv) & np.isfinite(target_likeness)
        if finite.sum() > 10:
            corr = float(np.corrcoef(target_likeness[finite], nv[finite])[0, 1])
            sp = float(spearmanr(target_likeness[finite], nv[finite]).statistic)

    result = AdversarialResult(
        roc_auc=auc, n_reference=int(len(reference)), n_target=int(len(target)),
        n_features=len(cols), top_features=top,
        novelty_correlation=corr, novelty_correlation_spearman=sp,
        verdict=_verdict(auc, corr),
    )
    log.info("adversarial validation: AUC=%.4f over %d reference vs %d target, %d features",
             auc, len(reference), len(target), len(cols))
    if corr is not None:
        log.info("novelty vs domain-likeness: pearson=%+.4f spearman=%+.4f", corr, sp)
    return result
