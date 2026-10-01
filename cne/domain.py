"""Domain matching and domain-shift monitoring (CNE v2, prompts 03 and 10).

Domain matching is the single highest-value architectural decision in CNE: the
same novelty metric on a domain-matched reference reaches AUC 0.818, and on an
out-of-domain reference only 0.610. That is worth more than any detector change.

Two separate concerns live here and must not be merged:

* **Reference building** (prompt 03) - construct a known-class reference that
  resembles the stream being scored, and say how well it matches.
* **Shift monitoring** (prompt 10) - watch the stream against the reference over
  time and degrade/abstain when it drifts.

Drift statistics are kept strictly OUT of the novelty score. A survey that gets
cloudier must produce lower confidence, not a queue full of "novel" candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .logging import get_logger

log = get_logger("domain")

#: Covariates that define "the same survey domain" for CNE. All are already in the
#: feature matrix, so matching needs no extra data.
DEFAULT_COVARIATES: Tuple[str, ...] = (
    "lc_snr_max",
    "lc_n_det",
    "lc_t_span",
    "lc_peak_mag",
    "phys_z",
    "phys_distmod",
    "q_n_bands",
    "q_det_frac",
)

MATCHED = "matched"
DEGRADED = "degraded"
ABSTAIN = "abstain"


def psi(reference: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index.

    < 0.1 negligible, 0.1-0.25 moderate, > 0.25 severe. Bins are quantiles of the
    reference so the statistic is invariant to a monotone transform of the
    covariate and to the covariate's absolute scale.
    """
    ref = np.asarray(reference, dtype="float64")
    tgt = np.asarray(target, dtype="float64")
    ref = ref[np.isfinite(ref)]
    tgt = tgt[np.isfinite(tgt)]
    if len(ref) < n_bins or len(tgt) < n_bins:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype("float64")
    tgt_counts = np.histogram(tgt, bins=edges)[0].astype("float64")
    eps = 1e-6
    p = np.clip(ref_counts / max(ref_counts.sum(), 1), eps, None)
    q = np.clip(tgt_counts / max(tgt_counts.sum(), 1), eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def ks_statistic(reference: np.ndarray, target: np.ndarray) -> float:
    ref = np.sort(np.asarray(reference, dtype="float64"))
    tgt = np.sort(np.asarray(target, dtype="float64"))
    ref = ref[np.isfinite(ref)]
    tgt = tgt[np.isfinite(tgt)]
    if not len(ref) or not len(tgt):
        return float("nan")
    grid = np.concatenate([ref, tgt])
    cdf_r = np.searchsorted(ref, grid, side="right") / len(ref)
    cdf_t = np.searchsorted(tgt, grid, side="right") / len(tgt)
    return float(np.max(np.abs(cdf_r - cdf_t)))


@dataclass
class DomainMatchScore:
    """How well the reference population represents the stream being scored."""

    score: float
    status: str
    per_covariate: Dict[str, Dict[str, float]] = field(default_factory=dict)
    strategy: str = "none"
    thresholds: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "status": self.status,
            "strategy": self.strategy,
            "thresholds": self.thresholds,
            "per_covariate": {
                k: {kk: (None if not np.isfinite(vv) else round(float(vv), 4)) for kk, vv in v.items()}
                for k, v in self.per_covariate.items()
            },
        }


class ReferenceBuilder:
    """Build a domain-matched known-class reference population.

    Three strategies, all deterministic:

    ``none``           Use the reference as given (CNE v1 behaviour).
    ``stratified``     Resample the reference so its covariate histogram matches
                       the stream's.
    ``density_ratio``  Keep every reference object, but weight it by an estimate
                       of p(stream)/p(reference), capped to control variance.
    """

    def __init__(self, covariates: Sequence[str] = DEFAULT_COVARIATES, weight_cap: float = 5.0, seed: int = 42):
        self.covariates = list(covariates)
        self.weight_cap = float(weight_cap)
        self.seed = seed

    # ------------------------------------------------------------------ tools
    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.covariates if c in frame.columns]
        if not cols:
            raise KeyError(f"none of the domain covariates {self.covariates} are present in the frame")
        return np.nan_to_num(frame[cols].to_numpy(dtype="float64"), nan=0.0)

    def density_ratio_weights(self, reference: pd.DataFrame, stream: pd.DataFrame) -> np.ndarray:
        """Capped source-to-target density ratio from a logistic discriminator."""
        x_ref, x_tgt = self._matrix(reference), self._matrix(stream)
        x = np.vstack([x_ref, x_tgt])
        y = np.r_[np.zeros(len(x_ref)), np.ones(len(x_tgt))]
        # Standardise on the pooled sample so the discriminator is scale-free.
        center, scale = x.mean(axis=0), np.clip(x.std(axis=0), 1e-9, None)
        model = LogisticRegression(max_iter=2000, C=1.0).fit((x - center) / scale, y)
        p_target = model.predict_proba((x_ref - center) / scale)[:, 1]
        ratio = np.clip(p_target, 1e-6, 1 - 1e-6) / np.clip(1 - p_target, 1e-6, 1 - 1e-6)
        # Scale-free first, then clip. The cap is the hard safety invariant - no
        # single reference object may dominate - so it is applied last; the mean
        # is therefore only approximately one and is reported alongside the ESS.
        weights = np.clip(ratio / ratio.mean(), 1.0 / self.weight_cap, self.weight_cap)
        log.info("density-ratio weights: min=%.3f max=%.3f mean=%.3f (cap=%.1f)",
                 weights.min(), weights.max(), weights.mean(), self.weight_cap)
        return weights

    def stratified_weights(self, reference: pd.DataFrame, stream: pd.DataFrame, n_bins: int = 4) -> np.ndarray:
        """Weight reference objects so their joint covariate histogram matches the stream."""
        cols = [c for c in self.covariates if c in reference.columns]
        if not cols:
            return np.ones(len(reference))
        keys = []
        for frame in (reference, stream):
            parts = []
            for col in cols[:3]:  # 3 covariates x 4 bins = 64 strata
                values = np.nan_to_num(frame[col].to_numpy(dtype="float64"), nan=0.0)
                edges = np.quantile(np.nan_to_num(stream[col].to_numpy(dtype="float64"), nan=0.0),
                                    np.linspace(0, 1, n_bins + 1))
                edges = np.unique(edges)
                parts.append(np.searchsorted(edges, values, side="right").astype(str))
            keys.append(["|".join(row) for row in np.stack(parts, axis=1).T])
        ref_keys, tgt_keys = keys
        tgt_counts = pd.Series(tgt_keys).value_counts(normalize=True)
        ref_counts = pd.Series(ref_keys).value_counts(normalize=True)
        weights = np.array([tgt_counts.get(k, 0.0) / max(ref_counts.get(k, 1e-9), 1e-9) for k in ref_keys])
        weights = np.clip(np.nan_to_num(weights, nan=1.0, posinf=self.weight_cap), 1.0 / self.weight_cap, self.weight_cap)
        return weights / max(weights.mean(), 1e-9)

    # ------------------------------------------------------------------- build
    def build(self, reference: pd.DataFrame, stream: pd.DataFrame, strategy: str = "density_ratio") -> Tuple[np.ndarray, Dict[str, Any]]:
        if strategy == "none":
            weights = np.ones(len(reference))
        elif strategy == "stratified":
            weights = self.stratified_weights(reference, stream)
        elif strategy == "density_ratio":
            weights = self.density_ratio_weights(reference, stream)
        else:
            raise ValueError(f"unknown reference strategy '{strategy}'")
        ess = float(weights.sum() ** 2 / max(np.sum(weights ** 2), 1e-12))
        info = {
            "strategy": strategy,
            "n_reference": int(len(reference)),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "effective_sample_size": ess,
            "ess_fraction": ess / max(len(reference), 1),
        }
        return weights, info

    # ------------------------------------------------------------------- score
    def match_score(self, reference: pd.DataFrame, stream: pd.DataFrame, strategy: str = "none",
                    psi_matched: float = 0.10, psi_abstain: float = 0.25) -> DomainMatchScore:
        """Composite domain-match score from per-covariate PSI and KS."""
        per_cov: Dict[str, Dict[str, float]] = {}
        for col in self.covariates:
            if col not in reference.columns or col not in stream.columns:
                continue
            ref = reference[col].to_numpy(dtype="float64")
            tgt = stream[col].to_numpy(dtype="float64")
            per_cov[col] = {"psi": psi(ref, tgt), "ks": ks_statistic(ref, tgt),
                            "ref_median": float(np.nanmedian(ref)) if np.isfinite(ref).any() else float("nan"),
                            "stream_median": float(np.nanmedian(tgt)) if np.isfinite(tgt).any() else float("nan")}
        if not per_cov:
            return DomainMatchScore(score=float("nan"), status=MATCHED, strategy=strategy,
                                    thresholds={"psi_matched": psi_matched, "psi_abstain": psi_abstain})
        psis = np.array([v["psi"] for v in per_cov.values()], dtype="float64")
        mean_psi = float(np.nanmean(psis))
        worst_psi = float(np.nanmax(psis))
        # Composite: mean drift dominates, the worst covariate can veto it.
        score = float(np.clip(1.0 - 0.5 * mean_psi / psi_abstain - 0.5 * worst_psi / (2 * psi_abstain), 0.0, 1.0))
        status = MATCHED if worst_psi < psi_matched else (ABSTAIN if worst_psi > psi_abstain else DEGRADED)
        result = DomainMatchScore(score=score, status=status, per_covariate=per_cov, strategy=strategy,
                                  thresholds={"psi_matched": psi_matched, "psi_abstain": psi_abstain})
        log.info("domain match: score=%.3f status=%s mean_psi=%.3f worst_psi=%.3f strategy=%s",
                 score, status, mean_psi, worst_psi, strategy)
        return result


class DomainShiftMonitor:
    """Track stream-vs-reference drift over time; never touches the novelty score.

    Validation ranges are versioned with the model, so "degraded" means
    "outside the range this model was validated on" - a factual statement, not a
    judgement about the sky.
    """

    def __init__(self, covariates: Sequence[str] = DEFAULT_COVARIATES, warn_psi: float = 0.10, fail_psi: float = 0.25):
        self.covariates = list(covariates)
        self.warn_psi = warn_psi
        self.fail_psi = fail_psi
        self.ranges_: Dict[str, Tuple[float, float]] = {}
        self.baseline_: Dict[str, np.ndarray] = {}

    def fit(self, reference: pd.DataFrame) -> "DomainShiftMonitor":
        for col in self.covariates:
            if col not in reference.columns:
                continue
            values = np.nan_to_num(reference[col].to_numpy(dtype="float64"), nan=0.0)
            self.ranges_[col] = (float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995)))
            self.baseline_[col] = values
        log.info("domain-shift monitor calibrated on %d covariates", len(self.ranges_))
        return self

    def health(self, stream: pd.DataFrame) -> Dict[str, Any]:
        rows = []
        for col, values in self.baseline_.items():
            if col not in stream.columns:
                continue
            target = np.nan_to_num(stream[col].to_numpy(dtype="float64"), nan=0.0)
            lo, hi = self.ranges_[col]
            outside = float(np.mean((target < lo) | (target > hi)))
            rows.append({
                "covariate": col,
                "psi": psi(values, target),
                "ks": ks_statistic(values, target),
                "fraction_outside_validated_range": outside,
                "valid_range": [round(lo, 4), round(hi, 4)],
                "stream_median": round(float(np.median(target)), 4),
                "reference_median": round(float(np.median(values)), 4),
            })
        frame = pd.DataFrame(rows)
        if frame.empty:
            return {"score": float("nan"), "status": MATCHED, "covariates": []}
        worst = float(np.nanmax(frame["psi"].to_numpy()))
        score = float(np.clip(1.0 - worst / self.fail_psi, 0.0, 1.0))
        status = MATCHED if worst < self.warn_psi else (ABSTAIN if worst > self.fail_psi else DEGRADED)
        return {
            "score": round(score, 4),
            "status": status,
            "worst_psi": round(worst, 4),
            "mean_psi": round(float(np.nanmean(frame["psi"].to_numpy())), 4),
            "thresholds": {"warn_psi": self.warn_psi, "fail_psi": self.fail_psi},
            "covariates": frame.to_dict(orient="records"),
        }
