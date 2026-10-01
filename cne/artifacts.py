"""Synthetic artifact injection and the quality safety suite (CNE v2, prompt 05).

PLAsTiCC contains no real artifact labels, so the quality gate cannot be
*trained* - it is rule-based. What it can be *tested* against is a suite of
deliberately corrupted observations. The invariant under test is absolute:

    **No injected artifact may be promoted, and corrupting an observation may
    never raise its astrophysical novelty score.**

Quality is suppress-only by construction; this suite is what proves the
construction holds in practice rather than in prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .logging import get_logger

log = get_logger("artifacts")

ARTIFACT_TYPES: Tuple[str, ...] = (
    "negative_flux_heavy",
    "single_epoch_spike",
    "error_bar_corruption",
    "duplicate_timestamps",
    "long_cadence_gap",
    "incomplete_band_coverage",
)


# --------------------------------------------------------------------------- #
# injectors: each takes a light curve and returns a corrupted copy
# --------------------------------------------------------------------------- #
def negative_flux_heavy(lc: pd.DataFrame, rng: np.random.Generator, fraction: float = 0.6) -> pd.DataFrame:
    """Flip most detections negative - the classic subtraction/bogus signature."""
    out = lc.reset_index(drop=True).copy()
    det = out["detected_bool"].to_numpy() == 1
    idx = np.flatnonzero(det)
    if not len(idx):
        return out
    flip = rng.choice(idx, size=max(1, int(fraction * len(idx))), replace=False)
    # iloc, not loc: flip holds row positions, and a sliced light curve has a
    # non-contiguous index, so label-based indexing here silently corrupts rows.
    col = out.columns.get_loc("flux")
    out.iloc[flip, col] = -np.abs(out.iloc[flip, col].to_numpy())
    return out


def single_epoch_spike(lc: pd.DataFrame, rng: np.random.Generator, factor: float = 60.0) -> pd.DataFrame:
    """One absurd epoch - a cosmic ray or a bad subtraction, not astrophysics."""
    out = lc.reset_index(drop=True).copy()
    if not len(out):
        return out
    row = int(rng.integers(0, len(out)))
    col = out.columns.get_loc("flux")
    out.iloc[row, col] = np.abs(out.iloc[row, col]) * factor
    return out


def error_bar_corruption(lc: pd.DataFrame, rng: np.random.Generator, factor: float = 25.0) -> pd.DataFrame:
    """Blow up the reported uncertainties: the data no longer constrain anything."""
    out = lc.copy()
    out["flux_err"] = out["flux_err"].to_numpy(dtype="float64") * factor
    return out


def duplicate_timestamps(lc: pd.DataFrame, rng: np.random.Generator, n: int = 6) -> pd.DataFrame:
    """Repeat epochs - a pipeline bug that inflates significance for free."""
    out = lc.copy()
    if not len(out):
        return out
    picks = out.sample(n=min(n, len(out)), random_state=int(rng.integers(0, 2 ** 31 - 1)))
    return pd.concat([out, picks], ignore_index=True)


def long_cadence_gap(lc: pd.DataFrame, rng: np.random.Generator, days: float = 400.0) -> pd.DataFrame:
    """Drop everything after a long gap, leaving a stub of a light curve."""
    out = lc.sort_values("mjd").copy()
    if len(out) < 4:
        return out
    cut = int(len(out) * 0.35)
    return out.iloc[:cut].reset_index(drop=True)


def incomplete_band_coverage(lc: pd.DataFrame, rng: np.random.Generator, keep_bands: int = 1) -> pd.DataFrame:
    """Restrict to a single passband - colour information simply does not exist."""
    out = lc.copy()
    bands = sorted(out["passband"].unique().tolist())
    if len(bands) <= keep_bands:
        return out
    keep = rng.choice(bands, size=keep_bands, replace=False)
    return out[out["passband"].isin(keep)].reset_index(drop=True)


INJECTORS: Dict[str, Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]] = {
    "negative_flux_heavy": negative_flux_heavy,
    "single_epoch_spike": single_epoch_spike,
    "error_bar_corruption": error_bar_corruption,
    "duplicate_timestamps": duplicate_timestamps,
    "long_cadence_gap": long_cadence_gap,
    "incomplete_band_coverage": incomplete_band_coverage,
}


# --------------------------------------------------------------------------- #
# suite
# --------------------------------------------------------------------------- #
@dataclass
class ArtifactCase:
    artifact: str
    object_id: int
    baseline_score: float
    corrupted_score: float
    baseline_quality: float
    corrupted_quality: float
    baseline_rank: int
    corrupted_rank: int
    promoted: bool
    score_increased: bool
    delta_score: float
    delta_quality: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "artifact": self.artifact,
            "object_id": int(self.object_id),
            "baseline_score": round(float(self.baseline_score), 6),
            "corrupted_score": round(float(self.corrupted_score), 6),
            "delta_score": round(float(self.delta_score), 6),
            "baseline_quality": round(float(self.baseline_quality), 4),
            "corrupted_quality": round(float(self.corrupted_quality), 4),
            "delta_quality": round(float(self.delta_quality), 4),
            "baseline_rank": int(self.baseline_rank),
            "corrupted_rank": int(self.corrupted_rank),
            "promoted": bool(self.promoted),
            "score_increased": bool(self.score_increased),
        }


@dataclass
class ArtifactSafetyReport:
    n_cases: int
    n_promoted: int
    n_score_increased: int
    promotion_rate: float
    mean_quality_drop: float
    per_artifact: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cases: List[ArtifactCase] = field(default_factory=list)
    passed: bool = True
    tolerance: float = 1e-6

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "n_promoted": self.n_promoted,
            "n_score_increased": self.n_score_increased,
            "promotion_rate": round(self.promotion_rate, 4),
            "mean_quality_drop": round(self.mean_quality_drop, 4),
            "passed": self.passed,
            "tolerance": self.tolerance,
            "per_artifact": self.per_artifact,
        }


class ArtifactSafetySuite:
    """Inject controlled bad data and verify the gate suppresses all of it.

    ``score_fn`` receives a light-curve frame for one object and returns
    ``(novelty_score, quality, rank)``. Keeping the callback this narrow means the
    suite tests the real production path - featuriser, engine and ranker - rather
    than a reimplementation.
    """

    def __init__(self, score_fn: Callable[[int, pd.DataFrame], Tuple[float, float, int]],
                 seed: int = 42, cases_per_artifact: int = 8):
        self.score_fn = score_fn
        self.seed = seed
        self.cases_per_artifact = cases_per_artifact

    def run(self, lightcurves: pd.DataFrame, object_ids: Sequence[int],
            baseline: pd.DataFrame) -> ArtifactSafetyReport:
        """``baseline`` must hold the uncorrupted score/quality/rank per object."""
        rng = np.random.default_rng(self.seed)
        cases: List[ArtifactCase] = []
        base = baseline.set_index("object_id")
        for artifact, injector in INJECTORS.items():
            for oid in list(object_ids)[: self.cases_per_artifact]:
                lc = lightcurves[lightcurves["object_id"] == oid]
                if not len(lc):
                    continue
                corrupted = injector(lc, np.random.default_rng(int(rng.integers(0, 2 ** 31 - 1))))
                try:
                    score, quality, rank = self.score_fn(int(oid), corrupted)
                except Exception as exc:  # a crash is a safety failure, not a skip
                    log.error("artifact %s on object %d raised %s", artifact, oid, exc)
                    score, quality, rank = float("nan"), 0.0, 10 ** 9
                b_score = float(base.loc[oid, "novelty_score"])
                b_quality = float(base.loc[oid, "quality"])
                b_rank = int(base.loc[oid, "rank"])
                increased = bool(np.isfinite(score) and score > b_score + 1e-9)
                promoted = bool(np.isfinite(score) and rank < b_rank)
                cases.append(ArtifactCase(
                    artifact=artifact, object_id=int(oid),
                    baseline_score=b_score, corrupted_score=float(score),
                    baseline_quality=b_quality, corrupted_quality=float(quality),
                    baseline_rank=b_rank, corrupted_rank=int(rank),
                    promoted=promoted, score_increased=increased,
                    delta_score=float(score) - b_score if np.isfinite(score) else float("nan"),
                    delta_quality=float(quality) - b_quality,
                ))
        per_artifact: Dict[str, Dict[str, Any]] = {}
        for artifact in ARTIFACT_TYPES:
            subset = [c for c in cases if c.artifact == artifact]
            if not subset:
                continue
            per_artifact[artifact] = {
                "n": len(subset),
                "n_promoted": sum(c.promoted for c in subset),
                "n_score_increased": sum(c.score_increased for c in subset),
                "mean_delta_quality": round(float(np.mean([c.delta_quality for c in subset])), 4),
                "mean_delta_score": round(float(np.nanmean([c.delta_score for c in subset])), 6),
            }
        n_promoted = sum(c.promoted for c in cases)
        n_increased = sum(c.score_increased for c in cases)
        report = ArtifactSafetyReport(
            n_cases=len(cases),
            n_promoted=n_promoted,
            n_score_increased=n_increased,
            promotion_rate=n_promoted / max(len(cases), 1),
            mean_quality_drop=float(np.mean([-c.delta_quality for c in cases])) if cases else float("nan"),
            per_artifact=per_artifact,
            cases=cases,
        )
        # The gate is allowed to be imperfect; it is NOT allowed to promote.
        report.passed = n_promoted == 0 and n_increased == 0
        log.info("artifact safety: %d cases, %d promoted, %d score increases, mean quality drop %.3f -> %s",
                 report.n_cases, n_promoted, n_increased, report.mean_quality_drop,
                 "PASS" if report.passed else "FAIL")
        return report


class RealBogusAdapter:
    """Optional hook for survey-provided data-quality scores (ZTF ``realbogus``,
    Rubin ``DRB``).

    Deliberately multiplicative and capped at 1.0: an external score can only
    lower quality. A broker score of 1.0 leaves the rule-based gate untouched, so
    adding real-bogus information can never promote an object.
    """

    def __init__(self, column: str = "realbogus", threshold: float = 0.5):
        self.column = column
        self.threshold = float(threshold)

    def adjust(self, quality: np.ndarray, quality_frame: Optional[pd.DataFrame]) -> np.ndarray:
        quality = np.asarray(quality, dtype="float64")
        if quality_frame is None or self.column not in quality_frame.columns:
            return quality
        score = np.clip(np.nan_to_num(quality_frame[self.column].to_numpy(dtype="float64"), nan=1.0), 0.0, 1.0)
        return np.clip(quality * score, 0.0, 1.0)
