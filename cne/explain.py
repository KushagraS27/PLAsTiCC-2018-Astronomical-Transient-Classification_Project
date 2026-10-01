"""Candidate explanation layer.

A ranked number an astronomer cannot interrogate is not useful. For every
candidate CNE returns:

* the evidence-channel breakdown, showing which channel drove the ranking;
* the nearest analogues in the known population, with class and distance;
* the specific features that fall outside the best-fit class's normal range,
  which is the sentence "this looks like a faint fast supernova, except the
  r-band decline is 3 sigma slower than any in that class";
* an explicit "no reliable analogue" statement when the neighbourhood is not
  trustworthy, rather than a silently fabricated explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .novelty import ALL_CHANNELS, CosmicNoveltyEngine
from .taxonomy import name_of

#: Features an astronomer actually reasons about. Ranked first in explanations.
PREFERRED_FEATURES: Tuple[str, ...] = (
    "phys_absmag_r", "phys_absmag_g", "phys_peak_luminosity", "phys_z",
    "col_gmr", "col_urm", "col_rmi",
    "r_rise_time", "r_decline_time", "r_decline_rate", "r_t_span",
    "lc_snr_max", "lc_n_det", "q_n_bands", "unc_absmag_std",
)


@dataclass
class FeatureDeviation:
    feature: str
    value: float
    class_mean: float
    class_std: float
    sigma: float
    direction: str

    def as_dict(self) -> Dict[str, Any]:
        sigma = _round(self.sigma, 2)
        return {
            "feature": self.feature,
            "value": _round(self.value),
            "class_mean": _round(self.class_mean),
            "class_std": _round(self.class_std),
            "sigma": sigma,
            "direction": self.direction,
            "sentence": (f"{self.feature} = {_round(self.value)} sits {abs(sigma)} sigma "
                         f"{'above' if self.direction == 'high' else 'below'} its normal range"),
        }


def _round(value: float, nd: int = 3) -> float:
    try:
        return round(float(value), nd)
    except (TypeError, ValueError):
        return float("nan")


def top_feature_deviations(row: pd.Series, engine: CosmicNoveltyEngine, best_fit_code: int,
                           n: int = 5, max_features: int = 220) -> List[FeatureDeviation]:
    """The features furthest outside the best-fit class's normal range."""
    stats = engine.prior_.class_stats_ if engine.prior_ is not None else None
    if stats is None or best_fit_code not in stats.index.get_level_values(0):
        return []
    candidates = [c for c in row.index if isinstance(c, str) and not c.startswith("q_") and c != "object_id"]
    # Preferred features first so the explanation leads with interpretable physics.
    ordered = [c for c in PREFERRED_FEATURES if c in candidates] + [c for c in candidates if c not in PREFERRED_FEATURES]
    deviations: List[FeatureDeviation] = []
    for feature in ordered[:max_features]:
        try:
            mean = float(stats.loc[best_fit_code][(feature, "mean")])
            std = float(stats.loc[best_fit_code][(feature, "std")])
        except KeyError:
            continue
        if not np.isfinite(std) or std < 1e-9:
            continue
        value = float(row[feature])
        if not np.isfinite(value):
            continue
        sigma = (value - mean) / std
        if abs(sigma) < 2.0:
            continue
        deviations.append(FeatureDeviation(
            feature=feature, value=value, class_mean=mean, class_std=std,
            sigma=float(sigma), direction="high" if sigma > 0 else "low",
        ))
    deviations.sort(key=lambda d: -abs(d.sigma))
    return deviations[:n]


def analogue_block(analogs: Sequence[Dict[str, Any]], min_reliable: int = 3,
                   max_distance: float = 12.0) -> Dict[str, Any]:
    """Summarise the nearest-neighbourhood, or say plainly that there is none."""
    rows = [a for a in analogs if float(a.get("distance", 1e9)) <= max_distance]
    if len(rows) < min_reliable:
        return {
            "reliable": False,
            "reason": f"only {len(rows)} analogue(s) within distance {max_distance}; "
                      "no trustworthy known-population match exists for this candidate",
            "n": len(rows),
            "class_distribution": {},
            "nearest": rows,
        }
    counts: Dict[str, int] = {}
    for row in rows:
        name = name_of(int(row["class_code"]))
        counts[name] = counts.get(name, 0) + 1
    purity = max(counts.values()) / len(rows)
    return {
        "reliable": True,
        "reason": "",
        "n": len(rows),
        "class_distribution": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "dominant_class": max(counts.items(), key=lambda kv: kv[1])[0],
        "class_purity": round(purity, 3),
        "mean_distance": round(float(np.mean([r["distance"] for r in rows])), 3),
        "nearest": rows,
    }


def explain_candidate(row: pd.Series, engine: CosmicNoveltyEngine, analogs: Sequence[Dict[str, Any]],
                      weights: Dict[str, float], features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Full explanation payload for one candidate."""
    best_code = int(row.get("best_fit_code", -1))
    runner_code = int(row.get("runner_up_code", -1))
    contributions = {
        channel: {
            "value": _round(float(row.get(channel, 0.0)), 4),
            "weight": _round(float(weights.get(channel, 0.0)), 4),
            "contribution": _round(float(row.get(channel, 0.0)) * float(weights.get(channel, 0.0)), 5),
        }
        for channel in ALL_CHANNELS
    }
    driver = max(contributions.items(), key=lambda kv: kv[1]["contribution"])[0] if contributions else ""
    deviations: List[FeatureDeviation] = []
    if features is not None and engine.prior_ is not None:
        match = features[features["object_id"].to_numpy() == int(row["object_id"])]
        if len(match):
            deviations = top_feature_deviations(match.iloc[0], engine, best_code)
    block = analogue_block(analogs)
    return {
        "best_fit_class": name_of(best_code),
        "best_fit_prob": _round(float(row.get("best_fit_prob", 0.0)), 4),
        "runner_up_class": name_of(runner_code),
        "runner_up_prob": _round(float(row.get("runner_up_prob", 0.0)), 4),
        "driving_channel": driver,
        "channel_contributions": contributions,
        "feature_deviations": [d.as_dict() for d in deviations],
        "analogs": block,
        "n_analogs_considered": len(analogs),
        "human_summary": _summary(row, best_code, driver, deviations, block),
    }


def _summary(row: pd.Series, best_code: int, driver: str,
             deviations: List[FeatureDeviation], block: Dict[str, Any]) -> str:
    """One plain-language sentence. Bounded vocabulary, by design."""
    name = name_of(best_code)
    prob = float(row.get("best_fit_prob", 0.0))
    parts = [f"Best explained as {name} (p={prob:.2f}), but the known-physics prior is poorly satisfied"]
    if driver:
        parts.append(f"the {driver.replace('_', ' ')} channel contributes most")
    if deviations:
        d = deviations[0]
        parts.append(f"{d.feature} is {abs(d.sigma):.1f} sigma outside that class's normal range")
    if not block.get("reliable", False):
        parts.append("no reliable known-population analogue was found")
    else:
        parts.append(f"nearest analogues are mostly {block.get('dominant_class', 'unknown')}")
    parts.append("potentially novel candidate - requires expert follow-up; "
                 "poorly explained by current known populations")
    return "; ".join(parts) + "."
