"""Ranking: the auditable, operator-tunable novelty score.

    novelty = sum_i w_i * evidence_i  x  quality^0.5  x  confidence^0.25  x  (1 + 0.15 * agreement)
    confidence = (1 - uncertainty) * quality

Properties this formula is chosen for:

* **Additive evidence, multiplicative trust.** Evidence adds up; data quality and
  confidence can only scale it down. Bad data can never promote an object.
* **Auditable.** Every term is exposed per candidate, so an astronomer can see
  *why* something is at the top of the queue.
* **Tunable without retraining.** Weights live in config, not in code.
* **Abstention is representable.** A candidate can be scored *and* refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CNEConfig
from .logging import get_logger
from .novelty import ALL_CHANNELS
from .schema import Candidate
from .taxonomy import name_of

log = get_logger("ranking")


@dataclass(frozen=True)
class RankingWeights:
    """Evidence-channel weights. Immutable and hashable so a run manifest can pin them."""

    values: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def v1(cls) -> "RankingWeights":
        """The frozen CNE v1.0.0 weights, preserved as ``baseline_v1``.

        Every value came from measured channel power. ``family_misfit`` and the
        unsupervised channels are measured at or below chance and are pinned to 0
        - kept in the config, never deleted, so the negative result stays auditable.
        """
        return cls({
            "taxonomy_gap": 1.00,
            "simplex_novelty": 0.20,
            "novelty_gap": 0.05,
            "prior_entropy": 0.0,
            "neighbor_entropy": 0.0,
            "anomaly_score": 0.0,
            "physics_gap": 0.0,
            "cc_weighted": 0.0,
            "family_misfit": 0.0,
        })

    @classmethod
    def from_config(cls, config: CNEConfig) -> "RankingWeights":
        return cls({c: float(config.weights.get(c, 0.0)) for c in ALL_CHANNELS})

    def normalised(self, channels: Sequence[str] = ALL_CHANNELS) -> Dict[str, float]:
        total = sum(self.values.get(c, 0.0) for c in channels)
        if total <= 0:
            raise ValueError("All evidence weights are zero; ranking is undefined")
        return {c: self.values.get(c, 0.0) / total for c in channels}

    def active_channels(self) -> List[str]:
        return [c for c in ALL_CHANNELS if self.values.get(c, 0.0) > 0]

    def as_dict(self) -> Dict[str, float]:
        return dict(self.values)


class NoveltyRanker:
    """Applies the ranking formula, assigns tiers and decides abstention."""

    def __init__(self, config: Optional[CNEConfig] = None, weights: Optional[RankingWeights] = None):
        self.cfg = config or CNEConfig.load()
        self.weights = weights or RankingWeights.from_config(self.cfg)
        self.norm_ = self.weights.normalised()
        self.agreement_bonus = float(self.cfg.ranking.agreement_bonus)
        self.quality_exponent = float(self.cfg.quality.exponent)
        self.uncertainty_exponent = float(self.cfg.uncertainty.exponent)
        self.tiers = dict(self.cfg.ranking.tiers)

    # ------------------------------------------------------------------ pieces
    def raw_evidence(self, evidence: pd.DataFrame) -> np.ndarray:
        """Weighted sum of the channels, already normalised to [0, 1]."""
        total = np.zeros(len(evidence), dtype="float64")
        for channel, weight in self.norm_.items():
            if weight <= 0:
                continue
            total += weight * evidence[channel].to_numpy(dtype="float64")
        return total

    def agreement(self, evidence: pd.DataFrame) -> np.ndarray:
        """Consensus among ACTIVE channels: do the top signals point the same way?

        1.0 when every active channel agrees this object is unusual, 0.0 when they
        disagree completely. Enters at 0.15 so it can refine, never dominate.
        """
        active = self.weights.active_channels()
        if len(active) < 2:
            return np.ones(len(evidence), dtype="float64")
        stack = np.stack([evidence[c].to_numpy(dtype="float64") for c in active], axis=1)
        # Coefficient of dispersion, NOT a rank of the spread. Ranking within the
        # scored batch would make agreement - and therefore the final score -
        # depend on which chunk an object landed in. Every active channel is
        # already on [0, 1], so the dispersion is directly interpretable:
        # 0 when all channels agree, 1 when they maximally disagree.
        mean = stack.mean(axis=1)
        spread = stack.std(axis=1)
        dispersion = spread / np.clip(mean + 0.5, 1e-9, None)
        return np.clip(1.0 - dispersion, 0.0, 1.0)

    def confidence(self, quality: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
        return np.clip((1.0 - np.clip(uncertainty, 0, 1)) * np.clip(quality, 0, 1), 0.0, 1.0)

    # -------------------------------------------------------------------- rank
    def rank(self, evidence: pd.DataFrame, uncertainty: Optional[np.ndarray] = None,
             reliability: Optional[np.ndarray] = None, explanations: Optional[Dict[int, Dict]] = None,
             labels: Optional[pd.Series] = None, novel_codes: Optional[Sequence[int]] = None) -> pd.DataFrame:
        quality = evidence["quality"].to_numpy(dtype="float64") if "quality" in evidence.columns else np.ones(len(evidence))
        unc = np.asarray(uncertainty, dtype="float64") if uncertainty is not None else evidence["prior_entropy"].to_numpy(dtype="float64")
        rel = np.asarray(reliability, dtype="float64") if reliability is not None else np.ones(len(evidence))

        raw = self.raw_evidence(evidence)
        agree = self.agreement(evidence)
        conf = self.confidence(quality, unc)
        score = raw * np.power(np.clip(quality, 1e-6, 1.0), self.quality_exponent) \
                    * np.power(np.clip(conf, 1e-6, 1.0), self.uncertainty_exponent) \
                    * (1.0 + self.agreement_bonus * agree)

        # Domain mismatch reduces confidence (CNE v2, prompt 03) but is kept out of
        # the evidence sum: a domain shift must never masquerade as novelty.
        if "domain_match" in evidence.columns:
            match = np.clip(evidence["domain_match"].to_numpy(dtype="float64"), 0.0, 1.0)
            score = score * (0.7 + 0.3 * match)

        order = np.argsort(-score, kind="mergesort")
        max_score = score.max() if len(score) else 1.0
        normalised = score / max(max_score, 1e-12)

        abstain, reasons = self.abstention(quality, unc, rel, normalised)
        tiers = self._tiers(normalised)

        out = pd.DataFrame({
            "object_id": evidence["object_id"].to_numpy(),
            "novelty_score": score,
            "novelty_score_normalised": normalised,
            "evidence_raw": raw,
            "agreement": agree,
            "quality": quality,
            "uncertainty": unc,
            "confidence": conf,
            "reliability": rel,
            "tier": tiers,
            "abstain": abstain,
            "abstain_reason": reasons,
            "rank": np.empty(len(score), dtype="int64"),
        })
        out["rank"] = np.empty(len(out), dtype="int64")
        out.loc[order, "rank"] = np.arange(1, len(out) + 1)
        for channel in ALL_CHANNELS:
            out[channel] = evidence[channel].to_numpy(dtype="float64")
        for extra in ("best_fit_code", "best_fit_prob", "runner_up_code", "runner_up_prob",
                      "neighbour_mean_distance", "neighbour_min_distance"):
            if extra in evidence.columns:
                out[extra] = evidence[extra].to_numpy()
        if labels is not None:
            lab = labels.reindex(out["object_id"]).to_numpy()
            out["true_code"] = lab
            out["is_novel"] = np.isin(lab, list(novel_codes or []))
        return out.sort_values("rank").reset_index(drop=True)

    # -------------------------------------------------------------- abstention
    def abstention(self, quality, uncertainty, reliability, normalised_score) -> Tuple[np.ndarray, np.ndarray]:
        """Refuse to rank when the data cannot support a claim.

        A discovery-assistance system that never says "I don't know" is worse than
        useless: it converts data problems into astrophysical candidates.
        """
        n = len(quality)
        abstain = np.zeros(n, dtype=bool)
        reasons = np.array([""] * n, dtype=object)
        if not self.cfg.v2.abstention:
            return abstain, reasons

        low_quality = quality < self.cfg.uncertainty.abstain_min_quality
        # Self-calibrating: abstain on the most uncertain tail OF THIS STREAM.
        # A fixed absolute threshold cannot work, because predictive uncertainty
        # is ~0.05 on a reference-like stream and far higher on a stream full of
        # genuinely odd objects - the same number means different things.
        unc_threshold = float(np.quantile(uncertainty, self.cfg.uncertainty.abstain_quantile))
        high_unc = uncertainty > unc_threshold
        unreliable = reliability < 0.05
        for mask, text in ((low_quality, "low data quality"),
                           (high_unc, "high predictive uncertainty"),
                           (unreliable, "no reliable distance information")):
            hit = mask & ~abstain
            abstain |= mask
            reasons[hit] = text
        # An abstained candidate that would otherwise top the queue is the most
        # dangerous case, so it is named explicitly.
        promoted = abstain & (normalised_score > 0.5)
        reasons[promoted] = reasons[promoted].astype(str) + " (would otherwise rank highly)"
        return abstain, reasons

    def _tiers(self, normalised: np.ndarray) -> np.ndarray:
        crit = self.tiers.get("critical", 0.95)
        high = self.tiers.get("high", 0.85)
        mod = self.tiers.get("moderate", 0.60)
        tiers = np.array(["routine"] * len(normalised), dtype=object)
        tiers[normalised >= mod] = "moderate"
        tiers[normalised >= high] = "high"
        tiers[normalised >= crit] = "critical"
        return tiers

    # ------------------------------------------------------------------ output
    def to_candidates(self, ranked: pd.DataFrame, explanations: Optional[Dict[int, Dict]] = None,
                      limit: Optional[int] = None) -> List[Candidate]:
        explanations = explanations or {}
        rows = ranked.head(limit) if limit else ranked
        candidates: List[Candidate] = []
        for _, row in rows.iterrows():
            oid = int(row["object_id"])
            candidates.append(Candidate(
                rank=int(row["rank"]),
                object_id=oid,
                novelty_score=float(row["novelty_score"]),
                tier=str(row["tier"]),
                evidence={c: float(row[c]) for c in ALL_CHANNELS},
                weighted_evidence={c: round(float(row[c]) * self.norm_.get(c, 0.0), 6) for c in ALL_CHANNELS},
                quality=float(row["quality"]),
                uncertainty=float(row["uncertainty"]),
                confidence=float(row["confidence"]),
                reliability=float(row["reliability"]),
                abstain=bool(row["abstain"]),
                abstain_reason=str(row["abstain_reason"] or ""),
                best_fit_class=name_of(int(row["best_fit_code"])) if "best_fit_code" in row else "",
                best_fit_prob=float(row.get("best_fit_prob", 0.0)),
                true_class=name_of(int(row["true_code"])) if pd.notna(row.get("true_code", np.nan)) else None,
                is_novel=bool(row["is_novel"]) if "is_novel" in row and pd.notna(row["is_novel"]) else None,
                explanation=explanations.get(oid, {}),
            ))
        return candidates
