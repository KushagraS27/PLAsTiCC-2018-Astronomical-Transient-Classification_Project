"""Broker-neutral data schema.

CNE never lets a survey-specific schema leak past the adapter boundary
(``cne/adapters.py``).  Everything downstream of the adapters speaks
:class:`AlertPacket`, which is deliberately a superset of what PLAsTiCC provides
so a real ZTF or Rubin packet can be replayed without touching scoring code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Canonical photometry columns in CNE's internal frame.
PHOTOMETRY_COLUMNS = ("object_id", "mjd", "passband", "flux", "flux_err", "detected_bool")

#: Metadata columns CNE actually reads from PLAsTiCC. Anything else is ignored,
#: and the true_* columns are NEVER passed to a feature or a model.
METADATA_COLUMNS = (
    "object_id",
    "ra",
    "decl",
    "ddf_bool",
    "hostgal_specz",
    "hostgal_photoz",
    "hostgal_photoz_err",
    "distmod",
    "mwebv",
)
LABEL_COLUMN = "target"

#: Metadata dtypes chosen to keep a 3.5M-row PLAsTiCC test table inside ~150 MB.
METADATA_DTYPES = {
    "object_id": "int32",
    "ra": "float32",
    "decl": "float32",
    "ddf_bool": "int8",
    "hostgal_specz": "float32",
    "hostgal_photoz": "float32",
    "hostgal_photoz_err": "float32",
    "distmod": "float32",
    "mwebv": "float32",
}
PHOTOMETRY_DTYPES = {
    "object_id": "int32",
    "mjd": "float32",
    "passband": "int8",
    "flux": "float32",
    "flux_err": "float32",
    "detected_bool": "int8",
}


@dataclass(slots=True)
class AlertPacket:
    """One object's alert history in broker-neutral form.

    ``context`` carries survey metadata (position, host galaxy, redshift) and
    ``quality`` carries survey-provided data-quality signals such as a ZTF
    ``realbogus`` score.  Both may be empty; missing modalities are first-class.
    """

    object_id: int
    survey: str
    mjd: np.ndarray
    passband: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    detected: np.ndarray
    context: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(len(self.mjd))

    @property
    def n_detections(self) -> int:
        return int(self.detected.sum())

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "object_id": np.full(self.n_points, self.object_id, dtype="int32"),
                "mjd": self.mjd.astype("float32"),
                "passband": self.passband.astype("int8"),
                "flux": self.flux.astype("float32"),
                "flux_err": self.flux_err.astype("float32"),
                "detected_bool": self.detected.astype("int8"),
            }
        )

    def validate(self) -> List[str]:
        """Return human-readable problems; an empty list means the packet is sane."""
        problems: List[str] = []
        n = self.n_points
        if n == 0:
            return ["empty light curve"]
        for name, arr in (("mjd", self.mjd), ("passband", self.passband), ("flux", self.flux), ("flux_err", self.flux_err)):
            if len(arr) != n:
                problems.append(f"{name} length {len(arr)} != mjd length {n}")
        if np.any(self.flux_err <= 0):
            problems.append("non-positive flux_err")
        if np.any(~np.isfinite(self.flux)):
            problems.append("non-finite flux")
        if np.any(~np.isfinite(self.mjd)):
            problems.append("non-finite mjd")
        if not set(np.unique(self.passband).tolist()) <= set(range(6)):
            problems.append("passband outside 0..5")
        return problems


@dataclass(slots=True)
class CandidateEvidence:
    """Per-channel evidence for one object, before ranking."""

    object_id: int
    channels: Dict[str, float] = field(default_factory=dict)
    best_fit_class: Optional[str] = None
    best_fit_prob: Optional[float] = None
    runner_up_class: Optional[str] = None
    runner_up_prob: Optional[float] = None
    quality: float = 1.0
    uncertainty: float = 0.5
    reliability: float = 1.0
    domain_match: float = 1.0
    flags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class Candidate:
    """A ranked, explainable novelty candidate.

    Note the deliberate vocabulary: CNE emits *candidates*, never discoveries.
    The string "NEW DISCOVERY CONFIRMED" is forbidden anywhere in this codebase.
    """

    rank: int
    object_id: int
    novelty_score: float
    tier: str
    evidence: Dict[str, float]
    weighted_evidence: Dict[str, float]
    quality: float
    uncertainty: float
    confidence: float
    reliability: float
    abstain: bool = False
    abstain_reason: str = ""
    best_fit_class: str = ""
    best_fit_prob: float = 0.0
    true_class: Optional[str] = None
    is_novel: Optional[bool] = None
    analogs: List[Dict[str, Any]] = field(default_factory=list)
    explanation: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "object_id": int(self.object_id),
            "novelty_score": round(float(self.novelty_score), 6),
            "tier": self.tier,
            "evidence": {k: round(float(v), 6) for k, v in self.evidence.items()},
            "weighted_evidence": {k: round(float(v), 6) for k, v in self.weighted_evidence.items()},
            "quality": round(float(self.quality), 4),
            "uncertainty": round(float(self.uncertainty), 4),
            "confidence": round(float(self.confidence), 4),
            "reliability": round(float(self.reliability), 4),
            "abstain": bool(self.abstain),
            "abstain_reason": self.abstain_reason,
            "best_fit_class": self.best_fit_class,
            "best_fit_prob": round(float(self.best_fit_prob), 4),
            "true_class": self.true_class,
            "is_novel": self.is_novel,
            "analogs": self.analogs,
            "explanation": self.explanation,
            "provenance": self.provenance,
        }


def empty_photometry_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=PHOTOMETRY_DTYPES[c]) for c in PHOTOMETRY_COLUMNS})


def photometry_from_packet(packet: AlertPacket) -> pd.DataFrame:
    return packet.to_frame()


def packets_to_frame(packets: Sequence[AlertPacket]) -> pd.DataFrame:
    if not packets:
        return empty_photometry_frame()
    return pd.concat([p.to_frame() for p in packets], ignore_index=True)
