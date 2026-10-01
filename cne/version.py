"""Version, seed and environment stamping for the Cosmic Novelty Engine.

Every artefact produced by CNE carries this stamp so that a number can always be
traced back to the code, the seed and the library versions that produced it.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from typing import Any, Dict

__version__ = "2.0.0"

#: Project-wide deterministic seed (inherited from CNE v1.0.0, deliberately unchanged).
SEED = 42

#: Codename for this release line.
CODENAME = "domain-matched prior, hardened evaluation"

#: CNE v1.0.0 reference numbers, frozen so v2 regressions are always comparable.
FROZEN_V1_BASELINE: Dict[str, Any] = {
    "cne_version": "1.0.0",
    "dataset": "PLAsTiCC (Zenodo 2539456)",
    "ai01_roc_auc_macro": 0.9700,
    "ai01_top3_accuracy": 0.9608,
    "ai05_roc_auc": 0.7291,
    "ai05_average_precision": 0.2681,
    "ai05_lift": 2.46,
    "ai05_precision_at_10": 0.80,
    "ai05_base_rate": 0.1089,
    "unsupervised_ensemble_roc_auc": 0.451,
    "domain_matched_prior_roc_auc": 0.818,
    "out_of_domain_prior_roc_auc": 0.610,
}


def library_versions() -> Dict[str, str]:
    """Collect the versions of every numerically load-bearing dependency."""
    out: Dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "pyarrow", "fastapi"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # pragma: no cover - optional dependency
            out[name] = "absent"
    return out


def stamp(**extra: Any) -> Dict[str, Any]:
    """Return the canonical provenance stamp for an artefact."""
    payload: Dict[str, Any] = {
        "cne_version": __version__,
        "codename": CODENAME,
        "seed": SEED,
        "libraries": library_versions(),
    }
    payload.update(extra)
    return payload


def fingerprint(payload: Dict[str, Any]) -> str:
    """Stable short hash of a provenance payload (order-independent)."""
    import json

    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
