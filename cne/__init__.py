"""Cosmic Novelty Engine (CNE) v2.0.0.

Ranks astronomical alerts by how badly they are explained by known astrophysics,
so a human astronomer can triage a high-volume alert stream down to a shortlist
worth spectroscopic follow-up.

This is a discovery-ASSISTANCE and prioritisation system. It does not, and must
never claim to, discover new physics or certify a new astronomical phenomenon.
The string ``NEW DISCOVERY CONFIRMED`` is forbidden anywhere in this codebase;
permitted language is "candidate anomaly", "potentially novel candidate",
"requires expert follow-up", "poorly explained by current known populations".

Typical use::

    from cne.config import load_config
    from cne.pipeline import CNEPipeline

    pipeline = CNEPipeline(load_config())
    pipeline.build_train_features()
    pipeline.make_manifest()
    pipeline.fit_engine()
    pipeline.select_weights()
"""

from __future__ import annotations

from .version import CODENAME, SEED, __version__

__all__ = ["__version__", "SEED", "CODENAME", "load_config", "CNEPipeline"]


def load_config(*args, **kwargs):  # pragma: no cover - thin re-export
    from .config import load_config as _load

    return _load(*args, **kwargs)


def CNEPipeline(*args, **kwargs):  # pragma: no cover - thin re-export
    from .pipeline import CNEPipeline as _Pipeline

    return _Pipeline(*args, **kwargs)
