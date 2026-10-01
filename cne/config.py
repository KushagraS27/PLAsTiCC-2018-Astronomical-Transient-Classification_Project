"""Typed, layered configuration.

The YAML file in ``configs/`` is the single source of hyperparameters; this
module loads it into a dataclass tree so a typo fails loudly at startup instead
of silently changing a model's behaviour three stages later.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .version import SEED

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    artefacts_dir: str = "data/artefacts"
    reports_dir: str = "reports"
    train_metadata: str = "plasticc_train_metadata.csv.gz"
    train_lightcurves: str = "plasticc_train_lightcurves.csv.gz"
    test_metadata: str = "plasticc_test_metadata.csv.gz"
    test_lightcurves: List[str] = field(
        default_factory=lambda: ["plasticc_test_lightcurves_01.csv.gz"]
    )
    read_chunk_rows: int = 2_000_000

    def raw(self, name: str) -> Path:
        return Path(self.raw_dir) / getattr(self, name)

    def test_lightcurve_paths(self) -> List[Path]:
        """Every present test chunk, in order. Missing files are skipped so a
        partial download still runs."""
        return [Path(self.raw_dir) / f for f in self.test_lightcurves if (Path(self.raw_dir) / f).exists()]


@dataclass
class FeatureConfig:
    global_prefix: str = "lc_"
    quality_prefix: str = "q_"
    physics_prefix: str = "phys_"
    uncertainty_prefix: str = "unc_"
    host_prefix: str = "host_"
    colour_prefix: str = "col_"
    mc_redshift_draws: int = 24
    min_detections_for_features: int = 1


@dataclass
class QualityConfig:
    enabled: bool = True
    exponent: float = 0.5
    max_negative_flux_fraction: float = 0.35
    min_median_snr: float = 3.0
    min_detections: int = 5
    max_single_band_fraction: float = 0.95
    low_quality_score: float = 0.25


@dataclass
class PriorConfig:
    n_folds: int = 4
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    class_balanced: bool = True
    max_ens_ref: int = 3500


@dataclass
class SimilarityConfig:
    k_neighbours: int = 25
    n_analogs: int = 5


@dataclass
class UncertaintyConfig:
    exponent: float = 0.25
    n_bootstrap_models: int = 6
    abstain_quantile: float = 0.9
    abstain_min_quality: float = 0.45


@dataclass
class RankingConfig:
    agreement_bonus: float = 0.15
    tiers: Dict[str, float] = field(default_factory=lambda: {"critical": 0.95, "high": 0.85, "moderate": 0.60})


@dataclass
class EvaluationConfig:
    precision_k: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    recall_k: List[int] = field(default_factory=lambda: [200, 500])
    bootstrap_draws: int = 1000
    min_n_for_class_report: int = 30
    base_rates: List[float] = field(default_factory=lambda: [0.10, 0.01, 0.001, 0.0001])


@dataclass
class V2Config:
    locked_test_manifest: bool = True
    nested_weight_selection: bool = True
    domain_matched_reference: bool = True
    base_rate_stress: bool = True
    artifact_safety_suite: bool = True
    uncertainty_propagation: bool = True
    abstention: bool = True
    domain_shift_monitor: bool = True
    replay_adapter: bool = True
    multimodal_adapter: bool = False


@dataclass
class CNEConfig:
    seed: int = SEED
    config_id: str = "cne-default"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    weights: Dict[str, float] = field(default_factory=dict)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    v2: V2Config = field(default_factory=V2Config)
    source_path: str = ""

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> "CNEConfig":
        src = Path(path) if path else _DEFAULT_PATH
        raw: Dict[str, Any] = yaml.safe_load(src.read_text()) if src.exists() else {}
        if overrides:
            raw = _deep_merge(raw, overrides)
        meta = raw.get("meta", {})
        kwargs: Dict[str, Any] = {"seed": int(meta.get("seed", SEED)), "config_id": str(meta.get("config_id", "cne-default"))}
        for key, klass in (
            ("data", DataConfig),
            ("features", FeatureConfig),
            ("quality", QualityConfig),
            ("prior", PriorConfig),
            ("similarity", SimilarityConfig),
            ("uncertainty", UncertaintyConfig),
            ("ranking", RankingConfig),
            ("evaluation", EvaluationConfig),
            ("v2", V2Config),
        ):
            section = raw.get(key) or {}
            valid = {f: v for f, v in section.items() if f in klass.__dataclass_fields__}
            dropped = set(section) - set(valid)
            if dropped:  # surface typos instead of ignoring them
                raise KeyError(f"Unknown key(s) {sorted(dropped)} in config section '{key}'")
            kwargs[key] = klass(**valid)
        kwargs["weights"] = dict(raw.get("weights") or {})
        kwargs["source_path"] = str(src)
        return cls(**kwargs)

    # -- introspection --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out.pop("source_path", None)
        return out

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def weight_vector(self, channels: List[str]) -> Dict[str, float]:
        """Weights restricted to ``channels``, with unknown channels reported."""
        missing = [c for c in channels if c not in self.weights]
        if missing:
            raise KeyError(f"Channels absent from config.weights: {missing}")
        return {c: float(self.weights[c]) for c in channels}


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> CNEConfig:
    return CNEConfig.load(path, overrides)
