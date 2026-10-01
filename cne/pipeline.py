"""Pipeline orchestration: build, fit, score, evaluate, persist.

Evaluation protocol (frozen before any tuning runs)
---------------------------------------------------
``prior_fit``        known-class objects that fit the known-physics prior.
``validation``       known + held-out objects used ONLY to select channel weights.
``test_matched``     locked test stream in the reference's own survey domain.
``test_fullscale``   locked test stream from PLAsTiCC test chunk 01 - a different
                     domain, and the one the domain-matching module must earn.

Nothing in the weight-selection or threshold-tuning path may read a locked test
id; :class:`cne.manifests.LeakageGuard` enforces that and a test asserts it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .adapters import ReplayRunner, write_sample_alerts
from .artifacts import ArtifactSafetySuite
from .classifier import TransientClassifier
from .config import CNEConfig, load_config
from .data import (
    LightCurveCache,
    harvest_stream,
    iter_lightcurve_chunks,
    rare_preserving_subsample,
    read_features,
    read_lightcurves,
    read_metadata,
    write_features,
)
from .domain import DomainShiftMonitor, ReferenceBuilder
from .evaluation import (
    channel_power,
    evaluate_ranking,
    false_positive_audit,
    nan_to_none,
    per_class_metrics,
    weight_selection_optimism,
    write_metrics,
)
from .explain import explain_candidate
from .features import LightCurveFeaturiser
from .logging import get_logger, stage
from .manifests import LeakageGuard, SplitManifest, object_level_split
from .novelty import ALL_CHANNELS, CosmicNoveltyEngine
from .ranking import NoveltyRanker, RankingWeights
from .stress import review_budget_curve, stress_test
from .taxonomy import KNOWN_CLASS_CODES, WITHHELD_FROM_PRIOR, name_of
from .uncertainty import BootstrapEnsemble, evaluate_uncertainty_value, feature_reliability
from .version import FROZEN_V1_BASELINE, fingerprint, stamp
from .weights import NestedWeightSelector

log = get_logger("pipeline")

FEATURISE_CHUNK = 8000


@dataclass
class BenchmarkResult:
    name: str
    ranked: pd.DataFrame
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        return self.ranked["is_novel"].to_numpy().astype(int)

    @property
    def scores(self) -> np.ndarray:
        return self.ranked["novelty_score"].to_numpy(dtype="float64")


def _n_astro_features(features: Optional[pd.DataFrame], cfg) -> int:
    """Astrophysical feature count, derived from the loaded matrix."""
    if features is None:
        return 0
    q = cfg.features.quality_prefix
    return int(sum(1 for c in features.columns if c != "object_id" and not c.startswith(q)))


def _n_quality_features(features: Optional[pd.DataFrame], cfg) -> int:
    if features is None:
        return 0
    return int(sum(1 for c in features.columns if c.startswith(cfg.features.quality_prefix)))


class CNEPipeline:
    """The whole experiment, in one reproducible object."""

    def __init__(self, config: Optional[CNEConfig] = None, root: Optional[Path] = None):
        self.cfg = config or load_config()
        self.root = Path(root) if root else Path.cwd()
        self.processed = self.root / self.cfg.data.processed_dir
        self.artefacts = self.root / self.cfg.data.artefacts_dir
        self.reports = self.root / self.cfg.data.reports_dir
        for path in (self.processed, self.artefacts, self.reports):
            path.mkdir(parents=True, exist_ok=True)
        self.featuriser_ = LightCurveFeaturiser(self.cfg.features)
        self.train_features_: Optional[pd.DataFrame] = None
        self.stream_features_: Optional[pd.DataFrame] = None
        self.train_labels_: Optional[pd.Series] = None
        self.stream_labels_: Optional[pd.Series] = None
        self.manifest_: Optional[SplitManifest] = None
        self.guard_ = LeakageGuard(locked_split="validation")
        self.engine_: Optional[CosmicNoveltyEngine] = None
        self.engine_reweighted_: Optional[CosmicNoveltyEngine] = None
        self.engine_mismatched_: Optional[CosmicNoveltyEngine] = None
        self.ensemble_: Optional[BootstrapEnsemble] = None
        self.monitor_: Optional[DomainShiftMonitor] = None
        self.weights_v1_ = RankingWeights.v1()
        self.weights_v2_: Optional[RankingWeights] = None
        self.lc_cache_ = LightCurveCache(self.processed / "lightcurve_cache.parquet")
        self.run_started_ = time.time()

    # ===================================================================== #
    # 1. data
    # ===================================================================== #
    def build_train_features(self, force: bool = False) -> pd.DataFrame:
        """Featurise the whole PLAsTiCC train split (7,848 objects, all labels)."""
        out = self.processed / "train_features.parquet"
        if out.exists() and not force:
            self.train_features_ = read_features(out)
            self._load_train_labels()
            return self.train_features_
        with stage("featurise_train", log):
            meta = read_metadata(self.cfg.data.raw("train_metadata"), with_label=True)
            lc = read_lightcurves(self.cfg.data.raw("train_lightcurves"))
            features = self._featurise_in_chunks(lc, meta)
            self.featuriser_.fit_fill_values(features)
            features = self._featurise_in_chunks(lc, meta)  # re-run with the learned fill policy
            write_features(out, features)
            meta.to_parquet(self.processed / "train_metadata.parquet", index=False)
            self.train_features_, self.train_labels_ = features, meta.set_index("object_id")["target"]
            log.info("train features: %s", features.shape)
        return features

    def build_stream_features(self, max_objects: Optional[int] = None, force: bool = False) -> pd.DataFrame:
        """Harvest and featurise every available PLAsTiCC test chunk.

        Chunk-at-a-time streaming keeps peak memory at one chunk regardless of how
        many chunks are configured, and each chunk is featurised before the next is
        read so the raw photometry is never fully resident.
        """
        out = self.processed / "stream_features.parquet"
        if out.exists() and not force:
            self.stream_features_ = read_features(out)
            self._load_stream_labels()
            return self.stream_features_
        paths = self.cfg.data.test_lightcurve_paths()
        if not paths:
            raise FileNotFoundError("no PLAsTiCC test light-curve chunks found - run scripts/00_download.py")
        log.info("stream chunks available: %s", [p.name for p in paths])
        meta_all: List[pd.DataFrame] = []
        feature_parts: List[pd.DataFrame] = []
        lc_part_paths: List[Path] = []
        for index, lc_path in enumerate(paths):
            with stage(f"harvest_chunk_{index}", log):
                available = self._stream_object_ids(lc_path)
                meta = read_metadata(self.cfg.data.raw("test_metadata"), object_ids=available,
                                     with_label=True, label_column="true_target")
                if max_objects is not None and len(meta) > max_objects:
                    keep = rare_preserving_subsample(meta.set_index("object_id")["target"], max_objects,
                                                     seed=self.cfg.seed)
                    meta = meta[meta["object_id"].isin(keep)]
                part = self.processed / f"stream_lightcurves_part{index}.parquet"
                harvest = harvest_stream(lc_path, set(int(v) for v in meta["object_id"]), part,
                                         chunk_rows=self.cfg.data.read_chunk_rows)
                log.info("chunk %s: %s", lc_path.name, harvest)
            with stage(f"featurise_chunk_{index}", log):
                # Featurise straight off the parquet, one row group at a time.
                # pd.read_parquet(part) loaded the whole chunk - ~114M rows for
                # PLAsTiCC chunk 02 - which is the second OOM in this path.
                feature_parts.append(self._featurise_parquet(part, meta))
                # The light curves are NOT accumulated in memory. Keeping every
                # chunk's raw photometry resident and then concatenating it needed
                # ~1.7 GB for three chunks and OOM-killed the harvest mid-run on a
                # 2 GB host. Each chunk's sorted photometry is appended to the
                # cache as its own row group instead; objects never span chunks,
                # so the concatenated file is still grouped by object_id.
                self.lc_cache_.append_sorted(part)
                lc_part_paths.append(part)
                meta_all.append(meta)
        meta = pd.concat(meta_all, ignore_index=True)
        features = pd.concat(feature_parts, ignore_index=True)
        del feature_parts
        with stage("finalise_stream", log):
            self.lc_cache_.finalise()
            write_features(out, features)
            meta.to_parquet(self.processed / "stream_metadata.parquet", index=False)
        self.stream_features_, self.stream_labels_ = features, meta.set_index("object_id")["target"]
        log.info("stream features: %s | novel objects: %d (%.3f%% base rate)", features.shape,
                 int(self.stream_labels_.isin(list(WITHHELD_FROM_PRIOR)).sum()),
                 100.0 * float(self.stream_labels_.isin(list(WITHHELD_FROM_PRIOR)).mean()))
        return features

    def _stream_object_ids(self, lc_path: Path) -> List[int]:
        ids: set = set()
        for chunk in iter_lightcurve_chunks(lc_path, chunk_rows=self.cfg.data.read_chunk_rows):
            ids.update(int(v) for v in np.unique(chunk["object_id"].to_numpy()))
        return sorted(ids)

    def _featurise_parquet(self, part: Path, meta: pd.DataFrame) -> pd.DataFrame:
        """Featurise a harvested parquet without ever holding the whole chunk.

        Row groups are read one at a time into a buffer. Because the harvest
        writes photometry in source order, an object can only straddle a row-group
        boundary, so every object except the buffer's last is complete and can be
        featurised and dropped. Peak memory is one row group plus one
        FEATURISE_CHUNK block of objects.
        """
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(part)
        wanted = set(int(v) for v in meta["object_id"].to_numpy())
        parts: List[pd.DataFrame] = []
        buffer: Optional[pd.DataFrame] = None
        for group in range(parquet.num_row_groups):
            block = parquet.read_row_group(group).to_pandas()
            block = block[block["object_id"].isin(wanted)]
            buffer = block if buffer is None else pd.concat([buffer, block], ignore_index=True)
            del block
            ids = np.unique(buffer["object_id"].to_numpy())
            if len(ids) <= 1:
                continue
            # hold back the last id: it may continue into the next row group
            ready, tail_id = ids[:-1], int(ids[-1])
            ready_mask = buffer["object_id"].to_numpy() != tail_id
            parts.append(self._featurise_in_chunks(buffer[ready_mask], meta))
            buffer = buffer[~ready_mask].reset_index(drop=True)
            del ready_mask
        if buffer is not None and len(buffer):
            parts.append(self._featurise_in_chunks(buffer, meta))
            del buffer
        return pd.concat(parts, ignore_index=True)

    def _featurise_in_chunks(self, lc: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        ids = np.unique(lc["object_id"].to_numpy())
        parts = []
        for start in range(0, len(ids), FEATURISE_CHUNK):
            block_ids = ids[start : start + FEATURISE_CHUNK]
            block = lc[lc["object_id"].isin(block_ids)]
            block_meta = meta[meta["object_id"].isin(block_ids)]
            parts.append(self.featuriser_.transform(block, block_meta))
            del block
        out = pd.concat(parts, ignore_index=True)
        del parts
        return out

    def _load_train_labels(self) -> None:
        path = self.processed / "train_metadata.parquet"
        if path.exists():
            meta = pd.read_parquet(path)
            self.train_labels_ = meta.set_index("object_id")["target"]

    def _load_stream_labels(self) -> None:
        path = self.processed / "stream_metadata.parquet"
        if path.exists():
            meta = pd.read_parquet(path)
            self.stream_labels_ = meta.set_index("object_id")["target"]

    # ===================================================================== #
    # 2. protocol
    # ===================================================================== #
    def make_manifest(self, force: bool = False) -> SplitManifest:
        """Freeze every split BEFORE anything is fitted or tuned."""
        path = self.reports / "manifests" / "splits_v2.json"
        if path.exists() and not force:
            self.manifest_ = SplitManifest.read(path)
            if not self.manifest_.verify(path):
                raise RuntimeError(
                    f"frozen manifest {path} fails its fingerprint check - the split "
                    "ids on disk no longer match the recorded hashes. Refusing to "
                    "evaluate against a manifest that may have been edited."
                )
            # The guard must be rebuilt here too. Returning early left the
            # manifest-less default from __init__ in place, so locked_split was
            # "validation" with locked_n=0 and the guard verified nothing at all -
            # a silent no-op on the one control that prevents test-set leakage.
            self.guard_ = LeakageGuard(self.manifest_, locked_split="test_matched")
            log.info("loaded frozen manifest %s (guard locked on test_matched, n=%d)",
                     self.manifest_.manifest_id, len(self.manifest_.split("test_matched")))
            return self.manifest_
        if self.train_features_ is None:
            self.build_train_features()
        labels = self.train_labels_
        known = labels[labels.isin(list(KNOWN_CLASS_CODES))]
        novel = labels[labels.isin(list(WITHHELD_FROM_PRIOR))]
        known_ids = np.asarray(known.index, dtype="int64")
        novel_ids = np.asarray(novel.index, dtype="int64")
        known_splits = object_level_split(known_ids, {"prior_fit": 0.60, "validation_known": 0.20, "test_known": 0.20}, self.cfg.seed)
        novel_splits = object_level_split(novel_ids, {"validation_novel": 0.30, "test_novel": 0.70}, self.cfg.seed)
        validation = np.concatenate([known_splits["validation_known"], novel_splits["validation_novel"]])
        test_matched = np.concatenate([known_splits["test_known"], novel_splits["test_novel"]])
        test_fullscale = np.asarray([], dtype="int64")
        if self.stream_labels_ is not None:
            test_fullscale = np.asarray(self.stream_labels_.index, dtype="int64")
        splits = {
            "prior_fit": known_splits["prior_fit"],
            "validation": validation,
            "test_matched": test_matched,
            "test_fullscale": test_fullscale,
        }
        manifest = SplitManifest.create(
            manifest_id="cne-v2-splits",
            splits=splits,
            dataset={
                "name": "PLAsTiCC",
                "zenodo": "2539456",
                "train_lightcurves": self.cfg.data.train_lightcurves,
                "test_lightcurves": self.cfg.data.test_lightcurves,
            },
            config_fingerprint=self.cfg.fingerprint(),
            seed=self.cfg.seed,
            notes="validation_known+validation_novel are the ONLY objects weight selection may read",
        )
        manifest.assert_disjoint("prior_fit", "validation", "test_matched")
        manifest.write(path, force=True)
        self.manifest_ = manifest
        # The guard protects the locked test streams; validation is what tuning sees.
        self.guard_ = LeakageGuard(manifest, locked_split="test_matched")
        return manifest

    # ===================================================================== #
    # 3. fit
    # ===================================================================== #
    def fit_engine(self) -> CosmicNoveltyEngine:
        """Fit the known-physics prior on the prior_fit split (12 known classes)."""
        with stage("fit_engine", log):
            ids = self.manifest_.split("prior_fit")
            reference = self.train_features_[self.train_features_["object_id"].isin(ids)].reset_index(drop=True)
            labels = self.train_labels_.reindex(reference["object_id"]).to_numpy()
            self.engine_ = CosmicNoveltyEngine(self.cfg, seed=self.cfg.seed)
            self.engine_.fit(reference, labels)
            self.ensemble_ = BootstrapEnsemble(
                self.cfg.prior, n_models=self.cfg.uncertainty.n_bootstrap_models, seed=self.cfg.seed
            ).fit(reference[self.engine_.feature_names_], labels, self.engine_.feature_names_)
            self.monitor_ = DomainShiftMonitor().fit(reference)
        return self.engine_

    def fit_mismatched_engine(self) -> CosmicNoveltyEngine:
        """Deliberately out-of-domain reference, to measure the degradation curve."""
        with stage("fit_mismatched_engine", log):
            ids = self.manifest_.split("prior_fit")
            reference = self.train_features_[self.train_features_["object_id"].isin(ids)].reset_index(drop=True)
            labels = pd.Series(self.train_labels_.reindex(reference["object_id"]).to_numpy())
            # Keep only the two most luminous, most slowly varying populations: a
            # reference that could not possibly represent a faint fast transient.
            biased_codes = [90, 88]
            keep = labels.isin(biased_codes).to_numpy()
            if keep.sum() < 100:
                keep = labels.isin(list(KNOWN_CLASS_CODES)[:2]).to_numpy()
            biased = reference[keep].reset_index(drop=True)
            self.engine_mismatched_ = CosmicNoveltyEngine(self.cfg, seed=self.cfg.seed)
            self.engine_mismatched_.fit(biased, labels[keep].to_numpy(), domain="mismatched")
            log.info("mismatched reference: %d objects, classes %s", len(biased),
                     sorted(set(labels[keep].tolist())))
        return self.engine_mismatched_

    def fit_reweighted_engine(self) -> CosmicNoveltyEngine:
        """Density-ratio reweighted reference aimed at the full-scale stream (prompt 03)."""
        with stage("fit_reweighted_engine", log):
            ids = self.manifest_.split("prior_fit")
            reference = self.train_features_[self.train_features_["object_id"].isin(ids)].reset_index(drop=True)
            labels = self.train_labels_.reindex(reference["object_id"]).to_numpy()
            covariates = [c for c in ("lc_snr_max", "lc_n_det", "lc_t_span", "lc_peak_mag",
                                      "phys_z", "phys_distmod", "q_n_bands", "q_det_frac")
                          if c in reference.columns and c in self.stream_features_.columns]
            builder = ReferenceBuilder(covariates=covariates, seed=self.cfg.seed)
            weights, info = builder.build(reference, self.stream_features_, strategy="density_ratio")
            self.domain_info_ = info
            self.domain_match_ = builder.match_score(reference, self.stream_features_, strategy="density_ratio")
            self.engine_reweighted_ = CosmicNoveltyEngine(self.cfg, seed=self.cfg.seed)
            self.engine_reweighted_.fit(reference, labels, sample_weight=weights, domain="density_ratio_matched")
        return self.engine_reweighted_

    # ===================================================================== #
    # 4. score
    # ===================================================================== #
    def _stream_metadata_for(self, features: pd.DataFrame) -> pd.DataFrame:
        """Metadata rows for a feature block, from whichever split it came from."""
        ids = set(int(v) for v in features["object_id"].to_numpy())
        for name in ("stream_metadata.parquet", "train_metadata.parquet"):
            path = self.processed / name
            if not path.exists():
                continue
            meta = pd.read_parquet(path)
            if ids <= set(int(v) for v in meta["object_id"].to_numpy()):
                return meta
        return pd.DataFrame(columns=["object_id"])

    def _labels_for(self, features: pd.DataFrame) -> Optional[pd.Series]:
        """Resolve true labels for a feature block, from whichever split it came from."""
        ids = set(int(v) for v in features["object_id"].to_numpy())
        for source in (self.stream_labels_, self.train_labels_):
            if source is not None and ids <= set(int(v) for v in source.index):
                return source
        return None

    def score_stream(self, features: pd.DataFrame, engine: CosmicNoveltyEngine,
                     weights: Optional[RankingWeights] = None,
                     domain_match: float = 1.0) -> pd.DataFrame:
        with stage("score_stream", log):
            evidence = engine.score(features, chunk=4000)
            uncertainty, _max_prob, _disagree = self.ensemble_.uncertainty(features[engine.feature_names_])
            reliability = feature_reliability(features)
            evidence["domain_match"] = float(domain_match)
            ranker = NoveltyRanker(self.cfg, weights or self.weights_v1_)
            return ranker.rank(
                evidence,
                uncertainty=uncertainty,
                reliability=reliability,
                labels=self._labels_for(features),
                novel_codes=list(WITHHELD_FROM_PRIOR),
            )

    # ===================================================================== #
    # 5. weights (validation only)
    # ===================================================================== #
    def select_weights(self) -> RankingWeights:
        if not self.cfg.v2.nested_weight_selection:
            self.weights_v2_ = self.weights_v1_
            return self.weights_v1_
        with stage("select_weights", log):
            ids = self.manifest_.split("validation")
            validation = self.train_features_[self.train_features_["object_id"].isin(ids)].reset_index(drop=True)
            self.guard_.stage("weight_search").seen_ids(validation["object_id"].to_numpy())
            evidence = self.engine_.score(validation, chunk=4000)
            y = np.isin(self.train_labels_.reindex(validation["object_id"]).to_numpy(),
                        list(WITHHELD_FROM_PRIOR)).astype(int)
            selector = NestedWeightSelector(guard=self.guard_)
            result = selector.search(evidence, y)
            result.n_test = len(self.manifest_.split("test_matched")) + len(self.manifest_.split("test_fullscale"))
            self.guard_.assert_clean()
            selector.write_config(result, self.root / "configs" / "weights_v2.yaml")
            selector.write_report(result, self.reports / "weight_selection.json")
            self.weight_search_result_ = result
            self.weights_v2_ = RankingWeights(result.weights)
            log.info("selected weights: %s", {k: v for k, v in result.weights.items() if v > 0})
        return self.weights_v2_

    # ===================================================================== #
    # 6. explain
    # ===================================================================== #
    def build_explanations(self, features: pd.DataFrame, ranked: pd.DataFrame, engine: CosmicNoveltyEngine,
                           limit: int = 60) -> Tuple[Dict[int, Dict[str, Any]], pd.DataFrame]:
        top = ranked.head(limit)
        subset = features[features["object_id"].isin(top["object_id"])].reset_index(drop=True)
        analogues = engine.analogue_payload(subset, k=self.cfg.similarity.n_analogs)
        weights = (self.weights_v2_ or self.weights_v1_).normalised()
        explanations: Dict[int, Dict[str, Any]] = {}
        analog_map: Dict[int, List[Dict[str, Any]]] = {}
        for _, row in top.iterrows():
            oid = int(row["object_id"])
            analogs = analogues.get(oid, [])
            explanations[oid] = explain_candidate(row, engine, analogs, weights, features=subset)
            analog_map[oid] = analogs
        # Assign the whole column at once. Per-row assignment through .loc with a
        # boolean mask and a list value makes pandas broadcast element-wise and
        # raise "Must have equal len keys and value when setting with an ndarray"
        # - which only surfaced once the top candidate had >1 analogue.
        enriched = ranked.copy()
        enriched["analogs"] = [analog_map.get(int(v), []) for v in enriched["object_id"]]
        return explanations, enriched

    # ===================================================================== #
    # 7. full evaluation
    # ===================================================================== #
    def run_full_evaluation(self, do_mismatched: bool = True, do_reweighted: bool = True,
                            bootstrap_draws: Optional[int] = None) -> Dict[str, Any]:
        """Run every v2 benchmark and write the consolidated metrics bundle."""
        from .experiments import (
            ablation_study,
            domain_degradation_study,
            summarise_benchmark,
            uncertainty_study,
        )
        from .stress import review_budget_curve, stress_test
        from .evaluation import write_metrics

        draws = bootstrap_draws or self.cfg.evaluation.bootstrap_draws
        cfg = self.cfg
        cfg.evaluation.bootstrap_draws = draws

        self.build_train_features()
        have_stream = (self.root / cfg.data.processed_dir / "stream_features.parquet").exists()
        if have_stream:
            self.build_stream_features()
        self.make_manifest()
        self.fit_engine()
        if do_mismatched:
            self.fit_mismatched_engine()
        if do_reweighted and have_stream:
            self.fit_reweighted_engine()
        weights = self.select_weights()

        out: Dict[str, Any] = {
            "provenance": stamp(
                config_id=cfg.config_id,
                config_fingerprint=cfg.fingerprint(),
                manifest=self.manifest_.manifest_id,
                splits={k: len(v) for k, v in self.manifest_.splits.items()},
                # Read the feature inventory from the matrix actually loaded, not
                # from self.featuriser_: when features come from a parquet the
                # featuriser never ran transform() in this process and its
                # feature_names() raises.
                n_features=_n_astro_features(self.train_features_, self.cfg),
                n_quality_features=_n_quality_features(self.train_features_, self.cfg),
                dataset=self.manifest_.dataset,
            ),
            "frozen_v1_baseline": FROZEN_V1_BASELINE,
            "prior": self.engine_.prior_.summary().as_dict(),
            "weight_search": getattr(self, "weight_search_result_", None)
            and self.weight_search_result_.as_dict(),
            "leakage_guard": self.guard_.report(),
            "benchmarks": {},
        }

        # ---------------- B1: domain-matched benchmark (headline) -------------
        ids = self.manifest_.split("test_matched")
        test = self.train_features_[self.train_features_["object_id"].isin(ids)].reset_index(drop=True)
        for label, w in (("baseline_v1", self.weights_v1_), ("nested_v2", weights)):
            ranked = self.score_stream(test, self.engine_, weights=w)
            codes = self.train_labels_.reindex(ranked["object_id"]).to_numpy()
            out["benchmarks"][f"matched_{label}"] = summarise_benchmark(
                f"matched_{label}", ranked, codes, cfg, w,
                extra={"protocol": "PLAsTiCC train split; reference drawn from the same survey domain",
                       "n_abstained": int(ranked["abstain"].sum()),
                       "abstention_rate": float(ranked["abstain"].mean())})
            if label == "nested_v2":
                self.ranked_matched_ = ranked
                out["stress_matched"] = stress_test(ranked["is_novel"].to_numpy().astype(int),
                                                    ranked["novelty_score"].to_numpy(dtype="float64"),
                                                    target_rates=cfg.evaluation.base_rates,
                                                    draws=min(draws, 400), seed=cfg.seed).as_dict()
                out["review_budget_matched"] = review_budget_curve(
                    ranked["is_novel"].to_numpy().astype(int),
                    ranked["novelty_score"].to_numpy(dtype="float64"), target_rate=0.004)
                out["ablation_matched"] = ablation_study(ranked, cfg, w).reset_index().to_dict(orient="records")
                out["uncertainty_matched"] = uncertainty_study(ranked)
                out["domain_health_matched"] = self.monitor_.health(test) if self.monitor_ else {}

        # ---------------- B2: full-scale, cross-domain ------------------------
        if have_stream and self.stream_features_ is not None:
            stream = self.stream_features_.reset_index(drop=True)
            configs = [("fullscale_naive", self.engine_, weights, 1.0,
                        "train reference applied to a different survey domain, unweighted")]
            if self.engine_reweighted_ is not None:
                match = getattr(self, "domain_match_", None)
                configs.append(("fullscale_domain_matched", self.engine_reweighted_, weights,
                                match.score if match else 1.0,
                                "train reference reweighted by capped density ratio to the stream domain"))
            for name, engine, w, dm, protocol in configs:
                ranked = self.score_stream(stream, engine, weights=w, domain_match=dm)
                codes = self.stream_labels_.reindex(ranked["object_id"]).to_numpy()
                out["benchmarks"][name] = summarise_benchmark(
                    name, ranked, codes, cfg, w,
                    extra={"protocol": protocol, "domain_match": getattr(self, "domain_match_", None)
                           and self.domain_match_.as_dict(),
                           "reference_build": getattr(self, "domain_info_", None)})
                if name == "fullscale_domain_matched" or self.engine_reweighted_ is None:
                    self.ranked_fullscale_ = ranked
                    y = ranked["is_novel"].to_numpy().astype(int)
                    s = ranked["novelty_score"].to_numpy(dtype="float64")
                    out["stress_fullscale"] = stress_test(y, s, target_rates=cfg.evaluation.base_rates,
                                                          draws=min(draws, 400), seed=cfg.seed).as_dict()
                    out["review_budget_fullscale"] = review_budget_curve(y, s, target_rate=0.004)
                    out["ablation_fullscale"] = ablation_study(ranked, cfg, w).reset_index().to_dict(orient="records")
                    out["uncertainty_fullscale"] = uncertainty_study(ranked)
                    out["domain_health_fullscale"] = self.monitor_.health(stream) if self.monitor_ else {}

            if self.engine_mismatched_ is not None:
                ranked = self.score_stream(stream, self.engine_mismatched_, weights=weights)
                codes = self.stream_labels_.reindex(ranked["object_id"]).to_numpy()
                out["benchmarks"]["fullscale_mismatched"] = summarise_benchmark(
                    "fullscale_mismatched", ranked, codes, cfg, weights,
                    extra={"protocol": "deliberately out-of-domain reference (SNIa + AGN only)"})
            # Three genuinely distinct references. Passing fullscale_domain_matched
            # twice - as both the "matched" and the "reweighted" slot - made two
            # of the three rows bit-identical and the comparison meaningless.
            out["domain_degradation"] = domain_degradation_study(
                out["benchmarks"].get("fullscale_naive", {}),
                out["benchmarks"].get("fullscale_domain_matched", {}),
                out["benchmarks"].get("fullscale_mismatched", {}),
            ).reset_index().to_dict(orient="records")

        # ---------------- weight-selection optimism ---------------------------
        ids = self.manifest_.split("validation")
        validation = self.train_features_[self.train_features_["object_id"].isin(ids)].reset_index(drop=True)
        evidence = self.engine_.score(validation, chunk=4000)
        y_val = np.isin(self.train_labels_.reindex(validation["object_id"]).to_numpy(),
                        list(WITHHELD_FROM_PRIOR)).astype(int)
        from .experiments import optimism_study

        out["weight_selection_optimism"] = optimism_study(evidence, y_val, weights.as_dict())

        # ---------------- candidate queue for the dashboard -------------------
        primary = getattr(self, "ranked_matched_", None)
        if primary is not None:
            features = self.train_features_
            explanations, enriched = self.build_explanations(features, primary, self.engine_, limit=80)
            enriched.head(400).to_parquet(self.artefacts / "candidates_matched.parquet", index=False)
            (self.artefacts / "explanations_matched.json").write_text(json.dumps(explanations, indent=1, default=str))
        fullscale = getattr(self, "ranked_fullscale_", None)
        if fullscale is not None:
            explanations, enriched = self.build_explanations(self.stream_features_, fullscale,
                                                             self.engine_reweighted_ or self.engine_, limit=80)
            enriched.head(400).to_parquet(self.artefacts / "candidates_fullscale.parquet", index=False)
            (self.artefacts / "explanations_fullscale.json").write_text(json.dumps(explanations, indent=1, default=str))

        out["runtime_s"] = round(time.time() - self.run_started_, 1)
        write_metrics(nan_to_none(out), self.reports / "metrics.json")
        log.info("metrics -> %s (runtime %.0fs)", self.reports / "metrics.json", out["runtime_s"])
        return out
