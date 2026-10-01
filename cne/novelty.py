"""The Cosmic Novelty Engine: known-physics prior, evidence channels, scoring.

The central finding this module encodes
---------------------------------------
Generic unsupervised outlier detection does not find subtle astronomical novelty.
Measured on held-out PLAsTiCC populations every generic detector sits at or below
chance (IF 0.451, PCA 0.465, autoencoder 0.478, Mahalanobis 0.467, kNN 0.488),
while a *domain-matched* model of known physics reaches 0.818. Calcium-rich
transients and ILOTs photometrically resemble faint fast supernovae - they are
not global outliers - while the loudest outliers in any feature space are bright
known variables. Outlierness and novelty are close to orthogonal here.

So novelty is defined as the failure of a domain-matched known-physics model, not
as distance from a density.

Channel definitions (higher always means "less well explained")
--------------------------------------------------------------
====================  =========================================================
taxonomy_gap          1 - max_k p_k. The prior's confusion. Primary signal.
prior_entropy         H(p)/log K. Diffuse assignment across known classes.
simplex_novelty       0.5 * ||p - c_nearest||_1: distance on the probability
                      simplex to the nearest known-class OOF probability
                      centroid. Asks whether the object's *confusion pattern*
                      resembles any confusion pattern the known population
                      actually produces.
novelty_gap           1 - (p1 - p2). Small decision margin = ambiguous.
neighbor_entropy      Normalised entropy of the k-NN class distribution in
                      standardised feature space.
anomaly_score         Rank-fused unsupervised ensemble. AUDIT ONLY, weight 0.
physics_gap           Analytic class-conditional chi^2 of physics features.
                      AUDIT ONLY, weight 0.
cc_weighted           Probability-weighted class-conditional Mahalanobis.
                      AUDIT ONLY, weight 0.
family_misfit         Gate-vs-neighbourhood galactic-family disagreement.
                      Measured anti-correlated with novelty; weight 0.
====================  =========================================================

Zero-weighted channels stay computed. That is a deliberate choice: they are the
audit trail for the negative result, and an operator can re-weight them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .classifier import TransientClassifier, _macro_auc
from .config import CNEConfig, PriorConfig, SimilarityConfig
from .logging import get_logger
from .models import (
    AnomalyEnsemble,
    ConditionalCalibrator,
    QualityModel,
    SimilarityIndex,
    Standardiser,
    QuantileMapper,
    _rank01,
)
from .taxonomy import family_of

log = get_logger("novelty")

#: Every evidence channel the engine can emit, in canonical order.
ALL_CHANNELS: Tuple[str, ...] = (
    "taxonomy_gap",
    "prior_entropy",
    "simplex_novelty",
    "novelty_gap",
    "neighbor_entropy",
    "anomaly_score",
    "physics_gap",
    "cc_weighted",
    "family_misfit",
)


@dataclass
class PriorSummary:
    classes: np.ndarray
    accuracy: float
    roc_auc: float
    n_objects: int
    n_folds: int
    domain: str = "matched"

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_classes": int(len(self.classes)),
            "classes": [str(c) for c in self.classes],
            "cv_accuracy": round(float(self.accuracy), 4),
            "cv_roc_auc_macro": round(float(self.roc_auc), 4),
            "n_objects": int(self.n_objects),
            "n_folds": int(self.n_folds),
            "domain": self.domain,
        }


class KnownPhysicsPrior:
    """A LightGBM model of the KNOWN classes, fit on a domain-matched reference.

    The reference population is the load-bearing decision in the whole system:
    the same metric on an out-of-domain reference falls from AUC 0.818 to 0.610.
    """

    def __init__(self, config: Optional[PriorConfig] = None, seed: int = 42):
        self.cfg = config or PriorConfig()
        self.seed = seed
        self.clf_ = TransientClassifier(self.cfg, seed=seed)
        self.feature_names_: List[str] = []
        self.centroids_: Optional[np.ndarray] = None
        self.class_stats_: Optional[pd.DataFrame] = None
        self.sample_weights_: Optional[np.ndarray] = None
        self.domain_: str = "matched"

    def fit(self, features: pd.DataFrame, labels: np.ndarray, feature_names: Sequence[str],
            sample_weight: Optional[np.ndarray] = None, domain: str = "matched") -> "KnownPhysicsPrior":
        self.feature_names_ = list(feature_names)
        self.sample_weights_ = sample_weight
        self.domain_ = domain
        if sample_weight is not None:
            # Density-ratio weighting (CNE v2 domain matching): per-sample
            # LightGBM weights, capped inside _fit_weighted.
            self._fit_weighted(features, labels, sample_weight)
        else:
            self.clf_.fit(features[self.feature_names_], labels, self.feature_names_)
        # Out-of-fold probability centroids per known class: the reference
        # "confusion patterns" that simplex_novelty compares against.
        oof = self.clf_.oof_proba_
        codes = np.unique(np.asarray(labels))
        self.centroids_ = np.stack([oof[np.asarray(labels) == c].mean(axis=0) for c in codes])
        self.class_stats_ = self._class_statistics(features, labels)
        log.info("known-physics prior fitted classes=%d n=%d domain=%s", len(codes), len(features), domain)
        return self

    def _fit_weighted(self, features, labels, sample_weight) -> None:
        """Weighted variant: rebuild the fold models with per-sample weights."""
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold

        x_arr = features[self.feature_names_].to_numpy(dtype="float32")
        w = np.clip(np.asarray(sample_weight, dtype="float64"), 1e-3, 20.0)
        classes, y_enc = np.unique(np.asarray(labels), return_inverse=True)
        y_enc = y_enc.ravel()
        params = self.clf_._params(len(classes), np.bincount(y_enc, minlength=len(classes)))
        n_splits = max(2, min(self.cfg.n_folds, int(np.min(np.bincount(y_enc, minlength=len(classes))))))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros((len(x_arr), len(classes)))
        models = []
        for tr, va in skf.split(x_arr, y_enc):
            model = lgb.LGBMClassifier(**params)
            model.fit(x_arr[tr], y_enc[tr], sample_weight=w[tr])
            oof[va] = model.predict_proba(x_arr[va])
            models.append(model)
        self.clf_.models_ = models
        self.clf_.classes_ = classes
        self.clf_.oof_proba_ = oof
        # Without this the wrapped classifier keeps the empty feature_names_ it
        # was constructed with, and probabilities() then selects zero columns:
        # "Found array with 0 feature(s) (shape=(10, 0))".
        self.clf_.feature_names_ = list(self.feature_names_)

        from sklearn.metrics import accuracy_score, log_loss

        self.clf_.metrics_ = {
            "accuracy": float(accuracy_score(y_enc, oof.argmax(axis=1))),
            "log_loss": float(log_loss(y_enc, oof, labels=np.arange(len(classes)))),
            # binary-safe: the mismatched reference has only two classes
            "roc_auc_macro": _macro_auc(y_enc, oof, len(classes)),
            "n_classes": int(len(classes)),
            "n_objects": int(len(x_arr)),
            "n_folds": int(n_splits),
        }

    @staticmethod
    def _class_statistics(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Per-class mean/std of every feature - used by physics_gap and by the
        explanation layer ("which features sit outside this class's normal range").
        """
        df = features.assign(_label=np.asarray(labels))
        grouped = df.groupby("_label")
        stats = grouped.agg(["mean", "std"])
        return stats

    def summary(self) -> PriorSummary:
        return PriorSummary(
            classes=self.clf_.classes_,
            accuracy=self.clf_.metrics_.get("accuracy", float("nan")),
            roc_auc=self.clf_.metrics_.get("roc_auc_macro", float("nan")),
            n_objects=self.clf_.metrics_.get("n_objects", 0),
            n_folds=self.clf_.metrics_.get("n_folds", 0),
            domain=self.domain_,
        )

    def probabilities(self, features: pd.DataFrame) -> np.ndarray:
        return self.clf_.predict_proba(features[self.feature_names_])

    def class_normal_range(self, class_code, feature: str) -> Tuple[float, float]:
        """Mean +/- 2 sigma of one feature within one known class."""
        stats = self.class_stats_
        if stats is None or class_code not in stats.index.get_level_values(0):
            return (float("nan"), float("nan"))
        mean = float(stats.loc[class_code][(feature, "mean")])
        std = float(stats.loc[class_code][(feature, "std")])
        return mean - 2 * std, mean + 2 * std


class NoveltyEvidence:
    """Turns prior probabilities and neighbourhoods into the evidence channels."""

    def __init__(self, prior: KnownPhysicsPrior, similarity: Optional[SimilarityIndex] = None,
                 anomaly: Optional[AnomalyEnsemble] = None, physics_columns: Optional[Sequence[str]] = None,
                 quantiles: Optional["QuantileMapper"] = None):
        self.prior = prior
        self.similarity = similarity
        self.anomaly = anomaly
        self.physics_columns = list(physics_columns or [])
        self.quantiles_ = quantiles

    # ------------------------------------------------------------- probability
    def _map_channel(self, name: str, values: np.ndarray) -> np.ndarray:
        """Map a raw channel onto the reference grid; fall back to local rank."""
        if self.quantiles_ is None or not self.quantiles_.fitted_:
            return _rank01(values)
        return self.quantiles_.transform(name, values)

    def from_probabilities(self, proba: np.ndarray) -> Dict[str, np.ndarray]:
        proba = np.clip(np.asarray(proba, dtype="float64"), 1e-12, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)
        sorted_p = -np.sort(-proba, axis=1)
        top1, top2 = sorted_p[:, 0], sorted_p[:, 1] if proba.shape[1] > 1 else np.zeros(len(proba))
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.sum(proba * np.log(proba), axis=1) / np.log(max(proba.shape[1], 2))
        # Distance on the probability simplex to the nearest known-class centroid.
        if self.prior.centroids_ is not None and len(self.prior.centroids_):
            centroids = self.prior.centroids_
            l1 = np.abs(proba[:, None, :] - centroids[None, :, :]).sum(axis=2)
            nearest = l1.min(axis=1)
        else:
            nearest = np.zeros(len(proba))
        return {
            "taxonomy_gap": 1.0 - top1,
            "prior_entropy": np.clip(entropy, 0.0, 1.0),
            "simplex_novelty": np.clip(0.5 * nearest, 0.0, 1.0),
            "novelty_gap": 1.0 - np.clip(top1 - top2, 0.0, 1.0),
        }

    # -------------------------------------------------------------- additional
    def neighbour_channels(self, similarity_result: Dict[str, np.ndarray], proba: np.ndarray) -> Dict[str, np.ndarray]:
        out = {"neighbor_entropy": np.clip(similarity_result["neighbour_entropy"], 0.0, 1.0)}
        classes = self.prior.clf_.classes_
        probs = similarity_result["class_probs"]
        neighbour_class_codes = np.asarray(similarity_result["classes"])
        galactic_neighbour = np.zeros(len(probs))
        for i, code in enumerate(neighbour_class_codes):
            if family_of(int(code)) == "galactic":
                galactic_neighbour += probs[:, i]
        gate_galactic = np.zeros(len(proba))
        for i, code in enumerate(classes):
            if family_of(int(code)) == "galactic":
                gate_galactic += proba[:, i]
        # Disagreement between the classifier's family assignment and the
        # neighbourhood's. Measured ANTI-correlated with novelty; weight 0.
        out["family_misfit"] = np.abs(gate_galactic - galactic_neighbour)
        return out

    def physics_channels_raw(self, features: pd.DataFrame, proba: np.ndarray) -> Dict[str, np.ndarray]:
        """Un-normalised physics misfit: the quantity the mapper is fitted on."""
        stats = self.prior.class_stats_
        if stats is None or not self.physics_columns:
            n = len(features)
            return {"physics_gap": np.zeros(n), "cc_weighted": np.zeros(n)}
        codes = self.prior.clf_.classes_
        chi2 = np.zeros((len(features), len(codes)))
        for j, code in enumerate(codes):
            if code not in stats.index.get_level_values(0):
                continue
            for k, col in enumerate(self.physics_columns):
                if (col, "mean") not in stats.columns:
                    continue
                mean = float(stats.loc[code][(col, "mean")])
                std = float(stats.loc[code][(col, "std")])
                if not np.isfinite(std) or std < 1e-9:
                    continue
                vals = features[col].to_numpy(dtype="float64")
                chi2[:, j] += np.nan_to_num(((vals - mean) / std) ** 2, nan=0.0, posinf=0.0)
        chi2 = chi2 / max(len(self.physics_columns), 1)
        # Probability-weighted expected misfit under the prior's own belief.
        cc = np.sum(proba * chi2, axis=1)
        gap = chi2.min(axis=1)
        return {"physics_gap": gap, "cc_weighted": cc}

    def physics_channels(self, features: pd.DataFrame, proba: np.ndarray) -> Dict[str, np.ndarray]:
        """Physics misfit mapped onto the reference quantile grid (batch-independent)."""
        raw = self.physics_channels_raw(features, proba)
        if self.quantiles_ is None or not self.quantiles_.fitted_:
            return {k: _rank01(v) for k, v in raw.items()}
        return {k: self.quantiles_.transform(k, v) for k, v in raw.items()}


class CosmicNoveltyEngine:
    """End-to-end scorer: fit on a reference, score a stream, emit evidence."""

    def __init__(self, config: Optional[CNEConfig] = None, seed: int = 42):
        self.cfg = config or CNEConfig.load()
        self.seed = seed
        self.prior_: Optional[KnownPhysicsPrior] = None
        self.standardiser_: Optional[Standardiser] = None
        self.similarity_: Optional[SimilarityIndex] = None
        self.anomaly_: Optional[AnomalyEnsemble] = None
        self.quality_ = QualityModel(self.cfg.quality)
        self.quantiles_: Optional[QuantileMapper] = None
        self.feature_names_: List[str] = []
        self.astro_columns_: List[str] = []
        self.physics_columns_: List[str] = []
        self.reference_ids_: Optional[np.ndarray] = None

    # --------------------------------------------------------------------- fit
    def fit(self, features: pd.DataFrame, labels: np.ndarray,
            sample_weight: Optional[np.ndarray] = None, domain: str = "matched") -> "CosmicNoveltyEngine":
        quality_cols = [c for c in features.columns if c.startswith(self.cfg.features.quality_prefix)]
        # Quality features are excluded from every astrophysical model input.
        self.astro_columns_ = [c for c in features.columns if c not in quality_cols and c != "object_id"]
        self.physics_columns_ = [c for c in self.astro_columns_ if c.startswith(self.cfg.features.physics_prefix)]
        self.feature_names_ = list(self.astro_columns_)
        self.reference_ids_ = features["object_id"].to_numpy()

        self.prior_ = KnownPhysicsPrior(self.cfg.prior, seed=self.seed)
        self.prior_.fit(features, labels, self.feature_names_, sample_weight=sample_weight, domain=domain)

        self.standardiser_ = Standardiser().fit(features, self.astro_columns_)
        x_ref = self.standardiser_.transform(features)
        sim_cfg = SimilarityConfig(k_neighbours=self.cfg.similarity.k_neighbours)
        self.similarity_ = SimilarityIndex(sim_cfg).fit(x_ref, labels, features["object_id"].to_numpy())
        self.anomaly_ = AnomalyEnsemble(seed=self.seed, max_ref=self.cfg.prior.max_ens_ref).fit(x_ref)

        # Fix the rank-normalisation grids on the reference population so that
        # channel values are batch-independent: without this, _rank01 makes an
        # object's score depend on which chunk of the stream it happened to be
        # scored in. Scoring the reference through itself is the honest choice -
        # no held-out novel labels are touched, and the grid is a property of the
        # frozen model, not of whatever stream arrives.
        ref_proba = self.prior_.probabilities(features)
        probe = NoveltyEvidence(self.prior_, self.similarity_, self.anomaly_,
                                self.physics_columns_, quantiles=None)
        raw = dict(probe.physics_channels_raw(features, ref_proba))
        raw["anomaly_score"] = self.anomaly_.score_fused(x_ref)
        self.quantiles_ = QuantileMapper().fit(raw)

        log.info("engine fitted on %d reference objects, %d features (%d physics); "
                 "quantile grids: %s",
                 len(features), len(self.feature_names_), len(self.physics_columns_),
                 sorted(self.quantiles_.grids_))
        return self

    # ------------------------------------------------------------------- score
    def score(self, features: pd.DataFrame, chunk: int = 4000) -> pd.DataFrame:
        """Compute every evidence channel for a stream.

        Chunked: the dense design matrix for a large stream plus the ensemble
        intermediates peaked at 1.28 GB unchunked on a 2 GB host. Chunking
        changes peak memory, never results - covered by a bit-identity test.
        """
        if self.prior_ is None:
            raise RuntimeError("CosmicNoveltyEngine.score() called before fit()")
        cols = ["object_id"] + self.astro_columns_
        parts: List[pd.DataFrame] = []
        for start in range(0, len(features), chunk):
            block = features.iloc[start : start + chunk]
            parts.append(self._score_block(block[cols]))
        out = pd.concat(parts, ignore_index=True)
        quality, penalties = self.quality_.score(features, self.cfg.features.quality_prefix)
        out["quality"] = quality
        for col in penalties.columns:
            out[f"qpen_{col}"] = penalties[col].to_numpy()
        return out

    def _score_block(self, block: pd.DataFrame) -> pd.DataFrame:
        proba = self.prior_.probabilities(block)
        evidence = NoveltyEvidence(self.prior_, self.similarity_, self.anomaly_,
                                   self.physics_columns_, quantiles=self.quantiles_)
        channels = evidence.from_probabilities(proba)
        x = self.standardiser_.transform(block)
        sim = self.similarity_.query(x)
        channels.update(evidence.neighbour_channels(sim, proba))
        channels.update(evidence.physics_channels(block, proba))
        channels["anomaly_score"] = evidence._map_channel("anomaly_score", self.anomaly_.score_fused(x))

        codes = self.prior_.clf_.classes_
        order = np.argsort(-proba, axis=1)
        out = pd.DataFrame({"object_id": block["object_id"].to_numpy()})
        for name, values in channels.items():
            out[name] = np.asarray(values, dtype="float64")
        out["best_fit_code"] = codes[order[:, 0]]
        out["best_fit_prob"] = proba[np.arange(len(proba)), order[:, 0]]
        out["runner_up_code"] = codes[order[:, 1]] if proba.shape[1] > 1 else codes[order[:, 0]]
        out["runner_up_prob"] = proba[np.arange(len(proba)), order[:, 1]] if proba.shape[1] > 1 else 0.0
        out["neighbour_mean_distance"] = sim["distances"].mean(axis=1)
        out["neighbour_min_distance"] = sim["distances"].min(axis=1)
        return out

    # --------------------------------------------------------------- artefacts
    def analogue_payload(self, features: pd.DataFrame, k: Optional[int] = None) -> Dict[int, List[Dict[str, object]]]:
        """Nearest analogues per object, for the explanation layer."""
        k = k or self.cfg.similarity.n_analogs
        x = self.standardiser_.transform(features)
        sim = self.similarity_.query(x)
        payload: Dict[int, List[Dict[str, object]]] = {}
        ids = features["object_id"].to_numpy()
        for i, oid in enumerate(ids):
            rows = []
            for j in range(min(k, sim["distances"].shape[1])):
                rows.append({
                    "object_id": int(sim["neighbour_ids"][i][j]),
                    "class_code": int(sim["neighbour_labels"][i][j]),
                    "distance": round(float(sim["distances"][i][j]), 4),
                })
            payload[int(oid)] = rows
        return payload
