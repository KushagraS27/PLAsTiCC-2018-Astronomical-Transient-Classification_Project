"""Model components: standardisation, anomaly detectors, similarity, quality, calibration.

Every class here is deliberately small, single-purpose and stateful only where it
must be. Two design rules carried over from CNE v1:

1. Anything fitted is fitted on the REFERENCE population only.
2. Quality is suppress-only. Nothing in this module can raise a novelty score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from .config import QualityConfig, SimilarityConfig
from .logging import get_logger

log = get_logger("models")


class Standardiser:
    """Robust z-scoring fitted on the reference population."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.columns_: List[str] = []
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, frame: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> "Standardiser":
        cols = list(columns) if columns is not None else [c for c in frame.columns if c != "object_id"]
        arr = frame[cols].to_numpy(dtype="float64")
        self.columns_ = cols
        self.center_ = np.nanmedian(arr, axis=0)
        spread = np.nanpercentile(arr, 84.13, axis=0) - np.nanpercentile(arr, 15.87, axis=0)
        fallback = np.nanstd(arr, axis=0)
        self.scale_ = np.where(np.isfinite(spread) & (spread > self.eps), spread, np.where(fallback > self.eps, fallback, 1.0))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.center_ is None:
            raise RuntimeError("Standardiser used before fit()")
        arr = frame[self.columns_].to_numpy(dtype="float64")
        out = (arr - self.center_) / self.scale_
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    def fit_transform(self, frame: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        return self.fit(frame, columns).transform(frame)


class DenseAutoencoder:
    """Hand-written numpy autoencoder (no torch dependency, per the v1 constraint).

    Trained with Adam on the reference population; the reconstruction residual is
    one of the audit channels. CNE v1 measured this channel at/below chance for
    novelty (AUC 0.478) - it stays in the codebase as a diagnostic, weighted zero.
    """

    def __init__(self, input_dim: int, hidden: Sequence[int] = (64, 24, 64), seed: int = 42,
                 epochs: int = 60, lr: float = 3e-3, batch: int = 256):
        self.hidden = tuple(hidden)
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.input_dim = input_dim
        rng = np.random.default_rng(seed)
        dims = (input_dim,) + self.hidden + (input_dim,)
        self.weights = [rng.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i]) for i in range(len(dims) - 1)]
        self.biases = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0.0)

    def _forward(self, x):
        acts = [x]
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            a = a @ w + b
            if i < len(self.weights) - 1:
                a = self._relu(a)
            acts.append(a)
        return acts

    def fit(self, x: np.ndarray) -> "DenseAutoencoder":
        """Adam on the reconstruction loss. Gradients are accumulated in the same
        order as ``self.weights + self.biases`` so momentum slots never mismatch."""
        rng = np.random.default_rng(self.seed)
        x = np.asarray(x, dtype="float64")
        n = len(x)
        params = self.weights + self.biases
        momentum = [np.zeros_like(par) for par in params]
        velocity = [np.zeros_like(par) for par in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        n_layers = len(self.weights)
        step = 0
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch):
                idx = order[start : start + self.batch]
                xb = x[idx]
                acts = self._forward(xb)
                m = max(len(xb), 1)
                delta = 2.0 * (acts[-1] - xb) / m          # dL/d(out)
                weight_grads: List[np.ndarray] = []
                bias_grads: List[np.ndarray] = []
                for i in reversed(range(n_layers)):
                    weight_grads.insert(0, acts[i].T @ delta)
                    bias_grads.insert(0, delta.sum(axis=0))
                    if i > 0:
                        delta = (delta @ self.weights[i].T) * (acts[i] > 0)
                grads = weight_grads + bias_grads
                step += 1
                for j, g in enumerate(grads):
                    momentum[j] = beta1 * momentum[j] + (1 - beta1) * g
                    velocity[j] = beta2 * velocity[j] + (1 - beta2) * g * g
                    mhat = momentum[j] / (1 - beta1 ** step)
                    vhat = velocity[j] / (1 - beta2 ** step)
                    params[j] = params[j] - self.lr * mhat / (np.sqrt(vhat) + eps)
        self.weights = params[:n_layers]
        self.biases = params[n_layers:]
        return self

    def residual(self, x: np.ndarray) -> np.ndarray:
        recon = self._forward(x)[-1]
        return np.sqrt(np.mean((recon - x) ** 2, axis=1))


class QuantileMapper:
    """Map raw channel values onto their quantile within a *fitted reference*.

    ``_rank01`` rank-normalises within whatever batch it is handed, so the same
    object scored in a chunk of 120 and in a chunk of 30 gets two different
    channel values - which makes "chunk the stream to cap memory" silently change
    the answer. This class fixes the mapping at fit time on the reference
    population, so a channel value means the same thing everywhere and is
    bit-identical regardless of how the stream is partitioned. It also reads
    better: "physics misfit at the 97th percentile of known objects".
    """

    __slots__ = ("grids_", "fitted_")

    def __init__(self) -> None:
        self.grids_: Dict[str, np.ndarray] = {}
        self.fitted_ = False

    def fit(self, values: Dict[str, np.ndarray]) -> "QuantileMapper":
        self.grids_ = {}
        for name, arr in values.items():
            clean = np.sort(np.asarray(arr, dtype="float64").ravel())
            clean = clean[np.isfinite(clean)]
            if len(clean) < 2:
                continue
            self.grids_[name] = clean
        self.fitted_ = bool(self.grids_)
        return self

    def transform(self, name: str, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype="float64")
        grid = self.grids_.get(name)
        if grid is None:                      # unregistered channel: fall back to local rank
            return _rank01(values)
        if len(values) == 0:
            return values
        # searchsorted gives the count of reference values strictly below; add half
        # the tied mass so a value sitting exactly on the median maps to 0.5.
        below = np.searchsorted(grid, values, side="left")
        upto = np.searchsorted(grid, values, side="right")
        position = below + 0.5 * (upto - below)
        return np.clip(position / len(grid), 0.0, 1.0)

    def state_dict(self) -> Dict[str, Any]:
        return {"grids": {k: v.tolist() for k, v in self.grids_.items()}}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "QuantileMapper":
        mapper = cls()
        mapper.grids_ = {k: np.asarray(v, dtype="float64") for k, v in (state or {}).get("grids", {}).items()}
        mapper.fitted_ = bool(mapper.grids_)
        return mapper


def _rank01(values: np.ndarray) -> np.ndarray:
    """Map to [0, 1] by rank, which makes heterogeneous channels comparable."""
    values = np.asarray(values, dtype="float64")
    if len(values) == 0:
        return values
    order = np.argsort(np.argsort(np.nan_to_num(values, nan=-np.inf)))
    return order / max(len(values) - 1, 1)


class AnomalyEnsemble:
    """Equal-weight rank ensemble of generic unsupervised detectors.

    This module exists to be *measured and then ignored*: CNE v1 showed every
    member sits at or below chance for held-out astronomical novelty (IF 0.451,
    PCA 0.465, AE 0.478, Mahalanobis 0.467, kNN 0.488). It is retained as an
    audit channel with production weight 0 so the negative result stays
    reproducible and an operator can re-target the system by re-weighting.
    """

    def __init__(self, seed: int = 42, max_ref: int = 3500, n_components: int = 24):
        self.seed = seed
        self.max_ref = max_ref
        self.n_components = n_components
        self.pca_: Optional[PCA] = None
        self.iforest_: Optional[IsolationForest] = None
        self.knn_: Optional[NearestNeighbors] = None
        self.ae_: Optional[DenseAutoencoder] = None
        self.mean_: Optional[np.ndarray] = None
        self.covar_inv_: Optional[np.ndarray] = None
        self.rank_grids_: Dict[str, np.ndarray] = {}

    def fit(self, x: np.ndarray) -> "AnomalyEnsemble":
        if self.max_ref and len(x) > self.max_ref:
            rng = np.random.default_rng(self.seed)
            x = x[rng.choice(len(x), self.max_ref, replace=False)]
        self.mean_ = x.mean(axis=0)
        cov = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * 1e-6
        self.covar_inv_ = np.linalg.pinv(cov)
        n_comp = min(self.n_components, x.shape[1], len(x) - 1)
        self.pca_ = PCA(n_components=max(n_comp, 2), random_state=self.seed).fit(x)
        self.iforest_ = IsolationForest(n_estimators=100, random_state=self.seed, n_jobs=1).fit(x)
        k = min(25, max(2, len(x) - 1))
        self.knn_ = NearestNeighbors(n_neighbors=k).fit(x)
        self.ae_ = DenseAutoencoder(x.shape[1], seed=self.seed).fit(x.astype("float32"))
        # Freeze each detector's rank grid on the reference so score_fused does
        # not depend on the size of whatever batch is being scored.
        self.rank_grids_ = {
            name: np.sort(np.nan_to_num(values.astype("float64"), nan=-np.inf, posinf=np.inf))
            for name, values in self.score(x).items()
        }
        return self

    def score(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        if self.mean_ is None:
            raise RuntimeError("AnomalyEnsemble used before fit()")
        d = x - self.mean_
        mahal = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", d, self.covar_inv_, d), 0, None))
        pca_res = np.linalg.norm(x - self.pca_.inverse_transform(self.pca_.transform(x)), axis=1)
        iso = -self.iforest_.score_samples(x)
        knn_d, _ = self.knn_.kneighbors(x)
        knn = knn_d.mean(axis=1)
        ae = self.ae_.residual(x.astype("float32"))
        return {"mahalanobis": mahal, "pca_residual": pca_res, "isolation_forest": iso,
                "knn_density": knn, "autoencoder": ae}

    def score_fused(self, x: np.ndarray) -> np.ndarray:
        """Mean of the five detector ranks, each mapped onto a REFERENCE grid.

        Ranking within the incoming batch - the obvious implementation - makes the
        fused score depend on how the stream is chunked: the same object scored in
        a block of 120 and in a block of 30 lands at different percentiles. The
        grids are frozen at fit time on the reference population instead, so this
        channel is a fixed property of the fitted model.
        """
        raw = self.score(x)
        if not self.rank_grids_:
            return np.mean(np.stack([_rank01(v) for v in raw.values()], axis=0), axis=0)
        parts = []
        for name, values in raw.items():
            grid = self.rank_grids_.get(name)
            parts.append(QuantileMapper().fit({name: grid}).transform(name, values)
                         if grid is not None else _rank01(values))
        return np.mean(np.stack(parts, axis=0), axis=0)


class SimilarityIndex:
    """k-nearest-neighbour retrieval over the reference population.

    Used for explanation (nearest analogues, class distribution) and for the
    neighbour-entropy channel. Brute-force exact search: at the reference sizes
    used here it is both fast enough and exactly reproducible.
    """

    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.cfg = config or SimilarityConfig()
        self.nn_: Optional[NearestNeighbors] = None
        self.labels_: Optional[np.ndarray] = None
        self.ids_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> "SimilarityIndex":
        self.nn_ = NearestNeighbors(n_neighbors=min(self.cfg.k_neighbours, len(x))).fit(x)
        self.labels_ = np.asarray(labels)
        self.ids_ = np.asarray(ids)
        return self

    def query(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        if self.nn_ is None:
            raise RuntimeError("SimilarityIndex used before fit()")
        dist, idx = self.nn_.kneighbors(x)
        neighbour_labels = self.labels_[idx]
        neighbour_ids = self.ids_[idx]
        classes = np.unique(self.labels_)
        counts = np.stack([np.sum(neighbour_labels == c, axis=1) for c in classes], axis=1).astype("float64")
        probs = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
        max_ent = np.log(max(len(classes), 2))
        return {
            "distances": dist,
            "indices": idx,
            "neighbour_labels": neighbour_labels,
            "neighbour_ids": neighbour_ids,
            "class_counts": counts,
            "class_probs": probs,
            "classes": classes,
            "neighbour_entropy": ent / max_ent,
        }


class QualityModel:
    """Rule-based, suppress-only data-quality gate.

    Returns ``quality`` in [0, 1]; 1.0 means the observations are trustworthy.
    It can only ever multiply a novelty score DOWN. Bad data must never make an
    object look more astrophysically interesting.
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        self.cfg = config or QualityConfig()

    def score(self, features: pd.DataFrame, prefix: str = "q_") -> Tuple[np.ndarray, pd.DataFrame]:
        """Return (quality in [0,1], per-reason penalty frame).

        Every input is a DETECTION-ONLY statistic. Two of these were originally
        computed over all epochs, which included PLAsTiCC's many non-detections;
        ``flux_err/|flux|`` then exploded and the median SNR collapsed to zero, so
        both penalties saturated and quality degenerated into a constant 0.125 for
        97% of objects. A gate that is constant is not a gate.
        """
        if not self.cfg.enabled:
            return np.ones(len(features)), pd.DataFrame(index=features.index)

        def col(name: str, default: float = np.nan) -> np.ndarray:
            key = f"{prefix}{name}"
            if key in features.columns:
                return features[key].to_numpy(dtype="float64")
            return np.full(len(features), default, dtype="float64")

        neg = col("neg_flux_frac", 0.0)
        det = col("n_det", 0.0)
        err_ratio = col("err_ratio_median", 0.0)
        n_bands_det = col("n_bands_detected", col("n_bands", 6.0))
        dominant_det = col("dominant_band_det_frac", col("single_band_frac", 0.0))
        dup = col("duplicate_epochs", 0.0)
        det_snr = col("det_snr_median", 10.0)

        penalties = pd.DataFrame(index=features.index)
        penalties["neg_flux"] = np.clip(neg / max(self.cfg.max_negative_flux_fraction, 1e-6), 0, 1)
        penalties["few_detections"] = np.clip(1.0 - det / max(self.cfg.min_detections, 1), 0, 1)
        penalties["dominant_band"] = np.clip(
            (dominant_det - 0.7) / max(self.cfg.max_single_band_fraction - 0.7, 1e-6), 0, 1)
        # A detection at SNR 5 has flux_err/|flux| = 0.20; SNR 3 gives 0.33.
        penalties["high_error"] = np.clip((err_ratio - 0.25) / 0.5, 0, 1)
        # Only penalise genuinely thin colour coverage: 4+ detected bands is fine.
        penalties["thin_coverage"] = np.clip((4.0 - n_bands_det) / 3.0, 0, 1)
        penalties["duplicates"] = np.clip(dup / 10.0, 0, 1)
        penalties["low_snr"] = np.clip(1.0 - det_snr / max(self.cfg.min_median_snr, 1e-6), 0, 1)

        values = penalties.to_numpy(dtype="float64")
        worst = values.max(axis=1)
        # Mean of the OTHER penalties in each row (an earlier revision dropped whole
        # columns that were maximal for any row, which is not a per-row operation).
        total = values.sum(axis=1)
        count = np.clip(values.shape[1] - 1, 1, None)
        others = (total - worst) / count
        severity = np.clip(0.8 * worst + 0.2 * others, 0.0, 1.0)
        quality = np.clip(1.0 - severity, 0.0, 1.0)
        floor = self.cfg.low_quality_score * 0.5
        quality = np.where(quality < self.cfg.low_quality_score, floor, quality)
        return quality, penalties


class ConditionalCalibrator:
    """Calibrate a confidence signal against realised error, adaptively binned.

    CNE v1 bug, preserved here as a regression test: a FIXED grid of 5 bins x 3
    nuisance variables leaves every one of the 125 cells below ``min_per_cell``,
    so the calibrator silently degenerates to global calibration without raising.
    Bins are therefore sized from quantiles of the observed data, and
    :attr:`coverage_` must be inspected after fitting.
    """

    def __init__(self, n_bins: int = 5, min_per_cell: int = 40, nuisance: Optional[Sequence[str]] = None):
        self.n_bins = n_bins
        self.min_per_cell = min_per_cell
        self.nuisance = list(nuisance or [])
        self.edges_: Dict[str, np.ndarray] = {}
        self.cell_mean_: Dict[Tuple[int, ...], float] = {}
        self.coverage_: float = 0.0
        self.global_mean_: float = 0.5
        self.degenerate_: bool = False

    def _bin_index(self, frame: pd.DataFrame, col: str) -> np.ndarray:
        edges = self.edges_[col]
        return np.clip(np.searchsorted(edges, frame[col].to_numpy(dtype="float64"), side="right") - 1, 0, len(edges) - 2)

    def fit(self, frame: pd.DataFrame, confidence_col: str, error_col: str) -> "ConditionalCalibrator":
        self.global_mean_ = float(np.nanmean(frame[error_col].to_numpy(dtype="float64")))
        for col in self.nuisance:
            values = frame[col].to_numpy(dtype="float64")
            # Adaptive: quantile edges, and collapse to a single bin if the
            # variable cannot support the requested resolution.
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            edges = np.unique(np.nanquantile(values, qs))
            self.edges_[col] = edges if len(edges) else np.array([np.nanmedian(values)])
        if not self.nuisance:
            self.cell_mean_[()] = self.global_mean_
            self.coverage_ = 1.0
            return self
        idx = np.stack([self._bin_index(frame, c) for c in self.nuisance], axis=1)
        keys = [tuple(int(v) for v in row) for row in idx]
        errors = frame[error_col].to_numpy(dtype="float64")
        ser = pd.Series(errors).groupby(pd.Series(keys)).agg(["mean", "size"])
        for key, row in ser.iterrows():
            if row["size"] >= self.min_per_cell:
                self.cell_mean_[key] = float(row["mean"])
        covered = sum(1 for k in keys if k in self.cell_mean_)
        self.coverage_ = covered / max(len(keys), 1)
        self.degenerate_ = self.coverage_ < 0.5
        if self.degenerate_:
            log.warning("ConditionalCalibrator coverage=%.2f (<0.5): degrading to global calibration", self.coverage_)
        return self

    def transform(self, frame: pd.DataFrame, confidence_col: str) -> np.ndarray:
        if not self.nuisance:
            return np.full(len(frame), self.global_mean_)
        idx = np.stack([self._bin_index(frame, c) for c in self.nuisance], axis=1)
        out = np.full(len(frame), self.global_mean_, dtype="float64")
        for i, row in enumerate(idx):
            out[i] = self.cell_mean_.get(tuple(int(v) for v in row), self.global_mean_)
        return out


@dataclass
class DiscriminativeNoveltyMetric:
    """Convenience container for the domain-matched prior's confusion signal."""

    classes: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: Optional[np.ndarray] = None
    taxonomy_gap: Optional[np.ndarray] = None
    entropy: Optional[np.ndarray] = None
    margin: Optional[np.ndarray] = None
