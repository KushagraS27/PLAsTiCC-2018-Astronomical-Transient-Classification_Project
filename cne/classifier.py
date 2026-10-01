"""Known-physics classifiers.

These are ordinary supervised classifiers. What makes CNE work is not the
classifier but the *question*: instead of asking "is this object an outlier?",
CNE asks "how badly does a domain-matched model of known astrophysics explain
this object?" The classifier's confusion is the novelty signal.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .config import PriorConfig
from .logging import get_logger
from .taxonomy import family_of

log = get_logger("classifier")


def _macro_auc(y_enc: np.ndarray, oof: np.ndarray, n_classes: int) -> float:
    """Macro OvR ROC-AUC that survives 1- and 2-class fits.

    ``multi_class="ovr"`` requires at least three classes: with exactly two,
    sklearn silently routes to *binary* scoring and rejects the (n, 2) score
    matrix with "y should be a 1d array, got an array of shape (1617, 2)". That
    is reachable in production - the deliberately out-of-domain reference used
    for the domain-degradation study has two classes.
    """
    y = np.asarray(y_enc).ravel()
    present = np.unique(y)
    if len(present) < 2:
        return float("nan")
    if len(present) == 2:
        column = 1 if oof.shape[1] > 1 else 0
        return float(roc_auc_score(y, oof[:, column]))
    return float(roc_auc_score(y, oof[:, present], labels=present,
                               multi_class="ovr", average="macro"))


class TransientClassifier:
    """LightGBM multiclass classifier with out-of-fold probability predictions.

    Out-of-fold probabilities are essential: an in-sample classifier is confident
    about everything it has seen, which would systematically *understate* the
    confusion of reference objects and inflate the apparent novelty of the scored
    stream.
    """

    def __init__(self, config: Optional[PriorConfig] = None, seed: int = 42):
        self.cfg = config or PriorConfig()
        self.seed = seed
        self.models_: List[lgb.LGBMClassifier] = []
        self.classes_: np.ndarray = np.array([])
        self.feature_names_: List[str] = []
        self.oof_proba_: Optional[np.ndarray] = None
        self.metrics_: Dict[str, float] = {}

    # ------------------------------------------------------------------ params
    def _params(self, n_classes: int, class_counts: Optional[np.ndarray] = None) -> Dict[str, object]:
        params: Dict[str, object] = {
            "objective": "multiclass",
            "num_class": n_classes,
            "n_estimators": self.cfg.n_estimators,
            "learning_rate": self.cfg.learning_rate,
            "num_leaves": self.cfg.num_leaves,
            "min_child_samples": self.cfg.min_child_samples,
            "subsample": self.cfg.subsample,
            "subsample_freq": 1,
            "colsample_bytree": self.cfg.colsample_bytree,
            "random_state": self.seed,
            "n_jobs": 2,
            "verbosity": -1,
        }
        # Class balancing is applied as per-row sample weights in fit(), not as a
        # class_weight dict here. LightGBM resolves class_weight keys against the
        # classes present in the *fold it is given*, so a fold missing a rare
        # class raises KeyError(14) - which happened with the 12-class prior on
        # small synthetic folds. Row weights are fold-safe by construction.
        return params

    def _row_weights(self, y_enc: np.ndarray, counts: np.ndarray) -> Optional[np.ndarray]:
        """Balanced per-row weights: each class contributes equally in total."""
        if not self.cfg.class_balanced or counts is None or not len(counts):
            return None
        total = float(np.sum(counts))
        per_class = total / (len(counts) * np.clip(counts, 1, None).astype("float64"))
        return per_class[y_enc]

    # --------------------------------------------------------------------- fit
    def fit(self, x: pd.DataFrame, y: np.ndarray, feature_names: Optional[Sequence[str]] = None) -> "TransientClassifier":
        self.feature_names_ = list(feature_names) if feature_names is not None else list(x.columns)
        x_arr = x[self.feature_names_].to_numpy(dtype="float32")
        y_arr = np.asarray(y)
        self.classes_, y_enc = np.unique(y_arr, return_inverse=True)
        y_enc = y_enc.ravel()
        counts = np.bincount(y_enc, minlength=len(self.classes_))
        params = self._params(len(self.classes_), counts)
        row_w = self._row_weights(y_enc, counts)

        n_splits = max(2, min(self.cfg.n_folds, int(np.min(counts))))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros((len(x_arr), len(self.classes_)), dtype="float64")
        self.models_ = []
        for fold, (tr, va) in enumerate(skf.split(x_arr, y_enc)):
            model = lgb.LGBMClassifier(**params)
            model.fit(x_arr[tr], y_enc[tr],
                      sample_weight=None if row_w is None else row_w[tr])
            # A fold may omit a rare class entirely, so predict_proba returns
            # fewer columns than there are global classes. Scatter each fold's
            # columns back onto their global class index instead of assigning
            # the block directly, which raises a shape mismatch.
            proba = model.predict_proba(x_arr[va])
            fold_cols = getattr(model, "classes_", np.arange(proba.shape[1]))
            block = np.zeros((len(va), len(self.classes_)), dtype="float64")
            block[:, np.asarray(fold_cols, dtype=int)] = proba
            oof[va] = block
            self.models_.append(model)
            log.debug("prior fold=%d train=%d valid=%d", fold, len(tr), len(va))

        self.oof_proba_ = oof
        pred = oof.argmax(axis=1)
        self.metrics_ = {
            "accuracy": float(accuracy_score(y_enc, pred)),
            "log_loss": float(log_loss(y_enc, oof, labels=np.arange(len(self.classes_)))),
            "roc_auc_macro": _macro_auc(y_enc, oof, len(self.classes_)),
            "n_classes": int(len(self.classes_)),
            "n_objects": int(len(x_arr)),
            "n_folds": int(n_splits),
        }
        log.info("known-physics prior cv_accuracy=%.4f auc=%.4f classes=%d n=%d",
                 self.metrics_["accuracy"], self.metrics_["roc_auc_macro"], len(self.classes_), len(x_arr))
        return self

    # ----------------------------------------------------------------- predict
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Average the fold models. Fold-averaging reduces variance in the
        confusion estimate, which is exactly the quantity novelty is built on."""
        if not self.models_:
            raise RuntimeError("TransientClassifier used before fit()")
        x_arr = x[self.feature_names_].to_numpy(dtype="float32")
        # Folds that omitted a rare class return fewer columns, so each fold is
        # scattered onto its own global class index before averaging. Stacking
        # directly raises "all input arrays must have the same shape".
        blocks = []
        for model in self.models_:
            proba = model.predict_proba(x_arr)
            fold_cols = np.asarray(getattr(model, "classes_", np.arange(proba.shape[1])), dtype=int)
            block = np.zeros((len(x_arr), len(self.classes_)), dtype="float64")
            block[:, fold_cols] = proba
            # renormalise: a missing class means the fold's mass sums to 1 over
            # fewer columns, and the zero column must not dilute the others.
            block /= np.clip(block.sum(axis=1, keepdims=True), 1e-12, None)
            blocks.append(block)
        return np.mean(np.stack(blocks, axis=0), axis=0)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(x).argmax(axis=1)]

    def class_names(self, indices: np.ndarray) -> List[str]:
        return [str(self.classes_[int(i)]) for i in np.atleast_1d(indices)]


class HierarchicalTransientClassifier(TransientClassifier):
    """Galactic / extragalactic gate, then a specialist per branch.

    Mirrors the AI-01 pipeline in CNE v1 (macro OvR ROC-AUC 0.9700). The gate is
    a cheap binary classifier; each branch sees only its own feature subset
    statistics, which is what recovers the fine-grained classes.
    """

    def __init__(self, config: Optional[PriorConfig] = None, seed: int = 42):
        super().__init__(config, seed)
        self.gate_: Optional[lgb.LGBMClassifier] = None
        self.branches_: Dict[str, TransientClassifier] = {}

    def fit(self, x: pd.DataFrame, y: np.ndarray, feature_names: Optional[Sequence[str]] = None) -> "HierarchicalTransientClassifier":
        self.feature_names_ = list(feature_names) if feature_names is not None else list(x.columns)
        codes = np.asarray(y, dtype="int64")
        is_galactic = np.array([family_of(c) == "galactic" for c in codes])
        params = self._params(2, np.bincount(is_galactic.astype(int), minlength=2))
        params.pop("class_weight", None) if len(params.get("class_weight", {})) != 2 else None
        self.gate_ = lgb.LGBMClassifier(**{**params, "num_class": 2})
        self.gate_.fit(x[self.feature_names_].to_numpy(dtype="float32"), is_galactic.astype(int))

        self.branches_ = {}
        for name, mask in (("galactic", is_galactic), ("extragalactic", ~is_galactic)):
            sub_codes, remap = np.unique(codes[mask], return_inverse=True)
            if len(sub_codes) < 2:
                continue
            branch = TransientClassifier(self.cfg, seed=self.seed)
            branch.fit(x.loc[mask], sub_codes, self.feature_names_)
            self.branches_[name] = branch
        # Keep the flat OOF metrics for comparison with the non-hierarchical prior.
        super().fit(x, y, self.feature_names_)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        flat = super().predict_proba(x)
        if self.gate_ is None:
            return flat
        p_gal = self.gate_.predict_proba(x[self.feature_names_].to_numpy(dtype="float32"))[:, 1]
        # Soft-blend the gate with the flat model instead of hard-routing: a hard
        # gate discards probability mass and wrecks calibration on objects that
        # sit between the galactic and extragalactic populations.
        branch_weight = np.stack(
            [p_gal if family_of(int(code)) == "galactic" else 1.0 - p_gal for code in self.classes_], axis=1
        )
        reweighted = flat * branch_weight
        reweighted = reweighted / np.clip(reweighted.sum(axis=1, keepdims=True), 1e-12, None)
        blend = 0.5 * flat + 0.5 * reweighted
        return blend / np.clip(blend.sum(axis=1, keepdims=True), 1e-12, None)
