"""The v2 experiment suite: benchmarks, ablations, stress tests, safety checks.

Every function here returns a plain dict/DataFrame that goes straight into
``reports/``. Nothing is tuned inside these functions - they only measure.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .artifacts import ArtifactSafetySuite
from .data import LightCurveCache
from .evaluation import (
    channel_power,
    evaluate_ranking,
    false_positive_audit,
    nan_to_none,
    per_class_metrics,
    precision_at_k,
    recall_at_k,
    roc_auc,
    average_precision,
    weight_selection_optimism,
)
from .explain import explain_candidate
from .features import LightCurveFeaturiser
from .logging import get_logger, stage
from .novelty import ALL_CHANNELS, CosmicNoveltyEngine
from .ranking import NoveltyRanker, RankingWeights
from .stress import review_budget_curve, stress_test
from .taxonomy import WITHHELD_FROM_PRIOR, name_of
from .uncertainty import evaluate_uncertainty_value

log = get_logger("experiments")

EARLY_WINDOWS = (15, 30, 60, 120)


# --------------------------------------------------------------------------- #
# benchmarks
# --------------------------------------------------------------------------- #
def summarise_benchmark(name: str, ranked: pd.DataFrame, codes: np.ndarray, cfg,
                        weights: RankingWeights, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Full metric block for one locked benchmark."""
    y = ranked["is_novel"].to_numpy().astype(int)
    scores = ranked["novelty_score"].to_numpy(dtype="float64")
    metrics = evaluate_ranking(y, scores, precision_k=cfg.evaluation.precision_k,
                               recall_k=cfg.evaluation.recall_k, draws=cfg.evaluation.bootstrap_draws,
                               seed=cfg.seed).as_dict()
    payload: Dict[str, Any] = {
        "benchmark": name,
        "weights": {k: v for k, v in weights.as_dict().items() if v > 0},
        "metrics": metrics,
        "false_positive_audit": false_positive_audit(ranked, k=100),
        "per_class": per_class_metrics(codes, scores, k=200, min_n=cfg.evaluation.min_n_for_class_report,
                                       novel_codes=WITHHELD_FROM_PRIOR).to_dict(orient="records"),
        "channel_power": channel_power(y, ranked, ALL_CHANNELS).to_dict(orient="records"),
        "tier_counts": ranked["tier"].value_counts().to_dict(),
        "n_abstained": int(ranked["abstain"].sum()),
    }
    if extra:
        payload.update(extra)
    return nan_to_none(payload)


def ablation_study(ranked: pd.DataFrame, cfg, base_weights: RankingWeights,
                   ks: Sequence[int] = (20, 50)) -> pd.DataFrame:
    """Drop or re-add one channel at a time and measure the effect.

    This is the test every component has to pass to stay enabled: if removing a
    channel does not hurt, the channel does not earn its place.
    """
    y = ranked["is_novel"].to_numpy().astype(int)
    rows = []

    def record(label: str, weights: RankingWeights) -> None:
        ranker = NoveltyRanker(cfg, weights=weights)
        scores = ranker.raw_evidence(ranked.reset_index(drop=True))
        row = {"configuration": label, "roc_auc": roc_auc(y, scores), "average_precision": average_precision(y, scores)}
        for k in ks:
            row[f"P@{k}"] = precision_at_k(y, scores, k)
        rows.append(row)

    record("full", base_weights)
    for channel in base_weights.active_channels():
        if channel == "taxonomy_gap":
            continue  # removing the primary signal is a different experiment
        reduced = {k: (0.0 if k == channel else v) for k, v in base_weights.as_dict().items()}
        if sum(reduced.values()) <= 0:
            continue
        record(f"without {channel}", RankingWeights(reduced))
    for channel in ("prior_entropy", "anomaly_score", "physics_gap", "neighbor_entropy", "family_misfit", "cc_weighted"):
        if base_weights.as_dict().get(channel, 0.0) > 0:
            continue
        added = dict(base_weights.as_dict())
        added[channel] = 0.2
        record(f"with {channel} (+0.2)", RankingWeights(added))
    only = {k: 0.0 for k in ALL_CHANNELS}
    only["taxonomy_gap"] = 1.0
    record("taxonomy_gap only", RankingWeights(only))
    return pd.DataFrame(rows).set_index("configuration")


def domain_degradation_study(matched: Dict[str, Any], reweighted: Dict[str, Any],
                             mismatched: Dict[str, Any]) -> pd.DataFrame:
    """Matched vs reweighted vs badly-mismatched reference, on one frozen protocol."""
    rows = []
    for label, block in (("unweighted_train_reference", matched),
                         ("density_ratio_reweighted", reweighted),
                         ("deliberately_mismatched", mismatched)):
        if not block:
            continue
        rows.append({
            "reference": label,
            "prior_cv_accuracy": block.get("prior", {}).get("cv_accuracy"),
            "roc_auc": block.get("metrics", {}).get("roc_auc"),
            "average_precision": block.get("metrics", {}).get("average_precision"),
            "P@50": block.get("metrics", {}).get("precision_at_k", {}).get("P@50"),
            "n_objects": block.get("metrics", {}).get("n_objects"),
            "base_rate": block.get("metrics", {}).get("base_rate"),
        })
    return pd.DataFrame(rows).set_index("reference")


# --------------------------------------------------------------------------- #
# early detection
# --------------------------------------------------------------------------- #
def early_detection_study(pipeline, features: pd.DataFrame, lightcurves: pd.DataFrame,
                          engine: CosmicNoveltyEngine, weights: RankingWeights,
                          windows: Sequence[int] = EARLY_WINDOWS) -> Dict[str, Any]:
    """Lower-bound early warning: full-light-curve models applied to truncated views.

    Models are NOT refit per window, so these numbers are a legitimate lower
    bound. Refitting would raise them; not refitting keeps the comparison honest
    and leakage-free, because no future observation can enter training.
    """
    code_series = pipeline._labels_for(features).reindex(features["object_id"])
    meta = pipeline._stream_metadata_for(features)
    results: Dict[str, Any] = {
        "protocol": "full-lightcurve models scored on truncated views (leakage-safe lower bound)",
        "windows": [],
    }
    for window in list(windows) + [None]:
        label = "full" if window is None else f"{window}d"
        if window is None:
            truncated = lightcurves
        else:
            first = lightcurves.groupby("object_id")["mjd"].transform("min")
            truncated = lightcurves[lightcurves["mjd"].to_numpy() <= (first.to_numpy() + float(window))]
        started = time.perf_counter()
        block = pipeline._featurise_in_chunks(truncated, meta)
        block = _align_columns(block, features)
        ranked = pipeline.score_stream(block, engine, weights=weights)
        y = np.isin(code_series.reindex(ranked["object_id"]).to_numpy(), list(WITHHELD_FROM_PRIOR)).astype(int)
        scores = ranked["novelty_score"].to_numpy(dtype="float64")
        entry = {
            "window": label,
            "n_objects": int(len(ranked)),
            "n_novel": int(y.sum()),
            "roc_auc": roc_auc(y, scores),
            "average_precision": average_precision(y, scores),
            "P@50": precision_at_k(y, scores, 50),
            "R@200": recall_at_k(y, scores, 200),
            "elapsed_s": round(time.perf_counter() - started, 1),
        }
        results["windows"].append(entry)
        log.info("early detection %-5s n=%d AUC=%.3f AP=%.3f P@50=%.3f",
                 label, entry["n_objects"], entry["roc_auc"], entry["average_precision"], entry["P@50"])
        del truncated, block
    return nan_to_none(results)


def _align_columns(block: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the truncated featurisation has exactly the reference's columns.

    A truncated light curve can lose a whole passband, which changes the column
    set; without this the engine would be handed a different design matrix than
    the one it was fitted on.
    """
    for col in reference.columns:
        if col not in block.columns:
            block[col] = 0.0
    return block[list(reference.columns)]


# --------------------------------------------------------------------------- #
# safety
# --------------------------------------------------------------------------- #
def artifact_safety_study(pipeline, features: pd.DataFrame, lightcurves: pd.DataFrame,
                          engine: CosmicNoveltyEngine, weights: RankingWeights,
                          n_objects: int = 8) -> Dict[str, Any]:
    """Inject controlled bad data and verify nothing gets promoted."""
    known_ids = features["object_id"].to_numpy()
    sample = known_ids[: max(n_objects * 6, 48)]
    subset = features[features["object_id"].isin(sample)].reset_index(drop=True)
    ranked = pipeline.score_stream(subset, engine, weights=weights)
    baseline = ranked[["object_id", "novelty_score", "quality", "rank"]].copy()
    rank_lookup = dict(zip(ranked["object_id"], ranked["rank"]))
    score_lookup = dict(zip(ranked["object_id"], ranked["novelty_score"]))

    meta_all = pipeline._stream_metadata_for(subset)
    lc_all = lightcurves[lightcurves["object_id"].isin(sample)]

    def score_fn(object_id: int, corrupted: pd.DataFrame) -> Tuple[float, float, int]:
        """Re-score the WHOLE queue with one object corrupted.

        Scoring the corrupted object on its own is not comparable to the
        baseline: the evidence channels are rank-normalised against the scored
        batch, so a solo object gets a completely different score. Measured on
        40 objects, solo-vs-batch novelty differed by a mean of -0.243 (max
        |delta| 0.523) and not one score matched. Comparing a 2106-object
        baseline against solo corrupted scores therefore measured batch size,
        not data quality, and reported 16/36 spurious promotions.
        """
        parts = [corrupted if oid == object_id else grp
                 for oid, grp in lc_all.groupby("object_id", sort=False)]
        block = pipeline.featuriser_.transform(pd.concat(parts, ignore_index=True), meta_all)
        block = _align_columns(block, features)
        scored = pipeline.score_stream(block, engine, weights=weights)
        row = scored.loc[scored["object_id"] == object_id].iloc[0]
        return float(row["novelty_score"]), float(row["quality"]), int(row["rank"])

    suite = ArtifactSafetySuite(score_fn, seed=pipeline.cfg.seed, cases_per_artifact=n_objects)
    report = suite.run(lightcurves[lightcurves["object_id"].isin(sample)], sample, baseline)
    # Re-express "promoted" against the original queue, which is the real question.
    promoted = 0
    for case in report.cases:
        original_rank = rank_lookup.get(case.object_id, 10 ** 9)
        original_score = score_lookup.get(case.object_id, float("nan"))
        case.baseline_rank = int(original_rank)
        case.baseline_score = float(original_score)
        case.promoted = bool(np.isfinite(case.corrupted_score) and case.corrupted_score > original_score)
        case.score_increased = case.promoted
        promoted += int(case.promoted)
    report.n_promoted = promoted
    report.n_score_increased = promoted
    report.promotion_rate = promoted / max(report.n_cases, 1)
    report.passed = promoted == 0
    # Rebuild the per-artifact breakdown from the re-expressed cases. The table
    # built inside suite.run() still used the pre-re-expression promotion rule
    # and reported n_promoted=5 for all six artifact types, contradicting the
    # per-case records it was supposed to summarise.
    per: Dict[str, Dict[str, float]] = {}
    for case in report.cases:
        a = per.setdefault(case.artifact, {"n": 0, "n_promoted": 0, "dq": 0.0, "ds": 0.0,
                                           "d_rank": 0, "climbed": 0})
        a["n"] += 1
        a["n_promoted"] += int(case.promoted)
        a["dq"] += float(case.corrupted_quality - case.baseline_quality)
        a["ds"] += float(case.corrupted_score - case.baseline_score)
        a["d_rank"] += int(case.baseline_rank - case.corrupted_rank)
        a["climbed"] += int(case.baseline_rank > case.corrupted_rank)
    report.per_artifact = {
        k: {"n": v["n"], "n_promoted": v["n_promoted"],
            "mean_delta_quality": v["dq"] / v["n"], "mean_delta_score": v["ds"] / v["n"],
            "mean_rank_gain": v["d_rank"] / v["n"], "n_climbed": v["climbed"]}
        for k, v in per.items()
    }
    payload = report.as_dict()
    payload["cases"] = [c.as_dict() for c in report.cases]
    payload["passed"] = report.passed
    return nan_to_none(payload)


# --------------------------------------------------------------------------- #
# uncertainty
# --------------------------------------------------------------------------- #
def uncertainty_study(ranked: pd.DataFrame) -> Dict[str, Any]:
    y = ranked["is_novel"].to_numpy().astype(int)
    scores = ranked["novelty_score"].to_numpy(dtype="float64")
    unc = ranked["uncertainty"].to_numpy(dtype="float64")
    out = evaluate_uncertainty_value(y, scores, unc)
    out["uncertainty_vs_novelty_auc"] = roc_auc(y, unc)
    out["mean_uncertainty_novel"] = float(unc[y == 1].mean()) if (y == 1).any() else None
    out["mean_uncertainty_known"] = float(unc[y == 0].mean()) if (y == 0).any() else None
    out["abstention"] = {
        "n_abstained": int(ranked["abstain"].sum()),
        "abstention_rate": float(ranked["abstain"].mean()),
        "reasons": ranked.loc[ranked["abstain"], "abstain_reason"].value_counts().to_dict(),
        "novel_lost_to_abstention": int(((ranked["abstain"]) & (ranked["is_novel"].astype(bool))).sum()),
    }
    return nan_to_none(out)


# --------------------------------------------------------------------------- #
# weight-selection optimism
# --------------------------------------------------------------------------- #
def optimism_study(evidence: pd.DataFrame, y: np.ndarray, pinned: Dict[str, float]) -> Dict[str, Any]:
    result = weight_selection_optimism(evidence, y, grid={
        "simplex_novelty": (0.0, 0.1, 0.2, 0.4),
        "novelty_gap": (0.0, 0.05, 0.15),
        "prior_entropy": (0.0, 0.1, 0.25),
    }, pinned=pinned, seeds=(0, 1, 2, 3, 4))
    result.pop("detail", None)
    return nan_to_none(result)
