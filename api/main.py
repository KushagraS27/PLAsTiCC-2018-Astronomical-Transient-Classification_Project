"""FastAPI service for the Cosmic Novelty Engine.

Serves the frozen artefacts written by ``scripts/``: the candidate queue, the
light-curve cache, the evaluation reports and the operator feedback log. The
models are loaded once at startup and never retrained by a request - a service
whose ranking quietly changes between page loads is not auditable.

Every endpoint that could be read as a scientific claim returns the provenance
stamp alongside it (dataset version, split manifest, seed, config id, base rate),
because a number without those is not a result.

    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from cne.logging import get_logger  # noqa: E402
from cne.taxonomy import (  # noqa: E402
    KNOWN_CLASS_CODES,
    WITHHELD_FROM_PRIOR,
    describe,
    family_of,
    is_held_out,
    name_of,
)
from cne.version import CODENAME, __version__, stamp  # noqa: E402

log = get_logger("api")

app = FastAPI(
    title="Cosmic Novelty Engine",
    version=__version__,
    description=(
        "Candidate anomaly triage for transient surveys. Nothing served here is a "
        "confirmed discovery: every candidate requires expert follow-up."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
ARTEFACTS = ROOT / "data" / "artefacts"
WEB_DIST = ROOT / "web" / "dist"
FEEDBACK_LOG = ARTEFACTS / "operator_feedback.jsonl"

DISCLAIMER = (
    "This is a candidate anomaly, not a confirmed discovery. It is poorly explained "
    "by current known populations and requires expert follow-up."
)


# --------------------------------------------------------------------------- #
# artefact loading
# --------------------------------------------------------------------------- #
class Store:
    """Lazily loads and caches the frozen artefacts the API serves."""

    def __init__(self) -> None:
        self._candidates: Optional[pd.DataFrame] = None
        self._explanations: Optional[Dict[str, Any]] = None
        self._lc_index: Optional[Dict[int, tuple]] = None
        self._lc_table: Optional[pd.DataFrame] = None
        self._metrics: Optional[Dict[str, Any]] = None

    # -- candidates ------------------------------------------------------ #
    @property
    def candidates(self) -> pd.DataFrame:
        if self._candidates is None:
            for name in ("candidates_fullscale.parquet", "candidates_matched.parquet"):
                path = ARTEFACTS / name
                if path.exists():
                    frame = pd.read_parquet(path)
                    frame["_source"] = name
                    self._candidates = frame
                    log.info("serving candidates from %s (%d rows)", name, len(frame))
                    break
            else:
                self._candidates = pd.DataFrame(columns=["object_id"])
        return self._candidates

    @property
    def explanations(self) -> Dict[str, Any]:
        if self._explanations is None:
            for name in ("explanations_fullscale.json", "explanations_matched.json"):
                path = ARTEFACTS / name
                if path.exists():
                    self._explanations = json.loads(path.read_text())
                    break
            else:
                self._explanations = {}
        return self._explanations

    @property
    def metrics(self) -> Dict[str, Any]:
        if self._metrics is None:
            path = REPORTS / "metrics.json"
            self._metrics = json.loads(path.read_text()) if path.exists() else {}
        return self._metrics

    # -- light curves ---------------------------------------------------- #
    def _ensure_lc(self) -> None:
        if self._lc_table is not None:
            return
        path = PROCESSED / "lightcurve_cache.parquet"
        if not path.exists():
            raise HTTPException(503, "light-curve cache not built - run scripts/01_featurise.py")
        self._lc_table = pd.read_parquet(path)
        index: Dict[int, tuple] = {}
        ids = self._lc_table["object_id"].to_numpy()
        i, n = 0, len(ids)
        while i < n:
            j = i
            while j + 1 < n and ids[j + 1] == ids[i]:
                j += 1
            index[int(ids[i])] = (i, j - i + 1)
            i = j + 1
        self._lc_index = index

    def lightcurve(self, object_id: int) -> pd.DataFrame:
        self._ensure_lc()
        span = (self._lc_index or {}).get(int(object_id))
        if span is None:
            raise HTTPException(404, f"no cached light curve for object {object_id}")
        start, length = span
        return self._lc_table.iloc[start : start + length]  # type: ignore[union-attr]


store = Store()


def _jsonable(value: Any) -> Any:
    """Coerce numpy/pandas scalars so FastAPI can serialise them."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def _row_payload(row: pd.Series) -> Dict[str, Any]:
    payload = {k: _jsonable(v) for k, v in row.to_dict().items() if k != "analogs"}
    payload["class_name"] = name_of(int(row.get("best_fit_code", -1)))
    payload["family"] = family_of(int(row.get("best_fit_code", -1)))
    payload["disclaimer"] = DISCLAIMER
    explanation = store.explanations.get(str(int(row["object_id"])))
    if explanation is not None:
        payload["explanation"] = explanation
    if "analogs" in row and isinstance(row["analogs"], (list, np.ndarray)):
        payload["analogs"] = _jsonable(list(row["analogs"]))
    return payload


# --------------------------------------------------------------------------- #
# endpoints (the v1 surface is preserved verbatim)
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "candidates_loaded": int(len(store.candidates)),
        "explanations_loaded": len(store.explanations),
        "lightcurve_cache": (PROCESSED / "lightcurve_cache.parquet").exists(),
        "metrics": bool(store.metrics),
    }


@app.get("/api/version")
def version() -> Dict[str, Any]:
    return {"version": __version__, "codename": CODENAME, **stamp()}


@app.get("/api/candidates")
def candidates(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    tier: Optional[str] = None,
    family: Optional[str] = None,
    abstained: Optional[bool] = None,
) -> Dict[str, Any]:
    frame = store.candidates
    if not len(frame):
        return {"total": 0, "candidates": [], "disclaimer": DISCLAIMER}
    view = frame
    if tier is not None and "tier" in view:
        view = view[view["tier"] == tier]
    if family is not None and "best_fit_code" in view:
        wanted = {c for c in view["best_fit_code"].unique() if family_of(int(c)) == family}
        view = view[view["best_fit_code"].isin(wanted)]
    if abstained is not None and "abstain" in view:
        view = view[view["abstain"].astype(bool) == abstained]
    total = int(len(view))
    page = view.iloc[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "candidates": [_row_payload(row) for _, row in page.iterrows()],
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/candidate/{object_id}")
def candidate(object_id: int) -> Dict[str, Any]:
    frame = store.candidates
    hit = frame[frame["object_id"] == object_id]
    if not len(hit):
        raise HTTPException(404, f"object {object_id} is not in the candidate queue")
    return _row_payload(hit.iloc[0])


@app.get("/api/lightcurve/{object_id}")
def lightcurve(object_id: int) -> Dict[str, Any]:
    frame = store.lightcurve(object_id)
    detections = frame[frame["detected_bool"] == 1]
    return {
        "object_id": int(object_id),
        "n_points": int(len(frame)),
        "n_detections": int(len(detections)),
        "mjd_range": [_jsonable(frame["mjd"].min()), _jsonable(frame["mjd"].max())],
        "points": [
            {
                "mjd": _jsonable(r["mjd"]),
                "passband": int(r["passband"]),
                "flux": _jsonable(r["flux"]),
                "flux_err": _jsonable(r["flux_err"]),
                "detected": bool(r["detected_bool"]),
            }
            for _, r in frame.iterrows()
        ],
    }


@app.get("/api/evaluation")
def evaluation() -> Dict[str, Any]:
    if not store.metrics:
        raise HTTPException(503, "no metrics.json - run scripts/03_evaluate.py")
    return store.metrics


@app.get("/api/summary")
def summary() -> Dict[str, Any]:
    frame = store.candidates
    metrics = store.metrics
    out: Dict[str, Any] = {
        "version": __version__,
        "codename": CODENAME,
        "n_candidates": int(len(frame)),
        "disclaimer": DISCLAIMER,
    }
    if len(frame):
        if "tier" in frame:
            out["tiers"] = {k: int(v) for k, v in frame["tier"].value_counts().items()}
        if "abstain" in frame:
            out["n_abstained"] = int(frame["abstain"].astype(bool).sum())
            out["abstention_rate"] = round(float(frame["abstain"].astype(bool).mean()), 4)
        if "novelty_score" in frame:
            out["score_range"] = [_jsonable(frame["novelty_score"].min()),
                                  _jsonable(frame["novelty_score"].max())]
    if metrics:
        benchmarks = metrics.get("benchmarks", {})
        out["benchmarks"] = benchmarks
        # base_rate and dataset are nested, not top-level: they live under
        # benchmarks.<name>.metrics.base_rate and provenance.dataset. Reading
        # metrics["base_rate"] returned None and the dashboard footer printed
        # "unrecorded" for a dataset that was in fact recorded.
        primary = (benchmarks.get("matched_nested_v2")
                   or benchmarks.get("fullscale_domain_matched")
                   or next(iter(benchmarks.values()), {}))
        out["base_rate"] = (primary.get("metrics") or {}).get("base_rate")
        out["base_rate_source"] = primary.get("benchmark") or next(iter(benchmarks), None)
        out["dataset"] = (metrics.get("provenance") or {}).get("dataset")
        out["manifest_id"] = (metrics.get("provenance") or {}).get("manifest")
        out["config_id"] = (metrics.get("provenance") or {}).get("config_id")
    return out


@app.get("/api/classes")
def classes() -> Dict[str, Any]:
    return {
        "n_populations": len(describe()),
        "n_known": len(KNOWN_CLASS_CODES),
        "n_held_out": len(WITHHELD_FROM_PRIOR),
        "classes": describe(),
        "note": (
            "Held-out populations are never seen by the prior, the weight search, the "
            "threshold tuning or the feature selection. Class 6 (muLens-Single) has only "
            "151 training examples and is disclosed as weakly represented in every report."
        ),
    }


@app.get("/api/rescore")
def rescore(object_id: int) -> Dict[str, Any]:
    """Re-serve a candidate's stored scores.

    Deliberately NOT a live re-scoring endpoint: recomputing on request would make
    the queue depend on whatever code is deployed at that moment, and two operators
    could see different numbers for the same alert. The ranking is frozen at
    pipeline time; this returns exactly what was frozen.
    """
    return candidate(object_id)


@app.get("/api/feedback")
def feedback(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    if not FEEDBACK_LOG.exists():
        return {"total": 0, "entries": []}
    lines = FEEDBACK_LOG.read_text().strip().splitlines()
    entries = [json.loads(line) for line in lines[-limit:]]
    return {"total": len(lines), "entries": list(reversed(entries))}


@app.post("/api/feedback")
async def submit_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Record an operator verdict.

    Feedback is append-only and never fed back into the model automatically: a
    loop that retrains on its own labels is how a triage system quietly convinces
    itself. These entries are an audit trail for the next scheduled review.
    """
    object_id = payload.get("object_id")
    verdict = payload.get("verdict")
    if object_id is None or verdict not in ("novel", "known", "artifact", "unclear"):
        raise HTTPException(422, "object_id and verdict in {novel, known, artifact, unclear} are required")
    entry = {
        "object_id": int(object_id),
        "verdict": verdict,
        "comment": str(payload.get("comment", ""))[:2000],
        "operator": str(payload.get("operator", "anonymous"))[:80],
        "received_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": __version__,
    }
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    log.info("operator feedback recorded for %s: %s", object_id, verdict)
    return {"recorded": True, "entry": entry}


# --------------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------------- #
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        """Serve the built React app; unknown paths fall back to index.html."""
        candidate_path = (WEB_DIST / full_path).resolve()
        if full_path and candidate_path.is_file() and candidate_path.is_relative_to(WEB_DIST):
            return FileResponse(candidate_path)
        return FileResponse(WEB_DIST / "index.html")

else:

    @app.get("/")
    def no_dashboard() -> JSONResponse:
        return JSONResponse(
            {"detail": "web/dist not built - run `cd web && npm install && npm run build`",
             "api_docs": "/docs"},
            status_code=200,
        )
