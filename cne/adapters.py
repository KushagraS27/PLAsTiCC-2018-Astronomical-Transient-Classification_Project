"""Broker-neutral alert adapters and offline replay (CNE v2, prompt 09).

The rule this module enforces: **no core scoring code may know what survey an
alert came from.** Everything downstream of an adapter speaks
:class:`cne.schema.AlertPacket`.

Live broker endpoints were unreachable from the development environment, so this
is a *replay* adapter: it reads stored ZTF-like or Rubin-like alert packets from
local files. That is enough to prove schema portability, per-alert failure
isolation and throughput, which are the properties that actually transfer to a
live integration. It cannot produce measurable novelty metrics - nobody knows the
true label of a genuine unknown - and this module never claims otherwise.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .logging import get_logger
from .schema import AlertPacket

log = get_logger("adapters")

ADAPTER_VERSION = "1.0.0"

#: LSST/Rubin passbands and the ZTF equivalents. ZTF has only g, r, i.
ZTF_FID_TO_PASSBAND = {1: 1, 2: 2, 3: 3}          # ztf fid 1=g 2=r 3=i  -> ugrizy 1,2,3
RUBIN_BAND_TO_PASSBAND = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}


class AdapterError(Exception):
    """Raised for a packet that cannot be interpreted. Never fatal to a stream."""


@dataclass
class ReplayFailure:
    alert_id: str
    survey: str
    error: str

    def as_dict(self) -> Dict[str, str]:
        return {"alert_id": self.alert_id, "survey": self.survey, "error": self.error}


@dataclass
class ReplayReport:
    survey: str
    adapter_version: str
    n_packets: int = 0
    n_succeeded: int = 0
    n_failed: int = 0
    elapsed_s: float = 0.0
    failures: List[ReplayFailure] = field(default_factory=list)

    @property
    def throughput_per_second(self) -> float:
        return self.n_packets / self.elapsed_s if self.elapsed_s > 0 else float("nan")

    @property
    def mean_latency_ms(self) -> float:
        return 1000.0 * self.elapsed_s / max(self.n_packets, 1)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "survey": self.survey,
            "adapter_version": self.adapter_version,
            "n_packets": self.n_packets,
            "n_succeeded": self.n_succeeded,
            "n_failed": self.n_failed,
            "failure_rate": self.n_failed / max(self.n_packets, 1),
            "elapsed_s": round(self.elapsed_s, 3),
            "throughput_per_second": round(self.throughput_per_second, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "failures": [f.as_dict() for f in self.failures[:20]],
        }


class AlertAdapter:
    """Base class: subclass and implement :meth:`to_packet`."""

    survey = "unknown"

    def to_packet(self, raw: Dict[str, Any]) -> AlertPacket:
        raise NotImplementedError

    def provenance(self, raw: Dict[str, Any], alert_id: str) -> Dict[str, Any]:
        return {
            "survey": self.survey,
            "alert_id": alert_id,
            "adapter": type(self).__name__,
            "adapter_version": ADAPTER_VERSION,
        }


class ZTFLikeAdapter(AlertAdapter):
    """ZTF-shaped packets: ``candid``, ``candidate`` with jd/flux/fluxerr/fid,
    plus ``prv_candidates`` history and a ``drb`` real-bogus score.

    ZTF reports Julian Dates and fluxes in nMgy; CNE works in MJD, so the adapter
    converts at the boundary and records the conversion in provenance.
    """

    survey = "ztf-like"
    JD_OFFSET = 2_400_000.5

    def to_packet(self, raw: Dict[str, Any]) -> AlertPacket:
        candidate = raw.get("candidate") or {}
        alert_id = str(raw.get("candid") or candidate.get("candid") or "unknown")
        history = list(raw.get("prv_candidates") or [])
        mjd: List[float] = []
        band: List[int] = []
        flux: List[float] = []
        err: List[float] = []
        det: List[int] = []
        # The triggering candidate is itself an observation.
        for entry in [candidate] + history:
            if not isinstance(entry, dict):
                continue
            if entry.get("jd") is None or entry.get("fid") is None:
                continue
            fid = int(entry["fid"])
            if fid not in ZTF_FID_TO_PASSBAND:
                continue
            f = entry.get("flux")
            fe = entry.get("fluxerr") or entry.get("flux_err")
            if f is None or fe is None or float(fe) <= 0:
                continue
            mjd.append(float(entry["jd"]) - self.JD_OFFSET)
            band.append(ZTF_FID_TO_PASSBAND[fid])
            flux.append(float(f))
            err.append(float(fe))
            det.append(1 if entry.get("isdiffpos", 1) in (1, "t", True) else 0)
        if not mjd:
            raise AdapterError("packet contains no usable observations")
        quality: Dict[str, Any] = {}
        for key in ("drb", "realbogus", "rb"):
            if key in candidate:
                quality["realbogus"] = float(candidate[key])
                break
        context = {
            "ra": candidate.get("ra"),
            "decl": candidate.get("dec"),
            "hostgal_photoz": candidate.get("zphot"),
            "hostgal_photoz_err": candidate.get("zphot_err"),
            "mwebv": candidate.get("mwebv"),
            "distmod": candidate.get("distmod"),
            "ddf_bool": 0,
            "hostgal_specz": candidate.get("z"),
        }
        packet = AlertPacket(
            object_id=int(raw.get("objectId_hash") or _hash_id(str(raw.get("objectId", alert_id)))),
            survey=self.survey,
            mjd=np.asarray(mjd, dtype="float32"),
            passband=np.asarray(band, dtype="int8"),
            flux=np.asarray(flux, dtype="float32"),
            flux_err=np.asarray(err, dtype="float32"),
            detected=np.asarray(det, dtype="int8"),
            context={k: v for k, v in context.items() if v is not None},
            quality=quality,
            provenance=self.provenance(raw, alert_id),
        )
        problems = packet.validate()
        if problems:
            raise AdapterError("; ".join(problems))
        return packet


class RubinLikeAdapter(AlertAdapter):
    """Rubin/LSST-shaped packets: ``alertId``, ``diaSource`` with midPointTai,
    ``band`` letters, ``psfFlux``/``psfFluxErr`` and a ``diaObject`` context."""

    survey = "rubin-like"

    def to_packet(self, raw: Dict[str, Any]) -> AlertPacket:
        alert_id = str(raw.get("alertId") or raw.get("diaSource", {}).get("diaSourceId") or "unknown")
        sources = raw.get("diaSourceHistory") or raw.get("prvDiaForcedSources") or []
        first = raw.get("diaSource") or {}
        entries = [first] + list(sources)
        mjd, band, flux, err, det = [], [], [], [], []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            letter = entry.get("band") or entry.get("filterName")
            if letter not in RUBIN_BAND_TO_PASSBAND:
                continue
            f = entry.get("psfFlux", entry.get("flux"))
            fe = entry.get("psfFluxErr", entry.get("fluxErr"))
            t = entry.get("midPointTai", entry.get("midpointMjdTai"))
            if f is None or fe is None or t is None or float(fe) <= 0:
                continue
            mjd.append(float(t))
            band.append(RUBIN_BAND_TO_PASSBAND[letter])
            flux.append(float(f))
            err.append(float(fe))
            det.append(0 if float(f) < 0 else 1)
        if not mjd:
            raise AdapterError("packet contains no usable diaSource observations")
        dia_object = raw.get("diaObject") or {}
        context = {
            "ra": first.get("ra", dia_object.get("ra")),
            "decl": first.get("decl", dia_object.get("dec")),
            "hostgal_photoz": dia_object.get("zphot"),
            "hostgal_photoz_err": dia_object.get("zphotErr"),
            "mwebv": dia_object.get("mwebv"),
            "distmod": dia_object.get("distmod"),
            "ddf_bool": 0,
            "hostgal_specz": dia_object.get("specz"),
        }
        packet = AlertPacket(
            object_id=int(dia_object.get("diaObjectId") or _hash_id(alert_id)),
            survey=self.survey,
            mjd=np.asarray(mjd, dtype="float32"),
            passband=np.asarray(band, dtype="int8"),
            flux=np.asarray(flux, dtype="float32"),
            flux_err=np.asarray(err, dtype="float32"),
            detected=np.asarray(det, dtype="int8"),
            context={k: v for k, v in context.items() if v is not None},
            quality={"drb": raw.get("drb")} if raw.get("drb") is not None else {},
            provenance=self.provenance(raw, alert_id),
        )
        problems = packet.validate()
        if problems:
            raise AdapterError("; ".join(problems))
        return packet


def _hash_id(text: str) -> int:
    import hashlib

    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


ADAPTERS: Dict[str, AlertAdapter] = {
    "ztf-like": ZTFLikeAdapter(),
    "rubin-like": RubinLikeAdapter(),
}


def detect_adapter(raw: Dict[str, Any]) -> AlertAdapter:
    """Pick an adapter from packet shape, so replay files need no metadata."""
    if "candid" in raw or "candidate" in raw:
        return ADAPTERS["ztf-like"]
    if "alertId" in raw or "diaSource" in raw:
        return ADAPTERS["rubin-like"]
    raise AdapterError("unrecognised alert packet shape")


class ReplayRunner:
    """Replay stored alert packets through a scoring callback with per-alert isolation.

    ``score_fn(packet) -> dict`` is the production scoring path. A packet that
    fails is recorded and skipped: one malformed alert must never stop a stream of
    a million.
    """

    def __init__(self, score_fn: Callable[[AlertPacket], Dict[str, Any]]):
        self.score_fn = score_fn

    def iter_packets(self, path: str | Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
        path = Path(path)
        text = path.read_text()
        if path.suffix == ".json":
            payload = json.loads(text)
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                yield str(item.get("candid") or item.get("alertId") or "unknown"), item
            return
        # JSON Lines
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                yield "malformed-line", {"_error": str(exc), "_raw": line[:200]}
                continue
            yield str(item.get("candid") or item.get("alertId") or "unknown"), item

    def run(self, path: str | Path, adapter_name: Optional[str] = None, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], ReplayReport]:
        started = time.perf_counter()
        report = ReplayReport(survey=adapter_name or "auto", adapter_version=ADAPTER_VERSION)
        results: List[Dict[str, Any]] = []
        for alert_id, raw in self.iter_packets(path):
            report.n_packets += 1
            if limit is not None and report.n_packets > limit:
                report.n_packets -= 1
                break
            try:
                if "_error" in raw:
                    raise AdapterError(raw["_error"])
                adapter = ADAPTERS[adapter_name] if adapter_name else detect_adapter(raw)
                report.survey = adapter.survey
                packet = adapter.to_packet(raw)
                payload = self.score_fn(packet)
                payload["provenance"] = packet.provenance
                results.append(payload)
                report.n_succeeded += 1
            except Exception as exc:
                report.n_failed += 1
                report.failures.append(ReplayFailure(alert_id=alert_id, survey=report.survey, error=f"{type(exc).__name__}: {exc}"))
                log.warning("replay: alert %s failed: %s", alert_id, exc)
        report.elapsed_s = time.perf_counter() - started
        log.info("replay complete: %d packets, %d ok, %d failed in %.2fs (%.1f/s)",
                 report.n_packets, report.n_succeeded, report.n_failed, report.elapsed_s, report.throughput_per_second)
        return results, report


def write_sample_alerts(path: str | Path, packets: Sequence[AlertPacket], survey: str = "ztf-like") -> Path:
    """Serialise internal packets back into a broker-shaped file.

    Used by the test suite and the demo so replay has fixtures without network
    access, and so the round trip (packet -> broker JSON -> packet) is verified.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for packet in packets:
        if survey == "ztf-like":
            candidate = {
                "candid": int(packet.object_id),
                "ra": packet.context.get("ra", 0.0),
                "dec": packet.context.get("decl", 0.0),
                "jd": float(packet.mjd[-1]) + ZTFLikeAdapter.JD_OFFSET,
                "fid": int(packet.passband[-1]),
                "flux": float(packet.flux[-1]),
                "fluxerr": float(packet.flux_err[-1]),
                "isdiffpos": int(packet.detected[-1]),
                "drb": packet.quality.get("realbogus", 0.9),
            }
            prv = [
                {
                    "jd": float(m) + ZTFLikeAdapter.JD_OFFSET,
                    "fid": int(b),
                    "flux": float(f),
                    "fluxerr": float(e),
                    "isdiffpos": int(d),
                }
                for m, b, f, e, d in zip(packet.mjd[:-1], packet.passband[:-1], packet.flux[:-1],
                                         packet.flux_err[:-1], packet.detected[:-1])
            ]
            lines.append(json.dumps({"candid": candidate["candid"], "objectId": f"ZTF{packet.object_id}",
                                     "candidate": candidate, "prv_candidates": prv}))
        else:
            first = {
                "diaSourceId": int(packet.object_id),
                "midPointTai": float(packet.mjd[-1]),
                "band": {v: k for k, v in RUBIN_BAND_TO_PASSBAND.items()}[int(packet.passband[-1])],
                "psfFlux": float(packet.flux[-1]),
                "psfFluxErr": float(packet.flux_err[-1]),
                "ra": packet.context.get("ra", 0.0),
                "decl": packet.context.get("decl", 0.0),
            }
            history = [
                {
                    "midPointTai": float(m),
                    "band": {v: k for k, v in RUBIN_BAND_TO_PASSBAND.items()}[int(b)],
                    "psfFlux": float(f),
                    "psfFluxErr": float(e),
                }
                for m, b, f, e in zip(packet.mjd[:-1], packet.passband[:-1], packet.flux[:-1], packet.flux_err[:-1])
            ]
            lines.append(json.dumps({"alertId": int(packet.object_id), "diaSource": first,
                                     "diaSourceHistory": history,
                                     "diaObject": {"diaObjectId": int(packet.object_id),
                                                   "zphot": packet.context.get("hostgal_photoz"),
                                                   "zphotErr": packet.context.get("hostgal_photoz_err")}}))
    path.write_text("\n".join(lines) + "\n")
    return path
