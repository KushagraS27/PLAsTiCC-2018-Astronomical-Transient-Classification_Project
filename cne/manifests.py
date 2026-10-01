"""Split manifests and leakage guards (CNE v2, prompt 01).

The failure mode this prevents is quiet and total: tune a weight on the same
objects you later report on, and every headline number is optimistic by an amount
you cannot estimate. CNE fixes the test set *before* any tuning code can see it,
records every object id the tuning stage touches, and fails a test if a locked
id ever appears there.

A manifest is write-once. Re-freezing requires an explicit ``force=True`` and is
logged, because a silently regenerated manifest is indistinguishable from
cheating after the fact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from .logging import get_logger
from .version import SEED, __version__

log = get_logger("manifests")


def _hash_ids(ids: Iterable[int]) -> str:
    arr = np.asarray(sorted(int(v) for v in ids), dtype="int64")
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


@dataclass
class SplitManifest:
    """A frozen record of which objects belong to which role in an experiment."""

    manifest_id: str
    created_utc: str
    seed: int
    cne_version: str
    config_fingerprint: str
    dataset: Dict[str, str]
    splits: Dict[str, List[int]] = field(default_factory=dict)
    locked_test: bool = True
    notes: str = ""

    # ------------------------------------------------------------------ create
    @classmethod
    def create(cls, manifest_id: str, splits: Dict[str, Sequence[int]], dataset: Dict[str, str],
               config_fingerprint: str = "", seed: int = SEED, locked_test: bool = True,
               notes: str = "") -> "SplitManifest":
        normalised = {name: sorted(int(v) for v in ids) for name, ids in splits.items()}
        return cls(
            manifest_id=manifest_id,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            seed=seed,
            cne_version=__version__,
            config_fingerprint=config_fingerprint,
            dataset=dataset,
            splits=normalised,
            locked_test=locked_test,
            notes=notes,
        )

    # ----------------------------------------------------------------- access
    @property
    def ids(self) -> Dict[str, np.ndarray]:
        return {name: np.asarray(values, dtype="int64") for name, values in self.splits.items()}

    def split(self, name: str) -> np.ndarray:
        if name not in self.splits:
            raise KeyError(f"split '{name}' not in manifest {self.manifest_id}; have {list(self.splits)}")
        return np.asarray(self.splits[name], dtype="int64")

    def overlap(self, a: str, b: str) -> Set[int]:
        return set(self.split(a).tolist()) & set(self.split(b).tolist())

    def assert_disjoint(self, *names: str) -> None:
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                shared = self.overlap(a, b)
                if shared:
                    raise AssertionError(f"splits '{a}' and '{b}' share {len(shared)} object ids, e.g. {sorted(shared)[:5]}")

    def fingerprints(self) -> Dict[str, Dict[str, Any]]:
        return {name: {"n": len(values), "sha256_16": _hash_ids(values)} for name, values in self.splits.items()}

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["fingerprints"] = self.fingerprints()
        return out

    def write(self, path: str | Path, force: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            raise FileExistsError(
                f"manifest {path} already exists and is frozen. Pass force=True to overwrite, "
                "and record why in the changelog - a silently regenerated manifest is how "
                "test-set leakage becomes undetectable."
            )
        if path.exists():
            path.chmod(0o644)  # a frozen manifest must still be overwritable when forced
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        path.chmod(0o444)
        log.info("manifest %s written to %s (splits: %s)", self.manifest_id, path,
                 {k: len(v) for k, v in self.splits.items()})
        return path

    @classmethod
    def read(cls, path: str | Path) -> "SplitManifest":
        raw = json.loads(Path(path).read_text())
        raw.pop("fingerprints", None)
        return cls(**raw)

    def verify(self, path: str | Path) -> bool:
        """Recompute the fingerprints from the ids actually on disk.

        Comparing the recorded block against a freshly computed one is what makes
        tampering detectable: comparing it against ``self.fingerprints()`` only
        proves the in-memory copy matches itself, and passes even after someone
        edits the split list.
        """
        stored = json.loads(Path(path).read_text())
        recorded = stored.get("fingerprints") or {}
        splits = stored.get("splits") or {}
        if set(recorded) != set(splits):
            return False
        for name, ids in splits.items():
            current = {"n": len(ids), "sha256_16": _hash_ids(np.asarray(ids, dtype="int64"))}
            if current != recorded[name]:
                log.warning("manifest %s split %r fails its fingerprint check", self.manifest_id, name)
                return False
        return True


class LeakageGuard:
    """Records every object id a tuning stage touches.

    Usage::

        guard = LeakageGuard(manifest)
        with guard.stage("weight_search"):
            ...  # pass guard.seen_ids(...) anything the stage reads
        guard.assert_clean()      # raises if a locked-test id was touched
    """

    def __init__(self, manifest: Optional[SplitManifest] = None, locked_split: str = "test"):
        self.manifest = manifest
        self.locked_split = locked_split
        self.seen: Dict[str, Set[int]] = {}
        self._stage: Optional[str] = None

    def stage(self, name: str) -> "LeakageGuard":
        self._stage = name
        self.seen.setdefault(name, set())
        return self

    def __enter__(self) -> "LeakageGuard":
        if self._stage is None:
            self.stage("default")
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def seen_ids(self, ids: Iterable[int]) -> np.ndarray:
        """Declare that the current stage read these ids."""
        arr = np.asarray([int(v) for v in ids], dtype="int64")
        self.seen.setdefault(self._stage or "default", set()).update(arr.tolist())
        return arr

    def violations(self) -> Dict[str, List[int]]:
        if self.manifest is None:
            return {}
        locked = set(self.manifest.split(self.locked_split).tolist())
        return {stage: sorted(seen & locked) for stage, seen in self.seen.items() if seen & locked}

    def assert_clean(self) -> None:
        bad = self.violations()
        if bad:
            detail = {k: v[:5] for k, v in bad.items()}
            raise AssertionError(f"locked '{self.locked_split}' objects were read during tuning: {detail}")

    def report(self) -> Dict[str, Any]:
        return {
            "stages": {stage: len(ids) for stage, ids in self.seen.items()},
            "violations": {stage: len(ids) for stage, ids in self.violations().items()},
            "locked_split": self.locked_split,
            "locked_n": len(self.manifest.split(self.locked_split)) if self.manifest else 0,
        }


def object_level_split(object_ids: Sequence[int], fractions: Dict[str, float], seed: int = SEED) -> Dict[str, np.ndarray]:
    """Deterministic split by OBJECT ID.

    Splitting by observation would let two epochs of the same transient land in
    both halves - the single easiest way to leak in time-domain astronomy.
    """
    ids = np.asarray(sorted({int(v) for v in object_ids}), dtype="int64")
    rng = np.random.default_rng(seed)
    keys = rng.random(len(ids))
    splits: Dict[str, np.ndarray] = {}
    cumulative = 0.0
    names = list(fractions)
    for i, name in enumerate(names):
        if i == len(names) - 1:
            splits[name] = ids[keys >= cumulative]
        else:
            frac = float(fractions[name])
            splits[name] = ids[(keys >= cumulative) & (keys < cumulative + frac)]
            cumulative += frac
    return splits
