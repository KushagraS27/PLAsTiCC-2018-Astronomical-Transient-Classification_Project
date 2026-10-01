"""Manifests and the leakage guard: the protocol that keeps numbers honest."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest

from cne.manifests import LeakageGuard, SplitManifest, object_level_split


@pytest.fixture
def manifest(tmp_path):
    splits = {
        "prior_fit": list(range(0, 600)),
        "validation": list(range(600, 800)),
        "test_matched": list(range(800, 1000)),
    }
    return SplitManifest.create("test-manifest", splits, dataset={"name": "synthetic"}, seed=42)


class TestSplitting:
    def test_split_is_disjoint_and_covers_every_object(self):
        ids = np.arange(1000)
        parts = object_level_split(ids, {"a": 0.6, "b": 0.2, "c": 0.2}, seed=42)
        union = np.concatenate([parts[k] for k in parts])
        assert len(set(union.tolist())) == 1000
        assert set(parts["a"]).isdisjoint(parts["b"])
        assert set(parts["b"]).isdisjoint(parts["c"])

    def test_proportions_are_respected(self):
        parts = object_level_split(np.arange(10_000), {"a": 0.6, "b": 0.4}, seed=1)
        assert len(parts["a"]) == pytest.approx(6000, abs=150)

    def test_deterministic_under_seed(self):
        a = object_level_split(np.arange(500), {"x": 0.5, "y": 0.5}, seed=9)
        b = object_level_split(np.arange(500), {"x": 0.5, "y": 0.5}, seed=9)
        assert a["x"].tolist() == b["x"].tolist()

    def test_duplicate_ids_are_collapsed(self):
        parts = object_level_split(np.array([1, 1, 2, 2, 3]), {"a": 0.5, "b": 0.5}, seed=0)
        union = np.concatenate([parts["a"], parts["b"]])
        assert sorted(union.tolist()) == [1, 2, 3]


class TestManifest:
    def test_write_once(self, manifest, tmp_path):
        path = tmp_path / "m.json"
        manifest.write(path)
        with pytest.raises(FileExistsError):
            manifest.write(path)

    def test_force_overwrite_is_allowed_but_explicit(self, manifest, tmp_path):
        path = tmp_path / "m.json"
        manifest.write(path)
        manifest.write(path, force=True)

    def test_round_trip_preserves_ids(self, manifest, tmp_path):
        path = tmp_path / "m.json"
        manifest.write(path)
        loaded = SplitManifest.read(path)
        assert loaded.split("test_matched").tolist() == manifest.split("test_matched").tolist()
        assert loaded.verify(path)

    def test_tampering_is_detected(self, manifest, tmp_path):
        import json

        path = tmp_path / "m.json"
        manifest.write(path, force=True)
        raw = json.loads(path.read_text())
        raw["splits"]["test_matched"].append(99999)
        path.chmod(0o644)
        path.write_text(json.dumps(raw))
        assert manifest.verify(path) is False

    def test_assert_disjoint_catches_overlap(self, tmp_path):
        bad = SplitManifest.create("bad", {"a": [1, 2, 3], "b": [3, 4]}, dataset={})
        with pytest.raises(AssertionError):
            bad.assert_disjoint("a", "b")

    def test_fingerprints_are_recorded(self, manifest):
        fp = manifest.fingerprints()
        assert set(fp) == set(manifest.splits)
        assert all(len(v["sha256_16"]) == 16 for v in fp.values())


class TestLeakageGuard:
    def test_clean_run_passes(self, manifest):
        guard = LeakageGuard(manifest, locked_split="test_matched")
        guard.stage("weight_search").seen_ids(manifest.split("validation"))
        guard.assert_clean()
        assert guard.violations() == {}

    def test_touching_a_locked_test_object_fails(self, manifest):
        guard = LeakageGuard(manifest, locked_split="test_matched")
        guard.stage("weight_search").seen_ids(np.concatenate([manifest.split("validation"), [850]]))
        violations = guard.violations()
        assert violations["weight_search"] == [850]
        with pytest.raises(AssertionError):
            guard.assert_clean()

    def test_violations_are_attributed_to_the_offending_stage(self, manifest):
        guard = LeakageGuard(manifest, locked_split="test_matched")
        guard.stage("feature_selection").seen_ids([801])
        guard.stage("weight_search").seen_ids(manifest.split("validation"))
        assert list(guard.violations()) == ["feature_selection"]

    def test_report_counts_everything(self, manifest):
        guard = LeakageGuard(manifest, locked_split="test_matched")
        guard.stage("a").seen_ids([1, 2, 3])
        report = guard.report()
        assert report["stages"]["a"] == 3
        assert report["locked_n"] == 200


needs_train_data = pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "data" / "processed" / "train_features.parquet").exists(),
    reason="requires data/processed/train_features.parquet - run scripts/00_download.py then 01_featurise.py",
)


@needs_train_data
class TestGuardIsActuallyArmed:
    """Regression: the cached-manifest path once returned before building the guard.

    The default guard from ``__init__`` had no manifest and locked ``validation``,
    so ``locked_n`` was 0 and ``assert_clean()`` could never fire - a silent no-op
    on the one control that prevents held-out test leakage.
    """

    def test_loading_a_frozen_manifest_arms_the_guard(self):
        from cne.pipeline import CNEPipeline

        pipe = CNEPipeline()
        pipe.build_train_features()
        pipe.make_manifest()  # takes the cached path when the file exists
        assert pipe.guard_.manifest is not None, "guard has no manifest - it verifies nothing"
        assert pipe.guard_.locked_split == "test_matched"
        assert len(pipe.guard_.manifest.split("test_matched")) > 0
        assert pipe.guard_.report()["locked_n"] == len(pipe.guard_.manifest.split("test_matched"))

    def test_guard_fires_when_a_locked_object_is_touched(self):
        from cne.pipeline import CNEPipeline

        pipe = CNEPipeline()
        pipe.build_train_features()
        manifest = pipe.make_manifest()
        locked = manifest.split("test_matched")[:5]
        pipe.guard_.stage("weight_search").seen_ids(locked)
        assert pipe.guard_.violations(), "guard did not record the violation"
        with pytest.raises(AssertionError):
            pipe.guard_.assert_clean()

    def test_reading_only_validation_is_clean(self):
        from cne.pipeline import CNEPipeline

        pipe = CNEPipeline()
        pipe.build_train_features()
        manifest = pipe.make_manifest()
        pipe.guard_.stage("weight_search").seen_ids(manifest.split("validation"))
        assert pipe.guard_.violations() == {}
        pipe.guard_.assert_clean()  # must not raise

    def test_a_tampered_manifest_is_refused(self, tmp_path, monkeypatch):
        import json

        from cne.pipeline import CNEPipeline

        pipe = CNEPipeline()
        pipe.build_train_features()
        manifest = pipe.make_manifest()
        path = pipe.reports / "manifests" / "splits_v2.json"
        raw = json.loads(path.read_text())
        raw["splits"]["test_matched"].append(999999)
        path.chmod(0o644)
        path.write_text(json.dumps(raw, sort_keys=True))
        try:
            fresh = CNEPipeline()
            fresh.build_train_features()
            with pytest.raises(RuntimeError, match="fingerprint"):
                fresh.make_manifest()
        finally:
            # Always restore both content and the frozen mode, or a failure here
            # leaves the write-once manifest editable for every later run.
            path.chmod(0o644)
            path.write_text(json.dumps(
                {**raw, "splits": {**raw["splits"], "test_matched": manifest.split("test_matched").tolist()}},
                sort_keys=True))
            path.chmod(0o444)
            assert manifest.verify(path), "failed to restore the frozen manifest"
