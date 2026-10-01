"""Data access: streaming correctness, sampling, and the blinded-file guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cne.data import (
    disjoint_object_split,
    download_url,
    iter_lightcurve_chunks,
    rare_preserving_subsample,
)


@pytest.fixture
def lc_file(tmp_path):
    """A light-curve file sorted by object_id, as PLAsTiCC ships it."""
    rows = []
    for oid in range(1, 21):
        for k in range(37):
            rows.append((oid, 60000.0 + k, k % 6, 100.0 + k, 2.0, 1))
    frame = pd.DataFrame(rows, columns=["object_id", "mjd", "passband", "flux", "flux_err", "detected_bool"])
    path = tmp_path / "lc.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    return path, frame


class TestStreaming:
    def test_every_object_appears_exactly_once(self, lc_file):
        path, frame = lc_file
        seen = []
        for chunk in iter_lightcurve_chunks(path, chunk_rows=100):
            seen.extend(chunk["object_id"].unique().tolist())
        assert sorted(seen) == sorted(frame["object_id"].unique().tolist())
        assert len(seen) == len(set(seen)), "an object was split across two chunks"

    def test_no_rows_are_lost_or_duplicated(self, lc_file):
        path, frame = lc_file
        total = sum(len(chunk) for chunk in iter_lightcurve_chunks(path, chunk_rows=137))
        assert total == len(frame)

    def test_row_count_is_independent_of_chunk_size(self, lc_file):
        path, frame = lc_file
        counts = {sum(len(c) for c in iter_lightcurve_chunks(path, chunk_rows=n)) for n in (37, 100, 997, 10_000)}
        assert counts == {len(frame)}

    def test_each_chunk_contains_whole_objects(self, lc_file):
        path, _ = lc_file
        for chunk in iter_lightcurve_chunks(path, chunk_rows=211):
            counts = chunk.groupby("object_id").size()
            assert (counts == 37).all(), "a chunk contained a partial object"


class TestSampling:
    def test_rare_classes_are_never_thinned(self):
        labels = pd.Series([1] * 5000 + [2] * 40 + [3] * 7, index=range(5047))
        keep = rare_preserving_subsample(labels, max_stream=1000, rare_cap=1500, seed=42)
        kept = labels.loc[keep].value_counts()
        assert kept[2] == 40 and kept[3] == 7

    def test_budget_is_respected(self):
        labels = pd.Series([1] * 5000 + [2] * 4000, index=range(9000))
        keep = rare_preserving_subsample(labels, max_stream=2000, rare_cap=1500, seed=42)
        assert len(keep) <= 2000

    def test_no_op_when_already_small(self):
        labels = pd.Series([1] * 10 + [2] * 10, index=range(20))
        assert len(rare_preserving_subsample(labels, max_stream=100)) == 20

    def test_deterministic(self):
        labels = pd.Series([1] * 5000 + [2] * 5000, index=range(10_000))
        a = rare_preserving_subsample(labels, 3000, seed=7)
        b = rare_preserving_subsample(labels, 3000, seed=7)
        assert list(a) == list(b)

    def test_split_is_disjoint_and_exhaustive(self):
        ids = np.arange(1000)
        a, b = disjoint_object_split(ids, reference_fraction=0.6, seed=42)
        assert set(a).isdisjoint(b)
        assert set(a) | set(b) == set(ids.tolist())
        assert 0.5 < len(a) / len(ids) < 0.7


class TestURLs:
    def test_uses_the_download_form_not_the_content_form(self):
        """Zenodo's /files/{name}/content returns 92-byte stubs."""
        url = download_url("plasticc_train_metadata.csv.gz")
        assert url.endswith("?download=1")
        assert "/content" not in url
