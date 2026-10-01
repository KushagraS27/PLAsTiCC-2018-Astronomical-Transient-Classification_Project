"""Data access: chunked streaming reads, memory-safe metadata, parquet caches.

Design constraints that shaped this module (all learned in CNE v1):

* 2 GB RAM ceiling. A PLAsTiCC test chunk is ~1 GB of CSV, so nothing here ever
  materialises the whole file as float64.
* Zenodo's ``/files/{f}/content`` URL form returns 92-byte stubs; only
  ``?download=1`` returns the real object.  ``download_url`` encodes that.
* Serving a light curve by scanning gzipped CSV per HTTP request took ~40 s in
  v1.  :class:`LightCurveCache` replaces it with a prebuilt parquet index (~ms).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .logging import get_logger
from .schema import (
    LABEL_COLUMN,
    METADATA_COLUMNS,
    METADATA_DTYPES,
    PHOTOMETRY_COLUMNS,
    PHOTOMETRY_DTYPES,
)

log = get_logger("data")

ZENODO_RECORD = "2539456"
ZENODO_URL_TEMPLATE = "https://zenodo.org/records/{record}/files/{name}?download=1"


# --------------------------------------------------------------------------- #
# downloading
# --------------------------------------------------------------------------- #
def download_url(name: str, record: str = ZENODO_RECORD) -> str:
    """Correct Zenodo download URL. The ``/content`` form returns 92-byte stubs."""
    return ZENODO_URL_TEMPLATE.format(record=record, name=name)


def download_file(name: str, dest_dir: str | Path, record: str = ZENODO_RECORD, min_bytes: int = 1024) -> Path:
    """Download one PLAsTiCC file, refusing to accept a stub response."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    url = download_url(name, record)
    log.info("download start name=%s url=%s", name, url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as handle:
        shutil.copyfileobj(response, handle)
    size = tmp.stat().st_size
    if size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise IOError(f"downloaded {name} is only {size} bytes - Zenodo stub, check the URL form")
    tmp.replace(dest)
    log.info("download done name=%s bytes=%d", name, size)
    return dest


def sha256_of_file(path: str | Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_frame(frame: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> str:
    """Content hash of a frame, order-independent over rows."""
    cols = list(columns) if columns else [c for c in frame.columns if frame[c].dtype.kind in "iuf"]
    view = frame[cols].copy()
    for col in view.columns:
        view[col] = np.round(view[col].to_numpy(dtype="float64"), 6)
    blob = np.ascontiguousarray(view.to_numpy(dtype="float64"))
    blob = blob[np.lexsort(blob.T[::-1][:1])] if blob.size else blob
    return hashlib.sha256(blob.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def read_metadata(
    path: str | Path,
    object_ids: Optional[Iterable[int]] = None,
    with_label: bool = False,
    chunk_rows: int = 2_000_000,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Memory-safe metadata read.

    Reading the 3.49M-row PLAsTiCC test metadata as float64 gets a 2 GB host
    OOM-killed.  We read only the columns we use, in narrow dtypes, and filter to
    ``object_ids`` while streaming so the full table is never resident.

    ``label_column`` matters more than it looks. The PLAsTiCC *train* metadata
    carries truth in ``target``; the *test* metadata ships ``target = 0`` for
    every object (the competition blinding) and the unblinded truth in
    ``true_target``.  Reading ``target`` from the test file silently yields an
    all-zero label vector and a benchmark that measures nothing.
    """
    path = Path(path)
    columns = list(METADATA_COLUMNS) + ([label_column] if with_label else [])
    dtypes = dict(METADATA_DTYPES)
    if with_label:
        dtypes[label_column] = "int16"
    keep = None if object_ids is None else np.unique(np.asarray(list(object_ids), dtype="int32"))

    parts: List[pd.DataFrame] = []
    reader = pd.read_csv(path, usecols=columns, dtype=dtypes, chunksize=chunk_rows)
    for chunk in reader:
        if keep is not None:
            chunk = chunk[np.isin(chunk["object_id"].to_numpy(), keep)]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in columns})
    out = pd.concat(parts, ignore_index=True)
    if with_label and label_column != LABEL_COLUMN:
        out = out.rename(columns={label_column: LABEL_COLUMN})
    if with_label and (out[LABEL_COLUMN].to_numpy() == 0).all():
        raise ValueError(
            f"{path.name}: every label is 0. This file is blinded - pass "
            f"label_column='true_target' for PLAsTiCC test metadata."
        )
    log.info("read_metadata file=%s rows=%d label_column=%s", path.name, len(out), label_column)
    return out


# --------------------------------------------------------------------------- #
# streaming photometry
# --------------------------------------------------------------------------- #
@dataclass
class StreamStats:
    rows: int = 0
    objects: int = 0
    chunks: int = 0
    peak_objects_per_chunk: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {"rows": self.rows, "objects": self.objects, "chunks": self.chunks,
                "peak_objects_per_chunk": self.peak_objects_per_chunk}


def iter_lightcurve_chunks(
    path: str | Path,
    chunk_rows: int = 2_000_000,
    object_filter: Optional[set] = None,
) -> Iterator[pd.DataFrame]:
    """Yield photometry grouped so that every object appears in exactly one chunk.

    PLAsTiCC light-curve files are sorted by ``object_id``, so an object can only
    straddle one chunk boundary.  We hold the trailing partial object over to the
    next chunk, which keeps peak memory at roughly one chunk regardless of file
    size and never splits an object across featurisation batches.
    """
    path = Path(path)
    carry = None
    stats = StreamStats()
    reader = pd.read_csv(path, dtype=PHOTOMETRY_DTYPES, chunksize=chunk_rows)
    for raw in reader:
        stats.rows += len(raw)
        stats.chunks += 1
        frame = raw if object_filter is None else raw[raw["object_id"].isin(object_filter)]
        if carry is not None and len(frame):
            frame = pd.concat([carry, frame], ignore_index=True)
            carry = None
        elif carry is not None:
            frame = carry
            carry = None
        if not len(frame):
            continue
        ids = frame["object_id"].to_numpy()
        last_id = ids[-1]
        boundary = np.flatnonzero(ids == last_id)[0]
        # The file may genuinely end here; the caller flushes the final chunk.
        carry = frame.iloc[boundary:].reset_index(drop=True)
        emit = frame.iloc[:boundary].reset_index(drop=True)
        if len(emit):
            stats.peak_objects_per_chunk = max(stats.peak_objects_per_chunk, emit["object_id"].nunique())
            stats.objects += emit["object_id"].nunique()
            yield emit
    if carry is not None and len(carry):
        stats.objects += carry["object_id"].nunique()
        stats.peak_objects_per_chunk = max(stats.peak_objects_per_chunk, carry["object_id"].nunique())
        yield carry.reset_index(drop=True)


def read_lightcurves(path: str | Path, object_ids: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Convenience whole-file read for small files (the 21 MB PLAsTiCC train set)."""
    frame = pd.read_csv(path, dtype=PHOTOMETRY_DTYPES)
    if object_ids is not None:
        frame = frame[frame["object_id"].isin(np.asarray(object_ids, dtype=frame["object_id"].dtype))]
    return frame.reset_index(drop=True)


def harvest_stream(
    lightcurve_path: str | Path,
    object_ids: set,
    out_path: str | Path,
    chunk_rows: int = 2_000_000,
    progress: Optional[Callable[[StreamStats], None]] = None,
) -> Dict[str, int]:
    """Stream a large light-curve file and persist only the selected objects.

    This is the low-memory harvest used to build the scored stream. Each input
    block is filtered and appended to the output parquet as its own row group,
    then released, so peak memory is one input block regardless of how many
    objects match.

    An earlier version accumulated every match with ``pd.concat`` and wrote once
    at the end. That is not low-memory: the ``del chunk`` freed only the raw
    input, while the buffer grew to the full result - ~114M rows for PLAsTiCC
    chunk 02 (345,997 objects), which OOM-killed the harvest on a 2 GB host.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    tmp = out_path.with_suffix(".parquet.part")
    writer: Optional[pq.ParquetWriter] = None
    seen_ids: set = set()
    rows_written = 0
    stats = StreamStats()
    try:
        for chunk in iter_lightcurve_chunks(lightcurve_path, chunk_rows=chunk_rows):
            stats.rows += len(chunk)
            stats.chunks += 1
            keep = chunk[chunk["object_id"].isin(object_ids)]
            if len(keep):
                seen_ids.update(int(v) for v in np.unique(keep["object_id"].to_numpy()))
                table = pa.Table.from_pandas(keep, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema, compression="snappy")
                writer.write_table(table)
                rows_written += len(keep)
                del table
            # Release the block explicitly: peak memory must stay at one block.
            del chunk, keep
            if progress:
                progress(stats)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"harvest selected no objects from {lightcurve_path}")
    tmp.replace(out_path)
    summary = {
        "rows_written": int(rows_written),
        "objects_written": len(seen_ids),
        "objects_requested": len(object_ids),
        "rows_scanned": stats.rows,
        "chunks_read": stats.chunks,
    }
    return summary


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def rare_preserving_subsample(
    labels: pd.Series,
    max_stream: int,
    rare_cap: int = 1500,
    seed: int = 42,
) -> pd.Index:
    """Keep every object of a scarce class; thin abundant classes proportionally.

    Never drops a scarce class to fit RAM.  Used whenever the scored stream has to
    be shrunk, so held-out populations stay measurable.
    """
    counts = labels.value_counts()
    if len(labels) <= max_stream:
        return labels.index
    rng = np.random.default_rng(seed)
    rare = counts[counts <= rare_cap].index
    keep: List[int] = []
    for code in rare:
        keep.extend(labels.index[labels == code].tolist())
    remaining_budget = max_stream - len(keep)
    abundant = counts[counts > rare_cap]
    if remaining_budget <= 0 or abundant.empty:
        return pd.Index(sorted(keep))
    frac = remaining_budget / int(abundant.sum())
    # Floor protects a class sitting just above the rare threshold from being
    # thinned to nothing, but it must never be allowed to blow the budget: when
    # EVERY class is abundant an unconditional floor of rare_cap+1 per class
    # overshoots max_stream (3,002 objects returned for a 2,000 budget).
    per_class_floor = min(rare_cap + 1, max(remaining_budget // max(len(abundant), 1), 1))
    takes = {code: int(min(max(per_class_floor, round(n * frac)), n)) for code, n in abundant.items()}
    total = sum(takes.values())
    if total > remaining_budget:
        scale = remaining_budget / total
        takes = {code: int(max(1, round(n * scale))) for code, n in takes.items()}
    for code, take in takes.items():
        idx = labels.index[labels == code].to_numpy()
        take = int(min(take, len(idx)))
        keep.extend(rng.choice(idx, size=take, replace=False).tolist())
    return pd.Index(sorted(set(int(v) for v in keep)))


def disjoint_object_split(object_ids: Sequence[int], reference_fraction: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Split by ``object_id`` so no object can appear in both halves."""
    ids = np.asarray(sorted(set(int(v) for v in object_ids)))
    rng = np.random.default_rng(seed)
    mask = rng.random(len(ids)) < reference_fraction
    return ids[mask], ids[~mask]


# --------------------------------------------------------------------------- #
# caches
# --------------------------------------------------------------------------- #
class LightCurveCache:
    """Prebuilt parquet light-curve store for low-latency dashboard serving."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._index: Optional[Dict[int, Tuple[int, int]]] = None
        self._table: Optional[pd.DataFrame] = None
        self._writer = None
        self._groups = 0

    def build(self, frame: pd.DataFrame) -> "LightCurveCache":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = frame.sort_values(["object_id", "mjd"], kind="mergesort")
        ordered.to_parquet(self.path, index=False)
        self._table = None
        self._index = None
        return self

    def _load(self) -> pd.DataFrame:
        if self._table is None:
            if not self.path.exists():
                raise FileNotFoundError(f"light-curve cache missing: {self.path} (run scripts/04_build_lc_cache.py)")
            self._table = pd.read_parquet(self.path)
            # Build a row-span index once; serving a light curve is then a slice.
            starts: Dict[int, Tuple[int, int]] = {}
            ids = self._table["object_id"].to_numpy()
            i = 0
            n = len(ids)
            while i < n:
                j = i
                while j + 1 < n and ids[j + 1] == ids[i]:
                    j += 1
                starts[int(ids[i])] = (i, j - i + 1)
                i = j + 1
            self._index = starts
        return self._table

    def append_sorted(self, source: str | Path) -> "LightCurveCache":
        """Append one harvested chunk to the cache as its own row group.

        Sorting each chunk on its own and appending keeps peak memory at a single
        chunk. PLAsTiCC chunks have disjoint object_id ranges, so a file built
        this way is still globally grouped by object_id and the row-span index
        below stays valid. This replaced a build that concatenated every chunk's
        raw photometry in memory, which needed ~1.7 GB for three chunks.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        source = Path(source)
        chunk = pd.read_parquet(source).sort_values(["object_id", "mjd"], kind="mergesort")
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self._writer.write_table(table)
        self._groups += 1
        self._table = None
        self._index = None
        del chunk, table
        return self

    def finalise(self) -> "LightCurveCache":
        """Close the row-group writer; the cache is then readable."""
        if self._writer is not None:
            self._writer.close()
            log.info("light-curve cache %s written (%d row groups)", self.path.name, self._groups)
        self._writer = None
        self._table = None
        self._index = None
        return self

    def frame(self, object_ids: Optional[Sequence[int]] = None) -> pd.DataFrame:
        """The whole cache, or just the requested objects.

        Passing ``object_ids`` matters: the full three-chunk PLAsTiCC stream is
        ~33M rows, and loading all of it to inject artifacts into a few hundred
        objects is what exhausted a 2 GB host. The row-span index makes a subset
        a cheap gather.
        """
        table = self._load()
        if object_ids is None:
            return table
        spans = [(s, n) for oid in object_ids for s, n in [(self._index or {}).get(int(oid))] if s is not None]
        if not spans:
            return table.iloc[0:0]
        starts = np.concatenate([np.arange(s, s + n) for s, n in spans])
        return table.iloc[np.sort(starts)].reset_index(drop=True)

    def get(self, object_id: int) -> pd.DataFrame:
        table = self._load()
        span = (self._index or {}).get(int(object_id))
        if span is None:
            return pd.DataFrame(columns=list(PHOTOMETRY_COLUMNS))
        start, length = span
        return table.iloc[start : start + length].reset_index(drop=True)

    def known_ids(self) -> List[int]:
        self._load()
        return sorted((self._index or {}).keys())


def write_features(path: str | Path, frame: pd.DataFrame) -> Path:
    """Persist a feature matrix, downcasting float64 columns to float32.

    402 features at float64 cost 3.2 KB per object, so the full-scale stream
    (~120k objects) does not fit alongside the models on a 2 GB host. float32
    halves that; every feature here is a summary statistic of noisy photometry,
    so the lost precision is far below the measurement error.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    floats = frame.select_dtypes(include=["float64"]).columns
    if len(floats):
        frame = frame.copy()
        frame[floats] = frame[floats].astype("float32")
    frame.to_parquet(path, index=False)
    return path


def read_features(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def disk_usage_mb(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:  # pragma: no cover
                pass
    return total / 1e6


def gzip_row_count(path: str | Path) -> int:
    """Row count of a gzipped CSV without decompressing into memory."""
    count = 0
    with gzip.open(path, "rb") as handle:
        buf = io.DEFAULT_BUFFER_SIZE
        while True:
            data = handle.read(buf)
            if not data:
                break
            count += data.count(b"\n")
    return max(count - 1, 0)
