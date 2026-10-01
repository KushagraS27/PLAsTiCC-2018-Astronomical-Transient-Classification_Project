"""Shared fixtures: a small, fast, deterministic synthetic population.

Real PLAsTiCC data is used by the integration tests only; unit tests run on this
fixture so the suite completes in seconds and never depends on a download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cne.config import load_config  # noqa: E402
from cne.taxonomy import KNOWN_CLASS_CODES, WITHHELD_FROM_PRIOR  # noqa: E402

BANDS = (0, 1, 2, 3, 4, 5)


def make_lightcurves(n_objects: int = 120, seed: int = 7, points_per_band: int = 24) -> pd.DataFrame:
    """Synthetic LSST-cadence-like photometry with a genuine transient shape."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_objects):
        oid = 1000 + i
        t0 = 60000.0 + rng.uniform(0, 200)
        amplitude = 10 ** rng.uniform(1.5, 3.5)
        timescale = rng.uniform(8.0, 60.0)
        for band in BANDS:
            band_amp = amplitude * 10 ** rng.normal(0, 0.25)
            for k in range(points_per_band):
                mjd = t0 + k * rng.uniform(1.5, 6.0)
                phase = (mjd - t0) / timescale
                shape = np.exp(-0.5 * phase ** 2) if phase < 0 else np.exp(-phase)
                flux = band_amp * shape + rng.normal(0, 3.0)
                err = max(rng.uniform(1.5, 12.0), 0.5)
                rows.append((oid, mjd, band, flux, err, int(flux / err > 5)))
    return pd.DataFrame(rows, columns=["object_id", "mjd", "passband", "flux", "flux_err", "detected_bool"]).astype(
        {"object_id": "int32", "mjd": "float32", "passband": "int8", "flux": "float32",
         "flux_err": "float32", "detected_bool": "int8"}
    )


def make_metadata(object_ids, seed: int = 11, galactic_fraction: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(object_ids)
    galactic = rng.random(n) < galactic_fraction
    z = np.where(galactic, 0.0, np.clip(rng.gamma(2.0, 0.25, n), 0.01, 2.5))
    return pd.DataFrame({
        "object_id": np.asarray(object_ids, dtype="int32"),
        "ra": rng.uniform(0, 360, n),
        "decl": rng.uniform(-60, 60, n),
        "ddf_bool": rng.integers(0, 2, n),
        "hostgal_specz": np.where(rng.random(n) < 0.2, z, np.nan),
        "hostgal_photoz": z,
        "hostgal_photoz_err": np.where(galactic, 0.0, np.clip(z * rng.uniform(0.02, 0.2, n), 1e-4, None)),
        "distmod": np.where(galactic, 0.0, 5 * np.log10(np.clip(z * 4300.0, 1e-3, None)) + 25.0),
        "mwebv": rng.uniform(0.0, 0.3, n),
    })


def make_labels(object_ids, novel_fraction: float = 0.25, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    known = sorted(KNOWN_CLASS_CODES)
    novel = sorted(WITHHELD_FROM_PRIOR)
    n = len(object_ids)
    is_novel = rng.random(n) < novel_fraction
    codes = np.where(is_novel,
                     np.asarray(novel)[rng.integers(0, len(novel), n)],
                     np.asarray(known)[rng.integers(0, len(known), n)])
    return codes.astype("int64")


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def synthetic():
    """A complete featurised synthetic population: (features, metadata, labels, lightcurves)."""
    from cne.features import LightCurveFeaturiser

    lc = make_lightcurves()
    ids = np.unique(lc["object_id"].to_numpy())
    meta = make_metadata(ids)
    featuriser = LightCurveFeaturiser()
    probe = featuriser.transform(lc[lc["object_id"].isin(ids[:40])], meta)
    featuriser.fit_fill_values(probe)
    features = featuriser.transform(lc, meta)
    labels = make_labels(ids)
    return {"features": features, "metadata": meta, "labels": labels, "lightcurves": lc,
            "featuriser": featuriser, "ids": ids}
