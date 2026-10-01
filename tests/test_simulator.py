"""Synthetic rare-transient simulator - schema and template sanity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cne.data import METADATA_DTYPES
from cne.simulator import (LC_COLUMNS, SIM_CLASSES, TEMPLATES, meta_for,
                           simulate_lightcurve)


def test_lightcurve_has_required_schema():
    lc = simulate_lightcurve("CaRT", 1, np.random.default_rng(0))
    assert list(lc.columns) == LC_COLUMNS
    assert lc["object_id"].nunique() == 1
    assert lc["passband"].between(0, 5).all()
    assert (lc["flux_err"] > 0).all()
    assert set(lc["detected_bool"].unique()) <= {0, 1}
    assert len(lc) > 0


@pytest.mark.parametrize("kind", ["CaRT", "KN", "SNII"])
def test_all_templates_produce_detections(kind):
    lc = simulate_lightcurve(kind, 2, np.random.default_rng(1))
    assert lc["detected_bool"].sum() > 0, f"{kind} must have some detections"


def test_cart_and_snii_templates_have_different_timescales():
    # CaRT is fast, SNII has a plateau - the defining physical difference.
    assert TEMPLATES["CaRT"].decay_days < TEMPLATES["SNII"].decay_days
    assert TEMPLATES["CaRT"].peak_abs_mag > TEMPLATES["SNII"].peak_abs_mag  # fainter


def test_kn_is_redder_than_cart():
    # KN brightens toward red bands (offset goes negative), CaRT does not.
    assert TEMPLATES["KN"].band_offsets[4] < TEMPLATES["CaRT"].band_offsets[4]


def test_meta_has_featuriser_columns():
    m = meta_for([1, 2, 3], np.random.default_rng(0))
    for col in METADATA_DTYPES:
        assert col in m.columns, f"missing {col}"
    assert len(m) == 3


def test_unknown_class_raises():
    with pytest.raises(KeyError):
        simulate_lightcurve("WOLF", 9, np.random.default_rng(0))


def test_sim_class_codes_match_held_out():
    assert SIM_CLASSES["CaRT"] == 993
    assert SIM_CLASSES["KN"] == 64
