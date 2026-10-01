"""Adversarial domain validation - the SETI-style 'is it novelty or domain?' check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cne.adversarial import AdversarialResult, adversarial_validation


def _frame(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "object_id": np.arange(n),
        "f1": rng.normal(shift, 1.0, n),
        "f2": rng.normal(shift, 1.0, n),
        "f3": rng.normal(shift, 1.0, n),
    })


def test_indistinguishable_populations_give_auc_near_half():
    ref, tgt = _frame(600, 1), _frame(600, 2)
    r = adversarial_validation(ref, tgt, ["f1", "f2", "f3"], seed=42)
    assert 0.4 < r.roc_auc < 0.6
    assert r.n_reference == 600 and r.n_target == 600
    assert "indistinguishable" in r.verdict


def test_shifted_populations_are_detected():
    ref, tgt = _frame(600, 1, shift=0.0), _frame(600, 2, shift=2.5)
    r = adversarial_validation(ref, tgt, ["f1", "f2", "f3"], seed=42)
    assert r.roc_auc > 0.9, f"expected clean separation, got {r.roc_auc}"
    assert "trivially separable" in r.verdict
    assert r.top_features, "should name the driving features"


def test_novelty_correlation_requires_row_alignment():
    ref, tgt = _frame(300, 1), _frame(300, 2)
    with pytest.raises(ValueError, match="aligned row-for-row"):
        adversarial_validation(ref, tgt, ["f1", "f2", "f3"],
                               novelty=pd.Series(np.zeros(5)), seed=42)


def test_correlation_is_reported_when_aligned():
    ref, tgt = _frame(400, 1), _frame(400, 2)
    nov = pd.Series(np.linspace(0, 1, len(tgt)))
    r = adversarial_validation(ref, tgt, ["f1", "f2", "f3"], novelty=nov, seed=42)
    assert r.novelty_correlation is not None
    assert -1.0 <= r.novelty_correlation <= 1.0
    assert isinstance(r.as_dict()["verdict"], str)


def test_no_shared_features_raises():
    ref, tgt = _frame(50, 1), _frame(50, 2)
    with pytest.raises(ValueError, match="no shared feature columns"):
        adversarial_validation(ref, tgt, ["does_not_exist"], seed=42)
