"""Synthetic rare-transient simulator - the SETI-style "magic #1" for CNE.

The headline negative result of this project is that CaRT (class 993) - 68% of the
held-out population - is undetectable, because its median observation (4
detections, SNR 7.57, 3 bands) is statistically identical to a core-collapse
supernova. That is a *coverage* finding: the reference has no CaRT examples, so the
prior cannot recognise them.

The winning SETI team faced the same shape of problem - a signal type present only
in test - and solved it by building a randomized signal generator and injecting the
missing signal into training. This module is the CNE analogue: a small, physically
plausible light-curve simulator for the withheld rare classes, plus a study that
asks the question that actually matters:

    **If the reference were given CaRT-like examples, could the engine learn to
    see them?**

If yes, the CaRT failure is proven to be a data-coverage requirement, not an
architecture defect - and the simulator is the tool that makes the fix measurable.

Everything here runs offline on synthetic data; no PLAsTiCC chunks are needed.
The simulator is deliberately simple (a single smooth transient template per class
plus realistic sampling and noise), because its job is to demonstrate the
*mechanism*, not to produce publication-quality light curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .logging import get_logger

log = get_logger("simulator")

LC_COLUMNS = ["object_id", "mjd", "passband", "flux", "flux_err", "detected_bool"]

# Held-out rare classes the simulator can stand in for, keyed to PLAsTiCC codes.
SIM_CLASSES: Dict[str, int] = {"CaRT": 993, "KN": 64}


@dataclass(frozen=True)
class Template:
    """A smooth transient template in absolute magnitude."""

    peak_abs_mag: float
    rise_days: float
    decay_days: float
    # Per-band relative brightness offsets (u,g,r,i,z,y). Redder = brighter in i/z.
    band_offsets: Dict[int, float]


TEMPLATES: Dict[str, Template] = {
    # Fast, faint, blue-grey - the calcium-rich signature. Short and dim, so it is
    # sampled sparsely: this is what makes it photometrically resemble a faint SN.
    "CaRT": Template(-16.5, 6.0, 18.0, {0: 0.4, 1: 0.2, 2: 0.0, 3: 0.1, 4: 0.3, 5: 0.6}),
    # Very fast and very red - kilonova-like.
    "KN": Template(-15.8, 3.0, 7.0, {0: 1.5, 1: 0.8, 2: 0.0, 3: -0.3, 4: -0.5, 5: -0.6}),
    # A known-like control (SN II-ish plateau) so the study has a sanity anchor.
    "SNII": Template(-17.0, 12.0, 90.0, {0: 0.3, 1: 0.1, 2: 0.0, 3: 0.0, 4: 0.1, 5: 0.2}),
}


def _abs_mag_to_flux(abs_mag: float, distmod: float) -> float:
    """Apparent mag -> nMgy-ish flux using the SNANA 27.5 zero point convention."""
    app = abs_mag + distmod
    return 10.0 ** (-0.4 * (app - 27.5))


def simulate_lightcurve(
    kind: str,
    object_id: int,
    rng: np.random.Generator,
    distmod: float = 36.0,
    n_epochs: int = 40,
    t_span: float = 90.0,
    floor_snr: float = 5.0,
) -> pd.DataFrame:
    """Draw one synthetic light curve for ``kind`` in {'CaRT','KN','SNII'}."""
    if kind not in TEMPLATES:
        raise KeyError(f"unknown simulator class {kind!r}; have {sorted(TEMPLATES)}")
    tpl = TEMPLATES[kind]

    mjd = np.sort(rng.uniform(0.0, t_span, n_epochs))
    peak = rng.uniform(0.2, 0.5) * t_span
    rows = []
    for pb in range(6):
        for t in mjd:
            dt = t - peak
            if dt < 0:
                mag = tpl.peak_abs_mag + 2.5 * (dt / tpl.rise_days) ** 2  # fast rise
            else:
                mag = tpl.peak_abs_mag + 1.0857 * (dt / tpl.decay_days)  # exp decay
            flux = _abs_mag_to_flux(mag + tpl.band_offsets[pb], distmod)
            # Noise floor so SNR is finite and sometimes sub-threshold.
            err = flux / rng.uniform(floor_snr, 30.0)
            detected = rng.random() < 0.8
            if not detected and rng.random() < 0.5:
                flux = rng.normal(0.0, err)
            rows.append((int(object_id), float(t), int(pb), float(flux), float(err),
                         int(detected)))
    return pd.DataFrame(rows, columns=LC_COLUMNS)


def meta_for(object_ids: List[int], rng: np.random.Generator) -> pd.DataFrame:
    """Plausible host metadata rows so the featuriser can run on synthetic curves."""
    return pd.DataFrame({
        "object_id": [int(i) for i in object_ids],
        "ra": rng.uniform(0, 360, len(object_ids)),
        "decl": rng.uniform(-60, 60, len(object_ids)),
        "mwebv": rng.uniform(0.02, 0.1, len(object_ids)),
        "hostgal_photoz": rng.uniform(0.05, 0.4, len(object_ids)),
        "hostgal_photoz_err": np.full(len(object_ids), 0.02),
        "hostgal_specz": np.full(len(object_ids), np.nan),
        "distmod": rng.uniform(34.0, 38.0, len(object_ids)),
        "ddf_bool": np.zeros(len(object_ids), dtype=int),
    })


def novelty_of_synthetic(
    pipeline,
    engine,
    weights,
    kind: str,
    n: int = 24,
    seed: int = 42,
) -> float:
    """Mean novelty score the given engine assigns to n synthetic ``kind`` objects."""
    rng = np.random.default_rng(seed)
    lcs = [simulate_lightcurve(kind, 900_000 + i, rng) for i in range(n)]
    lc = pd.concat(lcs, ignore_index=True)
    meta = meta_for([900_000 + i for i in range(n)], rng)
    feats = pipeline.featuriser_.transform(lc, meta)
    from .experiments import _align_columns
    feats = _align_columns(feats, pipeline.train_features_)
    scored = pipeline.score_stream(feats, engine, weights=weights)
    return float(scored["novelty_score"].mean())
