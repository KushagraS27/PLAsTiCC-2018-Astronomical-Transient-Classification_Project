"""Authoritative PLAsTiCC taxonomy (Zenodo 2539456, unblinded release).

The table below is transcribed from ``note2_modelNames.pdf`` shipped with the
PLAsTiCC record and from the train/test metadata ``target`` columns.  It is the
single source of truth for class names, astrophysical families and which
populations CNE is allowed to see during fitting.

Held-out populations are NEVER used for fitting the known-physics prior, for
weight selection, for threshold tuning or for feature selection.  They exist only
to be scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List

GALACTIC = "galactic"
EXTRAGALACTIC = "extragalactic"


@dataclass(frozen=True)
class PopulationSpec:
    """One astrophysical population in the PLAsTiCC simulation."""

    code: int
    name: str
    description: str
    family: str
    n_train: int
    n_test: int
    z_max: float
    #: Populations withheld from every fitting step and used as "unknowns".
    held_out: bool = False

    @property
    def is_galactic(self) -> bool:
        return self.family == GALACTIC

    @property
    def key(self) -> str:
        return f"{self.code}:{self.name}"


POPULATIONS: Dict[int, PopulationSpec] = {
    90: PopulationSpec(90, "SNIa", "White-dwarf detonation, Type Ia supernova", EXTRAGALACTIC, 2313, 1_659_831, 1.6),
    67: PopulationSpec(67, "SNIa-91bg", "Peculiar Type Ia (91bg-like)", EXTRAGALACTIC, 208, 40_193, 0.9),
    52: PopulationSpec(52, "SNIax", "Peculiar Type Iax supernova", EXTRAGALACTIC, 183, 63_664, 1.3),
    42: PopulationSpec(42, "SNII", "Core-collapse Type II supernova", EXTRAGALACTIC, 1193, 1_000_150, 2.0),
    62: PopulationSpec(62, "SNIbc", "Core-collapse Type Ibc supernova", EXTRAGALACTIC, 484, 175_094, 1.3),
    95: PopulationSpec(95, "SLSN-I", "Super-luminous supernova (magnetar)", EXTRAGALACTIC, 175, 35_782, 3.4),
    15: PopulationSpec(15, "TDE", "Tidal disruption event", EXTRAGALACTIC, 495, 13_555, 2.6),
    88: PopulationSpec(88, "AGN", "Active galactic nucleus variability", EXTRAGALACTIC, 370, 101_424, 3.4),
    92: PopulationSpec(92, "RRL", "RR Lyrae pulsating variable", GALACTIC, 239, 197_155, 0.0),
    65: PopulationSpec(65, "M-dwarf", "M-dwarf stellar flare", GALACTIC, 981, 93_494, 0.0),
    16: PopulationSpec(16, "EB", "Eclipsing binary star", GALACTIC, 924, 96_572, 0.0),
    53: PopulationSpec(53, "Mira", "Mira-type pulsating variable", GALACTIC, 30, 1_453, 0.0),
    6: PopulationSpec(6, "muLens-Single", "Microlensing, single lens", GALACTIC, 151, 1_303, 0.0),
    # --- populations that only ever appear as unknowns -------------------------
    64: PopulationSpec(64, "KN", "Kilonova (neutron-star merger)", EXTRAGALACTIC, 100, 131, 0.3, held_out=True),
    991: PopulationSpec(991, "muLens-Binary", "Microlensing, binary lens", GALACTIC, 0, 533, 0.0, held_out=True),
    992: PopulationSpec(992, "ILOT", "Intermediate-luminosity optical transient", EXTRAGALACTIC, 0, 1_702, 0.4, held_out=True),
    993: PopulationSpec(993, "CaRT", "Calcium-rich transient", EXTRAGALACTIC, 0, 9_680, 0.9, held_out=True),
    994: PopulationSpec(994, "PISN", "Pair-instability supernova", EXTRAGALACTIC, 0, 1_172, 1.9, held_out=True),
    995: PopulationSpec(995, "muLens-String", "Microlensing, cosmic string", GALACTIC, 0, 0, 0.0, held_out=True),
}

#: The twelve known classes the known-physics prior is allowed to model.
KNOWN_CLASS_CODES: FrozenSet[int] = frozenset(
    code for code, spec in POPULATIONS.items() if not spec.held_out and code != 6
)

#: ``muLens-Single`` (6) is treated as an unknown by CNE: only 151 training
#: examples exist and microlensing light curves are the hardest population to
#: separate from a fast, faint, red transient.  Keeping it out of the prior makes
#: the held-out protocol strictly harder and is disclosed in every report.
WITHHELD_FROM_PRIOR: FrozenSet[int] = frozenset({6}) | frozenset(
    code for code, spec in POPULATIONS.items() if spec.held_out
)


def name_of(code: int) -> str:
    spec = POPULATIONS.get(int(code))
    return spec.name if spec is not None else f"unknown-{code}"


def family_of(code: int) -> str:
    spec = POPULATIONS.get(int(code))
    return spec.family if spec is not None else "unknown"


def is_held_out(code: int) -> bool:
    return int(code) in WITHHELD_FROM_PRIOR


def known_class_codes() -> List[int]:
    return sorted(KNOWN_CLASS_CODES)


def describe() -> List[Dict[str, object]]:
    """Serialisable taxonomy, used by the API and the dashboard."""
    return [
        {
            "code": spec.code,
            "name": spec.name,
            "description": spec.description,
            "family": spec.family,
            "n_train": spec.n_train,
            "n_test": spec.n_test,
            "z_max": spec.z_max,
            "in_known_prior": spec.code in KNOWN_CLASS_CODES,
            "held_out": spec.code in WITHHELD_FROM_PRIOR,
        }
        for spec in sorted(POPULATIONS.values(), key=lambda s: s.code)
    ]


PASSBAND_NAMES = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}
PASSBAND_CODES = {v: k for k, v in PASSBAND_NAMES.items()}
#: LSST effective wavelengths in nm, used for temperature/extinction proxies.
PASSBAND_WAVELENGTH_NM = {0: 367.0, 1: 482.5, 2: 622.2, 3: 754.5, 4: 869.1, 5: 971.0}
#: Zero points on the AB system, identical across LSST passbands by construction.
PASSBAND_ZEROPOINT_AB = 8.90
