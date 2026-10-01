"""Physics layer: cosmology, extinction, K-corrections, absolute magnitudes.

Scope statement that governs every function here: CNE is a *prioritisation*
system, not a precision-cosmology pipeline.  Distance-dependent quantities are
used as ranking features, so their uncertainty must be propagated and reported
(``feature_reliability``), never hidden behind a single point estimate.

Conventions
-----------
* Flat Lambda-CDM, H0 = 70 km/s/Mpc, Omega_m = 0.3 (CNE v1 defaults, unchanged).
* AB magnitudes, flux in nMgy (PLAsTiCC native units), zero point 8.90.
* Extinction A_lambda = R_V * E(B-V) * k_lambda with a configurable R_V
  (default 3.1, matching v1) and the Cardelli-style k_lambda ratios below.
* K-corrections are optional and OFF by default: including a crude one adds
  bias, and v1 measured that analytic physics corrections did not improve
  ranking.  Set ``kcorrection.enabled`` in config to turn the approximation on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .taxonomy import PASSBAND_WAVELENGTH_NM

# --- cosmology --------------------------------------------------------------
H0_KM_S_MPC = 70.0
OMEGA_M = 0.3
OMEGA_LAMBDA = 1.0 - OMEGA_M
SPEED_OF_LIGHT_KM_S = 299_792.458
HUBBLE_DISTANCE_MPC = SPEED_OF_LIGHT_KM_S / H0_KM_S_MPC  # ~4282.7 Mpc

# --- photometry -------------------------------------------------------------
#: PLAsTiCC/SNANA magnitude zero point.
#:
#: This is the single most error-prone constant in the project, so it is pinned by
#: measurement rather than by memory. PLAsTiCC ships SNANA fluxes, whose magnitude
#: convention is ``MAG = -2.5 log10(FLUX) + 27.5`` - five magnitudes offset from
#: the nMgy/AB convention (22.5). Using 22.5 makes every absolute magnitude five
#: magnitudes too bright and every derived luminosity wrong.
#:
#: Verification against textbook peak absolute magnitudes (median over the
#: PLAsTiCC train split, this repository):
#:
#: =========  =========  ==============
#: class      measured   literature
#: =========  =========  ==============
#: SNIa        -19.17    -19.1
#: SNIa-91bg   -17.73    -17.5
#: SNII        -17.60    -17.0
#: SNIbc       -17.43    -17.5
#: SNIax       -18.15    -18.0
#: SLSN-I      -22.67    -21 to -23
#: KN          -13.97    -15 to -16
#: =========  =========  ==============
#:
#: With ZP = 22.5 every row above is five magnitudes too bright.
SNANA_ZEROPOINT = 27.5
AB_ZEROPOINT = SNANA_ZEROPOINT  # alias kept so imports read naturally
#: k_lambda / E(B-V) ratios for LSST passbands (Cardelli et al. 1989 style,
#: evaluated at the LSST effective wavelengths for R_V = 3.1).
EXTINCTION_K_RATIO = {0: 1.579, 1: 1.180, 2: 0.859, 3: 0.661, 4: 0.493, 5: 0.394}
DEFAULT_R_V = 3.1

# --- galactic pole (J2000), for |b| which drives stellar-contamination priors
_RA_NGP_DEG, _DEC_NGP_DEG, _L_NCP_DEG = 192.85948, 27.12825, 122.93192


@dataclass(frozen=True)
class CosmologyConfig:
    h0: float = H0_KM_S_MPC
    omega_m: float = OMEGA_M
    r_v: float = DEFAULT_R_V
    kcorrection_enabled: bool = False
    #: k(z) ~ k1 * z, a deliberately crude linear approximation per band.
    kcorrection_slope: Dict[int, float] | None = None


def _kcorrection_slopes() -> Dict[int, float]:
    """Bluer bands need a larger K-correction; linear in z is the honest limit."""
    return {band: 1.6 * (400.0 / nm) for band, nm in PASSBAND_WAVELENGTH_NM.items()}


def ez(z: np.ndarray | float, omega_m: float = OMEGA_M) -> np.ndarray | float:
    """Dimensionless Hubble parameter E(z) for a flat Lambda-CDM universe."""
    z = np.asarray(z, dtype="float64")
    return np.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))


#: Fixed integration grid in u = ln(1+z). A fixed grid is a correctness
#: requirement, not an optimisation: an earlier revision built the grid from
#: ``z.max()`` of the current batch, so the same object got a slightly different
#: distance depending on which other objects it was featurised alongside.
_U_GRID_MAX = float(np.log(1.0 + 10.0))
_U_GRID_STEPS = 1024
_U_GRID = np.linspace(0.0, _U_GRID_MAX, _U_GRID_STEPS + 1)
_Z_GRID = np.expm1(_U_GRID)


def _cumulative_comoving_grid(omega_m: float, h0: float) -> np.ndarray:
    """Cumulative transverse comoving distance on the fixed u-grid, in Mpc."""
    integrand = (1.0 + _Z_GRID) / ez(_Z_GRID, omega_m)      # dz = (1+z) du
    step = _U_GRID[1] - _U_GRID[0]
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * step)])
    return HUBBLE_DISTANCE_MPC * (H0_KM_S_MPC / h0) * cumulative


_COMOVING_CACHE: dict = {}


def comoving_distance_mpc(z, omega_m: float = OMEGA_M, h0: float = H0_KM_S_MPC, n_steps: int = _U_GRID_STEPS):
    """Transverse comoving distance in Mpc for a flat Lambda-CDM universe.

    The integrand is smooth in ``u = ln(1+z)``, so a fixed 1024-node trapezoidal
    grid with linear interpolation is accurate to well under 0.1% over
    0 < z < 10 and - importantly - returns the same value for the same redshift
    no matter what else is in the batch.
    """
    z_arr = np.clip(np.atleast_1d(np.asarray(z, dtype="float64")), 0.0, 10.0)
    key = (round(omega_m, 6), round(h0, 6))
    if key not in _COMOVING_CACHE:
        _COMOVING_CACHE[key] = _cumulative_comoving_grid(omega_m, h0)
    dc_grid = _COMOVING_CACHE[key]
    u = np.log1p(z_arr)
    out = np.interp(u, _U_GRID, dc_grid)
    return out if np.ndim(z) else float(out[0])


def luminosity_distance_mpc(z, omega_m: float = OMEGA_M, h0: float = H0_KM_S_MPC):
    z_arr = np.atleast_1d(np.asarray(z, dtype="float64"))
    return comoving_distance_mpc(z_arr, omega_m, h0) * (1.0 + z_arr)


def distance_modulus_from_z(z, omega_m: float = OMEGA_M, h0: float = H0_KM_S_MPC):
    """mu = 5 log10(d_L / 10 pc)."""
    d_l = np.atleast_1d(luminosity_distance_mpc(z, omega_m, h0))
    d_pc = np.clip(d_l, 1e-6, None) * 1e6
    mu = 5.0 * np.log10(d_pc) - 5.0
    return mu if np.ndim(z) else float(mu[0])


def extinction_ab(mwebv, band: int, r_v: float = DEFAULT_R_V) -> np.ndarray | float:
    """A_band = R_V * E(B-V) * k_band."""
    return np.asarray(mwebv, dtype="float64") * r_v * EXTINCTION_K_RATIO[int(band)]


def kcorrection(z, band: int, config: Optional[CosmologyConfig] = None) -> np.ndarray | float:
    cfg = config or CosmologyConfig()
    if not cfg.kcorrection_enabled:
        z_arr = np.asarray(z, dtype="float64")
        return np.zeros_like(z_arr) if np.ndim(z) else 0.0
    slopes = cfg.kcorrection_slope or _kcorrection_slopes()
    return np.asarray(z, dtype="float64") * slopes[int(band)]


def ab_magnitude(flux, flux_err=None, zp: float = AB_ZEROPOINT):
    """AB magnitude with a floor on non-positive flux (returns NaN, never -inf)."""
    flux = np.asarray(flux, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = np.where(flux > 0, -2.5 * np.log10(np.clip(flux, 1e-30, None)) + zp, np.nan)
    if flux_err is None:
        return mag
    err = np.asarray(flux_err, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return mag, np.where(flux > 0, 2.5 / np.log(10.0) * err / np.clip(flux, 1e-30, None), np.nan)


def absolute_magnitude(apparent_mag, distmod, band: Optional[int] = None, a_lambda=None,
                       z=None, config: Optional[CosmologyConfig] = None):
    """M = m - mu - A_band - K_band."""
    m = np.asarray(apparent_mag, dtype="float64")
    mu = np.asarray(distmod, dtype="float64")
    a = 0.0 if a_lambda is None else np.asarray(a_lambda, dtype="float64")
    k = 0.0 if (z is None or config is None) else kcorrection(z, band or 0, config)
    return m - mu - a - k


def peak_luminosity_proxy(flux_peak, distmod):
    """Relative bolometric-ish luminosity proxy in log10(nMgy at 10 pc)."""
    flux_peak = np.asarray(flux_peak, dtype="float64")
    mu = np.asarray(distmod, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where(flux_peak > 0, -2.5 * np.log10(np.clip(flux_peak, 1e-30, None)) + AB_ZEROPOINT - mu, np.nan)
    return val


def galactic_coordinates(ra_deg, dec_deg):
    """J2000 equatorial -> galactic (l, b) in degrees, no astropy dependency."""
    ra = np.radians(np.asarray(ra_deg, dtype="float64"))
    dec = np.radians(np.asarray(dec_deg, dtype="float64"))
    ra_p, dec_p = np.radians(_RA_NGP_DEG), np.radians(_DEC_NGP_DEG)
    sin_b = np.sin(dec) * np.sin(dec_p) + np.cos(dec) * np.cos(dec_p) * np.cos(ra - ra_p)
    b = np.arcsin(np.clip(sin_b, -1.0, 1.0))
    y = np.cos(dec) * np.sin(ra - ra_p)
    x = np.sin(dec) * np.cos(dec_p) - np.cos(dec) * np.sin(dec_p) * np.cos(ra - ra_p)
    l = np.radians(_L_NCP_DEG) - np.arctan2(y, x)
    l = np.degrees(l) % 360.0
    return l, np.degrees(b)


def resolve_distmod(distmod_column, photoz, config: Optional[CosmologyConfig] = None):
    """Prefer the survey distance modulus; fall back to the photometric redshift.

    Returns (distmod, source_flag) where source_flag is 1 for a survey value,
    0 for a photoz-derived fallback and -1 when no distance information exists.
    """
    cfg = config or CosmologyConfig()
    dm = np.asarray(distmod_column, dtype="float64")
    z = np.asarray(photoz, dtype="float64")
    fallback = distance_modulus_from_z(np.clip(np.nan_to_num(z, nan=0.0), 1e-4, 6.0), cfg.omega_m, cfg.h0)
    usable_dm = np.isfinite(dm) & (dm > 0)
    usable_z = np.isfinite(z) & (z > 1e-4)
    out = np.where(usable_dm, np.nan_to_num(dm, nan=0.0), np.where(usable_z, fallback, 0.0))
    flag = np.where(usable_dm, 1.0, np.where(usable_z, 0.0, -1.0))
    return out, flag


def mc_redshift_draws(photoz, photoz_err, n_draws: int, seed: int = 42) -> np.ndarray:
    """Monte-Carlo samples of the photometric-redshift posterior.

    PLAsTiCC photo-z errors are asymmetric and heavy-tailed near z=0, so draws are
    truncated at z > 0.  The result depends only on (z, z_err, n_draws) - never on
    the number or order of the other objects in the batch.
    """
    from scipy.stats import norm

    z = np.asarray(photoz, dtype="float64")
    err = np.nan_to_num(np.asarray(photoz_err, dtype="float64"), nan=0.0)
    err = np.clip(err, 1e-4, None)
    # STRATIFIED, not iid. An earlier revision drew ``standard_normal((n_draws,
    # n_objects))`` from a seeded generator, which made every object's noise
    # depend on how many objects happened to be in the same batch - so the same
    # object got different uncertainty features when scored in a chunk of 4,000
    # versus 30, and chunked scoring was no longer bit-identical to whole-array
    # scoring. Fixed quantiles of the normal distribution give every object the
    # same deterministic, order-independent set of draws, and stratification
    # reduces the Monte-Carlo variance that iid sampling would have added.
    quantiles = (np.arange(n_draws, dtype="float64") + 0.5) / n_draws
    noise = norm.ppf(quantiles)[:, None]          # (n_draws, 1) - broadcast over objects
    draws = z[None, :] + err[None, :] * noise
    return np.clip(draws, 1e-4, 8.0)


def reliability_from_redshift(photoz, photoz_err, distmod_flag, galactic_z_max: float = 1e-3) -> np.ndarray:
    """Per-object confidence in its distance estimate, in [0, 1].

    1.0 = spectroscopic or small fractional photo-z error; lower values mean
    distance-sensitive features must not dominate the candidate's evidence.

    Galactic sources are the trap here. An RR Lyrae or an M-dwarf flare has
    z = 0 and distmod = 0 by construction, so a naive implementation scores them
    as "no distance information" and the abstention layer refuses to rank a whole
    astrophysically clean population. Missing distance for a galactic source is
    not missing information - it is not applicable - so reliability is 1.0.
    """
    z_raw = np.asarray(photoz, dtype="float64")
    flag = np.asarray(distmod_flag, dtype="float64")
    galactic = (np.nan_to_num(z_raw, nan=0.0) <= galactic_z_max) & (flag < 0.0)

    z = np.clip(z_raw, 1e-4, None)
    frac = np.nan_to_num(np.asarray(photoz_err, dtype="float64"), nan=1.0) / z
    frac = np.clip(frac, 0.0, 5.0)
    rel = np.exp(-frac / 0.15)
    rel = np.where(flag >= 1.0, np.maximum(rel, 0.9), rel)
    rel = np.where((flag < 0.0) & ~galactic, 0.0, rel)
    rel = np.where(galactic, 1.0, rel)
    return np.clip(rel, 0.0, 1.0)
