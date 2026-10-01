"""Feature extraction: per-band morphology, colours, physics, quality, context.

Everything is vectorised with groupby aggregations - no Python loop over objects -
which is what makes the 33k-object PLAsTiCC harvest tractable on two cores.

Namespaces (the ``lc_`` prefix is load-bearing history: v1 used ``g_`` for global
all-band features, which collided with the ``g``-band namespace and cost a full
re-featurisation):

==================  =========================================================
``{u,g,r,i,z,y}_``  per-band morphology, cadence, SNR, magnitudes
``lc_``             global, all-band aggregates
``col_``            cross-band colours and colour evolution
``phys_``           distance-, extinction- and rest-frame-corrected quantities
``q_``              data-quality diagnostics. NEVER fed to the astrophysical
                    novelty channels - quality is a suppress-only gate.
``host_``           host-galaxy and sky context
``unc_``            uncertainty-propagated features (CNE v2)
==================  =========================================================

Irregular sampling is handled natively: every temporal statistic is computed from
the actual ``mjd`` values, and missingness is a feature, not an error.

Memory: call :meth:`LightCurveFeaturiser.transform` in object chunks (the
pipeline uses 8,000). A per-band working frame is ~19 float64 columns, so peak
memory scales with the chunk, never with the file.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .logging import get_logger
from .physics import (
    AB_ZEROPOINT,
    CosmologyConfig,
    distance_modulus_from_z,
    extinction_ab,
    galactic_coordinates,
    mc_redshift_draws,
    peak_luminosity_proxy,
    reliability_from_redshift,
    resolve_distmod,
)
from .taxonomy import PASSBAND_NAMES

log = get_logger("features")

BANDS = (0, 1, 2, 3, 4, 5)
BAND_LABELS = tuple(PASSBAND_NAMES[b] for b in BANDS)

#: Metadata columns required by the physics/host/uncertainty features.
REQUIRED_METADATA = ("ra", "decl", "ddf_bool", "hostgal_specz", "hostgal_photoz",
                     "hostgal_photoz_err", "distmod", "mwebv")


def _safe_div(num, den, fill: float = 0.0):
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(np.abs(den) > 1e-12, num / np.where(np.abs(den) > 1e-12, den, 1.0), fill)
    return np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)


def _mag(flux):
    flux = np.asarray(flux, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(flux > 0, -2.5 * np.log10(np.clip(flux, 1e-30, None)) + AB_ZEROPOINT, np.nan)


def _mag_snr_masked(flux, snr, min_snr: float = 3.0):
    """Magnitude, NaN unless the measurement is a real detection.

    Without this mask a 0.001 nMgy "peak" becomes a 30th-magnitude measurement and
    cross-band colours inherit five-figure outliers. Masking at source is the
    principled fix; a downstream clip is only a float32 safety net.
    """
    mag = _mag(flux)
    ok = np.isfinite(mag) & np.isfinite(np.asarray(snr, dtype="float64")) & (np.asarray(snr, dtype="float64") >= min_snr)
    return np.where(ok, mag, np.nan)


#: Defensive float32 safety net on magnitudes/colours, not a substitute for masking.
MAG_CLIP = (-5.0, 32.0)


def _central_moments(s1, s2, s3, s4, n):
    """Skewness and excess kurtosis from accumulated power sums."""
    n = np.clip(np.asarray(n, dtype="float64"), 1, None)
    m1 = s1 / n
    m2 = np.clip(s2 / n - m1 ** 2, 1e-30, None)
    m3 = s3 / n - 3 * m1 * m2 - m1 ** 3
    m4 = s4 / n - 4 * m1 * m3 - 6 * m1 ** 2 * m2 - m1 ** 4
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = np.where(m2 > 1e-24, m3 / np.power(m2, 1.5), 0.0)
        kurt = np.where(m2 > 1e-24, m4 / (m2 * m2) - 3.0, 0.0)
    return np.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)


def _quadratic_curvature(sums: Dict[str, np.ndarray]) -> np.ndarray:
    """Least-squares y = a + b*x + c*x^2 curvature from accumulated group sums.

    Accumulating x^k and y*x^k per group lets us fit a parabola to every object
    without materialising a per-object design matrix (Cramer's rule on the 3x3
    normal equations).
    """
    n = sums["n"].astype("float64")
    s1, s2, s3, s4 = (sums[k].astype("float64") for k in ("s1", "s2", "s3", "s4"))
    t0, t1, t2 = (sums[k].astype("float64") for k in ("t0", "t1", "t2"))
    # | n  S1 S2 |          Cramer's rule on the 3x3 normal equations for
    # | S1 S2 S3 | [a,b,c]^T = [T0,T1,T2]^T
    # | S2 S3 S4 |
    det = n * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s3 * s2) + s2 * (s1 * s3 - s2 * s2)
    det = np.where(np.abs(det) > 1e-9, det, np.nan)
    det_c = n * (s2 * t2 - t1 * s3) - s1 * (s1 * t2 - t1 * s2) + t0 * (s1 * s3 - s2 * s2)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = det_c / det
    return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)


class LightCurveFeaturiser:
    """Deterministic light-curve -> feature-matrix transformer."""

    def __init__(self, config: Optional[FeatureConfig] = None, cosmology: Optional[CosmologyConfig] = None):
        self.cfg = config or FeatureConfig()
        self.cosmo = cosmology or CosmologyConfig()
        self._fill: Dict[str, float] = {}
        self._names: Optional[List[str]] = None

    # ------------------------------------------------------------------ names
    def feature_names(self) -> List[str]:
        if self._names is None:
            raise RuntimeError("feature_names() is only known after transform(); featurise a batch first")
        return list(self._names)

    @property
    def quality_columns(self) -> List[str]:
        return [n for n in (self._names or []) if n.startswith(self.cfg.quality_prefix)]

    @property
    def astrophysical_columns(self) -> List[str]:
        """Everything except quality diagnostics - the prior/anomaly input space."""
        return [n for n in (self._names or []) if not n.startswith(self.cfg.quality_prefix)]

    # -------------------------------------------------------------- fit / fill
    def fit_fill_values(self, features: pd.DataFrame) -> "LightCurveFeaturiser":
        """Learn the NaN-fill policy on the REFERENCE population only.

        Filling from the scored stream would leak stream statistics into the
        reference; filling with a constant would silently bias colour features.
        """
        cols = [c for c in features.columns if c != "object_id"]
        med = features[cols].median(numeric_only=True)
        self._fill = {str(k): float(v) for k, v in med.items() if np.isfinite(v)}
        return self

    def _apply_fill(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features.copy()
        for col in (c for c in out.columns if c != "object_id"):
            fill = self._fill.get(col, 0.0)
            out[col] = out[col].astype("float32").fillna(np.float32(0.0 if not np.isfinite(fill) else fill))
        return out

    # --------------------------------------------------------------- transform
    def transform(self, photometry: pd.DataFrame, metadata: Optional[pd.DataFrame] = None,
                  apply_fill: bool = True) -> pd.DataFrame:
        """Build the full feature matrix for the objects present in ``photometry``."""
        lc = photometry
        ids = pd.Index(np.unique(lc["object_id"].to_numpy()), name="object_id")
        frames: List[pd.DataFrame] = []

        band_frames: Dict[str, pd.DataFrame] = {}
        for band in BANDS:
            frame = self._band_features(lc, band, BAND_LABELS[band], ids)
            band_frames[BAND_LABELS[band]] = frame
            frames.append(frame)

        frames.append(self._global_features(lc, ids))
        band_peaks = {b: band_frames[b][f"{b}_flux_max"].to_numpy(dtype="float64") for b in BAND_LABELS}
        band_peak_mjd = {b: band_frames[b][f"{b}_t_peak"].to_numpy(dtype="float64") for b in BAND_LABELS}
        frames.append(self._colour_features(ids, band_peaks, band_peak_mjd))

        meta = self._align_metadata(metadata, ids)
        frames.append(self._physics_features(ids, meta, band_peaks, band_frames))
        frames.append(self._host_features(ids, meta))
        frames.append(self._uncertainty_features(ids, meta, band_peaks))
        frames.append(self._quality_features(lc, ids))

        features = pd.concat(frames, axis=1)
        features.index.name = "object_id"
        self._names = list(features.columns)
        features = features.replace([np.inf, -np.inf], np.nan).reset_index()
        return self._apply_fill(features) if apply_fill else features

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _align_metadata(metadata: Optional[pd.DataFrame], ids: pd.Index) -> pd.DataFrame:
        if metadata is None or not len(metadata):
            empty = pd.DataFrame(index=ids)
            for col in REQUIRED_METADATA:
                empty[col] = np.nan
            return empty
        present = [c for c in REQUIRED_METADATA if c in metadata.columns]
        meta = metadata.set_index("object_id")[present].reindex(ids)
        for col in REQUIRED_METADATA:
            if col not in meta.columns:
                meta[col] = np.nan
        return meta

    # ---------------------------------------------------------------- per band
    def _band_features(self, lc: pd.DataFrame, band: int, label: str, ids: pd.Index) -> pd.DataFrame:
        pre = f"{label}_"
        sub = lc[lc["passband"].to_numpy() == band]
        if not len(sub):
            return pd.DataFrame(np.nan, index=ids, columns=[pre + s for s in self._band_suffixes()])

        sub = sub.sort_values(["object_id", "mjd"], kind="mergesort")
        oid = sub["object_id"].to_numpy()
        flux = sub["flux"].to_numpy(dtype="float64")
        err = sub["flux_err"].to_numpy(dtype="float64")
        mjd = sub["mjd"].to_numpy(dtype="float64")
        det = sub["detected_bool"].to_numpy(dtype="float64")
        t0 = pd.Series(mjd).groupby(pd.Series(oid)).transform("min").to_numpy()
        x = mjd - t0
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = np.where(err > 0, flux / np.where(err > 0, err, 1.0), np.nan)

        work = pd.DataFrame({
            "oid": oid, "flux": flux, "err": err, "mjd": mjd, "det": det, "snr": snr,
            "x": x, "x2": x * x, "x3": x ** 3, "x4": x ** 4,
            "xy": x * flux, "yx2": flux * x * x, "fmjd": flux * mjd,
            "f2": flux * flux, "f3": flux ** 3, "f4": flux ** 4,
            "dflux": np.where(det == 1, flux, np.nan),
            "dmjd": np.where(det == 1, mjd, np.nan),
        })
        work["gap"] = work.groupby("oid", sort=False)["mjd"].diff()

        agg = work.groupby("oid", sort=False).agg(
            n_obs=("flux", "size"), n_det=("det", "sum"),
            flux_max=("flux", "max"), flux_min=("flux", "min"), flux_mean=("flux", "mean"),
            flux_std=("flux", "std"), flux_sum=("flux", "sum"),
            err_mean=("err", "mean"), err_min=("err", "min"),
            t_min=("mjd", "min"), t_max=("mjd", "max"), t_mean=("mjd", "mean"), t_std=("mjd", "std"),
            snr_max=("snr", "max"), snr_mean=("snr", "mean"), snr_median=("snr", "median"),
            s1=("x", "sum"), s2=("x2", "sum"), s3=("x3", "sum"), s4=("x4", "sum"),
            t0=("flux", "sum"), t1=("xy", "sum"), t2=("yx2", "sum"), fmjd=("fmjd", "sum"),
            p1=("flux", "sum"), p2=("f2", "sum"), p3=("f3", "sum"), p4=("f4", "sum"),
            gap_median=("gap", "median"), gap_max=("gap", "max"),
            dflux_max=("dflux", "max"), dflux_mean=("dflux", "mean"), dflux_median=("dflux", "median"),
        ).reindex(ids)

        quant = work.groupby("oid", sort=False)["flux"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack().reindex(ids)
        snr_p90 = work.groupby("oid", sort=False)["snr"].quantile(0.9).reindex(ids)

        peak_pos = work.groupby("oid", sort=False)["flux"].idxmax()
        t_peak = work.loc[peak_pos].set_index("oid")["mjd"].reindex(ids)
        peak_snr = work.loc[peak_pos].set_index("oid")["snr"].reindex(ids)
        dpeak_pos = work.groupby("oid", sort=False)["dflux"].idxmax().dropna()
        dpeak_time = (work.loc[dpeak_pos.astype("int64")].set_index("oid")["mjd"].reindex(ids)
                      if len(dpeak_pos) else pd.Series(np.nan, index=ids))

        curvature = _quadratic_curvature({
            "n": agg["n_obs"].to_numpy(dtype="float64"),
            **{k: agg[k].to_numpy(dtype="float64") for k in ("s1", "s2", "s3", "s4", "t0", "t1", "t2")},
        })
        skew, kurt = _central_moments(
            agg["p1"].to_numpy(dtype="float64"), agg["p2"].to_numpy(dtype="float64"),
            agg["p3"].to_numpy(dtype="float64"), agg["p4"].to_numpy(dtype="float64"),
            agg["n_obs"].to_numpy(dtype="float64"),
        )

        n = agg["n_obs"].to_numpy(dtype="float64")
        span = (agg["t_max"] - agg["t_min"]).to_numpy(dtype="float64")
        s1, s2 = agg["s1"].to_numpy(dtype="float64"), agg["s2"].to_numpy(dtype="float64")
        t0s, t1s = agg["t0"].to_numpy(dtype="float64"), agg["t1"].to_numpy(dtype="float64")
        sumsq_y = agg["p2"].to_numpy(dtype="float64")
        denom = n * s2 - s1 * s1
        slope = _safe_div(n * t1s - s1 * t0s, denom)
        intercept = _safe_div(t0s - slope * s1, n)
        # Residual sum of squares around the LINEAR fit, from accumulated sums:
        # RSS = Sum(y^2) - 2a*Sum(y) - 2b*Sum(xy) + a^2 n + 2ab Sum(x) + b^2 Sum(x^2)
        fit_ss = intercept ** 2 * n + 2 * intercept * slope * s1 + slope ** 2 * s2
        rss = sumsq_y - 2 * intercept * t0s - 2 * slope * t1s + fit_ss
        resid_var = np.clip(_safe_div(rss, np.clip(n - 2, 1, None), np.nan), 0.0, None)
        n_neg = work.assign(neg=(work["flux"].to_numpy() < 0).astype("float64")).groupby("oid", sort=False)["neg"].sum().reindex(ids)

        out = pd.DataFrame(index=ids)
        out["n_obs"] = n
        out["n_det"] = agg["n_det"].to_numpy(dtype="float64")
        out["flux_max"] = agg["flux_max"].to_numpy(dtype="float64")
        out["flux_min"] = agg["flux_min"].to_numpy(dtype="float64")
        out["flux_mean"] = agg["flux_mean"].to_numpy(dtype="float64")
        out["flux_std"] = agg["flux_std"].to_numpy(dtype="float64")
        out["flux_sum"] = agg["flux_sum"].to_numpy(dtype="float64")
        out["flux_skew"] = skew
        out["flux_kurt"] = kurt
        out["flux_p10"] = quant[0.1].to_numpy(dtype="float64")
        out["flux_p25"] = quant[0.25].to_numpy(dtype="float64")
        out["flux_p50"] = quant[0.5].to_numpy(dtype="float64")
        out["flux_p75"] = quant[0.75].to_numpy(dtype="float64")
        out["flux_p90"] = quant[0.9].to_numpy(dtype="float64")
        out["err_mean"] = agg["err_mean"].to_numpy(dtype="float64")
        out["err_min"] = agg["err_min"].to_numpy(dtype="float64")
        out["snr_max"] = agg["snr_max"].to_numpy(dtype="float64")
        out["snr_mean"] = agg["snr_mean"].to_numpy(dtype="float64")
        out["snr_median"] = agg["snr_median"].to_numpy(dtype="float64")
        out["snr_p90"] = snr_p90.to_numpy(dtype="float64")
        out["t_min"] = agg["t_min"].to_numpy(dtype="float64")
        out["t_max"] = agg["t_max"].to_numpy(dtype="float64")
        out["t_span"] = span
        out["t_mean"] = agg["t_mean"].to_numpy(dtype="float64")
        out["t_std"] = agg["t_std"].to_numpy(dtype="float64")
        out["t_peak"] = t_peak.to_numpy(dtype="float64")
        out["peak_snr"] = peak_snr.to_numpy(dtype="float64")
        out["slope"] = slope
        out["curvature"] = curvature
        out["resid_std"] = np.sqrt(np.nan_to_num(resid_var, nan=0.0))
        out["gap_median"] = agg["gap_median"].to_numpy(dtype="float64")
        out["gap_max"] = agg["gap_max"].to_numpy(dtype="float64")
        out["dflux_max"] = agg["dflux_max"].to_numpy(dtype="float64")
        out["dflux_mean"] = agg["dflux_mean"].to_numpy(dtype="float64")
        out["dflux_median"] = agg["dflux_median"].to_numpy(dtype="float64")
        out["dpeak_time"] = dpeak_time.to_numpy(dtype="float64")
        out["det_frac"] = _safe_div(out["n_det"].to_numpy(), n, np.nan)
        out["flux_range"] = out["flux_max"].to_numpy() - out["flux_min"].to_numpy()
        out["flux_iqr"] = out["flux_p75"].to_numpy() - out["flux_p25"].to_numpy()
        out["t_centroid"] = _safe_div(agg["fmjd"].to_numpy(dtype="float64"), t0s, np.nan)
        out["rise_time"] = out["t_peak"].to_numpy() - out["t_min"].to_numpy()
        out["decline_time"] = out["t_max"].to_numpy() - out["t_peak"].to_numpy()
        out["rise_rate"] = _safe_div(out["flux_max"].to_numpy(), np.clip(out["rise_time"].to_numpy(), 0.5, None), np.nan)
        out["decline_rate"] = _safe_div(out["flux_max"].to_numpy(), np.clip(out["decline_time"].to_numpy(), 0.5, None), np.nan)
        out["variability"] = _safe_div(out["flux_std"].to_numpy(), np.abs(out["flux_mean"].to_numpy()), np.nan)
        out["integral"] = _safe_div(out["flux_sum"].to_numpy(), np.clip(span, 0.5, None), np.nan)
        out["neg_frac"] = _safe_div(n_neg.to_numpy(dtype="float64"), np.clip(out["n_det"].to_numpy(), 1, None), np.nan)
        out["peak_mag"] = _mag_snr_masked(out["flux_max"].to_numpy(), out["peak_snr"].to_numpy())
        out["det_peak_mag"] = _mag_snr_masked(out["dflux_max"].to_numpy(), out["peak_snr"].to_numpy())
        out["med_mag"] = _mag_snr_masked(out["flux_p50"].to_numpy(), out["snr_median"].to_numpy())
        for col in ("peak_mag", "det_peak_mag", "med_mag"):
            out[col] = np.clip(out[col].to_numpy(), *MAG_CLIP)
        out.columns = [pre + c for c in out.columns]
        return out.reindex(ids)

    @staticmethod
    def _band_suffixes() -> List[str]:
        return [
            "n_obs", "n_det", "flux_max", "flux_min", "flux_mean", "flux_std", "flux_sum",
            "flux_skew", "flux_kurt", "flux_p10", "flux_p25", "flux_p50", "flux_p75", "flux_p90",
            "err_mean", "err_min", "snr_max", "snr_mean", "snr_median", "snr_p90",
            "t_min", "t_max", "t_span", "t_mean", "t_std", "t_peak", "peak_snr", "slope",
            "curvature", "resid_std", "gap_median", "gap_max", "dflux_max", "dflux_mean",
            "dflux_median", "dpeak_time", "det_frac", "flux_range", "flux_iqr", "t_centroid",
            "rise_time", "decline_time", "rise_rate", "decline_rate", "variability", "integral",
            "neg_frac", "peak_mag", "det_peak_mag", "med_mag",
        ]

    # -------------------------------------------------------------- all bands
    def _global_features(self, lc: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
        p = self.cfg.global_prefix
        work = pd.DataFrame({
            "oid": lc["object_id"].to_numpy(),
            "flux": lc["flux"].to_numpy(dtype="float64"),
            "mjd": lc["mjd"].to_numpy(dtype="float64"),
            "det": lc["detected_bool"].to_numpy(dtype="float64"),
            "err": lc["flux_err"].to_numpy(dtype="float64"),
        })
        work["snr"] = np.where(work["err"].to_numpy() > 0, work["flux"].to_numpy() / np.clip(work["err"].to_numpy(), 1e-9, None), np.nan)
        work["dflux"] = np.where(work["det"] == 1, work["flux"], np.nan)
        agg = work.groupby("oid", sort=False).agg(
            n_obs=("flux", "size"), n_det=("det", "sum"), flux_max=("flux", "max"),
            flux_mean=("flux", "mean"), flux_std=("flux", "std"), t_min=("mjd", "min"),
            t_max=("mjd", "max"), snr_max=("snr", "max"), snr_median=("snr", "median"),
            dflux_max=("dflux", "max"), err_mean=("err", "mean"),
        ).reindex(ids)
        n = agg["n_obs"].to_numpy(dtype="float64")
        span = (agg["t_max"] - agg["t_min"]).to_numpy(dtype="float64")
        out = pd.DataFrame(index=ids)
        out[p + "n_obs"] = n
        out[p + "n_det"] = agg["n_det"].to_numpy(dtype="float64")
        out[p + "det_frac"] = _safe_div(agg["n_det"].to_numpy(dtype="float64"), n, np.nan)
        out[p + "flux_max"] = agg["flux_max"].to_numpy(dtype="float64")
        out[p + "flux_mean"] = agg["flux_mean"].to_numpy(dtype="float64")
        out[p + "flux_std"] = agg["flux_std"].to_numpy(dtype="float64")
        out[p + "snr_max"] = agg["snr_max"].to_numpy(dtype="float64")
        out[p + "snr_median"] = agg["snr_median"].to_numpy(dtype="float64")
        out[p + "err_mean"] = agg["err_mean"].to_numpy(dtype="float64")
        out[p + "t_span"] = span
        out[p + "t_min"] = agg["t_min"].to_numpy(dtype="float64")
        out[p + "t_max"] = agg["t_max"].to_numpy(dtype="float64")
        out[p + "dflux_max"] = agg["dflux_max"].to_numpy(dtype="float64")
        out[p + "peak_mag"] = np.clip(_mag_snr_masked(out[p + "flux_max"].to_numpy(), out[p + "snr_max"].to_numpy()), *MAG_CLIP)
        out[p + "det_peak_mag"] = np.clip(_mag_snr_masked(out[p + "dflux_max"].to_numpy(), out[p + "snr_max"].to_numpy()), *MAG_CLIP)
        out[p + "points_per_day"] = _safe_div(n, np.clip(span, 0.5, None), np.nan)
        out[p + "variability"] = _safe_div(out[p + "flux_std"].to_numpy(), np.abs(out[p + "flux_mean"].to_numpy()), np.nan)
        return out.reindex(ids)

    # ----------------------------------------------------------------- colours
    def _colour_features(self, ids, band_peaks: Dict[str, np.ndarray], band_peak_mjd: Dict[str, np.ndarray]) -> pd.DataFrame:
        p = self.cfg.colour_prefix
        out = pd.DataFrame(index=ids)
        # A colour is a difference of MAGNITUDES. Subtracting peak fluxes - which
        # is what an earlier revision of this function did - produces a quantity
        # with no physical meaning and a wildly skewed distribution.
        mags = {b: _mag(band_peaks[b]) for b in BAND_LABELS}
        for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("z", "y"),
                     ("g", "i"), ("r", "z"), ("u", "r"), ("g", "z")]:
            out[f"{p}{a}m{b}"] = np.clip(mags[a] - mags[b], -12.0, 12.0)
        for a, b in [("g", "r"), ("r", "i"), ("u", "g"), ("i", "z")]:
            out[f"{p}dt_{a}m{b}"] = band_peak_mjd[a] - band_peak_mjd[b]
        gr = np.abs(mags["g"] - mags["r"])
        # Bounded by construction. The naive 1/|g-r| explodes to ~1e3 for a
        # near-neutral colour, which makes one feature dominate any standardised
        # distance and any tree split that touches it.
        with np.errstate(invalid="ignore", divide="ignore"):
            out[p + "temp_proxy"] = np.where(np.isfinite(gr), 1.0 / (0.2 + np.clip(gr, 0.0, None)), np.nan)
        out[p + "n_bands_positive"] = np.sum(
            np.stack([np.nan_to_num(band_peaks[b], nan=-99.0) > 0 for b in BAND_LABELS]), axis=0
        ).astype("float64")
        return out.reindex(ids)

    # ----------------------------------------------------------------- physics
    def _physics_features(self, ids, meta: pd.DataFrame, band_peaks: Dict[str, np.ndarray],
                          band_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        p = self.cfg.physics_prefix
        out = pd.DataFrame(index=ids)
        z = np.nan_to_num(meta["hostgal_photoz"].to_numpy(dtype="float64"), nan=0.0)
        zerr = meta["hostgal_photoz_err"].to_numpy(dtype="float64")
        dm, flag = resolve_distmod(meta["distmod"].to_numpy(dtype="float64"), z, self.cosmo)
        mwebv = np.nan_to_num(meta["mwebv"].to_numpy(dtype="float64"), nan=0.0)
        out[p + "z"] = z
        out[p + "z_err"] = np.nan_to_num(zerr, nan=0.0)
        out[p + "distmod"] = dm
        out[p + "distmod_source"] = flag
        out[p + "mwebv"] = mwebv
        one_plus_z = np.clip(1.0 + z, 1.0, None)
        absmags = []
        for band in BANDS:
            label = BAND_LABELS[band]
            a_lam = extinction_ab(mwebv, band, self.cosmo.r_v)
            app = _mag(band_peaks[label])
            absmag = app - dm - a_lam
            out[f"{p}absmag_{label}"] = absmag
            out[f"{p}A_{label}"] = a_lam
            absmags.append(absmag)
            # Rest-frame timescales: observed timescale / (1 + z).
            rise = band_frames[label][f"{label}_rise_time"].to_numpy(dtype="float64")
            decline = band_frames[label][f"{label}_decline_time"].to_numpy(dtype="float64")
            out[f"{p}rf_rise_{label}"] = rise / one_plus_z
            out[f"{p}rf_decline_{label}"] = decline / one_plus_z
        stack = np.stack(absmags)
        out[p + "absmag_max"] = np.nanmax(stack, axis=0)
        out[p + "absmag_min"] = np.nanmin(stack, axis=0)
        peak_flux = np.nanmax(np.stack([band_peaks[b] for b in BAND_LABELS]), axis=0)
        out[p + "peak_luminosity"] = peak_luminosity_proxy(peak_flux, dm)
        return out.reindex(ids)

    # -------------------------------------------------------------------- host
    def _host_features(self, ids, meta: pd.DataFrame) -> pd.DataFrame:
        p = self.cfg.host_prefix
        out = pd.DataFrame(index=ids)
        ra = np.nan_to_num(meta["ra"].to_numpy(dtype="float64"), nan=0.0)
        dec = np.nan_to_num(meta["decl"].to_numpy(dtype="float64"), nan=0.0)
        lon, lat = galactic_coordinates(ra, dec)
        out[p + "ra"] = ra
        out[p + "decl"] = dec
        out[p + "gal_l"] = lon
        out[p + "gal_b"] = lat
        out[p + "abs_gal_b"] = np.abs(lat)
        out[p + "ddf"] = np.nan_to_num(meta["ddf_bool"].to_numpy(dtype="float64"), nan=0.0)
        out[p + "has_specz"] = np.isfinite(meta["hostgal_specz"].to_numpy(dtype="float64")).astype("float64")
        out[p + "specz"] = np.nan_to_num(meta["hostgal_specz"].to_numpy(dtype="float64"), nan=0.0)
        return out.reindex(ids)

    # ------------------------------------------------------------- uncertainty
    def _uncertainty_features(self, ids, meta: pd.DataFrame, band_peaks: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Monte-Carlo propagation of photometric-redshift uncertainty (CNE v2).

        Distance-sensitive features become distributions: mean, standard
        deviation and two quantiles, plus a per-object reliability score the
        ranker uses to refuse false precision.
        """
        p = self.cfg.uncertainty_prefix
        out = pd.DataFrame(index=ids)
        z = np.nan_to_num(meta["hostgal_photoz"].to_numpy(dtype="float64"), nan=0.0)
        zerr = meta["hostgal_photoz_err"].to_numpy(dtype="float64")
        _dm, flag = resolve_distmod(meta["distmod"].to_numpy(dtype="float64"), z, self.cosmo)
        draws = mc_redshift_draws(z, zerr, self.cfg.mc_redshift_draws, seed=42)
        dm_draws = distance_modulus_from_z(draws, self.cosmo.omega_m, self.cosmo.h0)
        peak_flux = np.nanmax(np.stack([band_peaks[b] for b in BAND_LABELS]), axis=0)
        peak_mag = _mag(peak_flux)
        absmag_draws = peak_mag[None, :] - dm_draws
        out[p + "absmag_std"] = np.nanstd(absmag_draws, axis=0)
        out[p + "absmag_q16"] = np.nanquantile(absmag_draws, 0.16, axis=0)
        out[p + "absmag_q84"] = np.nanquantile(absmag_draws, 0.84, axis=0)
        out[p + "absmag_mean"] = np.nanmean(absmag_draws, axis=0)
        out[p + "distmod_std"] = np.nanstd(dm_draws, axis=0)
        z_rel = reliability_from_redshift(z, zerr, flag)
        out[p + "z_reliability"] = z_rel
        out[p + "feature_reliability"] = self._feature_reliability(out[p + "absmag_std"].to_numpy(dtype="float64"), z_rel)
        del draws, dm_draws, absmag_draws
        return out.reindex(ids)

    @staticmethod
    def _feature_reliability(absmag_std, z_reliability) -> np.ndarray:
        spread_penalty = np.exp(-np.nan_to_num(absmag_std, nan=3.0) / 1.5)
        return np.clip(np.nan_to_num(z_reliability, nan=0.0) * spread_penalty, 0.0, 1.0)

    # ----------------------------------------------------------------- quality
    def _quality_features(self, lc: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
        """Data-quality diagnostics. Suppress-only: never fed to astrophysics."""
        p = self.cfg.quality_prefix
        work = pd.DataFrame({
            "oid": lc["object_id"].to_numpy(),
            "flux": lc["flux"].to_numpy(dtype="float64"),
            "err": lc["flux_err"].to_numpy(dtype="float64"),
            "mjd": lc["mjd"].to_numpy(dtype="float64"),
            "band": lc["passband"].to_numpy(),
            "det": lc["detected_bool"].to_numpy(dtype="float64"),
        })
        work["neg"] = ((work["det"].to_numpy() == 1) & (work["flux"].to_numpy() < 0)).astype("float64")
        # Detection-only diagnostics. Computing these over ALL epochs - including
        # the many non-detections PLAsTiCC records - is a real bug: |flux| is tiny
        # for a non-detection, so flux_err/|flux| explodes and the median SNR
        # collapses to ~0. Both statistics then saturate their penalty for ~90% of
        # objects and the quality gate degenerates into a constant.
        is_det = work["det"].to_numpy() == 1
        abs_flux = np.abs(work["flux"].to_numpy())
        with np.errstate(invalid="ignore", divide="ignore"):
            work["err_ratio"] = np.where(
                is_det & (abs_flux > 1e-9),
                work["err"].to_numpy() / np.where(abs_flux > 1e-9, abs_flux, 1.0),
                np.nan,
            )
            work["det_snr"] = np.where(
                is_det & (work["err"].to_numpy() > 0),
                work["flux"].to_numpy() / np.where(work["err"].to_numpy() > 0, work["err"].to_numpy(), 1.0),
                np.nan,
            )
        work["det_band"] = np.where(is_det, work["band"].to_numpy(), -1)
        dup_sizes = work.groupby(["oid", "band", "mjd"], sort=False).size()
        dups = dup_sizes[dup_sizes > 1].groupby(level=0).sum()
        agg = work.groupby("oid", sort=False).agg(
            n_obs=("flux", "size"), n_det=("det", "sum"), neg_sum=("neg", "sum"),
            err_ratio_mean=("err_ratio", "mean"), err_ratio_median=("err_ratio", "median"),
            err_ratio_max=("err_ratio", "max"), det_snr_median=("det_snr", "median"),
            det_snr_min=("det_snr", "min"),
            err_min=("err", "min"), t_min=("mjd", "min"), t_max=("mjd", "max"),
        ).reindex(ids)
        band_counts = (work.groupby(["oid", "band"], sort=False).size()
                       .unstack(fill_value=0).reindex(ids).fillna(0.0))
        det_band_counts = (work[work["det_band"].to_numpy() >= 0]
                           .groupby(["oid", "det_band"], sort=False).size()
                           .unstack(fill_value=0).reindex(ids).fillna(0.0))
        for band in BANDS:
            if band not in band_counts.columns:
                band_counts[band] = 0.0
            if band not in det_band_counts.columns:
                det_band_counts[band] = 0.0
        total = band_counts[list(BANDS)].sum(axis=1).to_numpy(dtype="float64")
        max_frac = _safe_div(band_counts[list(BANDS)].max(axis=1).to_numpy(dtype="float64"), total, np.nan)
        det_total = det_band_counts[list(BANDS)].sum(axis=1).to_numpy(dtype="float64")
        det_max_frac = _safe_div(det_band_counts[list(BANDS)].max(axis=1).to_numpy(dtype="float64"), det_total, np.nan)

        out = pd.DataFrame(index=ids)
        n = agg["n_obs"].to_numpy(dtype="float64")
        ndet = agg["n_det"].to_numpy(dtype="float64")
        out[p + "n_obs"] = n
        out[p + "n_det"] = ndet
        out[p + "neg_flux_frac"] = _safe_div(agg["neg_sum"].to_numpy(dtype="float64"), np.clip(ndet, 1, None), np.nan)
        out[p + "err_ratio_mean"] = agg["err_ratio_mean"].to_numpy(dtype="float64")
        out[p + "err_ratio_median"] = agg["err_ratio_median"].to_numpy(dtype="float64")
        out[p + "err_ratio_max"] = agg["err_ratio_max"].to_numpy(dtype="float64")
        out[p + "det_snr_median"] = agg["det_snr_median"].to_numpy(dtype="float64")
        out[p + "det_snr_min"] = agg["det_snr_min"].to_numpy(dtype="float64")
        out[p + "err_min"] = agg["err_min"].to_numpy(dtype="float64")
        out[p + "duplicate_epochs"] = dups.reindex(ids).fillna(0.0).to_numpy(dtype="float64")
        out[p + "single_band_frac"] = max_frac
        out[p + "dominant_band_det_frac"] = det_max_frac
        out[p + "n_bands"] = (band_counts[list(BANDS)] > 0).sum(axis=1).to_numpy(dtype="float64")
        out[p + "n_bands_detected"] = (det_band_counts[list(BANDS)] > 0).sum(axis=1).to_numpy(dtype="float64")
        out[p + "det_frac"] = _safe_div(ndet, n, np.nan)
        out[p + "t_span"] = (agg["t_max"] - agg["t_min"]).to_numpy(dtype="float64")
        for band in BANDS:
            out[f"{p}det_{BAND_LABELS[band]}"] = band_counts[band].to_numpy(dtype="float64")
        return out.reindex(ids)
