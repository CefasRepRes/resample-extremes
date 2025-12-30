#!/usr/bin/env python

# resample-extremes.py
#
# michael.taylor@cefas.gov.uk
# 29-Dec-2025

"""
Synthetic mixed-frequency riverine Total N time series with:
- ~weekly sampling outside 2000–2005 (≈4 samples/month)
- daily sampling during 2000–2005
- flash-event spikes preferentially in wetter UK months (Oct–Mar)

Extensions included:
1) CEEMDAN (noise-assisted EMD; more stable for hydrological signals)
2) Monthly confidence intervals (t-interval for sparse n; moving-block bootstrap for larger n)
3) Event detection + censoring (remove event days; compute baseline monthly means on censored data)
4) Outputs:
   - CSV: raw samples, monthly naive stats, monthly censored stats + CIs, event summaries, diagnostics
   - PNG: overlays, CI plots, detection diagnostic, CEEMDAN IMFs, wavelet power (optional), method comparison

Progress reporting:
- Section-level timestamps
- Optional tqdm progress bars (install with: pip install tqdm)

Important dependency note:
- CEEMDAN/CEEMDAN IMFs come from PyPI package "EMD-signal"
  pip install EMD-signal
  (Do NOT use package "pyemd" — it is unrelated.)

Usage:
  # very slow:
  python resample-extremes.py --outdir OUT --seed 0 --progress

  # slow:
  python resample-extremes.py --outdir OUT --seed 0 --progress --n-boot 200 --ceemdan-trials 10

  # faster:
  python resample-extremes.py --outdir OUT --seed 0 --progress --n-boot 200 --ceemdan-trials 10 --skip-wavelet
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pywt
from scipy import stats
from matplotlib.gridspec import GridSpec

try:
    # Provided by PyPI package: EMD-signal
    #   pip install EMD-signal
    from PyEMD import CEEMDAN
except Exception as e:  # pragma: no cover
    CEEMDAN = None  # type: ignore
    _CEEMDAN_IMPORT_ERROR = e

# Optional progress bars
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# Journal-style plotting defaults
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


# -----------------------------
# Consistent colour palette
# -----------------------------
# Use Matplotlib's default colour cycle (C0..C9) but name key series so
# colours are consistent across all figures.
COL = {
    "naive": "C0",
    "censored": "C1",
    "ceemdan": "C2",
    "baseline": "C3",
    "events": "C4",
    "band_event": "C0",
    "band_baseline": "C1",
    "obs_weekly": "0.25",  # dark gray
    "obs_daily": "0.55",  # light gray
}


# ----------------------------- Plot + output settings -----------------------------
# Toggle legacy CWT outputs (the three separate CWT figures). Requested default: False.
SAVE_CWT_LEGACY = False

# Colormap for the CWT log-power heatmap
CWT_CMAP = "viridis"

# Colour limits for CWT log10(power).
# If CWT_CLIM is not None, use (vmin, vmax). Otherwise use percentiles from the data.
CWT_CLIM = None  # e.g. (-2.5, 3.2)
CWT_CLIM_PERCENTILES = (2.0, 98.0)

# Mask non-significant regions in the wavelet heatmap (power below 95% red-noise threshold)
CWT_MASK_NONSIGNIFICANT = True
# How masked (non-significant) cells are rendered in the heatmap (RGBA). Use alpha=0 for transparent.
CWT_MASK_BAD_RGBA = (0.0, 0.0, 0.0, 0.0)

# Period guide lines drawn on wavelet panels (days)
PERIOD_GUIDES_DAYS = (7.0, 30.0, 365.0)
PERIOD_GUIDE_STYLE = dict(color="white", linestyle=":", linewidth=1.0, alpha=0.7)

# HF period rendering in plots: use two vertical dashed lines (no shaded span)
HF_MARKER = dict(color="white", linestyle="--", linewidth=1.2, alpha=0.95)

HF_SPAN = dict(color="lightsteelblue", alpha=0.12, zorder=0)

# Colourbar styling for inset colourbar drawn on top of the heatmap
CWT_COLORBAR = dict(
    width="38%",      # relative to heatmap axes
    height="3.0%",      # relative to heatmap axes
    loc="upper center",
    borderpad=1.0,
    label="log10(power)",
    tick_color="white",
    label_color="white",
    outline_color="white",
)



HF_START = pd.Timestamp("2000-01-01")
HF_END = pd.Timestamp("2005-12-31")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_section(msg: str):
    print(f"[{_now()}] {msg}", flush=True)


# -----------------------------
# Synthetic generator
# -----------------------------
def generate_synthetic_series(seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range("1980-01-01", "2025-12-31", freq="D")
    n = len(dates)
    rng = np.random.default_rng(seed)

    # Baseline trend (1980->2025): 5 -> 8 (arbitrary units)
    year = dates.year.to_numpy(dtype=float)
    doy = dates.dayofyear.to_numpy(dtype=float)
    year_frac = year + (doy - 1.0) / 365.0
    baseline = 5.0 + (8.0 - 5.0) * (year_frac - 1980.0) / (2025.0 - 1980.0)

    # Seasonality: winter-peaking cosine
    season = 0.5 * np.cos(2.0 * np.pi * (doy / 365.0))

    # Daily noise
    noise = 0.2 * rng.standard_normal(n)

    y_no_spike = (baseline + season + noise).astype(float)
    y = y_no_spike.copy()

    # Flash events during 2000–2005, more likely in wet months Oct–Mar
    hf_start, hf_end = pd.Timestamp("2000-01-01"), pd.Timestamp("2005-12-31")
    hf_mask = np.asarray((dates >= hf_start) & (dates <= hf_end), dtype=bool)
    months = np.asarray(dates.month)

    p_wet = 0.018
    p_dry = 0.004
    p = np.where(np.isin(months, [10, 11, 12, 1, 2, 3]), p_wet, p_dry)
    p = np.where(hf_mask, p, 0.0)

    event_day = rng.random(n) < p

    mag = np.zeros(n, dtype=float)
    mag[event_day] = rng.lognormal(mean=2.1, sigma=0.35, size=int(event_day.sum()))

    idxs = np.where(event_day)[0]
    for idx in idxs:
        peak = mag[idx]
        y[idx] += peak
        if idx > 0:
            y[idx - 1] += 0.5 * peak
        if idx < n - 1:
            y[idx + 1] += 0.3 * peak

    # --- Synthetic flow (m3/s), correlated with wet season and event days ---
    flow_season = 6.0 * np.cos(2.0 * np.pi * (doy / 365.0))  # winter-peaking
    flow_noise = rng.lognormal(mean=0.0, sigma=0.12, size=n)  # multiplicative variability
    flow_no_spike = (20.0 + flow_season) * flow_noise
    flow = flow_no_spike.copy()

    # Event-related flow pulses (storm hydrographs), aligned to concentration events
    flow_pulse = np.zeros(n, dtype=float)
    flow_pulse[event_day] = rng.lognormal(mean=2.2, sigma=0.35, size=int(event_day.sum()))
    for idx in idxs:
        peakq = flow_pulse[idx]
        flow[idx] += peakq
        if idx > 0:
            flow[idx - 1] += 0.6 * peakq
        if idx < n - 1:
            flow[idx + 1] += 0.4 * peakq

    df = pd.DataFrame(
        {
            "true_no_spikes": y_no_spike,
            "true_with_spikes": y,
            "is_event_day": event_day,
            "event_mag_peak": mag,
            "flow_no_spikes": flow_no_spike,
            "flow_with_spikes": flow,
        },
        index=dates,
    )
    return df


def build_observed_samples(df_daily: pd.DataFrame) -> pd.Series:
    weekly_idx = pd.date_range("1980-01-07", "2025-12-31", freq="7D")
    daily_hf_idx = pd.date_range("2000-01-01", "2005-12-31", freq="D")
    sample_idx = weekly_idx.union(daily_hf_idx)
    obs = df_daily["true_with_spikes"].loc[sample_idx].copy()
    obs.name = "Concentration"
    return obs


def build_observed_flow_samples(df_daily: pd.DataFrame) -> pd.Series:
    """Observed flow samples on the same mixed-frequency schedule as concentration."""
    weekly_idx = pd.date_range("1980-01-07", "2025-12-31", freq="7D")
    daily_hf_idx = pd.date_range("2000-01-01", "2005-12-31", freq="D")
    sample_idx = weekly_idx.union(daily_hf_idx)
    obs = df_daily["flow_with_spikes"].loc[sample_idx].copy()
    obs.name = "Flow"
    return obs


# -----------------------------
# Event detection + censoring
# -----------------------------
def rolling_baseline(daily: pd.Series, window_days: int = 31) -> pd.Series:
    return daily.rolling(window=window_days, center=True, min_periods=window_days // 2).median()


def detect_events(daily: pd.Series, baseline: pd.Series, k_mad: float = 6.0) -> pd.Series:
    resid = daily - baseline
    med = np.nanmedian(resid.values)
    mad = np.nanmedian(np.abs(resid.values - med))
    sigma = 1.4826 * mad if mad > 0 else np.nanstd(resid.values)
    thresh = med + k_mad * sigma
    events = (resid > thresh).fillna(False)
    return events


def censor_series(daily: pd.Series, event_mask: pd.Series, censor_buffer_days: int = 1) -> pd.Series:
    mask = event_mask.to_numpy().copy()
    if censor_buffer_days > 0:
        idx = np.where(mask)[0]
        for i in idx:
            lo = max(0, i - censor_buffer_days)
            hi = min(len(mask), i + censor_buffer_days + 1)
            mask[lo:hi] = True
    censored = daily.copy()
    censored.iloc[mask] = np.nan
    return censored


# -----------------------------
# CEEMDAN baseline extraction
# -----------------------------
def ceemdan_decompose(
    daily: np.ndarray,
    seed: int = 0,
    trials: int = 50,
    noise_width: float = 0.2,
    progress: bool = False,
) -> np.ndarray:
    if CEEMDAN is None:
        raise ImportError(
            "CEEMDAN requires the 'EMD-signal' package (module 'PyEMD'). "
            "Install with: pip install EMD-signal"
        ) from _CEEMDAN_IMPORT_ERROR
    ce = CEEMDAN()
    ce.noise_seed(int(seed))
    ce.trials = int(trials)
    ce.noise_width = float(noise_width)
    if progress:
        log_section(f"CEEMDAN started (n={len(daily)}, trials={ce.trials}, noise_width={ce.noise_width})")
    imfs = ce.ceemdan(daily.astype(float))
    if progress:
        log_section(f"CEEMDAN finished (IMFs={imfs.shape[0]})")
    return imfs


def choose_lowfreq_reconstruction(imfs: np.ndarray, keep_last: int = 2) -> np.ndarray:
    if imfs.ndim != 2 or imfs.shape[0] == 0:
        return imfs
    k = min(int(keep_last), imfs.shape[0])
    return np.sum(imfs[-k:, :], axis=0)


# -----------------------------
# Monthly resampling + CIs
# -----------------------------
@dataclass
class MonthlyCI:
    mean: float
    ci_lo: float
    ci_hi: float
    n: int
    method: str


def ci_t_interval(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    half = tcrit * s / np.sqrt(n)
    return (m - half, m + half)


def moving_block_bootstrap_ci(
    x: np.ndarray,
    block_len: int = 3,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        return (float(x[0]), float(x[0]))

    L = max(1, min(int(block_len), n))
    x_ext = np.concatenate([x, x[:L]])
    n_blocks = int(np.ceil(n / L))
    boot_means = np.empty(int(n_boot), dtype=float)

    iterable = range(int(n_boot))
    if show_progress and tqdm is not None:
        iterable = tqdm(iterable, total=int(n_boot), desc=progress_desc or "bootstrap", leave=False)

    for b in iterable:
        starts = rng.integers(0, n, size=n_blocks)
        sample = []
        for s in starts:
            sample.append(x_ext[s : s + L])
        samp = np.concatenate(sample)[:n]
        boot_means[b] = np.mean(samp)

    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def monthly_mean_and_ci(
    s: pd.Series,
    alpha: float = 0.05,
    block_len: int = 3,
    n_boot: int = 2000,
    min_n_bootstrap: int = 8,
    rng: np.random.Generator | None = None,
    progress: bool = False,
    inner_progress: bool = False,
    label: str = "monthly CI",
) -> pd.DataFrame:
    rows = []
    groups = list(s.groupby(pd.Grouper(freq="ME")))
    iterable = groups
    if progress and tqdm is not None:
        iterable = tqdm(groups, desc=label, total=len(groups))

    for month_end, grp in iterable:
        x = grp.dropna().to_numpy(dtype=float)
        n = len(x)
        if n == 0:
            rows.append((month_end, np.nan, np.nan, np.nan, 0, "none"))
            continue

        m = float(np.mean(x))
        if n >= int(min_n_bootstrap):
            lo, hi = moving_block_bootstrap_ci(
                x,
                block_len=block_len,
                n_boot=n_boot,
                alpha=alpha,
                rng=rng,
                show_progress=bool(inner_progress),
                progress_desc=f"{label} {month_end.strftime('%Y-%m')}",
            )
            method = f"mbb(L={block_len},B={n_boot})"
        else:
            lo, hi = ci_t_interval(x, alpha=alpha)
            method = "t"

        rows.append((month_end, m, lo, hi, n, method))

    return (
        pd.DataFrame(rows, columns=["MonthEnd", "Mean", "CI_Lo", "CI_Hi", "N", "CI_Method"])
        .set_index("MonthEnd")
        .sort_index()
    )


# -----------------------------
# Annual load uncertainty via Monte Carlo
# -----------------------------
def _sd_from_ci(lo: float, hi: float, z: float = 1.959963984540054) -> float:
    """Approximate SD from a symmetric 95% CI assuming normality."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan
    return float((hi - lo) / (2.0 * z))


def annual_load_monte_carlo(
    monthly_conc: pd.DataFrame,
    monthly_flow: pd.DataFrame,
    n_mc: int = 5000,
    seed: int = 0,
    dt_seconds: float = 86400.0,
) -> pd.DataFrame:
    """Propagate monthly concentration CI × flow CI to annual loads.

    Assumptions:
    - monthly_conc and monthly_flow contain columns Mean, CI_Lo, CI_Hi indexed by MonthEnd.
    - Flow is in m3/s; monthly volume is Mean * days_in_month * 86400.
    - Concentration units are synthetic; load units are synthetic-consistent.
    - Normal approximation for uncertainty with truncation at zero.
    """
    rng = np.random.default_rng(seed)

    idx = monthly_conc.index.intersection(monthly_flow.index)
    mc = monthly_conc.loc[idx]
    mf = monthly_flow.loc[idx]

    month_start = idx.to_period("M").to_timestamp()
    days = ((idx - month_start).days + 1).to_numpy(dtype=float)

    c_mean = mc["Mean"].to_numpy(dtype=float)
    c_sd = np.array([_sd_from_ci(lo, hi) for lo, hi in zip(mc["CI_Lo"], mc["CI_Hi"])], dtype=float)
    c_sd = np.where(np.isfinite(c_sd), c_sd, 0.10 * np.abs(c_mean))

    q_mean = mf["Mean"].to_numpy(dtype=float)
    q_sd = np.array([_sd_from_ci(lo, hi) for lo, hi in zip(mf["CI_Lo"], mf["CI_Hi"])], dtype=float)
    q_sd = np.where(np.isfinite(q_sd), q_sd, 0.10 * np.abs(q_mean))

    n_m = len(idx)
    loads = np.empty((n_mc, n_m), dtype=float)

    for i in range(n_mc):
        c = rng.normal(c_mean, c_sd)
        q = rng.normal(q_mean, q_sd)
        c = np.clip(c, 0.0, None)
        q = np.clip(q, 0.0, None)
        vol = q * days * dt_seconds
        loads[i, :] = c * vol

    years = idx.year.to_numpy(dtype=int)
    out = []
    for y in np.unique(years):
        sel = years == y
        annual = np.sum(loads[:, sel], axis=1)
        out.append(
            (
                int(y),
                float(np.mean(annual)),
                float(np.quantile(annual, 0.025)),
                float(np.quantile(annual, 0.975)),
            )
        )

    return pd.DataFrame(out, columns=["Year", "AnnualLoad_Mean", "AnnualLoad_CI_Lo", "AnnualLoad_CI_Hi"]).set_index("Year")


# -----------------------------
# Plotting helper
# -----------------------------
def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def mark_hf_period(ax, start, end, *, label=None, zorder=10):
    """Mark the HF window using two vertical dashed lines (high-contrast on the heatmap)."""
    ax.axvline(mdates.date2num(start), label=label, zorder=zorder, **HF_MARKER)
    ax.axvline(mdates.date2num(end), zorder=zorder, **HF_MARKER)

def mark_hf_span(ax, start, end, *, label=None, zorder=0):
    """Mark the HF window on standard (non-heatmap) plots using a shaded span."""
    kw = dict(HF_SPAN)
    # Avoid passing zorder twice (matplotlib raises TypeError).
    kw["zorder"] = zorder
    ax.axvspan(start, end, label=label, **kw)





def add_hf_label_in_heatmap(ax, start, end, *, y_period=2.5, x_pos=None, text="HF period"):
    """Add a small dashed guide + label inside the heatmap in data coordinates.

    This avoids a standard legend box, which can be unreadable over high-power regions.
    """
    x0 = mdates.date2num(start)
    x1 = mdates.date2num(end)
    if x_pos is None:
        x_pos = x0 + 0.02 * (x1 - x0)  # slightly inside the HF window
    # short horizontal dashed segment as a key
    seg = 0.10 * (x1 - x0)
    ax.plot([x_pos, x_pos + seg], [y_period, y_period], **HF_MARKER, zorder=15)
    ax.text(x_pos + 1.15 * seg, y_period, text, color="white", va="center", ha="left", zorder=15)

# -----------------------------
# Torrence & Compo-style wavelet significance + cone of influence (approximate)
# -----------------------------
def ar1_lag1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    x0 = x[:-1] - np.mean(x[:-1])
    x1 = x[1:] - np.mean(x[1:])
    denom = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
    if denom == 0:
        return 0.0
    return float(np.sum(x0 * x1) / denom)


def wavelet_significance_ar1(
    x: np.ndarray,
    period_days: np.ndarray,
    dt_days: float = 1.0,
    p: float = 0.95,
    dof: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale-dependent red-noise background and significance threshold for wavelet power.

    Returns
    -------
    signif : ndarray
        Scale-dependent 95% (or p) significance threshold (same units as power).
    background : ndarray
        AR(1) red-noise background spectrum (same units as power).
    alpha : float
        Estimated AR(1) lag-1 coefficient.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    variance = float(np.var(x, ddof=0))
    alpha = ar1_lag1(x)

    omega = 2.0 * np.pi * (dt_days / np.asarray(period_days, dtype=float))
    pk = (1.0 - alpha**2) / (1.0 + alpha**2 - 2.0 * alpha * np.cos(omega))
    background = variance * pk

    from scipy.stats import chi2

    chi2_thr = float(chi2.ppf(p, dof) / dof)
    signif = background * chi2_thr
    return signif, background, alpha


def cone_of_influence_period(
    n: int,
    dt_days: float = 1.0,
    omega0: float = 6.0,
) -> np.ndarray:
    """Cone of influence (COI) as equivalent period (days) vs time index for Morlet."""
    t = np.arange(n, dtype=float)
    dist = np.minimum(t, (n - 1) - t)
    coi_time = dist * dt_days

    fourier_factor = (4.0 * np.pi) / (omega0 + np.sqrt(2.0 + omega0**2))
    s = coi_time / (np.sqrt(2.0) * dt_days)
    return fourier_factor * s


# -----------------------------
# Main workflow
# -----------------------------
def main(
    outdir: Path,
    seed: int = 0,
    progress: bool = False,
    inner_progress: bool = False,
    n_boot: int = 2000,
    block_len: int = 3,
    min_n_bootstrap: int = 8,
    ceemdan_trials: int = 50,
    ceemdan_noise_width: float = 0.2,
    skip_wavelet: bool = False,
):
    outdir.mkdir(parents=True, exist_ok=True)

    if progress:
        log_section("1/6 Generating synthetic daily truth")
    df_daily = generate_synthetic_series(seed=seed)

    if progress:
        log_section("2/6 Building mixed-frequency observed samples")
    obs = build_observed_samples(df_daily)
    flow_obs = build_observed_flow_samples(df_daily)

    if progress:
        log_section("3/6 Computing naive monthly means + CIs")
    monthly_naive = monthly_mean_and_ci(
        obs,
        rng=np.random.default_rng(seed),
        n_boot=n_boot,
        block_len=block_len,
        min_n_bootstrap=min_n_bootstrap,
        progress=progress,
        inner_progress=inner_progress,
        label="naive monthly CI",
    )

    if progress:
        log_section("3b/6 Computing monthly flow means + CIs (for load propagation)")
    monthly_flow = monthly_mean_and_ci(
        flow_obs,
        rng=np.random.default_rng(seed + 101),
        n_boot=n_boot,
        block_len=block_len,
        min_n_bootstrap=min_n_bootstrap,
        progress=progress,
        inner_progress=inner_progress,
        label="flow monthly CI",
    )

    if progress:
        log_section("4/6 Event detection + censoring; censored monthly means + CIs")
    daily_truth = df_daily["true_with_spikes"]
    base_det = rolling_baseline(daily_truth, window_days=31)
    evt_mask = detect_events(daily_truth, base_det, k_mad=6.0)
    daily_censored = censor_series(daily_truth, evt_mask, censor_buffer_days=1)

    df_daily["det_baseline_med31"] = base_det
    df_daily["det_event_mask"] = evt_mask
    df_daily["censored_series"] = daily_censored

    monthly_censored = monthly_mean_and_ci(
        daily_censored,
        rng=np.random.default_rng(seed),
        n_boot=n_boot,
        block_len=block_len,
        min_n_bootstrap=min_n_bootstrap,
        progress=progress,
        inner_progress=inner_progress,
        label="censored monthly CI",
    )

    hf_months = (monthly_naive.index >= HF_START) & (monthly_naive.index <= HF_END)
    bias = monthly_naive["Mean"] - monthly_censored["Mean"]
    bias_hf_mean = float(np.nanmean(bias[hf_months].to_numpy()))

    if progress:
        log_section("5/6 CEEMDAN decomposition + baseline reconstruction")
    imfs = ceemdan_decompose(
        daily_truth.to_numpy(dtype=float),
        seed=seed,
        trials=ceemdan_trials,
        noise_width=ceemdan_noise_width,
        progress=progress,
    )
    baseline_ceemdan = choose_lowfreq_reconstruction(imfs, keep_last=2)
    baseline_ceemdan_s = pd.Series(baseline_ceemdan, index=df_daily.index, name="CEEMDAN_baseline")

    monthly_ceemdan = monthly_mean_and_ci(
        baseline_ceemdan_s,
        rng=np.random.default_rng(seed),
        n_boot=n_boot,
        block_len=block_len,
        min_n_bootstrap=min_n_bootstrap,
        progress=progress,
        inner_progress=inner_progress,
        label="CEEMDAN baseline monthly CI",
    )

    if progress:
        log_section("5b/6 Propagating uncertainty to annual loads (Monte Carlo)")
    annual_load_naive = annual_load_monte_carlo(monthly_naive, monthly_flow, n_mc=5000, seed=seed + 2001)
    annual_load_censored = annual_load_monte_carlo(monthly_censored, monthly_flow, n_mc=5000, seed=seed + 2002)
    annual_load_ceemdan = annual_load_monte_carlo(monthly_ceemdan, monthly_flow, n_mc=5000, seed=seed + 2003)

    if progress:
        log_section("6/6 Wavelet diagnostics (optional), writing CSVs and PNGs")

    # -----------------------------
    # Wavelet diagnostics (optional)
    # -----------------------------
    if not skip_wavelet:
        if progress:
            log_section(
                "Computing improved CWT diagnostics (1998–2007): log-power heatmap, global spectrum, band-power series"
            )

        subset = df_daily.loc["1998":"2007", "true_with_spikes"]
        x = subset.to_numpy(dtype=float)
        t = subset.index

        dt_days = 1.0
        #scales = np.arange(1, 128)
        scales = np.unique(np.logspace(np.log10(1), np.log10(1024), 200).astype(int))        
        wavelet = "morl"

        cwt_coef, _ = pywt.cwt(x, scales, wavelet, sampling_period=dt_days)
        power = np.abs(cwt_coef) ** 2

        freqs = pywt.scale2frequency(wavelet, scales) / dt_days
        period_days = 1.0 / freqs

        log_power = np.log10(power + 1e-12)
        vmin = float(np.nanquantile(log_power, 0.05))
        vmax = float(np.nanquantile(log_power, 0.995))

        years = t.year + (t.dayofyear - 1) / 365.0
        x_num = mdates.date2num(t.to_pydatetime())

        # Red-noise diagnostics (AR1 background + 95% significance)
        signif, background, ar1 = wavelet_significance_ar1(x, period_days, dt_days=dt_days, p=0.95, dof=2.0)
        signif_log = np.log10(signif + 1e-12)[:, None]

        # Optionally mask non-significant power (below 95% red-noise threshold)
        plot_log_power = log_power
        if CWT_MASK_NONSIGNIFICANT:
            plot_log_power = np.where(log_power < signif_log, np.nan, log_power)

        coi_period = cone_of_influence_period(len(t), dt_days=dt_days, omega0=6.0)
        
        # Clamp COI into the plotted/available period range
        coi_period = np.clip(coi_period, period_days.min(), period_days.max())                
        #coi_period = np.maximum(coi_period, period_days.min())

        # (1) Band-integrated power time series
        def bandpower(period_lo: float, period_hi: float) -> np.ndarray:
            band = (period_days >= period_lo) & (period_days <= period_hi)
            if not np.any(band):
                return np.full(power.shape[1], np.nan)
            return np.nanmean(power[band, :], axis=0)

        bp_event = bandpower(1.0, 7.0)
        bp_month = bandpower(20.0, 60.0)

        # (2) Global wavelet spectrum: time-mean power vs period + AR(1) curves
        global_power = np.nanmean(power, axis=1)

        # ---- Faceted CWT figure (heatmap + global spectrum + bandpower series) ----
        # Layout:
        #   [heatmap] [global spectrum]
        #   [bandpower time series    ]
        fig = plt.figure(figsize=(13.8, 7.2))
        gs = GridSpec(nrows=2, ncols=2, figure=fig, width_ratios=[5.0, 1.35], height_ratios=[3.0, 1.25])
        ax_hm = fig.add_subplot(gs[0, 0])
        ax_gs = fig.add_subplot(gs[0, 1], sharey=ax_hm)
        ax_bp = fig.add_subplot(gs[1, 0], sharex=ax_hm)

        # Heatmap
        cmap = plt.get_cmap(CWT_CMAP).copy() if hasattr(plt.get_cmap(CWT_CMAP), "copy") else plt.get_cmap(CWT_CMAP)
        try:
            cmap.set_bad(CWT_MASK_BAD_RGBA)
        except Exception:
            pass
        pcm = ax_hm.pcolormesh(
            x_num,
            period_days,
            plot_log_power,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        X, Y = np.meshgrid(x_num, period_days)
        ax_hm.contour(X, Y, log_power - signif_log, levels=[0.0], linewidths=0.9)
        mark_hf_period(ax_hm, HF_START, HF_END, label=None)
        # Period guide lines
        for _p in PERIOD_GUIDES_DAYS:
            ax_hm.axhline(_p, **PERIOD_GUIDE_STYLE)
        # Place HF label inside the heatmap around 2–3 days
        add_hf_label_in_heatmap(ax_hm, HF_START, HF_END, y_period=2.5, text="HF period")

        # Optional event-day guide lines (very light)
        if "det_event_mask" in df_daily.columns:
            evt = df_daily.loc["1998":"2007", "det_event_mask"]
            evt_times = evt.index[evt.values.astype(bool)]
            evt_x = mdates.date2num(evt_times.to_pydatetime())
            for xx in evt_x:
                ax_hm.axvline(float(xx), lw=0.25, alpha=0.18)

        ax_hm.plot(x_num, coi_period, color="white", lw=1.0)
        ax_hm.fill_between(x_num, coi_period, period_days.max(), color="white", alpha=0.35)
        ax_hm.set_yscale("log")
        ax_hm.invert_yaxis()
        ax_hm.set_ylabel("Period (days) [log scale]")
        ax_hm.xaxis.set_major_locator(mdates.YearLocator(1))
        ax_hm.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_hm.set_xlabel("Year")
        ax_hm.set_title(f"CWT wavelet log-power (1998–2007) with 95% red-noise significance; AR1={ar1:.2f}")
        # Global spectrum panel (same y-axis as heatmap)
        ax_gs.plot(global_power, period_days, lw=1.3)
        ax_gs.plot(background, period_days, lw=1.0, ls="--", alpha=0.95)
        ax_gs.plot(signif, period_days, lw=1.0, ls=":", alpha=0.95)
        for _p in PERIOD_GUIDES_DAYS:
            ax_gs.axhline(_p, **PERIOD_GUIDE_STYLE)
        ax_gs.set_yscale("log")
        ax_gs.invert_yaxis()
        ax_gs.set_xlabel("Mean power")
        ax_gs.set_title("Global\nspectrum")
        ax_gs.grid(True, alpha=0.25)
        # Keep the shared y formatting but hide duplicate labels
        plt.setp(ax_gs.get_yticklabels(), visible=False)
        ax_gs.tick_params(axis="y", which="both", length=0)
        ax_gs.legend(["Mean", f"AR(1) bg (a={ar1:.2f})", "95% thr"], loc="lower right", frameon=False)

        # Bandpower time series (aligned to heatmap x-axis)
        ax_bp.plot(x_num, bp_event, color=COL["band_event"], label="Bandpower 1–7 days (event)")
        ax_bp.plot(x_num, bp_month, color=COL["band_baseline"], label="Bandpower 20–60 days (baseline)")
        mark_hf_span(ax_bp, HF_START, HF_END, label="HF period")
        ax_bp.set_ylabel("Mean wavelet power")
        ax_bp.set_xlabel("Date")
        ax_bp.set_title("Band-integrated wavelet power time series")
        ax_bp.xaxis.set_major_locator(mdates.YearLocator(1))
        ax_bp.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_bp.legend(loc="upper left")

        # Inset horizontal colourbar on top of the heatmap (short, high-contrast)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax_hm, width=CWT_COLORBAR["width"], height=CWT_COLORBAR["height"],
                         loc=CWT_COLORBAR["loc"], borderpad=CWT_COLORBAR["borderpad"])
        cbar = fig.colorbar(pcm, cax=cax, orientation="horizontal")
        cbar.set_label(CWT_COLORBAR["label"], color=CWT_COLORBAR["label_color"])
        cbar.ax.tick_params(axis="x", colors=CWT_COLORBAR["tick_color"], labelcolor=CWT_COLORBAR["tick_color"])
        cbar.outline.set_edgecolor(CWT_COLORBAR["outline_color"])

        # Save facet figure and also retain the original individual panels for convenience
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(outdir / "cwt_facet_1998_2007.png", dpi=300)
        plt.close(fig)

    else:
        if progress:
            log_section("Skipping CWT diagnostics (--skip-wavelet)")

    # ------------------ CSV outputs ------------------
    df_daily_out = df_daily.copy()
    df_daily_out["ceemdan_baseline"] = baseline_ceemdan_s
    df_daily_out.to_csv(outdir / "daily_truth_and_diagnostics.csv", float_format="%.6f")

    obs.to_frame().to_csv(outdir / "raw_observed_samples.csv", float_format="%.6f")
    flow_obs.to_frame().to_csv(outdir / "raw_observed_flow_samples.csv", float_format="%.6f")

    monthly_naive.to_csv(outdir / "monthly_naive_with_ci.csv", float_format="%.6f")
    monthly_censored.to_csv(outdir / "monthly_censored_with_ci.csv", float_format="%.6f")
    monthly_ceemdan.to_csv(outdir / "monthly_ceemdan_baseline_with_ci.csv", float_format="%.6f")
    monthly_flow.to_csv(outdir / "monthly_flow_with_ci.csv", float_format="%.6f")

    annual_load_naive.to_csv(outdir / "annual_load_naive_mc.csv", float_format="%.6f")
    annual_load_censored.to_csv(outdir / "annual_load_censored_mc.csv", float_format="%.6f")
    annual_load_ceemdan.to_csv(outdir / "annual_load_ceemdan_mc.csv", float_format="%.6f")

    evt_df = pd.DataFrame({"event": evt_mask.astype(int)}, index=evt_mask.index)
    evt_df["month"] = evt_df.index.to_period("M")
    evt_summary = evt_df.groupby("month")["event"].sum().to_frame("n_event_days")
    evt_summary.to_csv(outdir / "event_days_by_month.csv")

    bias_df = pd.DataFrame(
        {
            "naive_mean": monthly_naive["Mean"],
            "censored_mean": monthly_censored["Mean"],
            "bias_naive_minus_censored": bias,
        }
    )
    bias_df.to_csv(outdir / "monthly_bias_naive_minus_censored.csv", float_format="%.6f")

    # ------------------ PNG outputs ------------------
    # 1) Raw observed vs monthly
    plt.figure(figsize=(11, 6))
    obs_before = obs[obs.index < "2000-01-01"]
    obs_hf = obs[(obs.index >= "2000-01-01") & (obs.index <= "2005-12-31")]
    obs_after = obs[obs.index > "2005-12-31"]

    plt.plot(
        obs_before.index,
        obs_before.values,
        "o",
        ms=2.5,
        color=COL["obs_weekly"],
        label="Observed (weekly)",
    )
    plt.plot(
        obs_hf.index,
        obs_hf.values,
        "-",
        lw=0.4,
        color=COL["obs_daily"],
        label="Observed (daily 2000–2005)",
    )
    plt.plot(
        obs_after.index,
        obs_after.values,
        "o",
        ms=2.5,
        color=COL["obs_weekly"],
        label="Observed (weekly)",
    )

    plt.plot(
        monthly_naive.index,
        monthly_naive["Mean"],
        "-",
        lw=1.2,
        color=COL["naive"],
        label="Monthly naive mean",
    )
    plt.plot(
        monthly_censored.index,
        monthly_censored["Mean"],
        "--",
        lw=1.2,
        color=COL["censored"],
        label="Monthly censored mean",
    )

    plt.axvspan(HF_START, HF_END, alpha=0.12, label="HF period")
    plt.xlabel("Date")
    plt.ylabel("Total N (synthetic units)")
    plt.title("Raw observations vs monthly resampling (naive vs event-censored)")
    plt.legend(loc="best")
    save_fig(outdir / "raw_vs_monthly_naive_vs_censored.png")

    # 2) Monthly means with CIs (match wavelet period: 1998–2007) + CEEMDAN baseline
    def _linear_trend(y: pd.Series) -> pd.Series:
        yy = y.dropna()
        if len(yy) < 3:
            return y * np.nan
        x = yy.index.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
        b1, b0 = np.polyfit(x, yy.to_numpy(dtype=float), 1)
        x_all = y.index.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
        return pd.Series(b1 * x_all + b0, index=y.index)

    plt.figure(figsize=(11, 5))
    window = slice("1998", "2007")
    m = monthly_naive.loc[window]
    mc = monthly_censored.loc[window]
    mce = monthly_ceemdan.loc[window]

    plt.plot(m.index, m["Mean"], "-", lw=1.35, color=COL["naive"], label="Naive mean")
    plt.fill_between(m.index, m["CI_Lo"], m["CI_Hi"], alpha=0.16, color=COL["naive"], label="Naive 95% CI")

    plt.plot(mc.index, mc["Mean"], "--", lw=1.35, color=COL["censored"], label="Censored mean")
    plt.fill_between(
        mc.index,
        mc["CI_Lo"],
        mc["CI_Hi"],
        alpha=0.16,
        color=COL["censored"],
        label="Censored 95% CI",
    )

    plt.plot(
        mce.index,
        mce["Mean"],
        "-.",
        lw=1.25,
        color=COL["ceemdan"],
        label="CEEMDAN baseline mean",
    )

    # Comparable trend lines (OLS) for naive vs censored
    plt.plot(m.index, _linear_trend(m["Mean"]), ":", lw=1.1, color=COL["naive"], alpha=0.9, label="Naive trend")
    plt.plot(
        mc.index,
        _linear_trend(mc["Mean"]),
        ":",
        lw=1.1,
        color=COL["censored"],
        alpha=0.9,
        label="Censored trend",
    )

    plt.axvspan(HF_START, HF_END, alpha=0.10, label="HF period")
    plt.xlabel("Date")
    plt.ylabel("Total N (synthetic units)")
    plt.title(f"Monthly means with confidence intervals (1998–2007; HF bias mean={bias_hf_mean:.3f})")
    plt.legend(loc="best", ncol=2)
    save_fig(outdir / "monthly_means_with_ci_1998_2007.png")

    # 3) Event detection diagnostic (full HF period)
    sel = df_daily.loc[HF_START:HF_END]
    plt.figure(figsize=(11, 5))
    plt.plot(
        sel.index,
        sel["true_with_spikes"],
        lw=0.7,
        color="0.25",
        label="Daily truth (with spikes)",
    )
    plt.plot(
        sel.index,
        sel["det_baseline_med31"],
        lw=1.2,
        color=COL["baseline"],
        label="Rolling median baseline (31d)",
    )
    plt.scatter(
        sel.index[sel["det_event_mask"]],
        sel.loc[sel["det_event_mask"], "true_with_spikes"],
        s=10,
        color=COL["events"],
        label="Detected events",
    )
    plt.axvspan(HF_START, HF_END, alpha=0.08, label="HF period")
    plt.xlabel("Date")
    plt.ylabel("Total N (synthetic units)")
    plt.title("Event detection (2000–2005)")
    plt.legend(loc="best")
    save_fig(outdir / "event_detection_2000_2005.png")

    # 4) CEEMDAN IMFs (first 6) + reconstruction
    n_plot = min(6, imfs.shape[0])
    fig, axes = plt.subplots(n_plot + 2, 1, figsize=(11, 2.2 * (n_plot + 2)), sharex=True)
    axes[0].plot(df_daily.index, daily_truth.values, lw=0.6)
    axes[0].set_title("Daily truth")
    for i in range(n_plot):
        axes[i + 1].plot(df_daily.index, imfs[i], lw=0.6)
        axes[i + 1].set_title(f"CEEMDAN IMF {i + 1}")
    axes[n_plot + 1].plot(df_daily.index, baseline_ceemdan, lw=0.8)
    axes[n_plot + 1].set_title("Low-frequency reconstruction (sum of last 2 IMFs)")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.suptitle("CEEMDAN decomposition", y=0.995)
    plt.tight_layout()
    fig.savefig(outdir / "ceemdan_imfs_and_baseline.png", dpi=200)
    plt.close(fig)

    # 5) Compare monthly baselines
    plt.figure(figsize=(11, 5))
    window = slice("1996", "2008")
    plt.plot(
        monthly_naive.loc[window].index,
        monthly_naive.loc[window, "Mean"],
        lw=1.2,
        color=COL["naive"],
        label="Naive monthly mean",
    )
    plt.plot(
        monthly_censored.loc[window].index,
        monthly_censored.loc[window, "Mean"],
        lw=1.2,
        ls="--",
        color=COL["censored"],
        label="Event-censored monthly mean",
    )
    plt.plot(
        monthly_ceemdan.loc[window].index,
        monthly_ceemdan.loc[window, "Mean"],
        lw=1.2,
        ls="-.",
        color=COL["ceemdan"],
        label="CEEMDAN baseline monthly mean",
    )
    plt.axvspan(HF_START, HF_END, alpha=0.10, label="HF period")
    plt.xlabel("Date")
    plt.ylabel("Total N (synthetic units)")
    plt.title("Monthly resampling approaches comparison")
    plt.legend(loc="best")
    save_fig(outdir / "monthly_comparison_naive_censored_ceemdan.png")

    log_section(f"Done. Outputs written to: {outdir}")
    log_section(f"Mean HF-period bias (naive - censored): {bias_hf_mean:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory for CSV/PNG files")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument("--progress", action="store_true", help="Show section timestamps and tqdm bars (if installed)")
    parser.add_argument("--inner-progress", action="store_true", help="Show inner bootstrap bars (very verbose)")

    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap replicates for monthly CI (default 2000)")
    parser.add_argument("--block-len", type=int, default=3, help="Moving-block bootstrap block length in days (default 3)")
    parser.add_argument("--min-n-bootstrap", type=int, default=8, help="Min n in month to use bootstrap CI (default 8)")

    parser.add_argument("--ceemdan-trials", type=int, default=50, help="CEEMDAN ensemble trials (default 50)")
    parser.add_argument("--ceemdan-noise-width", type=float, default=0.2, help="CEEMDAN noise width (default 0.2)")

    parser.add_argument("--skip-wavelet", action="store_true", help="Skip CWT wavelet spectrum (faster)")

    args = parser.parse_args()
    main(
        Path(args.outdir),
        seed=args.seed,
        progress=args.progress,
        inner_progress=args.inner_progress,
        n_boot=args.n_boot,
        block_len=args.block_len,
        min_n_bootstrap=args.min_n_bootstrap,
        ceemdan_trials=args.ceemdan_trials,
        ceemdan_noise_width=args.ceemdan_noise_width,
        skip_wavelet=args.skip_wavelet,
    )
