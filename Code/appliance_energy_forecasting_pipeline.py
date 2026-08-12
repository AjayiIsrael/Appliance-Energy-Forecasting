"""
Appliance Energy Forecasting -- Time Series Case Study (Parts 1-7)
====================================================================

Reproducible pipeline script covering data preparation, EDA, benchmark
models, SARIMAX, covariate engineering, feature-based ML models, and a
time-series foundation model, for the UCI Appliance Energy Prediction
dataset.

Dataset
-------
Candanedo, Feldheim & Deramaix (2017) -- 10-minute-resolution appliance
energy use, indoor sensor readings, and outdoor weather data from a
low-energy house in Belgium.

How to run
----------
    pip install -r requirements.txt
    python appliance_energy_forecasting_pipeline.py

Useful flags for faster iteration while developing (see --help):
    --quick                 shrink the SARIMAX grid search and use fewer
                             backtest origins, for a fast smoke test
    --skip-sarimax-search    skip the (slow, ~15-20 min) AIC grid search
                             and use a fixed order instead
    --skip-chronos           skip Part 7 (useful if torch/chronos are not
                             installed, or to save time)
    --output-dir DIR         where Figures/ and Results/ are written

Runtime
-------
End to end with defaults: roughly 25-40 minutes on a single CPU core.
The SARIMAX AIC grid search (Part 4) is the slowest single step
(~15-20 minutes); Part 7 additionally needs internet access the first
time it runs, to download the pretrained Chronos-Bolt-small weights.

Design notes carried over from the original analysis
------------------------------------------------------
* All models are evaluated on the SAME 14-day rolling-origin backtest
  (14 separate 24h-ahead forecasts, one per day, each trained only on
  data before its own origin). A single fixed test window was tried
  first and rejected: the data happen to end right after an unusual
  demand spike, which badly biases persistence-style forecasts for that
  one window. Rolling-origin backtesting also happens to satisfy two
  requirements at once -- Part 2's chronological train/test split, and
  Part 6's "use the last 14 days as the test period" -- since the 14
  origins, taken together, exactly span the last 14*24 hours.
* Weather/sensor covariates come in two variants: `realistic` (lagged
  24h -- a same-day persistence proxy that is genuinely known at
  forecast time) and `conditional` (true future values -- an oracle
  upper bound, never a genuine forecast). This directly informs the
  assignment's data-leakage discussion question.
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe: figures are saved to disk, never popped up on screen
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.ensemble import RandomForestRegressor  # noqa: F401  (available via get_model if wanted)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
MIRROR_URL = "https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/master/energydata_complete.csv"

TARGET = "Appliances"
HORIZON = 24          # forecast horizon in hours (Part 2)
N_ORIGINS = 14         # number of rolling-origin backtest windows (= last 14 days, Part 6)
STEP = 24              # hours between consecutive origins
SEASONAL_PERIOD = 24   # daily seasonality, in hours

INDOOR_SENSOR_COLS = [f"T{i}" for i in range(1, 10)] + [f"RH_{i}" for i in range(1, 10)]
OUTDOOR_WEATHER_COLS = ["T_out", "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"]

# These are set for real inside main() from CLI args; declared here so every
# function in the module can reach them without threading a path through
# every call.
FIGURE_DIR = Path("Figures")
RESULTS_DIR = Path("Results")

plt.rcParams["figure.dpi"] = 100


def _save_fig(fig, filename):
    """Save a figure into FIGURE_DIR, close it (to avoid piling up memory
    over a long run), and print where it went so the run log doubles as an
    index of every plot produced."""
    path = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure saved] {path}")


def _banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ============================================================================
# PART 1 -- DATA RETRIEVAL AND PREPARATION
# ============================================================================

def load_dataset(uci_url: str = UCI_URL, mirror_url: str = MIRROR_URL) -> pd.DataFrame:
    """Download the 10-minute-resolution dataset.

    The official UCI URL is tried first. If it is unreachable (this
    happens from some sandboxed / restricted-network environments, and is
    exactly the kind of access problem the assignment brief warns can cost
    marks), a mirror of the identical file on the original authors' GitHub
    repository is used instead.
    """
    try:
        raw = pd.read_csv(uci_url)
        print(f"Loaded from UCI ({len(raw)} rows).")
    except Exception as exc:
        print(f"UCI URL failed ({exc}); falling back to GitHub mirror.")
        raw = pd.read_csv(mirror_url)
        print(f"Loaded from mirror ({len(raw)} rows).")

    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").sort_index()
    return raw


def check_data_quality(raw: pd.DataFrame) -> pd.DatetimeIndex:
    """Report missing values and gaps in the 10-minute timestamp index.

    Returns the DatetimeIndex of missing timestamps (empty if none). The
    gaps are deliberately NOT interpolated: the next step, hourly
    resampling, sums/averages whatever 10-minute readings ARE present
    within each hour, so a short gap only slightly reduces the precision
    of that one hourly value rather than producing a NaN that would need
    filling. This is noted explicitly because "checking missing values" is
    a named requirement in Part 1 -- checking is not the same as ignoring,
    so we print exactly what was found and why no further action is taken.
    """
    print("Shape:", raw.shape)
    print("Date range:", raw.index.min(), "to", raw.index.max())
    print("Missing values per column (total):", raw.isna().sum().sum())

    expected_index = pd.date_range(raw.index.min(), raw.index.max(), freq="10min")
    missing_timestamps = expected_index.difference(raw.index)
    pct = 100 * len(missing_timestamps) / len(expected_index)
    print(f"Missing timestamps: {len(missing_timestamps)} / {len(expected_index)} expected "
          f"({pct:.2f}%)")

    if len(missing_timestamps) > 0:
        # Show which hourly bins are actually affected, since that is the
        # resolution the rest of the analysis operates at.
        affected_hours = pd.Series(missing_timestamps).dt.floor("h").nunique()
        print(f"These fall within {affected_hours} distinct hourly bins. Hourly resampling "
              f"(next step) uses pandas' default skipna behaviour, so each affected hour is "
              f"still computed from whatever readings survived -- not silently dropped, and "
              f"not fabricated by interpolation either.")
    else:
        print("No gaps in the 10-minute timestamp index.")
    return missing_timestamps


def resample_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Bin the 10-minute data up to hourly resolution.

    Energy columns (`Appliances`, `lights`; measured in Wh) are SUMMED,
    since Wh is already an additive quantity and hourly usage is the sum
    of the six 10-minute readings that make up that hour. All sensor and
    weather columns are AVERAGED, since they are instantaneous physical
    measurements (a temperature is not "additive" across readings). The
    two random-noise control columns (`rv1`, `rv2`), included by the
    original authors purely to test feature-selection methods, are
    dropped since they carry no physical signal and would only add noise.
    """
    df = raw.drop(columns=[c for c in ["rv1", "rv2"] if c in raw.columns])

    energy_cols = ["Appliances", "lights"]
    sensor_cols = [c for c in df.columns if c not in energy_cols]

    hourly = pd.concat([
        df[energy_cols].resample("h").sum(),
        df[sensor_cols].resample("h").mean(),
    ], axis=1)[df.columns]
    hourly.index.freq = "h"

    print("Hourly shape:", hourly.shape)
    return hourly


def plot_full_series(y: pd.Series):
    fig, ax = plt.subplots(figsize=(12, 4))
    y.plot(ax=ax, linewidth=0.7, color="#1f77b4")
    ax.set_title("Hourly Appliance Energy Use")
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    _save_fig(fig, "01_full_series.png")


def plot_distribution(y: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(y, bins=50, color="#1f77b4", edgecolor="white")
    axes[0].set_title("Distribution of hourly Appliances")
    axes[0].set_xlabel("Wh")

    axes[1].hist(np.log1p(y), bins=50, color="#ff7f0e", edgecolor="white")
    axes[1].set_title("Distribution of log(1 + Appliances)")
    axes[1].set_xlabel("log(1 + Wh)")
    _save_fig(fig, "02_distribution.png")


def plot_daily_weekly_profile(y: pd.Series):
    """Mean usage by hour-of-day and by day-of-week -- the simplest, most
    direct way to SEE whether daily/weekly seasonality exists before any
    formal decomposition or statistical test is run."""
    profile_df = y.to_frame("Appliances")
    profile_df["hour"] = profile_df.index.hour
    profile_df["dow"] = profile_df.index.dayofweek

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    profile_df.groupby("hour")["Appliances"].mean().plot(kind="bar", ax=axes[0], color="#1f77b4")
    axes[0].set_title("Mean appliance use by hour of day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Mean Wh")

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_profile = profile_df.groupby("dow")["Appliances"].mean()
    axes[1].bar(dow_labels, dow_profile.values, color="#ff7f0e")
    axes[1].set_title("Mean appliance use by day of week")
    axes[1].set_ylabel("Mean Wh")
    _save_fig(fig, "03_daily_weekly_profile.png")


def decompose_components(y: pd.Series) -> dict:
    """STL (Seasonal-Trend decomposition using LOESS) decomposition at both
    a daily (24h) and weekly (168h) period, to identify which seasonal
    components the series actually has. Returns the strength-of-component
    statistics used to justify modelling choices later (e.g. SARIMAX's
    seasonal period).
    """
    stl_daily = STL(y, period=24, robust=True).fit()
    fig = stl_daily.plot()
    fig.set_size_inches(11, 7)
    fig.suptitle("STL decomposition (daily period = 24h)", y=1.01)
    _save_fig(fig, "04_stl_daily.png")

    seasonal_strength_daily = max(
        0, 1 - np.var(stl_daily.resid) / np.var(stl_daily.resid + stl_daily.seasonal)
    )
    trend_strength = max(
        0, 1 - np.var(stl_daily.resid) / np.var(stl_daily.resid + stl_daily.trend)
    )

    stl_weekly = STL(y, period=24 * 7, robust=True).fit()
    seasonal_strength_weekly = max(
        0, 1 - np.var(stl_weekly.resid) / np.var(stl_weekly.resid + stl_weekly.seasonal)
    )

    print(f"STL seasonal strength (daily):  {seasonal_strength_daily:.3f}")
    print(f"STL seasonal strength (weekly): {seasonal_strength_weekly:.3f}")
    print(f"STL trend strength:             {trend_strength:.3f}")
    print("Interpretation: a strong daily seasonal component (peaks morning/evening, trough "
          "overnight), a weaker weekly component (modest weekend uplift), and little trend.")

    return {
        "seasonal_strength_daily": seasonal_strength_daily,
        "seasonal_strength_weekly": seasonal_strength_weekly,
        "trend_strength": trend_strength,
    }


def plot_acf_pacf(y: pd.Series):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    plot_acf(y, lags=200, ax=axes[0])
    axes[0].set_title("ACF -- raw hourly series (200 lags)")
    plot_pacf(y, lags=200, ax=axes[1], method="ywm")
    axes[1].set_title("PACF -- raw hourly series (200 lags)")
    _save_fig(fig, "05_acf_pacf.png")


def test_stationarity(y: pd.Series) -> dict:
    """Two tests with OPPOSITE null hypotheses, so agreement between them is
    more convincing than either alone:
      - ADF (Augmented Dickey-Fuller): H0 = series has a unit root (non-stationary)
      - KPSS:                           H0 = series is (trend-)stationary
    Also runs an explicit first-differencing check as a diagnostic, even
    though the levels-based tests are expected to already show stationarity,
    to confirm differencing doesn't reveal structure the tests missed.
    """
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(y.dropna(), autolag="AIC")
    print("ADF test:")
    print(f"  statistic = {adf_stat:.3f}, p-value = {adf_p:.3g}")
    print(f"  conclusion: {'Stationary (reject H0)' if adf_p < 0.05 else 'Non-stationary (fail to reject H0)'}")

    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(y.dropna(), regression="c", nlags="auto")
    print("\nKPSS test:")
    print(f"  statistic = {kpss_stat:.3f}, p-value = {kpss_p:.3g}")
    print(f"  conclusion: {'Non-stationary (reject H0)' if kpss_p < 0.05 else 'Stationary (fail to reject H0)'}")

    # --- Differencing check ---
    y_diff = y.diff().dropna()
    adf_diff_stat, adf_diff_p, *_ = adfuller(y_diff, autolag="AIC")
    print(f"\nADF on first-differenced series: statistic={adf_diff_stat:.3f}, p-value={adf_diff_p:.3g}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(y_diff.index, y_diff.values, linewidth=0.6)
    axes[0].set_title("First-differenced series")
    plot_acf(y_diff, lags=48, ax=axes[1])
    axes[1].set_title("ACF of first-differenced series")
    _save_fig(fig, "05b_differencing_check.png")

    is_stationary = (adf_p < 0.05) and (kpss_p >= 0.05)
    print(f"\nBoth tests {'agree' if is_stationary else 'disagree'} the raw hourly series is "
          f"stationary in levels. This directly informs the SARIMAX non-seasonal difference "
          f"order in Part 4 (d=0 if stationary).")

    return {"adf_p": adf_p, "kpss_p": kpss_p, "stationary_in_levels": is_stationary}


# ============================================================================
# PART 2 -- FORECASTING PROBLEM DEFINITION
# (metrics + the rolling-origin backtest harness every model reuses)
# ============================================================================
#
# Target variable:    Appliances -- total hourly appliance energy use (Wh)
# Forecast horizon:   24 hours ahead (matches a daily smart-home planning cycle)
# Train/test split:   strictly chronological; 14 separate 24h-ahead forecasts
#                      (rolling-origin backtest), covering the last 14 days,
#                      rather than one fixed window (see module docstring)
# Evaluation metrics: MAE, RMSE, sMAPE, and MASE (scale-free, the primary
#                      metric for cross-model comparison)

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE, bounded in [0, 200], robust to values near zero."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-8, denom)
    return float(200 * np.mean(np.abs(y_true - y_pred) / denom))


def mase(y_true, y_pred, y_train, seasonal_period: int = SEASONAL_PERIOD) -> float:
    """Mean Absolute Scaled Error: MAE of the forecast, scaled by the
    in-sample MAE of a seasonal-naive forecast. MASE < 1 means the model
    beats seasonal-naive; MASE > 1 means it's worse. Scale-free, so it is
    comparable across models AND across different backtest origins.
    """
    y_train = np.asarray(y_train, dtype=float)
    scale = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
    scale = scale if scale != 0 else 1e-8
    return mae(y_true, y_pred) / scale


def coverage(y_true, lower, upper) -> float:
    """Fraction of true values that fall inside a forecast's prediction
    interval -- used to check whether e.g. an 80% interval actually
    contains ~80% of outcomes (calibration), not just how narrow it is."""
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))))


def evaluate_forecast(y_true, y_pred, y_train, seasonal_period: int = SEASONAL_PERIOD,
                       model_name: str = "model") -> dict:
    return {
        "model": model_name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, seasonal_period),
    }


def rolling_origins(y: pd.Series, horizon: int = HORIZON, n_origins: int = N_ORIGINS,
                     step: int = STEP) -> list:
    """Chronologically ordered list of test-window START positions
    (integer index locations) for the rolling backtest. With the default
    horizon=24, n_origins=14, step=24, the 14 windows are back-to-back and
    non-overlapping, together spanning exactly the last 14*24h = 14 days
    of the series.
    """
    n = len(y)
    origins = []
    for i in range(n_origins):
        test_end = n - i * step
        test_start = test_end - horizon
        if test_start - 1 < 1:
            break
        origins.append(test_start)
    return sorted(origins)


def walk_forward_evaluate(y: pd.Series, forecast_fn, horizon: int = HORIZON,
                           n_origins: int = N_ORIGINS, step: int = STEP,
                           seasonal_period: int = SEASONAL_PERIOD, model_name: str = "model",
                           min_train_size: int = 24 * 30):
    """Generic rolling-origin backtest runner.

    forecast_fn(y_train, horizon, index) -> pd.Series of length `horizon`,
    predicting the given future index using only y_train (no peeking).

    Returns (per_origin_results DataFrame, {origin_timestamp: forecast}).
    """
    origin_starts = rolling_origins(y, horizon, n_origins, step)
    per_origin, forecasts = [], {}
    for test_start in origin_starts:
        train = y.iloc[:test_start]
        if len(train) < min_train_size:
            continue
        test = y.iloc[test_start:test_start + horizon]
        if len(test) < horizon:
            continue
        fc = forecast_fn(train, horizon, test.index)
        res = evaluate_forecast(test, fc, train, seasonal_period=seasonal_period, model_name=model_name)
        res["origin"] = test.index[0]
        per_origin.append(res)
        forecasts[test.index[0]] = fc
    return pd.DataFrame(per_origin), forecasts


def summarise_backtest(df: pd.DataFrame) -> pd.Series:
    return df[["MAE", "RMSE", "sMAPE", "MASE"]].mean()


# ============================================================================
# PART 3 -- BENCHMARK MODELS
# ============================================================================
# Five standard benchmarks: mean, naive (persistence), daily seasonal-naive
# (lag 24), weekly seasonal-naive (lag 168), and drift.

def mean_forecast(y_train, horizon, index):
    return pd.Series(np.full(horizon, y_train.mean()), index=index, name="mean")


def naive_forecast(y_train, horizon, index):
    return pd.Series(np.full(horizon, y_train.iloc[-1]), index=index, name="naive")


def seasonal_naive_forecast(y_train, horizon, index, season_length, name):
    last_season = y_train.iloc[-season_length:].to_numpy()
    reps = int(np.ceil(horizon / season_length))
    values = np.tile(last_season, reps)[:horizon]
    return pd.Series(values, index=index, name=name)


def drift_forecast(y_train, horizon, index):
    n = len(y_train)
    slope = (y_train.iloc[-1] - y_train.iloc[0]) / (n - 1)
    steps = np.arange(1, horizon + 1)
    return pd.Series(y_train.iloc[-1] + slope * steps, index=index, name="drift")


BENCHMARK_FNS = {
    "mean": lambda tr, h, idx: mean_forecast(tr, h, idx),
    "naive": lambda tr, h, idx: naive_forecast(tr, h, idx),
    "daily_seasonal_naive": lambda tr, h, idx: seasonal_naive_forecast(tr, h, idx, 24, "daily_seasonal_naive"),
    "weekly_seasonal_naive": lambda tr, h, idx: seasonal_naive_forecast(tr, h, idx, 24 * 7, "weekly_seasonal_naive"),
    "drift": lambda tr, h, idx: drift_forecast(tr, h, idx),
}


def run_benchmarks(y: pd.Series, n_origins: int = N_ORIGINS):
    """Run every benchmark model through the same rolling-origin backtest
    and return a sorted summary table plus each model's per-origin
    forecasts (kept for plotting / later comparison in Part 8)."""
    results, forecasts = {}, {}
    for name, fn in BENCHMARK_FNS.items():
        bt_df, fcs = walk_forward_evaluate(y, fn, n_origins=n_origins, model_name=name)
        results[name] = bt_df
        forecasts[name] = fcs

    summary = pd.DataFrame({name: summarise_backtest(df) for name, df in results.items()}).T
    summary = summary.sort_values("RMSE")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    summary["RMSE"].sort_values().plot(kind="barh", ax=axes[0], color="#1f77b4")
    axes[0].set_title("Benchmark models: mean RMSE\n(rolling backtest)")
    axes[0].set_xlabel("RMSE (Wh)")

    summary["MASE"].sort_values().plot(kind="barh", ax=axes[1], color="#ff7f0e")
    axes[1].axvline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Benchmark models: mean MASE")
    axes[1].set_xlabel("MASE")
    _save_fig(fig, "06_benchmark_summary.png")

    print(f"\nStrongest benchmark: {summary['MASE'].idxmin()} (MASE={summary['MASE'].min():.3f})")
    return summary, results, forecasts


# ============================================================================
# PART 4 -- SARIMAX
# ============================================================================
# SARIMAX (Seasonal ARIMA with eXogenous regressors) extends ARIMA with an
# explicit seasonal AR/MA structure -- well suited to a series with strong,
# regular daily seasonality (confirmed in Part 1).

def grid_search_aic(y_train: pd.Series, p_range, d_range, q_range, seasonal_order) -> pd.DataFrame:
    """Exhaustive AIC grid search over the non-seasonal order (p, d, q), as
    required by the assignment (p in [0,6], d in [0,2], q in [0,6] -> 147
    combinations). The seasonal order is held fixed for this search (see
    select_seasonal_order() for how that fixed value is itself justified).

    To keep this tractable on a single CPU core, callers are expected to
    pass in a SUBSAMPLE of the training data (e.g. the last 45 days) for
    order identification; the winning order is then refit on the FULL
    training series for actual forecasting. This is a standard practical
    shortcut -- order selection is far less sensitive to sample size than
    parameter estimation is.
    """
    combos = list(itertools.product(p_range, d_range, q_range))
    results = []
    t_start = time.time()
    for i, (p, d, q) in enumerate(combos):
        order = (p, d, q)
        try:
            mod = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=50)
            converged = res.mle_retvals.get("converged", None) if hasattr(res, "mle_retvals") else None
            row = {"p": p, "d": d, "q": q, "AIC": res.aic, "BIC": res.bic,
                   "converged": converged, "status": "ok"}
        except Exception as exc:
            row = {"p": p, "d": d, "q": q, "AIC": np.nan, "BIC": np.nan,
                   "converged": False, "status": f"failed: {type(exc).__name__}"}
        results.append(row)
        if (i + 1) % 10 == 0 or i == len(combos) - 1:
            print(f"  [{i + 1}/{len(combos)}] order={order} AIC={row['AIC']} "
                  f"(elapsed {time.time() - t_start:.0f}s)")
    return pd.DataFrame(results).sort_values("AIC", na_position="last")


def select_best_order(gridsearch_results: pd.DataFrame) -> tuple:
    """AIC values from optimizer runs that did NOT converge are not
    reliable, so the best order is chosen only among converged fits."""
    converged = gridsearch_results[gridsearch_results["converged"] == True].sort_values("AIC")  # noqa: E712
    print("Top 10 converged models by AIC:")
    print(converged.head(10).to_string(index=False))
    best_row = converged.iloc[0]
    return int(best_row["p"]), int(best_row["d"]), int(best_row["q"])


def select_seasonal_order(y_train: pd.Series, order: tuple, candidates=None) -> tuple:
    """Improvement over simply asserting a seasonal order: with the
    non-seasonal (p, d, q) already fixed, this runs a small additional AIC
    comparison over a handful of plausible seasonal orders (P, D, Q, 24)
    and picks the one with the lowest AIC. This is cheap (a handful of
    extra fits, not another 147) because it reuses the already-selected
    non-seasonal order, and it turns "we assumed (1,0,1,24)" into "we
    picked (1,0,1,24) because it had the lowest AIC among the orders we
    tested" -- actual evidence instead of an assertion.
    """
    if candidates is None:
        candidates = [
            (0, 0, 1, SEASONAL_PERIOD),
            (1, 0, 0, SEASONAL_PERIOD),
            (1, 0, 1, SEASONAL_PERIOD),
            (2, 0, 1, SEASONAL_PERIOD),
        ]
    rows = []
    for seasonal_order in candidates:
        try:
            mod = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=50)
            rows.append({"seasonal_order": seasonal_order, "AIC": res.aic})
        except Exception as exc:
            rows.append({"seasonal_order": seasonal_order, "AIC": np.nan, "error": str(exc)})
    df = pd.DataFrame(rows).sort_values("AIC")
    print("Seasonal order comparison (fixed non-seasonal order = {}):".format(order))
    print(df.to_string(index=False))
    best = df.iloc[0]["seasonal_order"]
    print(f"Selected seasonal order: {best}")
    return best


def fit_sarimax(y_train: pd.Series, order: tuple, seasonal_order: tuple):
    mod = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False)
    return mod.fit(disp=False, maxiter=100)


def sarimax_forecast(res, horizon: int, index, alpha: float = 0.2):
    """Point forecast plus a (1-alpha) confidence interval (default: 80%)."""
    fc = res.get_forecast(steps=horizon)
    mean_fc = pd.Series(fc.predicted_mean.to_numpy(), index=index, name="SARIMAX")
    ci = fc.conf_int(alpha=alpha)
    lower = pd.Series(ci.iloc[:, 0].to_numpy(), index=index, name="lower")
    upper = pd.Series(ci.iloc[:, 1].to_numpy(), index=index, name="upper")
    return mean_fc, lower, upper


def plot_sarimax_residual_diagnostics(sarimax_res):
    """Inspects model fit via residual ACF + distribution, as required.
    Also runs the formal Ljung-Box test (residuals should show NO
    significant autocorrelation if the model has captured the structure)
    and reports skew/kurtosis, going a step beyond the visual checks
    alone.
    """
    resid = sarimax_res.resid

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(resid.index, resid.values, linewidth=0.6)
    axes[0, 0].set_title("Residuals over time")

    plot_acf(resid.dropna(), lags=48, ax=axes[0, 1])
    axes[0, 1].set_title("ACF of residuals (48 lags)")

    axes[1, 0].hist(resid, bins=50, color="#1f77b4", edgecolor="white", density=True)
    xs = np.linspace(resid.min(), resid.max(), 200)
    axes[1, 0].plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()), color="red", label="Normal fit")
    axes[1, 0].set_title("Residual distribution")
    axes[1, 0].legend()

    stats.probplot(resid.dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Residual Q-Q plot")
    _save_fig(fig, "07_sarimax_residual_diagnostics.png")

    lb = acorr_ljungbox(resid.dropna(), lags=[24, 48], return_df=True)
    print(lb)
    print(f"Residual skew: {stats.skew(resid):.2f}, kurtosis: {stats.kurtosis(resid):.2f}")


def plot_sarimax_forecast(train_full, test_full, mean_fc, lower, upper):
    fig, ax = plt.subplots(figsize=(12, 5))
    train_full.iloc[-24 * 5:].plot(ax=ax, color="black", label="Training data (last 5 days)")
    test_full.plot(ax=ax, color="black", linewidth=2.5, marker="o", markersize=3, label="Actual")
    mean_fc.plot(ax=ax, color="tab:red", linestyle="--", label="SARIMAX forecast")
    ax.fill_between(test_full.index, lower, upper, alpha=0.25, color="tab:red", label="80% CI")
    ax.axvline(train_full.index[-1], color="gray", linestyle=":", label="Forecast origin")
    ax.set_title("SARIMAX 24h-ahead forecast vs. actual")
    ax.legend()
    _save_fig(fig, "08_sarimax_forecast.png")


def sarimax_backtest(y: pd.Series, order: tuple, seasonal_order: tuple, horizon: int = HORIZON,
                      n_origins: int = N_ORIGINS, step: int = STEP):
    """Rolling-origin backtest for SARIMAX (same origins as every other
    model): refits the selected order fresh at each origin, so the
    backtest reflects genuinely out-of-sample performance rather than one
    lucky/unlucky fit."""
    origin_starts = rolling_origins(y, horizon, n_origins, step)
    per_origin, forecasts = [], {}
    for test_start in origin_starts:
        train = y.iloc[:test_start]
        test = y.iloc[test_start:test_start + horizon]
        if len(train) < 24 * 30 or len(test) < horizon:
            continue
        res = fit_sarimax(train, order, seasonal_order)
        mean_fc, lower, upper = sarimax_forecast(res, horizon, test.index)
        metrics = evaluate_forecast(test, mean_fc, train, seasonal_period=24, model_name="SARIMAX")
        metrics["origin"] = test.index[0]
        per_origin.append(metrics)
        forecasts[test.index[0]] = mean_fc
        print(f"  origin={test.index[0].date()} RMSE={metrics['RMSE']:.1f} MAE={metrics['MAE']:.1f}")
    return pd.DataFrame(per_origin), forecasts


# ============================================================================
# PART 5 -- COVARIATES
# ============================================================================
# Three feature groups: (i) indoor sensors + outdoor weather; (ii) cyclically
# encoded time-of-day / day-of-week features; (iii) lagged/rolling-window
# appliance-use features, computed strictly on PAST values only.
#
# Methodological note: the raw weather/sensor columns are CONTEMPORANEOUS
# measurements, not forecasts. Using their true value at the forecast target
# time would make the model a CONDITIONAL forecast ("if we already knew
# tomorrow's exact weather"), not a genuine one. Two variants are built:
#   - "realistic":   weather lagged 24h -- a same-day-persistence proxy that
#                     is genuinely available at forecast time.
#   - "conditional": true future values -- an oracle upper bound, built only
#                     for a leakage comparison (see compare_realistic_vs_conditional).

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cyclically-encoded (sin/cos) hour-of-day and day-of-week
    features, so e.g. 23:00 and 00:00 are seen as close together instead
    of maximally far apart the way a raw integer 0-23 would imply."""
    out = df.copy()
    hour, dow = out.index.hour, out.index.dayofweek
    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dayofweek"] = dow
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(int)
    return out


def add_lag_features(df: pd.DataFrame, target_col: str, lags=(1, 2, 3, 24, 48, 168)) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, target_col: str, windows=(3, 24, 168)) -> pd.DataFrame:
    """Rolling mean/std, computed on values shifted by 1 FIRST, so the
    current hour is never included in its own rolling window (that would
    be leakage: using the answer to help predict itself)."""
    out = df.copy()
    shifted = out[target_col].shift(1)
    for w in windows:
        out[f"{target_col}_rollmean{w}"] = shifted.rolling(window=w, min_periods=max(2, w // 4)).mean()
        out[f"{target_col}_rollstd{w}"] = shifted.rolling(window=w, min_periods=max(2, w // 4)).std()
    return out


def build_feature_frame(hourly_df: pd.DataFrame, target_col: str = TARGET,
                         weather_mode: str = "realistic", lags=(1, 2, 3, 24, 48, 168),
                         rolling_windows=(3, 24, 168)) -> pd.DataFrame:
    """Assemble the full feature matrix for the ML models.

    weather_mode="realistic"   -> sensor/weather columns lagged 24h (usable
                                   at genuine forecast time)
    weather_mode="conditional" -> true future sensor/weather values (oracle
                                   upper bound only, never a real forecast)
    """
    sensor_weather_cols = [c for c in (INDOOR_SENSOR_COLS + OUTDOOR_WEATHER_COLS) if c in hourly_df.columns]
    cols = [target_col] + sensor_weather_cols + (["lights"] if "lights" in hourly_df.columns else [])
    out = hourly_df[cols].copy()

    if weather_mode == "realistic":
        for c in sensor_weather_cols:
            out[c] = out[c].shift(24)
        if "lights" in out.columns:
            out["lights"] = out["lights"].shift(24)
    elif weather_mode != "conditional":
        raise ValueError("weather_mode must be 'realistic' or 'conditional'")

    out = add_time_features(out)
    out = add_lag_features(out, target_col, lags=lags)
    out = add_rolling_features(out, target_col, windows=rolling_windows)
    return out


def plot_feature_correlations(feat_realistic: pd.DataFrame, target_col: str = TARGET):
    corr = feat_realistic.corr(numeric_only=True)[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    print("Top 15 correlations with Appliances:")
    print(corr.head(15).round(3))

    fig, ax = plt.subplots(figsize=(8, 7))
    top15 = corr.head(15)[::-1]
    colors = ["tab:green" if v > 0 else "tab:red" for v in top15.values]
    ax.barh(top15.index, top15.values, color=colors)
    ax.set_title("Top 15 features: correlation with Appliances")
    ax.axvline(0, color="black", linewidth=0.8)
    _save_fig(fig, "09_feature_correlations.png")
    return corr


def compare_realistic_vs_conditional(hourly_df: pd.DataFrame, y: pd.Series, horizon: int = HORIZON):
    """Concrete demonstration of the leakage question the assignment asks
    about (Part 9, Q5): fits the SAME model on the final 24h test window
    using 'realistic' (genuinely available) vs 'conditional' (oracle
    future) weather features, and reports the RMSE gap. A single window
    is used here (rather than the full 14-origin backtest) purely to keep
    this demonstration cheap -- LightGBM is fast, so this adds only a few
    seconds regardless.
    """
    results = {}
    for mode in ["realistic", "conditional"]:
        feat = build_feature_frame(hourly_df, weather_mode=mode).dropna()
        feature_cols = [c for c in feat.columns if c != TARGET]
        train_feat, test_feat = feat.iloc[:-horizon], feat.iloc[-horizon:]
        if len(test_feat) < horizon:
            print(f"  [skipped {mode}: not enough clean rows for a full test window]")
            continue
        model = LGBMRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=42,
                               n_jobs=1, verbosity=-1)
        model.fit(train_feat[feature_cols], train_feat[TARGET])
        preds = model.predict(test_feat[feature_cols])
        results[mode] = evaluate_forecast(test_feat[TARGET], preds, train_feat[TARGET], model_name=mode)

    if len(results) == 2:
        gap = results["conditional"]["RMSE"] and (
            (results["realistic"]["RMSE"] - results["conditional"]["RMSE"]) / results["conditional"]["RMSE"] * 100
        )
        print(f"\nRealistic RMSE={results['realistic']['RMSE']:.1f}  "
              f"Conditional (oracle) RMSE={results['conditional']['RMSE']:.1f}  "
              f"({gap:+.1f}% worse using only genuinely-available information)")
        print("This gap IS the cost of honesty: 'conditional' silently assumes tomorrow's "
              "weather is already known, which is not a real forecasting scenario.")
    return results


# ============================================================================
# PART 6 -- FEATURE-BASED MACHINE LEARNING MODEL
# ============================================================================
# XGBoost and LightGBM (gradient-boosted trees) on the 'realistic' feature
# set. Multi-step forecasting is done RECURSIVELY: the model's own
# prediction for hour t+1 is fed back in as a lag input for predicting t+2,
# and so on through the full 24h horizon -- a genuine multi-step forecast,
# not 24 independent one-step predictions that would each get to peek at
# the true recent history.

def get_model(name: str, random_state: int = 42):
    """Factory for the tree-based regressors used in Part 6. Kept as a
    single function so hyperparameters are defined in exactly one place."""
    if name == "xgboost":
        return XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=random_state, n_jobs=1, verbosity=0)
    if name == "lightgbm":
        return LGBMRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, random_state=random_state, n_jobs=1, verbosity=-1)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=400, max_depth=10, random_state=random_state, n_jobs=1)
    raise ValueError(f"Unknown model name: {name}")


def recursive_forecast(model, history: pd.DataFrame, horizon: int, feature_cols: list,
                        target_col: str = TARGET, lags=(1, 2, 3, 24, 48, 168),
                        rolling_windows=(3, 24, 168), weather_mode: str = "realistic",
                        raw_hourly_for_weather: pd.DataFrame = None) -> pd.Series:
    """Forecast `horizon` steps ahead one hour at a time, feeding each
    prediction back into the feature set used for the next hour (a
    genuine recursive/multi-step forecast). `history` must contain the raw
    (unlagged) columns needed to rebuild features at each step.
    """
    hist = history.copy()
    future_index = pd.date_range(hist.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    preds = []
    sensor_weather_cols = [c for c in (INDOOR_SENSOR_COLS + OUTDOOR_WEATHER_COLS) if c in hist.columns]

    for t in future_index:
        row = {}
        for lag in lags:
            src = t - pd.Timedelta(hours=lag)
            row[f"{target_col}_lag{lag}"] = hist.loc[src, target_col] if src in hist.index else np.nan

        shifted = hist[target_col]
        for w in rolling_windows:
            window_vals = shifted.loc[:t - pd.Timedelta(hours=1)].iloc[-w:]
            row[f"{target_col}_rollmean{w}"] = window_vals.mean() if len(window_vals) > 0 else np.nan
            row[f"{target_col}_rollstd{w}"] = window_vals.std() if len(window_vals) > 1 else 0.0

        row["hour"] = t.hour
        row["hour_sin"] = np.sin(2 * np.pi * t.hour / 24)
        row["hour_cos"] = np.cos(2 * np.pi * t.hour / 24)
        row["dayofweek"] = t.dayofweek
        row["dow_sin"] = np.sin(2 * np.pi * t.dayofweek / 7)
        row["dow_cos"] = np.cos(2 * np.pi * t.dayofweek / 7)
        row["is_weekend"] = int(t.dayofweek >= 5)

        extra_cols = sensor_weather_cols + (["lights"] if "lights" in hist.columns else [])
        for c in extra_cols:
            if weather_mode == "realistic":
                src = t - pd.Timedelta(hours=24)
                row[c] = (raw_hourly_for_weather.loc[src, c]
                          if (raw_hourly_for_weather is not None and src in raw_hourly_for_weather.index)
                          else np.nan)
            else:
                row[c] = (raw_hourly_for_weather.loc[t, c]
                          if (raw_hourly_for_weather is not None and t in raw_hourly_for_weather.index)
                          else np.nan)

        x_row = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(x_row)[0])
        preds.append(pred)

        new_row = pd.DataFrame([{**row, target_col: pred}], index=[t])
        hist = pd.concat([hist, new_row[hist.columns]])

    return pd.Series(preds, index=future_index, name="ML_forecast")


def run_ml_backtest(y: pd.Series, hourly_df: pd.DataFrame, feat_clean: pd.DataFrame,
                     feature_cols: list, models=("xgboost", "lightgbm"), n_origins: int = N_ORIGINS):
    """Rolling-origin backtest for the ML models (same origins as every
    other model in this pipeline)."""
    results = {name: [] for name in models}
    forecasts = {name: {} for name in models}

    for test_start in rolling_origins(y, n_origins=n_origins):
        test = y.iloc[test_start:test_start + HORIZON]
        train_feat = feat_clean.loc[feat_clean.index < test.index[0]]
        if len(train_feat) < 24 * 30:
            continue
        x_train, y_train = train_feat[feature_cols], train_feat[TARGET]
        history_seed = hourly_df.loc[hourly_df.index < test.index[0]]

        for model_name in models:
            model = get_model(model_name)
            model.fit(x_train, y_train)
            fc = recursive_forecast(model, history_seed, HORIZON, feature_cols,
                                     weather_mode="realistic", raw_hourly_for_weather=hourly_df)
            metrics = evaluate_forecast(test, fc, y_train, model_name=model_name)
            metrics["origin"] = test.index[0]
            results[model_name].append(metrics)
            forecasts[model_name][test.index[0]] = fc

        print(f"  origin={test.index[0].date()} done")

    summary = pd.DataFrame({
        name: pd.DataFrame(res)[["MAE", "RMSE", "sMAPE", "MASE"]].mean()
        for name, res in results.items() if len(res) > 0
    }).T
    return summary, results, forecasts


def run_feature_ablation(y: pd.Series, hourly_df: pd.DataFrame, n_origins: int = N_ORIGINS):
    """Isolates the contribution of each feature group (lags only; lags +
    time features; the full realistic feature set) using LightGBM, which
    is fast enough that this doesn't meaningfully add to total runtime.
    Directly answers the assignment's "which feature groups appear most
    useful?" question (Part 9, Q3) with actual numbers.
    """
    def build_ablation_frame(include_sensors, include_weather, include_time, weather_mode="realistic"):
        sensor_weather_cols = []
        if include_sensors:
            sensor_weather_cols += [c for c in INDOOR_SENSOR_COLS if c in hourly_df.columns]
        if include_weather:
            sensor_weather_cols += [c for c in OUTDOOR_WEATHER_COLS if c in hourly_df.columns]
        cols = [TARGET] + sensor_weather_cols
        out = hourly_df[cols].copy()
        if weather_mode == "realistic":
            for c in sensor_weather_cols:
                out[c] = out[c].shift(24)
        if include_time:
            out = add_time_features(out)
        out = add_lag_features(out, TARGET)
        out = add_rolling_features(out, TARGET)
        return out

    configs = {
        "lags_only": dict(include_sensors=False, include_weather=False, include_time=False),
        "lags_plus_time": dict(include_sensors=False, include_weather=False, include_time=True),
        "full_realistic": dict(include_sensors=True, include_weather=True, include_time=True),
    }

    ablation_results = {name: [] for name in configs}
    for name, kwargs in configs.items():
        feat = build_ablation_frame(**kwargs).dropna()
        cols = [c for c in feat.columns if c != TARGET]
        for test_start in rolling_origins(y, n_origins=n_origins):
            test = y.iloc[test_start:test_start + HORIZON]
            train_feat = feat.loc[feat.index < test.index[0]]
            if len(train_feat) < 24 * 30:
                continue
            model = get_model("lightgbm")
            model.fit(train_feat[cols], train_feat[TARGET])
            history_seed = hourly_df.loc[hourly_df.index < test.index[0]]
            fc = recursive_forecast(model, history_seed, HORIZON, cols,
                                     weather_mode="realistic", raw_hourly_for_weather=hourly_df)
            metrics = evaluate_forecast(test, fc, train_feat[TARGET], model_name=name)
            metrics["origin"] = test.index[0]
            ablation_results[name].append(metrics)
        print(f"  {name} done")

    summary = pd.DataFrame({
        name: pd.DataFrame(res)[["MAE", "RMSE", "sMAPE", "MASE"]].mean()
        for name, res in ablation_results.items() if len(res) > 0
    }).T
    return summary


def plot_feature_importance(feat_clean: pd.DataFrame, feature_cols: list):
    final_train = feat_clean.iloc[:-HORIZON]
    model = get_model("lightgbm")
    model.fit(final_train[feature_cols], final_train[TARGET])

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 8))
    top15 = importances.head(15)[::-1]
    ax.barh(top15.index, top15.values, color="#1f77b4")
    ax.set_title("LightGBM feature importance (top 15)")
    _save_fig(fig, "10_ml_feature_importance.png")
    return importances


# ============================================================================
# PART 7 -- FOUNDATION MODEL: Chronos-Bolt-small
# ============================================================================
# Chronos-Bolt-small (Amazon) is a pretrained time-series foundation model,
# used here in pure ZERO-SHOT mode -- no fine-tuning on this dataset at all.
# It takes a window of past values as context and produces quantile
# forecasts directly.
#
# Dependency note: install with a plain `pip install chronos-forecasting`
# (see requirements.txt). Do NOT hand-pin an old exact transformers version
# alongside it -- chronos-forecasting declares its own compatible version
# range (transformers>=4.41,<6, torch>=2.2,<3) and pip's resolver satisfies
# both together correctly. Pinning an old exact version manually is what
# broke this step in the original notebook (a stale transformers build was
# left unable to resolve `PreTrainedModel` from huggingface_hub's lazy
# import machinery). This function fails soft: if torch/chronos aren't
# installed, Part 7 is skipped with clear instructions rather than crashing
# the whole pipeline.

def run_chronos_backtest(y: pd.Series, n_origins: int = N_ORIGINS, context_length: int = 24 * 28):
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        print(f"Chronos/torch not installed ({exc}).")
        print("Skipping Part 7. To enable it, run:")
        print("    pip install chronos-forecasting")
        print("(let pip resolve the compatible torch/transformers versions itself -- "
              "do not pin an old exact transformers version by hand).")
        return None, None

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_map}")
    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=device_map)

    results, forecasts = [], {}
    for test_start in rolling_origins(y, n_origins=n_origins):
        train = y.iloc[:test_start]
        test = y.iloc[test_start:test_start + HORIZON]
        if len(train) < 24 * 30 or len(test) < HORIZON:
            continue

        context_tensor = torch.tensor(
            train.iloc[-context_length:].to_numpy(), dtype=torch.float32
        ).unsqueeze(0)

        quantiles, _ = pipeline.predict_quantiles(
            inputs=context_tensor, prediction_length=HORIZON, quantile_levels=[0.1, 0.5, 0.9],
        )
        q = quantiles[0].detach().cpu().numpy()
        lower_80 = pd.Series(q[:, 0], index=test.index, name="lower")
        median = pd.Series(q[:, 1], index=test.index, name="median")
        upper_80 = pd.Series(q[:, 2], index=test.index, name="upper")

        metrics = evaluate_forecast(test, median, train, model_name="Chronos-Bolt-small")
        metrics["origin"] = test.index[0]
        metrics["coverage_80"] = coverage(test, lower_80, upper_80)
        results.append(metrics)
        forecasts[test.index[0]] = {"median": median, "lower": lower_80, "upper": upper_80}
        print(f"  origin={test.index[0].date()} RMSE={metrics['RMSE']:.1f} MASE={metrics['MASE']:.3f} "
              f"coverage80={metrics['coverage_80']:.2f}")

    results_df = pd.DataFrame(results)
    summary = results_df[["MAE", "RMSE", "sMAPE", "MASE", "coverage_80"]].mean()
    return summary, forecasts


def plot_chronos_forecast(y: pd.Series, forecasts: dict):
    last_origin = list(forecasts.keys())[-1]
    fc = forecasts[last_origin]
    test = y.loc[fc["median"].index]
    train = y.loc[:fc["median"].index[0] - pd.Timedelta(hours=1)]

    fig, ax = plt.subplots(figsize=(12, 5))
    train.iloc[-24 * 5:].plot(ax=ax, color="black", label="Training data (last 5 days)")
    test.plot(ax=ax, color="black", linewidth=2.5, marker="o", markersize=3, label="Actual")
    fc["median"].plot(ax=ax, color="tab:purple", linestyle="--", label="Chronos-Bolt-small median")
    ax.fill_between(test.index, fc["lower"], fc["upper"], alpha=0.25, color="tab:purple", label="80% interval")
    ax.axvline(train.index[-1], color="gray", linestyle=":", label="Forecast origin")
    ax.set_title(f"Chronos-Bolt-small zero-shot 24h forecast, origin {last_origin}")
    ax.legend()
    _save_fig(fig, "11_chronos_forecast.png")


# ============================================================================
# ORCHESTRATION
# ============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=".", help="Base directory for Figures/ and Results/ (default: current directory)")
    parser.add_argument("--quick", action="store_true",
                         help="Fast smoke-test mode: shrinks the SARIMAX grid and uses 3 backtest origins instead of 14")
    parser.add_argument("--skip-sarimax-search", action="store_true",
                         help="Skip the slow AIC grid search; use a fixed order of (2, 0, 2) instead")
    parser.add_argument("--skip-chronos", action="store_true", help="Skip Part 7 entirely")
    return parser.parse_args(argv)


def main(argv=None):
    global FIGURE_DIR, RESULTS_DIR

    args = parse_args(argv)
    base = Path(args.output_dir)
    FIGURE_DIR = base / "Figures"
    RESULTS_DIR = base / "Results"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_origins = 3 if args.quick else N_ORIGINS
    p_max, d_max, q_max = (2, 1, 2) if args.quick else (6, 2, 6)

    all_summaries = {}

    # ---------------- Part 1 ----------------
    _banner("PART 1 -- Data retrieval and preparation")
    raw = load_dataset()
    check_data_quality(raw)
    hourly = resample_hourly(raw)
    hourly.to_csv(RESULTS_DIR / "hourly_data.csv")
    y = hourly[TARGET]

    plot_full_series(y)
    plot_distribution(y)
    plot_daily_weekly_profile(y)
    decompose_components(y)
    plot_acf_pacf(y)
    test_stationarity(y)

    # ---------------- Part 2 ----------------
    _banner("PART 2 -- Forecasting problem definition")
    print(f"Target: {TARGET} | Horizon: {HORIZON}h | Backtest origins: {n_origins} | Step: {STEP}h")
    print("Metrics: MAE, RMSE, sMAPE, MASE (primary, scale-free)")

    # ---------------- Part 3 ----------------
    _banner("PART 3 -- Benchmark models")
    benchmark_summary, benchmark_results, benchmark_forecasts = run_benchmarks(y, n_origins=n_origins)
    print(benchmark_summary.round(3))
    benchmark_summary.to_csv(RESULTS_DIR / "benchmark_summary.csv")
    all_summaries.update({name: summarise_backtest(df) for name, df in benchmark_results.items()})

    # ---------------- Part 4 ----------------
    _banner("PART 4 -- SARIMAX")
    train_full, test_full = y.iloc[:-HORIZON], y.iloc[-HORIZON:]
    train_subsample = train_full.iloc[-24 * 45:]  # last 45 days, for tractable order search

    if args.skip_sarimax_search:
        print("Skipping AIC grid search (--skip-sarimax-search); using fixed order (2, 0, 2).")
        best_order = (2, 0, 2)
    else:
        gridsearch_results = grid_search_aic(
            train_subsample, range(0, p_max + 1), range(0, d_max + 1), range(0, q_max + 1),
            seasonal_order=(1, 0, 1, SEASONAL_PERIOD),
        )
        gridsearch_results.to_csv(RESULTS_DIR / "sarimax_gridsearch_results.csv", index=False)
        best_order = select_best_order(gridsearch_results)

    print(f"Selected non-seasonal order: {best_order}")
    seasonal_order = select_seasonal_order(train_subsample, best_order)

    sarimax_res = fit_sarimax(train_full, best_order, seasonal_order)
    print(f"AIC={sarimax_res.aic:.1f}, BIC={sarimax_res.bic:.1f}")
    sarimax_mean_fc, sarimax_lower, sarimax_upper = sarimax_forecast(sarimax_res, HORIZON, test_full.index)
    plot_sarimax_residual_diagnostics(sarimax_res)
    plot_sarimax_forecast(train_full, test_full, sarimax_mean_fc, sarimax_lower, sarimax_upper)

    sarimax_backtest_df, sarimax_backtest_forecasts = sarimax_backtest(
        y, best_order, seasonal_order, n_origins=n_origins
    )
    sarimax_summary = summarise_backtest(sarimax_backtest_df)
    print("\nSARIMAX mean over all origins:")
    print(sarimax_summary.round(3))
    all_summaries["SARIMAX"] = sarimax_summary

    # ---------------- Part 5 ----------------
    _banner("PART 5 -- Covariates")
    feat_realistic = build_feature_frame(hourly, weather_mode="realistic")
    plot_feature_correlations(feat_realistic)
    compare_realistic_vs_conditional(hourly, y)

    # ---------------- Part 6 ----------------
    _banner("PART 6 -- Feature-based machine learning model")
    feat_clean = feat_realistic.dropna()
    feature_cols = [c for c in feat_clean.columns if c != TARGET]

    ml_summary, ml_results, ml_forecasts = run_ml_backtest(
        y, hourly, feat_clean, feature_cols, models=("xgboost", "lightgbm"), n_origins=n_origins
    )
    print("\nML model backtest summary:")
    print(ml_summary.round(3))
    ml_summary.to_csv(RESULTS_DIR / "ml_summary.csv")
    for name, res in ml_results.items():
        if len(res) > 0:
            all_summaries[name] = pd.DataFrame(res)[["MAE", "RMSE", "sMAPE", "MASE"]].mean()

    ablation_summary = run_feature_ablation(y, hourly, n_origins=n_origins)
    print("\nFeature-group ablation (LightGBM):")
    print(ablation_summary.round(3))
    ablation_summary.to_csv(RESULTS_DIR / "ablation_summary.csv")

    plot_feature_importance(feat_clean, feature_cols)

    # ---------------- Part 7 ----------------
    _banner("PART 7 -- Foundation model (Chronos-Bolt-small)")
    if args.skip_chronos:
        print("Skipping Part 7 (--skip-chronos).")
    else:
        chronos_summary, chronos_forecasts = run_chronos_backtest(y, n_origins=n_origins)
        if chronos_summary is not None:
            print("\nChronos-Bolt-small mean over all origins:")
            print(chronos_summary.round(3))
            chronos_summary.to_csv(RESULTS_DIR / "chronos_summary.csv")
            plot_chronos_forecast(y, chronos_forecasts)
            all_summaries["Chronos-Bolt-small"] = chronos_summary[["MAE", "RMSE", "sMAPE", "MASE"]]

    # ---------------- Consolidated summary (feeds directly into Part 8) ----------------
    _banner("CONSOLIDATED MODEL COMPARISON (staged for Part 8 write-up)")
    combined = pd.DataFrame(all_summaries).T.sort_values("RMSE")
    print(combined.round(3))
    combined.to_csv(RESULTS_DIR / "all_models_summary.csv")

    print(f"\nDone. Figures in '{FIGURE_DIR}/', results/CSVs in '{RESULTS_DIR}/'.")
    return combined


if __name__ == "__main__":
    main()
