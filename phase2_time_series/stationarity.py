"""
================================================================================
STEP 2 & 3: STATIONARITY ANALYSIS  +  ACF / PACF
================================================================================
Purpose:
    - Check if the time series is stationary (required for ARIMA/SARIMA).
    - Visualize rolling statistics (mean & std) to detect trends / variance change.
    - Perform the Augmented Dickey-Fuller (ADF) test for a formal hypothesis test.
    - If non-stationary, apply differencing (and optional log transform) until stationary.
    - Plot ACF and PACF to guide (p, d, q) parameter selection.

Background – What is Stationarity?
    A time series is stationary when its statistical properties (mean, variance,
    autocorrelation) are constant over time.  ARIMA models assume stationarity
    of the differenced series because their linear coefficients are fixed.  If
    the series has a changing mean or variance, the model's parameter estimates
    become unreliable.

ADF Test Interpretation:
    H0 : The series has a unit root (non-stationary).
    H1 : The series is stationary.
    If p-value < 0.05, we reject H0 → series IS stationary.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                              
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import os

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (14, 5),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def adf_test(series: pd.Series, label: str = "") -> dict:
    """
    Run the Augmented Dickey-Fuller test and print a clear interpretation.

    Returns a dict with test statistic, p-value, and stationarity verdict.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    stats = {
        "label": label,
        "test_statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_observations": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }

    output_str = (
        f"\n{'='*60}\n"
        f"  ADF Test — {label}\n"
        f"{'='*60}\n"
        f"  Test Statistic : {stats['test_statistic']:.6f}\n"
        f"  p-value        : {stats['p_value']:.6f}\n"
        f"  Lags Used      : {stats['lags_used']}\n"
        f"  Observations   : {stats['n_observations']}\n"
    )
    for k, v in stats["critical_values"].items():
        output_str += f"  Critical Value ({k}): {v:.4f}\n"
    verdict = "STATIONARY" if stats["is_stationary"] else "NON-STATIONARY"
    output_str += f"\n  ➜ Conclusion: {verdict}  (p {'<' if stats['is_stationary'] else '>'} 0.05)\n"
    output_str += f"{'='*60}\n"

    print(output_str)

    try:
        with open("model_results.txt", "a", encoding="utf-8") as f:
            f.write(output_str)
    except Exception as e:
        print(f"Could not write to model_results.txt: {e}")

    return stats

def plot_rolling_statistics(ts: pd.Series, window: int = 24,
                            save_path: str | None = None):
    """
    Plot the original series alongside its rolling mean and rolling std.

    WHY: Visual inspection of rolling statistics reveals trends (changing mean)
         and heteroscedasticity (changing variance), both signs of
         non-stationarity.  A window of 24 captures one full daily cycle.
    """
    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(ts, color="#4A90D9", alpha=0.5, label="Original")
    ax.plot(rolling_mean, color="#E74C3C", linewidth=2, label=f"Rolling Mean (w={window})")
    ax.plot(rolling_std, color="#2ECC71", linewidth=2, label=f"Rolling Std  (w={window})")
    ax.set_title("Rolling Mean & Standard Deviation")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand Count")
    ax.legend(loc="upper left")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  → Saved rolling statistics plot to {save_path}")
    plt.close(fig)

def make_stationary(ts: pd.DataFrame, col: str = "demand_count",
                    save_dir: str | None = None):
    """
    Attempt to make the series stationary via differencing (and log transform
    if variance is non-constant).

    Returns
    -------
    result : dict  with keys:
        'original_adf'    – ADF result on original series
        'stationary_series' – the stationary pd.Series (possibly differenced)
        'd'               – order of differencing applied (0, 1, or 2)
        'log_applied'     – whether log1p was used
        'transformations' – list of string descriptions
    """
    series = ts[col].copy()
    transformations = []
    log_applied = False
    d = 0

    orig_adf = adf_test(series, label="Original Series")

    if save_dir:
        plot_rolling_statistics(series, save_path=os.path.join(save_dir, "rolling_original.png"))

    working = series.copy()
    if not orig_adf["is_stationary"]:
                                                                           
        if series.std() > series.mean() * 0.5:
            working = np.log1p(series)                            
            log_applied = True
            transformations.append("log1p transform (stabilise variance)")
            adf_test(working, label="After Log Transform")

    if not adf_test(working, label="Pre-differencing check")["is_stationary"]:
        working = working.diff().dropna()
        d = 1
        transformations.append("1st order differencing")
        adf_after_d1 = adf_test(working, label="After 1st Differencing")

        if save_dir:
            plot_rolling_statistics(working,
                                   save_path=os.path.join(save_dir, "rolling_diff1.png"))

        if not adf_after_d1["is_stationary"]:
            working = working.diff().dropna()
            d = 2
            transformations.append("2nd order differencing")
            adf_test(working, label="After 2nd Differencing")

            if save_dir:
                plot_rolling_statistics(working,
                                       save_path=os.path.join(save_dir, "rolling_diff2.png"))

    print(f"\n[STATIONARITY] Transformations applied: {transformations or 'None (already stationary)'}")
    print(f"[STATIONARITY] Final differencing order d = {d}")
    print(f"[STATIONARITY] Log applied = {log_applied}\n")

    return {
        "original_adf": orig_adf,
        "stationary_series": working,
        "d": d,
        "log_applied": log_applied,
        "transformations": transformations,
    }

def plot_acf_pacf(series: pd.Series, lags: int = 50,
                  save_path: str | None = None):
    """
    Plot ACF and PACF side-by-side.

    How to read the plots (for ARIMA parameter selection):
    ─────────────────────────────────────────────────────
    • ACF  → determines q (MA order).
      - If ACF cuts off sharply after lag q → use MA(q).
      - If ACF decays slowly               → AR component dominates.

    • PACF → determines p (AR order).
      - If PACF cuts off sharply after lag p → use AR(p).
      - If PACF decays slowly                → MA component dominates.

    Seasonal spikes at multiples of 24 (hourly data) indicate daily
    seasonality, guiding the seasonal order (P, D, Q, 24) for SARIMA.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_acf(series.dropna(), lags=lags, ax=axes[0], title="Autocorrelation Function (ACF)")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm",
              title="Partial Autocorrelation Function (PACF)")

    for ax in axes:
        ax.set_xlabel("Lag")
        ax.set_ylabel("Correlation")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  → Saved ACF/PACF plot to {save_path}")
    plt.close(fig)

def plot_time_series_decomposition(series: pd.Series, period: int = 24, 
                                   save_path: str | None = None):
    """
    Perform seasonal decomposition using an Additive model.
    
    Academic Context:
    - Trend: Extracts the underlying long-term direction of the data.
    - Seasonal: Extracts the strict repeating periodic pattern (hourly data over a 24-hour cycle).
    - Residual: The random "noise" left over after calculating Trend and Seasonal components.
    
    Why this justifies SARIMA/Holt-Winters: 
    If a clear cyclic repeating pattern exists in the 'Seasonal' subplot and the residuals
    look like random noise, it proves mathematical necessity to deploy models capable
    of absorbing seasonal parameters (SARIMA with s=24 or Holt-Winters).
    """
    import warnings
                                                                         
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
                                       
        if series.index.freq is None:
            series = series.asfreq('h')

        decomposition = seasonal_decompose(series.dropna(), model='additive', period=period)

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    decomposition.observed.plot(ax=axes[0], color="#2C3E50", title="Observed (Raw Data)")
    decomposition.trend.plot(ax=axes[1], color="#E74C3C", title="Trend Component")
    decomposition.seasonal.plot(ax=axes[2], color="#2ECC71", title="Seasonal Component (Period=24)")
    decomposition.resid.plot(ax=axes[3], color="#8E44AD", style=".", title="Residuals (Noise)")
    
    for ax in axes:
        ax.set_ylabel("Demand")
    axes[3].set_xlabel("Time")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  → Saved Time Series Decomposition to {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    from preprocessing import load_and_preprocess

    ts = load_and_preprocess()
    result = make_stationary(ts, save_dir=".")
    plot_acf_pacf(result["stationary_series"], save_path="acf_pacf.png")
