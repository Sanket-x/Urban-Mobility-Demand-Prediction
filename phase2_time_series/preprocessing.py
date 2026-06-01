"""
================================================================================
STEP 1: DATA PREPROCESSING
================================================================================
Purpose:
    - Load the raw hourly demand dataset
    - Convert columns to proper datetime format and create a datetime index
    - Aggregate demand across ALL pickup areas into a single city-wide hourly
      time series (suited for univariate time series models like ARIMA/SARIMA)
    - Handle missing hours, duplicates, and missing values
    - Return a clean, chronologically sorted, hourly time series

Why each step matters:
    1. Datetime index   → Required by statsmodels for time series decomposition,
                          ACF/PACF, and ARIMA fitting.
    2. Aggregation      → Classical univariate models operate on a single series.
                          Summing across areas gives total city demand per hour.
    3. Missing-hour fill→ Time series models assume a regular (gapless) frequency.
                          Missing hours are filled with 0 (no rides observed).
    4. Sorting           → Ensures temporal ordering for correct differencing
                          and train/test splits.
================================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

def load_and_preprocess(data_path: str | None = None) -> pd.DataFrame:
    """
    Load the hourly demand dataset and return a clean, aggregated hourly
    time series with a DatetimeIndex.

    Parameters
    ----------
    data_path : str or None
        Path to the CSV file. If None, auto-detects relative to this file.

    Returns
    -------
    ts : pd.DataFrame
        DataFrame with DatetimeIndex (freq='h') and column 'demand_count'.
    """

    if data_path is None:
                                                                                  
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data", "hourly_demand_dataset.csv")

    print(f"[PREPROCESSING] Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    print(f"  → Raw shape: {df.shape}")
    print(f"  → Columns  : {list(df.columns)}")
    print(f"  → Sample rows:\n{df.head()}\n")

    missing_before = df.isnull().sum()
    print(f"[PREPROCESSING] Missing values per column (before):\n{missing_before}\n")

    df = df.dropna(subset=["Demand_Count"])
    print(f"  → Shape after dropping NaN targets: {df.shape}")

    df["datetime"] = pd.to_datetime(df["Date_Only"]) + pd.to_timedelta(df["Hour"], unit="h")
    print(f"[PREPROCESSING] Created datetime column.  Range: "
          f"{df['datetime'].min()} → {df['datetime'].max()}")

    n_dup = df.duplicated().sum()
    print(f"[PREPROCESSING] Exact duplicates found: {n_dup}")
    if n_dup > 0:
        df = df.drop_duplicates()
        print(f"  → Shape after removing duplicates: {df.shape}")

    ts = (
        df.groupby("datetime")["Demand_Count"]
        .sum()
        .reset_index()
        .rename(columns={"Demand_Count": "demand_count"})
    )
    ts = ts.set_index("datetime")
    ts = ts.sort_index()                              

    full_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq="h")
    ts = ts.reindex(full_range, fill_value=0)
    ts.index.name = "datetime"
    ts.index.freq = "h"                            

    print(f"\n[PREPROCESSING] Final time series shape: {ts.shape}")
    print(f"  → Date range : {ts.index.min()} → {ts.index.max()}")
    print(f"  → Frequency  : {ts.index.freq}")
    print(f"  → Total hours : {len(ts)}")
    print(f"  → Demand stats:\n{ts['demand_count'].describe()}\n")

    return ts

if __name__ == "__main__":
    ts = load_and_preprocess()
    print(ts.head(30))
