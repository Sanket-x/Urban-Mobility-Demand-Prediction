"""
================================================================================
STEP 4 & 5: CLASSICAL TIME SERIES MODELS & FORECASTING
================================================================================
Models Implemented:
    1. ARIMA (AutoRegressive Integrated Moving Average)
       - Handles non-stationary data via differencing (d).
       - Uses past values (AR - p) and past errors (MA - q) for prediction.

    2. SARIMA (Seasonal ARIMA)
       - Extends ARIMA to model periodic seasonality.
       - Parameters: (p,d,q) x (P,D,Q,s).  We use s=24 for daily seasonality
         in hourly data.

    3. Holt-Winters (Exponential Smoothing)
       - Uses exponentially decreasing weights for past observations.
       - Extracts Level, Trend, and Seasonal components.

Assumptions:
    - Classical models assume future patterns are functions of past patterns.
    - ARIMA assumes residuals (errors) are white noise (normally distributed,
      uncorrelated).
    - Stationarity (or capability to achieve it via differencing) is required
      for ARIMA/SARIMA.
================================================================================
"""

import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from prophet import Prophet
import xgboost as xgb
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def build_arima(train: pd.Series, order: tuple = (1, 1, 1)):
    """
    Build and train an ARIMA model.

    Parameters
    ----------
    train : pd.Series
        Training data.
    order : tuple
        The (p, d, q) order of the model.
        p: AR terms (lags of stationarized series)
        d: Integration order (number of nonseasonal differences)
        q: MA terms (lags of forecast errors)

    Returns
    -------
    model_fit
        Fitted statsmodels ARIMA object.
    """
    print(f"\n[MODEL] Training ARIMA{order}...")
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    print("  → Training complete.")
                                                                        
    return model_fit

def build_sarima(train: pd.Series, order: tuple = (1, 1, 1),
                 seasonal_order: tuple = (1, 1, 1, 24)):
    """
    Build and train a SARIMA model.

    Parameters
    ----------
    train : pd.Series
        Training data.
    order : tuple
        The (p, d, q) order for the non-seasonal component.
    seasonal_order : tuple
        The (P, D, Q, s) order for the seasonal component.
        s=24 implies hourly data with a daily repeating pattern.

    Returns
    -------
    model_fit
        Fitted statsmodels SARIMAX object.
    """
    print(f"\n[MODEL] Training SARIMA{order}x{seasonal_order}...")
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    print("  → Training complete.")
    return model_fit

def build_holt_winters(train: pd.Series, seasonal_periods: int = 24,
                       trend: str = "add", seasonal: str = "add"):
    """
    Build and train a Holt-Winters Exponential Smoothing model.

    Parameters
    ----------
    train : pd.Series
        Training data.
    seasonal_periods : int
        Number of periods in a seasonal cycle (24 for daily seasonality).
    trend/seasonal : str {'add', 'mul'}
        Type of trend and seasonal component (Additive or Multiplicative).
        We default to 'add'. If series contains 0s, 'mul' may fail or
        require a small offset.

    Returns
    -------
    model_fit
        Fitted statsmodels ExponentialSmoothing object.
    """
    print(f"\n[MODEL] Training Holt-Winters (Trend='{trend}', Seasonal='{seasonal}', periods={seasonal_periods})...")
                                                                        
    if seasonal == "mul" and (train <= 0).any():
        print("  → Warning: Multiplicative method requires strictly positive data. Swapping to Additive.")
        seasonal = "add"

    model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal,
                                 seasonal_periods=seasonal_periods,
                                 initialization_method="estimated")
    model_fit = model.fit()
    print("  → Training complete.")
    return model_fit

def make_forecast(model_fit, steps: int) -> pd.Series:
    """
    Generate forecasting for *steps* into the future.
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast

def build_prophet(train: pd.Series):
    print("\n[MODEL] Training Prophet...")
    df = pd.DataFrame({'ds': train.index, 'y': train.values})
    if df['ds'].dt.tz is not None:
        df['ds'] = df['ds'].dt.tz_localize(None)
    model = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
    model.fit(df)
    print("  → Training complete.")
    return model

def make_prophet_forecast(model, steps: int, freq: str = 'h'):
    future = model.make_future_dataframe(periods=steps, freq=freq)
    forecast = model.predict(future)
    return pd.Series(forecast['yhat'].iloc[-steps:].values)

def create_supervised_features(series: pd.Series, lags: int = 24):
    df = pd.DataFrame(series)
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df.iloc[:, 0].shift(i)
    df.dropna(inplace=True)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    return X, y, df

def build_xgboost(train: pd.Series, lags: int = 24):
    print(f"\n[MODEL] Training XGBoost (using {lags} lags)...")
    X_train, y_train, _ = create_supervised_features(train, lags)
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    print("  → Training complete.")
    return model, lags

def make_xgboost_forecast(model, train: pd.Series, steps: int, lags: int = 24):
    history = list(train.values[-lags:])
    predictions = []
    for _ in range(steps):
        X_test = np.array(history[-lags:]).reshape(1, -1)
        pred = model.predict(X_test)[0]
        predictions.append(pred)
        history.append(pred)
    return pd.Series(predictions)

def build_lstm(train: pd.Series, lags: int = 24, epochs: int = 15):
    print(f"\n[MODEL] Training LSTM (using {lags} lags for {epochs} epochs)...")
    tf.random.set_seed(42)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lags, len(scaled_data)):
        X.append(scaled_data[i-lags:i, 0])
        y.append(scaled_data[i, 0])
        
    X_train, y_train = np.array(X), np.array(y)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
    print("  → Training complete.")
    return model, scaler, lags

def make_lstm_forecast(model, scaler, train: pd.Series, steps: int, lags: int = 24):
    history = list(train.values[-lags:])
    predictions = []
    for _ in range(steps):
        history_scaled = scaler.transform(np.array(history[-lags:]).reshape(-1, 1))
        X_test = np.reshape(history_scaled, (1, lags, 1))
        pred_scaled = model.predict(X_test, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred)
        history.append(pred)
    return pd.Series(predictions)
