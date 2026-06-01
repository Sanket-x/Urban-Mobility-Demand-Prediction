"""
================================================================================
STEP 8: ORCHESTRATION & INTERPRETATION (MAIN PIPELINE)
================================================================================
This script stitches together the entire Phase 2 workflow:
    1. Preprocessing & Aggregation
    2. Stationarity Checks & ACF/PACF Analysis
    3. Splitting into Train/Test subsets (Time-based, NOT random)
    4. Model Training (ARIMA, SARIMA, Holt-Winters)
    5. Forecasting & Evaluation (Metrics table + Plots generation)
    6. Interpretation printing
================================================================================
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_and_preprocess
from stationarity import make_stationary, plot_acf_pacf, plot_time_series_decomposition
from models import build_arima, build_sarima, build_holt_winters, make_forecast, build_prophet, make_prophet_forecast, build_xgboost, make_xgboost_forecast, build_lstm, make_lstm_forecast
from evaluation import calculate_metrics, display_metrics_table, plot_forecast, plot_residual_analysis, plot_residual_acf

def define_interpretation(best_model="SARIMA"):
    """
    Step 8: Interpret results in a real-world urban mobility context.
    """
    print(f"\n================================================================================")
    print(f"STEP 8: BUSINESS INTERPRETATION & REAL-WORLD CONTEXT")
    print(f"================================================================================")
    
    interpretation = f"""
1. Peak Demand & Weekly Trends:
   - The ACF/PACF and rolling statistic plots show strong cyclic behavior every 
     24 hours (daily seasonality).
   - Demand predictably surges during traditional commute hours and drops 
     substantially overnight.

2. Model Behavior & Comparison:
   - ARIMA captures immediate autocorrelation but generally fails to carry the 
     daily oscillating pattern deep into the forecast period.
   - Holt-Winters and {best_model} successfully isolate the periodic nature of urban 
     mobility. The seasonal component correctly forces the forecast to rise and 
     fall with the daily clock.
     
3. Relate to Urban Mobility Operations:
   - Surge Pricing: Predicting these sharp peaks accurately allows platform algorithms
     to enact surge pricing preemptively, distributing demand more evenly.
   - Fleet Planning / Driver Allocation: Drivers organically aggregate in high-demand
     zones. Time-series forecasting allows companies to nudge idle drivers toward
     areas anticipating a spike, lowering Customer ETA and increasing driver earnings.
     
CONCLUSION: Classical univariate time-series models (like {best_model}) establish a 
strong baseline. They mathematically validate the intuitive assumption that urban 
transport follows rigid societal clocks, proving suitable for mid-term tactical planning.
"""
    print(interpretation)
    print(f"================================================================================\n")

def main():
                                      
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\n[START] PHASE 2: TIME SERIES ANALYSIS PIPELINE\n")
    
    print(">>> STEP 1: Preprocessing Data...")
                                                                   
    base_dir = os.path.dirname(script_dir)
    data_path = os.path.join(base_dir, "data", "hourly_demand_dataset.csv")
    
    try:
        ts = load_and_preprocess(data_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find dataset at {data_path}")
        print("Please ensure 'hourly_demand_dataset.csv' exists in the appropriate location.")
        return

    series = ts['demand_count']

    print("\n>>> STEP 2 & 3: Analyzing Stationarity, ACF & PACF, and Decomposition...")
    stat_result = make_stationary(ts, save_dir=plot_dir)
    
    print("  → Generating Time Series Decomposition plot...")
    plot_time_series_decomposition(series, period=24, save_path=os.path.join(plot_dir, "decomposition.png"))

    stat_series = stat_result['stationary_series']
    plot_acf_pacf(stat_series, save_path=os.path.join(plot_dir, "acf_pacf.png"))

    forecast_steps = 48
    train = series.iloc[:-forecast_steps]
    test = series.iloc[-forecast_steps:]
    
    print(f"\n[DATA SPLIT] Training Data: {len(train)} hours")
    print(f"[DATA SPLIT] Testing Data : {len(test)} hours (Forecast Horizon)")

    print("\n>>> STEP 4 & 5: Training Models & Forecasting...")
    results = []

    d = stat_result['d'] if stat_result['d'] > 0 else 1 
    arima_order = (2, d, 2)
    arima_model = build_arima(train, order=arima_order)
    arima_fc = make_forecast(arima_model, steps=forecast_steps)
    arima_fc.index = test.index
    
    plot_residual_analysis(arima_model.resid, "ARIMA", save_path=os.path.join(plot_dir, "residual_arima.png"))
    plot_residual_acf(arima_model.resid, "ARIMA", save_path=os.path.join(plot_dir, "residual_acf_arima.png"))
    
    results.append(calculate_metrics(test, arima_fc, "ARIMA"))
    plot_forecast(train, test, arima_fc, "ARIMA", save_path=os.path.join(plot_dir, "forecast_arima.png"))

    sarima_order = (1, d, 1)
    sarima_seasonal = (1, 1, 1, 24)
    sarima_model = build_sarima(train, order=sarima_order, seasonal_order=sarima_seasonal)
    sarima_fc = make_forecast(sarima_model, steps=forecast_steps)
    sarima_fc.index = test.index
    
    plot_residual_analysis(sarima_model.resid, "SARIMA", save_path=os.path.join(plot_dir, "residual_sarima.png"))
    plot_residual_acf(sarima_model.resid, "SARIMA", save_path=os.path.join(plot_dir, "residual_acf_sarima.png"))
    
    results.append(calculate_metrics(test, sarima_fc, "SARIMA"))
    plot_forecast(train, test, sarima_fc, "SARIMA", save_path=os.path.join(plot_dir, "forecast_sarima.png"))

    hw_model = build_holt_winters(train, seasonal_periods=24, trend="add", seasonal="add")
    hw_fc = make_forecast(hw_model, steps=forecast_steps)
    hw_fc.index = test.index
    
    plot_residual_analysis(hw_model.resid, "Holt-Winters", save_path=os.path.join(plot_dir, "residual_hw.png"))
    plot_residual_acf(hw_model.resid, "Holt-Winters", save_path=os.path.join(plot_dir, "residual_acf_hw.png"))

    results.append(calculate_metrics(test, hw_fc, "Holt-Winters"))
    plot_forecast(train, test, hw_fc, "Holt-Winters", save_path=os.path.join(plot_dir, "forecast_hw.png"))

    try:
        prophet_model = build_prophet(train)
        prophet_fc = make_prophet_forecast(prophet_model, steps=forecast_steps)
        prophet_fc.index = test.index
        results.append(calculate_metrics(test, prophet_fc, "Prophet"))
        plot_forecast(train, test, prophet_fc, "Prophet", save_path=os.path.join(plot_dir, "forecast_prophet.png"))
    except Exception as e:
        print(f"Error running Prophet: {e}")

    try:
        xgb_model, xgb_lags = build_xgboost(train, lags=24)
        xgb_fc = make_xgboost_forecast(xgb_model, train, steps=forecast_steps, lags=xgb_lags)
        xgb_fc.index = test.index
        results.append(calculate_metrics(test, xgb_fc, "XGBoost"))
        plot_forecast(train, test, xgb_fc, "XGBoost", save_path=os.path.join(plot_dir, "forecast_xgboost.png"))
    except Exception as e:
        print(f"Error running XGBoost: {e}")

    try:
        lstm_model, lstm_scaler, lstm_lags = build_lstm(train, lags=24, epochs=15)
        lstm_fc = make_lstm_forecast(lstm_model, lstm_scaler, train, steps=forecast_steps, lags=lstm_lags)
        lstm_fc.index = test.index
        results.append(calculate_metrics(test, lstm_fc, "LSTM"))
        plot_forecast(train, test, lstm_fc, "LSTM", save_path=os.path.join(plot_dir, "forecast_lstm.png"))
    except Exception as e:
        print(f"Error running LSTM: {e}")

    print("\n>>> STEP 6 & 7: Model Comparison & Evaluation...")
    df_results = display_metrics_table(results)
    
    best_idx = df_results['RMSE'].idxmin()
    best_model_name = df_results.loc[best_idx, 'Model']
    print(f"[BEST MODEL] Best Performing Model identified as: {best_model_name} (Lowest RMSE)")
    
    results_path = os.path.join(script_dir, "model_results.txt")
    with open(results_path, "w") as f:
        f.write(df_results.to_string(index=False))

    define_interpretation(best_model=best_model_name)
    
    print(f"[COMPLETE] PHASE 2 COMPLETE. Check the '{plot_dir}' directory for visualizations.")

if __name__ == "__main__":
    main()
