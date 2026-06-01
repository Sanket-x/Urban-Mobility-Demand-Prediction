import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
                         
script_dir = os.path.dirname(os.path.abspath(__file__))
                                                                            
data_path = os.path.join(script_dir, '..', 'data', 'hourly_demand_dataset.csv')

def load_and_preprocess(filepath):
    """Loads data, encodes categorical features, and prepares X, y."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    le = LabelEncoder()
    df['Location_Encoded'] = le.fit_transform(df['Pickup Location'])
    
    df['is_weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x in [5, 6] else 0)
    df['peak_hour'] = df['Hour'].apply(lambda x: 1 if x in [8, 9, 18, 19] else 0)
    
    df['Date_Only'] = pd.to_datetime(df['Date_Only'])
    
    df = df.sort_values(by=['Pickup Location', 'Date_Only', 'Hour']).reset_index(drop=True)
    
    grouped = df.groupby('Pickup Location')
    
    df['lag_1'] = grouped['Demand_Count'].shift(1)
    df['lag_24'] = grouped['Demand_Count'].shift(24)
    df['lag_168'] = grouped['Demand_Count'].shift(168)
    
    df['rolling_mean_3'] = grouped['Demand_Count'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['rolling_mean_6'] = grouped['Demand_Count'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    
    df['momentum'] = df['Demand_Count'] - df['lag_1']
    
    df['target'] = grouped['Demand_Count'].shift(-1)
    
    df['sin_hour'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['sin_day'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    df['cos_day'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
    
    df['rolling_std_3'] = grouped['Demand_Count'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    
    feature_store_df = df[df['target'].isna()].copy()
    
    df = df.dropna()
    
    df = df.sort_values(by=['Date_Only', 'Hour']).reset_index(drop=True)
    
    features = [
        'Hour', 'Day_of_Week', 'Location_Encoded', 'is_weekend', 'peak_hour',
        'lag_1', 'lag_24', 'lag_168', 'rolling_mean_3', 'rolling_mean_6', 'momentum',
        'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'rolling_std_3'
    ]
    target = 'target'
    
    X = df[features]
    y = np.log1p(df[target])                      
    
    return X, y, le, features, feature_store_df

print("Splitting data into 80% train and 20% test using time-based split...")
X, y, le, feature_names, feature_store_df = load_and_preprocess(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Starting Hyperparameter Tuning for Random Forest Regressor...")
                                                             
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)

weights_train = np.where(np.expm1(y_train) >= 4, 3, 1)

grid_search.fit(X_train, y_train, sample_weight=weights_train)
best_model = grid_search.best_estimator_

print(f"\nBest Hyperparameters found: {grid_search.best_params_}")

print("\nTraining Tuned XGBoost Regressor...")
xgb_model = XGBRegressor(
    n_estimators=300, 
    max_depth=6, 
    learning_rate=0.05, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42
)
xgb_model.fit(X_train, y_train, sample_weight=weights_train)

print("\n--- Training LSTM (Deep Learning) ---")
try:
                         
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_scaled, epochs=15, batch_size=64, verbose=0)
    
    lstm_pred_scaled = lstm_model.predict(X_test_lstm, verbose=0)
    lstm_pred = np.expm1(scaler_y.inverse_transform(lstm_pred_scaled).flatten())
    print("LSTM Training and Prediction complete.")
except Exception as e:
    print(f"Error training LSTM: {e}")
    lstm_pred = None

print("\n--- Training Prophet (Modern Statistical) ---")
                                                               
prophet_pred = None
try:
                                                                         
    p_df = pd.DataFrame({
        'ds': pd.date_range(start='2026-01-01', periods=len(y_train), freq='H'),
        'y': y_train.values
    })
    p_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    p_model.fit(p_df)
    
    p_future = p_model.make_future_dataframe(periods=len(y_test), freq='H')
    p_forecast = p_model.predict(p_future)
    prophet_pred = np.expm1(p_forecast['yhat'].tail(len(y_test)).values)
    print("Prophet Training and Prediction complete.")
except Exception as e:
    print(f"Prophet failed (likely Stan/Windows compatibility): {e}")
    prophet_pred = None

print("\n--- Training Classical Models (ARIMA, SARIMA, HW) ---")
                                                                   
try:
                   
    arima_model = ARIMA(y_train, order=(1,1,1)).fit()
    arima_pred = np.expm1(arima_model.forecast(steps=len(y_test)))
    print("ARIMA complete.")
except Exception as e:
    print(f"ARIMA failed: {e}"); arima_pred = None

try:
                                                               
    sarima_model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(0,1,1,24)).fit(disp=False)
    sarima_pred = np.expm1(sarima_model.forecast(steps=len(y_test)))
    print("SARIMA complete.")
except Exception as e:
    print(f"SARIMA failed: {e}"); sarima_pred = None

try:
                  
    hw_model = ExponentialSmoothing(y_train, seasonal_periods=24, trend='add', seasonal='add').fit()
    hw_pred = np.expm1(hw_model.forecast(steps=len(y_test)))
    print("Holt-Winters complete.")
except Exception as e:
    print(f"Holt-Winters failed: {e}"); hw_pred = None

print("\n--- Baseline Comparison ---")
                                                                          
baseline_pred = X_test['lag_1'].values
baseline_mae = mean_absolute_error(np.expm1(y_test), baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), baseline_pred))
print(f"Naive Baseline (lag_1) - MAE: {baseline_mae:.2f}, RMSE: {baseline_rmse:.2f}")

print("\nEvaluating Models on Test Set (Reversing Log Transform)...")
y_test_orig = np.expm1(y_test)

rf_pred = np.expm1(best_model.predict(X_test))
rf_mae = mean_absolute_error(y_test_orig, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test_orig, rf_pred))

xgb_pred = np.expm1(xgb_model.predict(X_test))
xgb_mae = mean_absolute_error(y_test_orig, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test_orig, xgb_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score

print("\n--- Training Spike Classification Model ---")
                                             
y_train_orig = np.expm1(y_train)
y_train_spike = np.where(y_train_orig >= 4, 1, 0)
y_test_spike = np.where(y_test_orig >= 4, 1, 0)

spike_classifier = XGBClassifier(
    n_estimators=200, 
    max_depth=5, 
    learning_rate=0.05, 
    scale_pos_weight=(len(y_train_spike) - sum(y_train_spike)) / max(1, sum(y_train_spike)),
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
spike_classifier.fit(X_train, y_train_spike)

print("\n--- Evaluating Spike Classifier ---")
spike_pred_class = spike_classifier.predict(X_test)
acc = accuracy_score(y_test_spike, spike_pred_class)
prec = precision_score(y_test_spike, spike_pred_class, zero_division=0)
rec = recall_score(y_test_spike, spike_pred_class, zero_division=0)

print(f"Spike Classifier Accuracy:  {acc:.2%}")
print(f"Spike Classifier Precision: {prec:.4f}")
print(f"Spike Classifier Recall:    {rec:.4f}")

spike_mask = y_test_orig >= 4
spike_y = y_test_orig[spike_mask]

rf_mae_spike = mean_absolute_error(spike_y, rf_pred[spike_mask]) if len(spike_y) > 0 else 0
xgb_mae_spike = mean_absolute_error(spike_y, xgb_pred[spike_mask]) if len(spike_y) > 0 else 0
baseline_mae_spike = mean_absolute_error(spike_y, baseline_pred[spike_mask]) if len(spike_y) > 0 else 0

print("\n--- Random Forest ---")
print(f"Overall MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")
print(f"Spike MAE (Demand >= 4): {rf_mae_spike:.2f}")

print("--- XGBoost ---")
print(f"Overall MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}")
print(f"Spike MAE (Demand >= 4): {xgb_mae_spike:.2f}")

print("--- Naive Baseline ---")
print(f"Overall MAE: {baseline_mae:.2f}")
print(f"Spike MAE (Demand >= 4): {baseline_mae_spike:.2f}")

if xgb_mae < rf_mae and xgb_rmse < rf_rmse:
    best_overall_name = "XGBoost Regressor"
    final_pred = xgb_pred
    final_model = xgb_model
    final_spike_mae = xgb_mae_spike
elif rf_mae < xgb_mae and rf_rmse < xgb_rmse:
    best_overall_name = "Random Forest Regressor"
    final_pred = rf_pred
    final_model = best_model
    final_spike_mae = rf_mae_spike
else:
    best_overall_name = "Mixed (Depends on metric) - Defaulting to XGBoost"
    final_pred = xgb_pred if xgb_mae < rf_mae else rf_pred
    final_model = xgb_model if xgb_mae < rf_mae else best_model
    final_spike_mae = xgb_mae_spike if xgb_mae < rf_mae else rf_mae_spike

print(f"\nBest Performing ML Model: {best_overall_name}")
                    
diff_baseline = baseline_mae - min(xgb_mae, rf_mae)
print(f"Explanation: ML Model beats naive baseline by {diff_baseline:.2f} MAE globally.")

print("\nFirst 10 Predictions (Best Model) vs Actual Values:")
results = pd.DataFrame({
    'Actual Demand': y_test_orig.values[:10],
    'Predicted Demand': np.round(final_pred[:10]).astype(int)
})
print(results)

print(f"\nFeature Importance Analysis ({best_overall_name}):")
importances = final_model.feature_importances_
for feature, imp in zip(feature_names, importances):
    print(f" - {feature}: {imp*100:.2f}%")

model_path = os.path.join(script_dir, 'random_forest_model.pkl')
le_path = os.path.join(script_dir, 'label_encoder.pkl')
print(f"\nSaving model to {model_path}...")
joblib.dump(best_model, model_path)
print(f"Saving LabelEncoder to {le_path}...")
joblib.dump(le, le_path)

results_path = os.path.join(script_dir, 'model_results.txt')

comparison_data = [
    {"Model": "Naive Baseline", "MAE": baseline_mae, "RMSE": baseline_rmse, "MAPE": calculate_mape(y_test_orig, baseline_pred)},
]

if arima_pred is not None:
    comparison_data.append({"Model": "ARIMA", "MAE": mean_absolute_error(y_test_orig, arima_pred), "RMSE": np.sqrt(mean_squared_error(y_test_orig, arima_pred)), "MAPE": calculate_mape(y_test_orig, arima_pred)})
if sarima_pred is not None:
    comparison_data.append({"Model": "SARIMA", "MAE": mean_absolute_error(y_test_orig, sarima_pred), "RMSE": np.sqrt(mean_squared_error(y_test_orig, sarima_pred)), "MAPE": calculate_mape(y_test_orig, sarima_pred)})
if hw_pred is not None:
    comparison_data.append({"Model": "Holt-Winters", "MAE": mean_absolute_error(y_test_orig, hw_pred), "RMSE": np.sqrt(mean_squared_error(y_test_orig, hw_pred)), "MAPE": calculate_mape(y_test_orig, hw_pred)})

comparison_data.append({"Model": "Random Forest", "MAE": rf_mae, "RMSE": rf_rmse, "MAPE": calculate_mape(y_test_orig, rf_pred)})
comparison_data.append({"Model": "XGBoost", "MAE": xgb_mae, "RMSE": xgb_rmse, "MAPE": calculate_mape(y_test_orig, xgb_pred)})

if lstm_pred is not None:
    comparison_data.append({"Model": "LSTM (DL)", "MAE": mean_absolute_error(y_test_orig, lstm_pred), "RMSE": np.sqrt(mean_squared_error(y_test_orig, lstm_pred)), "MAPE": calculate_mape(y_test_orig, lstm_pred)})

if prophet_pred is not None:
    comparison_data.append({"Model": "Prophet", "MAE": mean_absolute_error(y_test_orig, prophet_pred), "RMSE": np.sqrt(mean_squared_error(y_test_orig, prophet_pred)), "MAPE": calculate_mape(y_test_orig, prophet_pred)})

df_comp = pd.DataFrame(comparison_data)

with open(results_path, 'w') as f:
    f.write(f"--- Final Project: Model Comparison Results ---\n")
    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("MASTER COMPARISON TABLE (All Metrics):\n")
    f.write(df_comp.to_string(index=False) + "\n\n")
    
    f.write(f"Best Performing Model: {best_overall_name}\n")
    f.write(f"Explanation: ML Model improves baseline globally by {diff_baseline:.2f} MAE.\n\n")
    
    f.write("Sample Predictions (Best Model) vs Actual Values:\n")
    f.write(results.to_string() + "\n\n")
    
    f.write(f"Feature Importance Analysis ({best_overall_name}):\n")
    for feature, imp in zip(feature_names, importances):
        f.write(f" - {feature}: {imp*100:.2f}%\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("SPIKE CLASSIFIER (ANOMALY DETECTION) PERFORMANCE:\n")
    f.write(f" - Accuracy:  {acc*100:.2f}%\n")
    f.write(f" - Precision: {prec:.4f}\n")
    f.write(f" - Recall:    {rec:.4f}\n")
    f.write("="*60 + "\n\n")
    
    f.write("="*60 + "\n")
    f.write("ACADEMIC CHALLENGES & LIMITATIONS:\n")
    f.write("- Prophet: Implementation attempted but often faces Stan/C++ compiler issues on Windows.\n")
    f.write("- LSTM: Requires significant computational resources and data sequence tuning.\n")
    f.write("="*60 + "\n")

print(f"\nResults successfully saved to: {results_path}")

model_save_path = os.path.join(script_dir, 'random_forest_model.pkl')
le_save_path = os.path.join(script_dir, 'label_encoder.pkl')
feature_store_path = os.path.join(script_dir, 'feature_store.pkl')
spike_model_path = os.path.join(script_dir, 'spike_model.pkl')

joblib.dump(best_model, model_save_path)
joblib.dump(le, le_save_path)
joblib.dump(feature_store_df, feature_store_path)
joblib.dump(spike_classifier, spike_model_path)

print(f"\nSaved Random Forest model to: {model_save_path}")
print(f"Saved Label Encoder to: {le_save_path}")
print(f"Saved Feature Store to: {feature_store_path}")
print(f"Saved Spike Classifier to: {spike_model_path}")
