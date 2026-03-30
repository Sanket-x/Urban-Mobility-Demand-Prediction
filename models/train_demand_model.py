import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import joblib
# --- 1. Load Dataset ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes the script is inside the 'models' folder and data is in '../data/'
data_path = os.path.join(script_dir, '..', 'data', 'hourly_demand_dataset.csv')

def load_and_preprocess(filepath):
    """Loads data, encodes categorical features, and prepares X, y."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # 2. Encode Pickup Location
    le = LabelEncoder()
    df['Location_Encoded'] = le.fit_transform(df['Pickup Location'])
    
    # 3. Add New Features
    df['is_weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x in [5, 6] else 0)
    df['peak_hour'] = df['Hour'].apply(lambda x: 1 if x in [8, 9, 18, 19] else 0)
    
    # 4. Define Features and Target
    features = ['Hour', 'Day_of_Week', 'Location_Encoded', 'is_weekend', 'peak_hour']
    target = 'Demand_Count'
    
    X = df[features]
    y = df[target]
    
    return X, y, le, features

# --- 5. Split Data ---
print("Splitting data into 80% train and 20% test...")
X, y, le, feature_names = load_and_preprocess(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 6. Train Random Forest (with Hyperparameter Tuning) ---
print("Starting Hyperparameter Tuning for Random Forest Regressor...")
# Keeping the parameter grid small so it runs reasonably fast
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, scoring='neg_mean_absolute_error', n_jobs=1)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"\nBest Hyperparameters found: {grid_search.best_params_}")

# --- 7. Train XGBoost ---
print("\nTraining XGBoost Regressor...")
xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# --- 8. Evaluate Models ---
print("\nEvaluating Models on Test Set...")
rf_pred = best_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

xgb_pred = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

print("\n--- Random Forest ---")
print(f"MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")

print("--- XGBoost ---")
print(f"MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}")

# Compare and pick best
if xgb_mae < rf_mae and xgb_rmse < rf_rmse:
    best_overall_name = "XGBoost Regressor"
    final_pred = xgb_pred
    final_model = xgb_model
elif rf_mae < xgb_mae and rf_rmse < xgb_rmse:
    best_overall_name = "Random Forest Regressor"
    final_pred = rf_pred
    final_model = best_model
else:
    best_overall_name = "Mixed (Depends on metric)"
    final_pred = xgb_pred if xgb_mae < rf_mae else rf_pred
    final_model = xgb_model if xgb_mae < rf_mae else best_model

print(f"\nBest Performing Model: {best_overall_name}")
# Simple explanation
diff_mae = abs(rf_mae - xgb_mae)
winner = "XGBoost" if xgb_mae < rf_mae else "Random Forest"
print(f"Explanation: {winner} performs better because its MAE is lower by {diff_mae:.2f}.")

# --- 9. Show first 10 predictions vs actual ---
print("\nFirst 10 Predictions (Best Model) vs Actual Values:")
results = pd.DataFrame({
    'Actual Demand': y_test.values[:10],
    'Predicted Demand': np.round(final_pred[:10]).astype(int)
})
print(results)

# --- Feature Importance Analysis ---
print(f"\nFeature Importance Analysis ({best_overall_name}):")
importances = final_model.feature_importances_
for feature, imp in zip(feature_names, importances):
    print(f" - {feature}: {imp*100:.2f}%")

# --- Save Model and LabelEncoder ---
model_path = os.path.join(script_dir, 'random_forest_model.pkl')
le_path = os.path.join(script_dir, 'label_encoder.pkl')
print(f"\nSaving model to {model_path}...")
joblib.dump(best_model, model_path)
print(f"Saving LabelEncoder to {le_path}...")
joblib.dump(le, le_path)

# --- Save Results to File ---
results_path = os.path.join(script_dir, 'model_results.txt')
with open(results_path, 'w') as f:
    f.write(f"--- Model Comparison Results ---\n\n")
    f.write(f"Random Forest - MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}\n")
    f.write(f"XGBoost       - MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}\n\n")
    f.write(f"Best Performing Model: {best_overall_name}\n")
    f.write(f"Explanation: {winner} predictions are closer to actual demand by {diff_mae:.2f} rides per hour on average.\n\n")
    f.write("First 10 Predictions vs Actual Values:\n")
    f.write(results.to_string() + "\n\n")
    f.write(f"Feature Importance Analysis ({best_overall_name}):\n")
    for feature, imp in zip(feature_names, importances):
        f.write(f" - {feature}: {imp*100:.2f}%\n")

print(f"\nResults successfully saved to: {results_path}")

# --- 10. Save Models ---
model_save_path = os.path.join(script_dir, 'random_forest_model.pkl')
le_save_path = os.path.join(script_dir, 'label_encoder.pkl')

joblib.dump(best_model, model_save_path)
joblib.dump(le, le_save_path)

print(f"\nSaved Random Forest model to: {model_save_path}")
print(f"Saved Label Encoder to: {le_save_path}")
