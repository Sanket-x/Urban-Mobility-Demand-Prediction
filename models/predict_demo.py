import pandas as pd
import joblib
import os

def load_and_predict(input_data):
    """
    Demonstrates how to load the saved Random Forest model and LabelEncoder
    to make predictions on new data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'random_forest_model.pkl')
    le_path = os.path.join(script_dir, 'label_encoder.pkl')
    
    # 1. Load the model and LabelEncoder
    print("Loading model and LabelEncoder...")
    rf_model = joblib.load(model_path)
    le = joblib.load(le_path)
    
    # input_data is a DataFrame with raw features: 
    # ['Hour', 'Day_of_Week', 'Pickup Location']
    df = input_data.copy()
    
    # 2. Preprocess the input data
    # Encode Location (must have been seen during training)
    df['Location_Encoded'] = le.transform(df['Pickup Location'])
    
    # Add engineered features
    df['is_weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x in [5, 6] else 0)
    df['peak_hour'] = df['Hour'].apply(lambda x: 1 if x in [8, 9, 18, 19] else 0)
    
    features = ['Hour', 'Day_of_Week', 'Location_Encoded', 'is_weekend', 'peak_hour']
    X_new = df[features]
    
    # 3. Make predictions
    print("Making predictions...")
    predictions = rf_model.predict(X_new)
    df['Predicted_Demand'] = predictions.round().astype(int)
    
    return df

if __name__ == "__main__":
    # Example usage:
    new_data = pd.DataFrame({
        'Hour': [8, 14, 19],
        'Day_of_Week': [0, 5, 6], # Monday, Saturday, Sunday
        'Pickup Location': ['Area-1', 'Area-10', 'Area-50']
    })
    
    try:
        results = load_and_predict(new_data)
        print("\nPredicted Demands:")
        print(results[['Hour', 'Day_of_Week', 'Pickup Location', 'Predicted_Demand']])
    except FileNotFoundError:
        print("\nError: Model files not found. Please run train_demand_model.py first to generate the .pkl files.")
