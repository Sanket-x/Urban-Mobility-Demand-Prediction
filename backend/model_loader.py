import joblib
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.feature_store = None
        self.spike_model = None
        self.raw_data = None

    def load_artifacts(self):
        """
        Load the model, encoder, feature store, spike classifier, and raw CSV data.
        """
                                                      
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', 'models', 'random_forest_model.pkl')
        encoder_path = os.path.join(current_dir, '..', 'models', 'label_encoder.pkl')
        feature_store_path = os.path.join(current_dir, '..', 'models', 'feature_store.pkl')
        spike_model_path = os.path.join(current_dir, '..', 'models', 'spike_model.pkl')
        raw_data_path = os.path.join(current_dir, '..', 'data', 'Bengaluru Ola.csv')
        
        try:
            logger.info(f"Loading Random Forest model from {model_path}...")
            self.model = joblib.load(model_path)
            logger.info("Successfully loaded Random Forest model.")
            
            logger.info(f"Loading Label Encoder from {encoder_path}...")
            self.encoder = joblib.load(encoder_path)
            logger.info("Successfully loaded Label Encoder.")
            
            logger.info(f"Loading Feature Store from {feature_store_path}...")
            self.feature_store = joblib.load(feature_store_path)
            logger.info("Successfully loaded Feature Store.")
            
            logger.info(f"Loading Spike Classifier from {spike_model_path}...")
            self.spike_model = joblib.load(spike_model_path)
            logger.info("Successfully loaded Spike Classifier.")
            
            logger.info(f"Loading raw data from {raw_data_path}...")
            raw_df = pd.read_csv(raw_data_path)
            raw_df['Datetime'] = pd.to_datetime(raw_df['Date'] + ' ' + raw_df['Time'], dayfirst=True)
            raw_df['Hour'] = raw_df['Datetime'].dt.hour
            raw_df['Day_of_Week'] = raw_df['Datetime'].dt.dayofweek
            self.raw_data = raw_df
            logger.info(f"Successfully loaded raw data: {len(raw_df)} rows.")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise RuntimeError(f"Could not load model artifacts: {e}")

ml = ModelLoader()
