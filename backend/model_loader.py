import joblib
import os
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.encoder = None

    def load_artifacts(self):
        """
        Load the model and encoder from the models directory.
        """
        # Determine paths relative to the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', 'models', 'random_forest_model.pkl')
        encoder_path = os.path.join(current_dir, '..', 'models', 'label_encoder.pkl')
        
        try:
            logger.info(f"Loading Random Forest model from {model_path}...")
            self.model = joblib.load(model_path)
            logger.info("Successfully loaded Random Forest model.")
            
            logger.info(f"Loading Label Encoder from {encoder_path}...")
            self.encoder = joblib.load(encoder_path)
            logger.info("Successfully loaded Label Encoder.")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise RuntimeError(f"Could not load model artifacts: {e}")

# Create a singleton instance
ml = ModelLoader()
