import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import numpy as np

from model_loader import ml
from schemas import PredictRequest, PredictResponse
from utils import encode_location, extract_features, get_demand_level

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ml_service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application...")
    ml.load_artifacts()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application...")

app = FastAPI(
    title="Demand Prediction API",
    description="API for predicting machine learning demand system using Random Forest",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", summary="Health Check")
async def health_check():
    """
    Health check endpoint to ensure the service is running and models are loaded.
    """
    if ml.model is None or ml.encoder is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly.")
    return {"status": "ok", "message": "Service is healthy and models are loaded."}

@app.post("/predict", response_model=PredictResponse, summary="Predict Demand")
async def predict_demand(request: PredictRequest):
    """
    Predict demand based on hour, day of the week, and pickup location.
    
    ### Example Request:
    ```bash
    curl -X POST http://127.0.0.1:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"hour": 8, "day_of_week": 1, "location": "Manhattan"}'
    ```
    """
    logger.info(f"Received prediction request: hour={request.hour}, day={request.day_of_week}, location={request.location}")
    
    # 1. Encode location
    encoded_loc = encode_location(request.location, ml.encoder)
    
    # 2. Extract engineered features
    is_weekend, peak_hour = extract_features(request.hour, request.day_of_week)
    
    # 3. Prepare feature array for prediction
    # Features must match the training order: ['Hour', 'Day_of_Week', 'Location_Encoded', 'is_weekend', 'peak_hour']
    features = np.array([[request.hour, request.day_of_week, encoded_loc, is_weekend, peak_hour]])
    
    # 4. Predict
    try:
        prediction = ml.model.predict(features)[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")
    
    # 5. Classify demand level
    demand_level = get_demand_level(prediction)
    
    logger.info(f"Prediction successful: {prediction:.2f} ({demand_level})")
    
    return PredictResponse(
        predicted_demand=round(float(prediction), 2),
        demand_level=demand_level
    )
