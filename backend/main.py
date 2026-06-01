import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import numpy as np

from model_loader import ml
from schemas import PredictRequest, PredictResponse
from utils import encode_location, extract_features, get_demand_level, build_feature_array, compute_bi_insights, compute_vehicle_demand, compute_city_avg_demand, generate_explainability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ml_service")

@asynccontextmanager
async def lifespan(app: FastAPI):
             
    logger.info("Starting up FastAPI application...")
    ml.load_artifacts()
    yield
              
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
    logger.info(f"Received prediction request: hour={request.hour}, day={request.day_of_week}, location={request.location}")
    
    encoded_loc = encode_location(request.location, ml.encoder)
    
    features = build_feature_array(request.location, encoded_loc, request.hour, request.day_of_week, ml.feature_store)
    
    try:
                                
        prediction_log = ml.model.predict(features)[0]
        prediction = float(np.expm1(prediction_log))
        
        spike_prob = float(ml.spike_model.predict_proba(features)[0][1])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")
        
    base_prediction = prediction
    if spike_prob > 0.6:
        spike_risk = "⚠️ High"
        prediction *= 1.2
    elif spike_prob > 0.3:
        spike_risk = "⚡ Moderate"
    else:
        spike_risk = "✅ Low"

    forecast_hours = []
    forecast_demands = []
    
    for i in range(1, 6):               
        f_hour = (request.hour + i) % 24
        f_day = request.day_of_week
        if (request.hour + i) >= 24:
            f_day = (f_day + 1) % 7
            
        f_feats = build_feature_array(request.location, encoded_loc, f_hour, f_day, ml.feature_store)
        f_pred_log = ml.model.predict(f_feats)[0]
        f_pred = float(np.expm1(f_pred_log))
        
        f_spike_prob = float(ml.spike_model.predict_proba(f_feats)[0][1])
        if f_spike_prob > 0.6:
            f_pred *= 1.2
            
        forecast_hours.append(f_hour)
        forecast_demands.append(round(f_pred, 2))
        
    actual_hours = []
    actual_demands = []
    
    loc_data = ml.feature_store[ml.feature_store['Pickup Location'] == request.location]
    if not loc_data.empty:
        last_few = loc_data.tail(5)
        actual_hours = last_few['Hour'].tolist()
        actual_demands = last_few['Demand_Count'].tolist()
    
    bi = compute_bi_insights(request.location, float(prediction), ml.feature_store)
    
    veh_demand = compute_vehicle_demand(
        request.location, request.hour, float(prediction), ml.raw_data
    )
    
    city_avg = compute_city_avg_demand(request.hour, request.day_of_week, ml.raw_data)
    
    explain_bullets = generate_explainability(
        request.location, request.hour, request.day_of_week,
        float(prediction), ml.feature_store, ml.raw_data
    )
    
    demand_level = get_demand_level(prediction)
    logger.info(f"Prediction successful: {prediction:.2f} ({demand_level})")
    
    return PredictResponse(
        base_predicted_demand=round(float(base_prediction), 2),
        predicted_demand=round(float(prediction), 2),
        demand_level=demand_level,
        forecast_hours=forecast_hours,
        forecast_demands=forecast_demands,
        actual_hours=actual_hours,
        actual_demands=actual_demands,
        vehicle_recommendation=bi["vehicle_recommendation"],
        surge_alert=bi["surge_alert"],
        area_warning=bi["area_warning"],
        spike_risk=spike_risk,
        spike_probability=spike_prob,
        vehicle_demand=veh_demand,
        city_avg_demand=city_avg,
        explainability=explain_bullets
    )
