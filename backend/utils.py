from sklearn.preprocessing import LabelEncoder
from fastapi import HTTPException
from typing import Tuple, Dict, Any
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def encode_location(location: str, le: LabelEncoder) -> int:
    if location not in le.classes_:
        logger.warning(f"Unknown location requested: {location}")
        raise HTTPException(status_code=400, detail=f"Unknown location: '{location}'. Please provide a valid location.")
    return int(le.transform([location])[0])

def extract_features(hour: int, day_of_week: int) -> Tuple[int, int]:
    is_weekend = 1 if day_of_week in [5, 6] else 0
    peak_hour = 1 if hour in [8, 9, 18, 19] else 0
    return is_weekend, peak_hour

def get_demand_level(predicted_demand: float) -> str:
    if predicted_demand < 2:
        return "Low"
    elif predicted_demand <= 3:
        return "Medium"
    else:
        return "High"

def build_feature_array(location: str, location_enc: int, hour: int, day_of_week: int, feature_store: pd.DataFrame) -> np.ndarray:
    """
    Build the 16-feature array for prediction: 
    ['Hour', 'Day_of_Week', 'Location_Encoded', 'is_weekend', 'peak_hour', 'lag_1', 'lag_24', 'lag_168', 'rolling_mean_3', 'rolling_mean_6', 'momentum', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'rolling_std_3']
    """
    is_weekend, peak_hour = extract_features(hour, day_of_week)
    
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)
    sin_day = np.sin(2 * np.pi * day_of_week / 7)
    cos_day = np.cos(2 * np.pi * day_of_week / 7)
    
    loc_data = feature_store[feature_store['Pickup Location'] == location]
    
    lag_1, lag_24, lag_168 = 0, 0, 0
    rm_3, rm_6, momentum, rstd_3 = 0, 0, 0, 0
    
    if not loc_data.empty:
                                                                          
        last_row = loc_data.iloc[-1]
        lag_1 = last_row.get('lag_1', 0)
        lag_24 = last_row.get('lag_24', 0)
        lag_168 = last_row.get('lag_168', 0)
        rm_3 = last_row.get('rolling_mean_3', 0)
        rm_6 = last_row.get('rolling_mean_6', 0)
        momentum = last_row.get('momentum', 0)
        rstd_3 = last_row.get('rolling_std_3', 0)
        
        lag_1 = lag_1 if not pd.isna(lag_1) else 0
        lag_24 = lag_24 if not pd.isna(lag_24) else 0
        lag_168 = lag_168 if not pd.isna(lag_168) else 0
        rm_3 = rm_3 if not pd.isna(rm_3) else 0
        rm_6 = rm_6 if not pd.isna(rm_6) else 0
        momentum = momentum if not pd.isna(momentum) else 0
        rstd_3 = rstd_3 if not pd.isna(rstd_3) else 0
        
    feature_dict = {
        'Hour': [hour],
        'Day_of_Week': [day_of_week],
        'Location_Encoded': [location_enc],
        'is_weekend': [is_weekend],
        'peak_hour': [peak_hour],
        'lag_1': [lag_1],
        'lag_24': [lag_24],
        'lag_168': [lag_168],
        'rolling_mean_3': [rm_3],
        'rolling_mean_6': [rm_6],
        'momentum': [momentum],
        'sin_hour': [sin_hour],
        'cos_hour': [cos_hour],
        'sin_day': [sin_day],
        'cos_day': [cos_day],
        'rolling_std_3': [rstd_3]
    }
    return pd.DataFrame(feature_dict)

def compute_bi_insights(location: str, predicted_demand: float, feature_store: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute Vehicle Recommendation, Surge Alert, and Area Warning based on the current state and forecast.
    """
    loc_data = feature_store[feature_store['Pickup Location'] == location]
    momentum = 0
    
    if not loc_data.empty:
        last_row = loc_data.iloc[-1]
        momentum = last_row.get('momentum', 0)
        if pd.isna(momentum): momentum = 0
        
    if predicted_demand > 3 and momentum > 0:
        vehicle_rec = "Auto: High | Mini: Medium | Sedan: Low"
    elif predicted_demand <= 2:
        vehicle_rec = "Auto: Low | Mini: Low | Sedan: Medium"
    else:
        vehicle_rec = "Auto: Medium | Mini: Medium | Sedan: Medium"
        
    simulated_ctat = 5 + (predicted_demand * 1.5) + (momentum * 2)
    simulated_cancel_rate = min(100, max(0, (momentum * 5) + (predicted_demand * 4)))
    
    is_demand_up = predicted_demand > 3 or momentum > 1
    surge_alert = bool(is_demand_up and simulated_cancel_rate > 15 and simulated_ctat > 10)
    
    area_warning = bool(predicted_demand > 5 or simulated_cancel_rate > 20)
    
    return {
        "vehicle_recommendation": vehicle_rec,
        "surge_alert": surge_alert,
        "area_warning": area_warning
    }

def compute_vehicle_demand(location: str, hour: int, predicted_demand: float, raw_data: pd.DataFrame) -> Dict[str, float]:
    """
    Compute per-vehicle-type estimated demand using historical proportions
    from the raw dataset, grouped by (Pickup Location, Hour, Vehicle Type).
    The proportions are then scaled by the model's total predicted demand.
    
    Vehicle types are standardized to 4 categories: Auto, Mini, Sedan, Bike.
    Premium variants (Prime Sedan, Prime SUV) are mapped into Sedan.
    """
    VEHICLE_TYPE_MAP = {
        "Auto": "Auto",
        "Mini": "Mini",
        "Sedan": "Sedan",
        "Bike": "Bike",
        "Prime Sedan": "Sedan",
        "Prime SUV": "SUV",
        "SUV": "SUV",
    }
    STANDARD_TYPES = ["Auto", "Mini", "Sedan", "SUV", "Bike"]
    
    if raw_data is None or raw_data.empty:
        return {}
    
    loc_hour_data = raw_data[
        (raw_data['Pickup Location'] == location) & 
        (raw_data['Hour'] == hour)
    ]
    
    if loc_hour_data.empty:
                                          
        loc_hour_data = raw_data[raw_data['Pickup Location'] == location]
    
    if loc_hour_data.empty:
                                                                
        loc_hour_data = raw_data[raw_data['Hour'] == hour]
    
    if loc_hour_data.empty:
        return {}
    
    mapped_types = loc_hour_data['Vehicle Type'].map(VEHICLE_TYPE_MAP).fillna('Sedan')
    veh_counts = mapped_types.value_counts()
    total_count = veh_counts.sum()
    
    if total_count == 0:
        return {}
    
    vehicle_demand = {}
    for vtype in STANDARD_TYPES:
        count = veh_counts.get(vtype, 0)
        proportion = count / total_count
        vehicle_demand[vtype] = round(proportion * predicted_demand, 2)
    
    current_sum = sum(vehicle_demand.values())
    residual = round(predicted_demand - current_sum, 2)
    if residual != 0 and vehicle_demand:
                                              
        largest_type = max(vehicle_demand, key=vehicle_demand.get)
        vehicle_demand[largest_type] = round(vehicle_demand[largest_type] + residual, 2)
    
    return vehicle_demand

def compute_city_avg_demand(hour: int, day_of_week: int, raw_data: pd.DataFrame) -> float:
    """
    Compute the city-wide average demand per area for the given hour and day_of_week
    from historical data.
    """
    if raw_data is None or raw_data.empty:
        return 0.0
    
    matched = raw_data[
        (raw_data['Hour'] == hour) & 
        (raw_data['Day_of_Week'] == day_of_week)
    ]
    
    if matched.empty:
                             
        matched = raw_data[raw_data['Hour'] == hour]
    
    if matched.empty:
        return 0.0
    
    n_areas = matched['Pickup Location'].nunique()
    n_dates = matched['Date'].nunique() if 'Date' in matched.columns else 1
    
    if n_areas == 0 or n_dates == 0:
        return 0.0
    
    avg_demand = len(matched) / n_areas / max(n_dates, 1)
    return round(avg_demand, 2)

def generate_explainability(location: str, hour: int, day_of_week: int, 
                           predicted_demand: float, feature_store: pd.DataFrame,
                           raw_data: pd.DataFrame) -> list:
    """
    Generate 2-3 dynamic explanation bullets based on actual feature values.
    No hardcoded text — every bullet is conditionally generated from data.
    """
    bullets = []
    
    loc_data = feature_store[feature_store['Pickup Location'] == location]
    
    momentum = 0
    lag_1 = 0
    lag_24 = 0
    rolling_mean_3 = 0
    rolling_mean_6 = 0
    
    if not loc_data.empty:
        last_row = loc_data.iloc[-1]
        momentum = last_row.get('momentum', 0)
        if pd.isna(momentum): momentum = 0
        lag_1 = last_row.get('lag_1', 0)
        if pd.isna(lag_1): lag_1 = 0
        lag_24 = last_row.get('lag_24', 0)
        if pd.isna(lag_24): lag_24 = 0
        rolling_mean_3 = last_row.get('rolling_mean_3', 0)
        if pd.isna(rolling_mean_3): rolling_mean_3 = 0
        rolling_mean_6 = last_row.get('rolling_mean_6', 0)
        if pd.isna(rolling_mean_6): rolling_mean_6 = 0
    
    is_peak = hour in [8, 9, 18, 19]
    if raw_data is not None and not raw_data.empty:
        hour_total = len(raw_data[raw_data['Hour'] == hour])
        overall_avg = len(raw_data) / 24
        if hour_total > overall_avg * 1.2:
            bullets.append(f"📊 Historically high demand at {hour}:00 — this hour sees {((hour_total / overall_avg) - 1) * 100:.0f}% more bookings than the daily average")
        elif hour_total < overall_avg * 0.8:
            bullets.append(f"📉 Historically lower demand at {hour}:00 — this hour sees {(1 - (hour_total / overall_avg)) * 100:.0f}% fewer bookings than average")
        elif is_peak:
            bullets.append(f"⏰ Peak hour detected ({hour}:00) — rush hour traffic drives elevated demand")
    
    if momentum > 0.5:
        bullets.append(f"📈 Recent upward momentum detected (+{momentum:.1f}) — demand has been rising in recent hours")
    elif momentum < -0.5:
        bullets.append(f"📉 Recent downward trend detected ({momentum:.1f}) — demand has been declining in recent hours")
    else:
        if rolling_mean_3 > rolling_mean_6 * 1.1:
            bullets.append(f"📈 Short-term average ({rolling_mean_3:.1f}) exceeds long-term ({rolling_mean_6:.1f}) — rising demand pattern")
        elif rolling_mean_3 < rolling_mean_6 * 0.9:
            bullets.append(f"📉 Short-term average ({rolling_mean_3:.1f}) below long-term ({rolling_mean_6:.1f}) — cooling demand pattern")
    
    is_weekend = day_of_week in [5, 6]
    day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    day_name = day_names.get(day_of_week, "")
    if is_weekend:
        bullets.append(f"🗓️ Weekend pattern ({day_name}) — leisure and social mobility typically drives different demand profiles")
    else:
        bullets.append(f"🗓️ Weekday pattern ({day_name}) — commuter-driven demand dominates the profile")
    
    if lag_24 > 0 and abs(predicted_demand - lag_24) > 0.5:
        if predicted_demand > lag_24:
            bullets.append(f"⬆️ Predicted demand ({predicted_demand:.1f}) is higher than same hour yesterday ({lag_24:.1f})")
        else:
            bullets.append(f"⬇️ Predicted demand ({predicted_demand:.1f}) is lower than same hour yesterday ({lag_24:.1f})")
    
    return bullets[:3] if len(bullets) >= 2 else bullets
