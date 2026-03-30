from sklearn.preprocessing import LabelEncoder
from fastapi import HTTPException
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def encode_location(location: str, le: LabelEncoder) -> int:
    """
    Encode the pickup location using the fitted LabelEncoder.
    Raises a 400 exception if the location is unknown (not seen during training).
    """
    if location not in le.classes_:
        logger.warning(f"Unknown location requested: {location}")
        raise HTTPException(status_code=400, detail=f"Unknown location: '{location}'. Please provide a valid location.")
    
    # transform requires an array-like structure and returns an array
    return int(le.transform([location])[0])

def extract_features(hour: int, day_of_week: int) -> Tuple[int, int]:
    """
    Extract is_weekend and peak_hour features from hour and day_of_week.
    
    is_weekend: 1 if day_of_week is 5 (Saturday) or 6 (Sunday), else 0
    peak_hour: 1 if hour in [8, 9, 18, 19], else 0
    """
    is_weekend = 1 if day_of_week in [5, 6] else 0
    peak_hour = 1 if hour in [8, 9, 18, 19] else 0
    return is_weekend, peak_hour

def get_demand_level(predicted_demand: float) -> str:
    """
    Classify the predicted demand into Low, Medium, or High categories.
    - Low: < 2
    - Medium: 2-3
    - High: > 3
    """
    if predicted_demand < 2:
        return "Low"
    elif predicted_demand <= 3:
        return "Medium"
    else:
        return "High"
