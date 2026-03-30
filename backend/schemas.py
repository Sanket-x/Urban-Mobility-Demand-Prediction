from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of the week (0=Monday, 6=Sunday)")
    location: str = Field(..., description="Pickup Location")

class PredictResponse(BaseModel):
    predicted_demand: float = Field(..., description="Predicted demand count")
    demand_level: str = Field(..., description="Demand level category (Low, Medium, High)")
