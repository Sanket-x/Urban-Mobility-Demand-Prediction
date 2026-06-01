from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of the week (0=Monday, 6=Sunday)")
    location: str = Field(..., description="Pickup Location")

class PredictResponse(BaseModel):
    base_predicted_demand: float = Field(..., description="Original raw predicted demand before boost")
    predicted_demand: float = Field(..., description="Final predicted demand count for next hour")
    demand_level: str = Field(..., description="Demand level category (Low, Medium, High)")
    forecast_hours: list[int] = Field(default=[], description="Array of forecasted hours (e.g., [18, 19, 20])")
    forecast_demands: list[float] = Field(default=[], description="Array of forecasted demand values")
    actual_hours: list[int] = Field(default=[], description="Array of recent historical hours")
    actual_demands: list[float] = Field(default=[], description="Array of recent historical demand values")
    vehicle_recommendation: str = Field(default="", description="Vehicle recommendation text")
    surge_alert: bool = Field(default=False, description="Whether to show surge alert")
    area_warning: bool = Field(default=False, description="Whether to show high demand area warning")
    spike_risk: str = Field(default="✅ Low Spike Risk", description="Spike risk level")
    spike_probability: float = Field(default=0.0, description="Raw probability of a demand spike")
    vehicle_demand: dict = Field(default={}, description="Per-vehicle-type estimated demand values")
    city_avg_demand: float = Field(default=0.0, description="City-wide average demand for this hour+day")
    explainability: list[str] = Field(default=[], description="Dynamic explanation bullets for the prediction")
