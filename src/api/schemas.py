from pydantic import BaseModel, Field
from typing import List, Literal

class PricingRequest(BaseModel):
    basket_value: float = Field(gt=0, lt=50000, description="Order basket value in INR")
    basket_margin: float = Field(ge=0, description="Gross margin on basket in INR")
    basket_weight_kg: float = Field(gt=0, lt=100)
    num_items: int = Field(ge=1, le=100)
    distance_km: float = Field(gt=0, lt=50)
    estimated_delivery_time_min: int = Field(ge=1, le=180)
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    traffic_level: Literal["low", "medium", "high"]
    price_sensitivity_score: float = Field(ge=0.0, le=1.0)
    delivery_cost_potential: float = Field(ge=0)
    conversion_prob_stage1: float = Field(ge=0.0, le=1.0)

class PricingResponse(BaseModel):
    optimal_fee: float
    expected_conversion: float
    expected_cm2: float
    baseline_conversion: float
    cm2_uplift_pct: float

class BatchPricingRequest(BaseModel):
    requests: List[PricingRequest] = Field(..., max_length=100)

class BatchPricingResponse(BaseModel):
    results: List[PricingResponse]
