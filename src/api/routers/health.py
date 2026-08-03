from fastapi import APIRouter
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    version: str

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", version="0.1.0")
