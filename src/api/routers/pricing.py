from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Header, Response
from src.api.schemas import PricingRequest, PricingResponse, BatchPricingRequest, BatchPricingResponse
from src.domain.pricing_engine import PricingEngine
from src.api.idempotency import idempotency_store

router = APIRouter()

# Dependency to inject the pricing engine
def get_pricing_engine() -> PricingEngine:
    try:
        return PricingEngine()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded")

@router.post("/optimize", response_model=PricingResponse)
def optimize_fee(
    request: PricingRequest, 
    response: Response,
    x_idempotency_key: Optional[str] = Header(None, alias="X-Idempotency-Key"),
    engine: PricingEngine = Depends(get_pricing_engine)
):
    try:
        if x_idempotency_key:
            cached_data = idempotency_store.get(x_idempotency_key)
            if cached_data:
                response.headers["X-Cache"] = "HIT"
                return PricingResponse(**cached_data)

        res = engine.optimize_fee(request.model_dump())
        response_model = PricingResponse(**res)

        if x_idempotency_key:
            response.headers["X-Cache"] = "MISS"
            idempotency_store.set(x_idempotency_key, response_model.model_dump())

        return response_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize/batch", response_model=BatchPricingResponse)
def optimize_fee_batch(
    request: BatchPricingRequest, 
    response: Response,
    x_idempotency_key: Optional[str] = Header(None, alias="X-Idempotency-Key"),
    engine: PricingEngine = Depends(get_pricing_engine)
):
    try:
        if x_idempotency_key:
            cached_data = idempotency_store.get(x_idempotency_key)
            if cached_data:
                response.headers["X-Cache"] = "HIT"
                return BatchPricingResponse(**cached_data)

        results = []
        for req in request.requests:
            res = engine.optimize_fee(req.model_dump())
            results.append(PricingResponse(**res))
        
        batch_response = BatchPricingResponse(results=results)

        if x_idempotency_key:
            response.headers["X-Cache"] = "MISS"
            idempotency_store.set(x_idempotency_key, batch_response.model_dump())

        return batch_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
