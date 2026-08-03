from fastapi import FastAPI
from src.api.routers import health, pricing

app = FastAPI(
    title="QCom Margin Optimization Engine",
    description="Dynamic delivery fee optimization API",
    version="0.1.0"
)

from fastapi.responses import RedirectResponse

app.include_router(health.router, tags=["Health"])
app.include_router(pricing.router, prefix="/v1", tags=["Pricing"])

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
