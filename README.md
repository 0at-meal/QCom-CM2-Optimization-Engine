# Quick Commerce CM2 Optimization Engine

> **Aim: Maximize Delivery Margin Without Sacrificing Conversion**
> 
> An intelligent dynamic pricing system that optimizes delivery fees in real-time to improve CM2 (Contribution Margin 2) while maintaining healthy conversion rates.

---

## Problem:

In quick-commerce logistics, pricing delivery fees is a high-stakes game: 

- **If you Charge too much** → Customers abandon orders thus conversion drops.
- **If you Charge too little** → Margins erode, profitability suffers
- **If pricing is static** → Ignores real-time context (traffic, distance, customer sensitivity)

Quick Commerce is bleeding money, and traditional flat-fee models leaving margin on the table is a significant reason for the same. 

---

## Proposed Solution:

This engine uses **dynamic pricing using machine learning** to find the optimal fee for each order:

1. **Predicts conversion probability** for any delivery fee using XGBoost
2. **Optimizes CM2** by testing fee candidates (0–100 rupees)
3. **Respects business constraints**:  conversion drop ≤ 3%
4. **Adapts in real-time** based on traffic, distance, basket value, customer price sensitivity, and competitor pricing

**Result:** a double digit expected CM2 improvement in percentage terms without hurting conversion rates.

---

## Key Features

### ML-Powered Pricing
- **XGBoost Classification Model** trained on 10K+ historical orders
- **Feature Engineering** with 15+ engineered signals
  - Margin per km (cost-distance alignment)
  - Price sensitivity × basket value (customer elasticity)
  - Traffic-adjusted delivery dynamics
  - Competitor pricing context

### Offline Policy Evaluation
- Compares **historic fee → expected CM2** vs **optimized fee → expected CM2**
- Uses same conversion model for fair apples-to-apples comparison
- Simulates 500–1000 order impact

### Interactive Dashboard
- Real-time pricing recommendations
- Visual KPI tracking (CM2 uplift, conversion impact)
- Fee distribution analysis by traffic/basket segment
- Built with Streamlit

### Production Architecture & Setup

The optimizer solves a constrained CM2 maximization problem:

$$\max_{\text{fee}} \mathbb{E}[\text{CM2} \mid \text{fee}] \quad \text{subject to} \quad P(\text{convert} \mid \text{fee}) \ge P(\text{convert} \mid \text{fee}=0) - 0.03$$

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   CLIENT (curl / OMS)                   │
└─────────────────────────┬───────────────────────────────┘
                          │ POST /v1/optimize
                          │ POST /v1/optimize/batch
                          │ GET  /health
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  FASTAPI SERVICE                        │
│                                                         │
│  PricingRequest (Pydantic v2)                           │
│       → Validation → PricingEngine                      │
│                           │                             │
│             ┌─────────────▼────────────┐                │
│             │   feature_pipeline.py    │  ← Single src  │
│             │   (train + serve same)   │                 │
│             └─────────────┬────────────┘                │
│                           │                             │
│             ┌─────────────▼────────────┐                │
│             │   XGBoost Inference      │                 │
│             │   Argmax CM2 subject to  │                 │
│             │   Δconversion ≤ 3%       │                 │
│             └─────────────┬────────────┘                │
│                           │                             │
│             ┌─────────────▼────────────┐                │
│             │   PricingResponse        │                 │
│             │   (Pydantic v2)          │                 │
│             └──────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴──────────────┐
          ▼                              ▼
┌──────────────────┐          ┌──────────────────────┐
│  MLflow Tracking │          │  GitHub Actions CI   │
│                  │          │                      │
│  - Params        │          │  lint & type check   │
│  - AUC / Loss    │          │  pytest (15 tests)   │
│  - Model artifact│          │  docker build        │
│  - Run history   │          │                      │
└──────────────────┘          └──────────────────────┘
```

---

## Quickstart

### 1. Local Development
```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests (15 unit + integration tests)
pytest tests/ -v

# Train MLflow-tracked model
python src/train.py

# Start API server
uvicorn src.api.main:app --reload
```
Visit Swagger UI at `http://127.0.0.1:8000/docs`.

### 2. Docker Setup
Run the entire production stack (API + MLflow Tracking Server) with Docker Compose:
```bash
docker compose up --build
```
- **API Endpoint & Swagger UI**: `http://127.0.0.1:8000/docs`
- **MLflow Tracking UI**: `http://127.0.0.1:5000`

---

## 💡 Synthetic Data & Production Considerations

> [!NOTE]
> **Data Note**: Quick-commerce order and pricing logs are proprietary IP. This project utilizes a synthetically generated dataset modeled after real-world Pune quick-commerce logistics heuristics (incorporating distance decay, peak-hour traffic multipliers, and price elasticity curves).

### Production Realities & Next Steps:
1. **Unobserved Confounders**: Synthetic data assumes known elasticity. In production, unobserved variables (e.g. sudden weather changes, hyper-local competitor discounts) affect conversion.
2. **Logging Policy Bias & Off-Policy Evaluation (OPE)**: Production deployment requires correcting for historical logging policy bias using techniques like **Inverse Propensity Scoring (IPS)** or **Doubly Robust Estimation** before rolling out new pricing policies.


