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

### Production-Ready API
- FastAPI-based REST endpoint for real-time optimization
- Batch processing support
- Constraint enforcement (no sub-cost pricing, fee bounds)

---

