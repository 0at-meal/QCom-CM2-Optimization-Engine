from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}

def test_optimize_endpoint_valid_request():
    payload = {
        "basket_value": 500,
        "basket_margin": 100,
        "basket_weight_kg": 2.5,
        "num_items": 5,
        "distance_km": 4.0,
        "estimated_delivery_time_min": 30,
        "hour_of_day": 18,
        "day_of_week": 0,
        "traffic_level": "medium",
        "price_sensitivity_score": 0.5,
        "delivery_cost_potential": 40,
        "conversion_prob_stage1": 0.8
    }
    response = client.post("/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "optimal_fee" in data
    assert "expected_conversion" in data
    assert "expected_cm2" in data

def test_optimize_endpoint_invalid_input():
    payload = {
        "basket_value": -500,  # Invalid: should be positive
        "basket_margin": 100,
        "basket_weight_kg": 2.5,
        "num_items": 5,
        "distance_km": 4.0,
        "estimated_delivery_time_min": 30,
        "hour_of_day": 18,
        "day_of_week": 0,
        "traffic_level": "medium",
        "price_sensitivity_score": 0.5,
        "delivery_cost_potential": 40,
        "conversion_prob_stage1": 0.8
    }
    response = client.post("/v1/optimize", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation error)

def test_batch_optimize_valid():
    payload = {
        "requests": [
            {
                "basket_value": 500,
                "basket_margin": 100,
                "basket_weight_kg": 2.5,
                "num_items": 5,
                "distance_km": 4.0,
                "estimated_delivery_time_min": 30,
                "hour_of_day": 18,
                "day_of_week": 0,
                "traffic_level": "medium",
                "price_sensitivity_score": 0.5,
                "delivery_cost_potential": 40,
                "conversion_prob_stage1": 0.8
            },
            {
                "basket_value": 800,
                "basket_margin": 150,
                "basket_weight_kg": 3.0,
                "num_items": 8,
                "distance_km": 2.0,
                "estimated_delivery_time_min": 20,
                "hour_of_day": 12,
                "day_of_week": 3,
                "traffic_level": "low",
                "price_sensitivity_score": 0.2,
                "delivery_cost_potential": 30,
                "conversion_prob_stage1": 0.9
            }
        ]
    }
    response = client.post("/v1/optimize/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert "optimal_fee" in data["results"][0]

def test_root_redirect():
    # TestClient doesn't automatically follow redirects by default unless follow_redirects=True
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
