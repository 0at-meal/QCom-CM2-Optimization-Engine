import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.idempotency import idempotency_store

client = TestClient(app)

@pytest.fixture(autouse=True)
def clear_store():
    idempotency_store.clear()

def get_sample_payload():
    return {
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

def test_idempotency_first_request_is_miss():
    headers = {"X-Idempotency-Key": "key-test-1"}
    response = client.post("/v1/optimize", json=get_sample_payload(), headers=headers)
    assert response.status_code == 200
    assert response.headers.get("x-cache") == "MISS"

def test_idempotency_subsequent_request_is_hit():
    headers = {"X-Idempotency-Key": "key-test-2"}
    payload = get_sample_payload()
    
    # First call: MISS
    res1 = client.post("/v1/optimize", json=payload, headers=headers)
    assert res1.status_code == 200
    assert res1.headers.get("x-cache") == "MISS"
    
    # Second call with same key: HIT
    res2 = client.post("/v1/optimize", json=payload, headers=headers)
    assert res2.status_code == 200
    assert res2.headers.get("x-cache") == "HIT"
    assert res1.json() == res2.json()

def test_different_keys_do_not_collide():
    payload = get_sample_payload()
    
    res1 = client.post("/v1/optimize", json=payload, headers={"X-Idempotency-Key": "key-a"})
    res2 = client.post("/v1/optimize", json=payload, headers={"X-Idempotency-Key": "key-b"})
    
    assert res1.headers.get("x-cache") == "MISS"
    assert res2.headers.get("x-cache") == "MISS"

def test_request_without_key_works_normally():
    response = client.post("/v1/optimize", json=get_sample_payload())
    assert response.status_code == 200
    assert "x-cache" not in response.headers
