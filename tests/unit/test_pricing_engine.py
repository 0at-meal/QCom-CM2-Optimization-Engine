import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from src.domain.pricing_engine import PricingEngine
from config.settings import settings

# Mock model that returns decreasing conversion probability as fee increases
class MockModel:
    def predict_proba(self, X):
        # Base prob - fee_penalty
        fees = X['delivery_fee_charged'].values
        probs = 0.8 - (fees * 0.005) # Drop 0.5% per rupee
        probs = np.clip(probs, 0.0, 1.0)
        
        # Return [prob_class_0, prob_class_1]
        return np.column_stack((1 - probs, probs))

@pytest.fixture
def mock_engine():
    with patch('joblib.load', return_value=MockModel()):
        with patch('pathlib.Path.exists', return_value=True):
            engine = PricingEngine()
            return engine

@pytest.fixture
def sample_request():
    return {
        'basket_value': 500, 
        'basket_margin': 100, 
        'basket_weight_kg': 2.0,
        'num_items': 5, 
        'distance_km': 3.0, 
        'estimated_delivery_time_min': 25,
        'hour_of_day': 18, 
        'day_of_week': 0, 
        'traffic_level': 'medium',
        'price_sensitivity_score': 0.5, 
        'delivery_cost_potential': 40,
        'conversion_prob_stage1': 0.8
    }

def test_optimal_fee_is_non_negative(mock_engine, sample_request):
    res = mock_engine.optimize_fee(sample_request)
    assert res['optimal_fee'] >= 0

def test_optimal_fee_respects_max_bound(mock_engine, sample_request):
    res = mock_engine.optimize_fee(sample_request)
    assert res['optimal_fee'] <= settings.fee_max

def test_conversion_constraint_is_respected(mock_engine, sample_request):
    res = mock_engine.optimize_fee(sample_request)
    # constraint is baseline_conversion - drop_budget
    min_allowed = res['baseline_conversion'] - settings.conversion_drop_budget
    assert res['expected_conversion'] >= min_allowed

def test_zero_basket_margin(mock_engine, sample_request):
    req = sample_request.copy()
    req['basket_margin'] = 0
    res = mock_engine.optimize_fee(req)
    assert 'optimal_fee' in res

def test_batch_results_match_individual_results(mock_engine, sample_request):
    res1 = mock_engine.optimize_fee(sample_request)
    
    # We can check candidates
    assert len(res1['candidates']) > 0
    assert 'fee' in res1['candidates'][0]
    assert 'expected_cm2' in res1['candidates'][0]

def test_invalid_traffic_level_handled(mock_engine, sample_request):
    req = sample_request.copy()
    req['traffic_level'] = 'unknown' # pipeline should fillna with 1 (medium)
    res = mock_engine.optimize_fee(req)
    assert res['optimal_fee'] >= 0
