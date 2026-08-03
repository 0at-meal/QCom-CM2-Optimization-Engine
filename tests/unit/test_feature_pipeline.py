import pytest
import pandas as pd
import numpy as np

from src.domain.feature_pipeline import build_features, FEATURE_COLS

def get_base_df():
    return pd.DataFrame({
        'basket_value': [500],
        'basket_margin': [100],
        'basket_weight_kg': [2.5],
        'num_items': [5],
        'distance_km': [4.0],
        'estimated_delivery_time_min': [30],
        'hour_of_day': [18],
        'day_of_week': [0],
        'traffic_level': ['medium'],
        'price_sensitivity_score': [0.5],
        'delivery_cost_potential': [40],
        'conversion_prob_stage1': [0.8],
        'delivery_fee_charged': [25]
    })

def test_build_features_creates_correct_columns():
    df = get_base_df()
    features = build_features(df)
    assert list(features.columns) == FEATURE_COLS

def test_margin_per_km_calculation():
    df = get_base_df()
    df = pd.concat([df, df], ignore_index=True)
    df.loc[0, 'basket_margin'] = 100
    df.loc[0, 'distance_km'] = 4.9 # (4.9 + 0.1) = 5
    df.loc[1, 'basket_margin'] = 0
    df.loc[1, 'distance_km'] = 1.9 # (1.9 + 0.1) = 2
    features = build_features(df)
    assert features['margin_per_km'].iloc[0] == 20.0
    assert features['margin_per_km'].iloc[1] == 0.0

def test_traffic_numeric_mapping():
    df = get_base_df()
    df = pd.concat([df, df, df, df], ignore_index=True)
    df['traffic_level'] = ['low', 'medium', 'high', np.nan]
    features = build_features(df)
    assert list(features['traffic_numeric']) == [0, 1, 2, 1]  # nan should fill with 1 (medium)

def test_missing_column_imputation():
    df = get_base_df()
    df = df.drop(columns=['distance_km'])
    features = build_features(df)
    assert 'distance_km' in features.columns
    assert features['distance_km'].iloc[0] == 0.0
