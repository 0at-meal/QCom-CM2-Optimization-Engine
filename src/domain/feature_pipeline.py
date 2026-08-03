import pandas as pd

# The feature cols expected by the model
FEATURE_COLS = [
    'basket_value', 'basket_margin', 'basket_weight_kg',
    'num_items', 'distance_km', 'estimated_delivery_time_min',
    'hour_of_day', 'day_of_week', 'is_weekend', 'traffic_numeric',
    'price_sensitivity_score', 'margin_per_km', 'margin_x_distance',
    'sensitivity_x_value', 'delivery_fee_charged', 'base_conversion_prob'
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single source of truth for feature engineering.
    Used identically during offline training and online serving to prevent skew.
    """
    data = df.copy()
    
    # Ensure all required base features are present before computing derived features
    for col in FEATURE_COLS:
        if col not in data.columns:
            data[col] = 0.0
            
    # Cost-distance alignment
    data['margin_per_km'] = data['basket_margin'] / (data['distance_km'] + 0.1)
    data['margin_x_distance'] = data['basket_margin'] * data['distance_km']
    
    # Temporal
    if 'day_of_week' in data.columns:
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # Customer elasticity
    data['sensitivity_x_value'] = data['price_sensitivity_score'] * data['basket_value']

    # Traffic
    traffic_map = {'low': 0, 'medium': 1, 'high': 2}
    data['traffic_numeric'] = data['traffic_level'].map(traffic_map).fillna(1) if 'traffic_level' in data.columns else 1
    
    # Base conversion fallbacks
    if 'conversion_prob_stage1' in data.columns:
        data['base_conversion_prob'] = data['conversion_prob_stage1']
    else:
        data['base_conversion_prob'] = 0.5  # default baseline if missing

    return data[FEATURE_COLS]
