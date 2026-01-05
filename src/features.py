import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def preprocess_features(df):
    """
    Feature engineering and preprocessing.
    """
    data = df.copy()
    
    # 1. Interaction Features
    # Margin per km: High margin orders at long distance might justify lower fee? 
    # Or actually: Cost is high for long distance.
    data['margin_per_km'] = data['basket_margin'] / (data['distance_km'] + 0.1)
    
    # Margin x Distance (Weighted Load)
    data['margin_x_distance'] = data['basket_margin'] * data['distance_km']
    
    # 2. Time Features
    # already has hour_of_day, day_of_week
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # 3. Customer Features
    # price_sensitivity_score is key. 
    # Interaction: Sensitivity x Basket Value (High value orders might be less sensitive relative to total?)
    data['sensitivity_x_value'] = data['price_sensitivity_score'] * data['basket_value']
    
    # 4. Traffic Encoding (Simple Ordinal)
    traffic_map = {'low': 0, 'medium': 1, 'high': 2}
    data['traffic_numeric'] = data['traffic_level'].map(traffic_map).fillna(1)
    
    # 5. Base Probability (Available before fee?)
    # Assuming conversion_prob_stage1 is pre-fee interest
    if 'conversion_prob_stage1' in data.columns:
        data['base_conversion_prob'] = data['conversion_prob_stage1']
    else:
        data['base_conversion_prob'] = 0.5

    # Select Features for Model
    feature_cols = [
        'basket_value', 'basket_margin', 'basket_weight_kg',
        'num_items', 'distance_km', 'estimated_delivery_time_min',
        'hour_of_day', 'day_of_week', 'is_weekend', 'traffic_numeric',
        'price_sensitivity_score', 'margin_per_km', 'margin_x_distance',
        'sensitivity_x_value', 'delivery_fee_charged',
        'base_conversion_prob'
    ]
    
    target_col = 'order_placed'
    
    # Drop rows where target is missing (should be none based on EDA)
    data = data.dropna(subset=[target_col])
    
    return data[feature_cols], data[target_col]

if __name__ == "__main__":
    df = load_data(r"d:\QCom Margin Optimization Engine\data\qcom_pune_dataset.csv")
    X, y = preprocess_features(df)
    print("Features Shape:", X.shape)
    print("Target Mean:", y.mean())
    print("Features Head:\n", X.head())
