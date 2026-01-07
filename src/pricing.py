import pandas as pd
import numpy as np
import joblib
import os

class PricingEngine:
    def __init__(self, model_path='models/conversion_model.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = joblib.load(model_path)
        self.feature_cols = [
            'basket_value', 'basket_margin', 'basket_weight_kg',
            'num_items', 'distance_km', 'estimated_delivery_time_min',
            'hour_of_day', 'day_of_week', 'is_weekend', 'traffic_numeric',
            'price_sensitivity_score', 'margin_per_km', 'margin_x_distance',
            'sensitivity_x_value', 'delivery_fee_charged', 'base_conversion_prob'
        ]

    def _prepare_features(self, request_data, fee_candidates):
        """
        Prepare DataFrame for batch prediction across fee candidates.
        """

        base_df = pd.DataFrame([request_data])
        
        df = base_df.loc[base_df.index.repeat(len(fee_candidates))].reset_index(drop=True)
        df['delivery_fee_charged'] = fee_candidates
        
        df['margin_per_km'] = df['basket_margin'] / (df['distance_km'] + 0.1)
        df['margin_x_distance'] = df['basket_margin'] * df['distance_km']
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['sensitivity_x_value'] = df['price_sensitivity_score'] * df['basket_value']
        
        traffic_map = {'low': 0, 'medium': 1, 'high': 2}
        df['traffic_numeric'] = df['traffic_level'].map(traffic_map).fillna(1)
        
        if 'base_conversion_prob' not in df.columns:

             df['base_conversion_prob'] = request_data.get('conversion_prob_stage1', 0.8) # Dataset mean approx

        return df[self.feature_cols]

    def optimize_fee(self, request_data, min_fee=0, max_fee=100, step=5):
        """
        Find optimal fee optimizing CM2 subject to conversion constraint.
        request_data: dict containing feature values.
        """
        fee_candidates = np.arange(min_fee, max_fee + step, step)
        
        features_df = self._prepare_features(request_data, fee_candidates)
        probs = self.model.predict_proba(features_df)[:, 1]
        
        margin = request_data['basket_margin']
        cost = request_data.get('delivery_cost_potential', 60)
                
        expected_cm2 = probs * (margin + fee_candidates - cost)

        results = pd.DataFrame({
            'fee': fee_candidates,
            'prob': probs,
            'expected_cm2': expected_cm2
        })

        baseline_prob = results.loc[results['fee'] == 0, 'prob'].values[0] if 0 in results['fee'].values else probs.max()

        min_prob = baseline_prob - 0.03
        valid_results = results[results['prob'] >= min_prob]
        
        if valid_results.empty:
            best_row = results.loc[results['prob'].idxmax()]
        else:
            best_row = valid_results.loc[valid_results['expected_cm2'].idxmax()]
            
        return {
            'optimal_fee': float(best_row['fee']),
            'expected_conversion': float(best_row['prob']),
            'expected_cm2': float(best_row['expected_cm2']),
            'baseline_conversion': float(baseline_prob),
            'candidates': results.to_dict(orient='records')
        }

if __name__ == "__main__":

    engine = PricingEngine()
    test_req = {
        'basket_value': 500, 'basket_margin': 100, 'basket_weight_kg': 5,
        'num_items': 5, 'distance_km': 3, 'estimated_delivery_time_min': 20,
        'hour_of_day': 18, 'day_of_week': 0, 'traffic_level': 'medium',
        'price_sensitivity_score': 0.5, 'delivery_cost_potential': 40,
        'conversion_prob_stage1': 0.85
    }
    res = engine.optimize_fee(test_req)
    print("Optimization Result:", res['optimal_fee'], "CM2:", res['expected_cm2'])
