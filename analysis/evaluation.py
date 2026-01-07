import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pricing import PricingEngine
from features import load_data, preprocess_features 

def evaluate_performance():
    print("Loading Data...")
    df = load_data(r"d:\QCom Margin Optimization Engine\data\qcom_pune_dataset.csv")

    test_df = df.sample(1000, random_state=42)
    
    engine = PricingEngine(model_path='models/conversion_model.pkl')
    
    results = []
    
    print(f"Evaluating on {len(test_df)} orders...")
    
    for _, row in test_df.iterrows():

        req = row.to_dict()

        opt_res = engine.optimize_fee(req)

        actual_fee = row['delivery_fee_charged']
        
        hist_check = engine._prepare_features(req, [actual_fee])
        hist_prob = engine.model.predict_proba(hist_check)[:, 1][0]
        
        margin = row['basket_margin']
        cost = row['delivery_cost_potential'] if not pd.isna(row['delivery_cost_potential']) else 60
        
        hist_exp_cm2 = hist_prob * (margin + actual_fee - cost)
        
        results.append({
            'order_id': row['order_id'],
            'hist_fee': actual_fee,
            'hist_prob': hist_prob,
            'hist_exp_cm2': hist_exp_cm2,
            'opt_fee': opt_res['optimal_fee'],
            'opt_prob': opt_res['expected_conversion'],
            'opt_exp_cm2': opt_res['expected_cm2'],
            'baseline_prob': opt_res['baseline_conversion']
        })
        
    res_df = pd.DataFrame(results)
    
    total_hist_cm2 = res_df['hist_exp_cm2'].sum()
    total_opt_cm2 = res_df['opt_exp_cm2'].sum()
    
    avg_hist_conv = res_df['hist_prob'].mean()
    avg_opt_conv = res_df['opt_prob'].mean()
    
    uplift_cm2 = (total_opt_cm2 - total_hist_cm2) / total_hist_cm2 * 100
    conv_impact = (avg_opt_conv - avg_hist_conv) * 100
    
    print("\n=== Evaluation Results ===")
    print(f"Total Orders: {len(test_df)}")
    print(f"Historic Total Exp CM2: {total_hist_cm2:.2f}")
    print(f"Optimized Total Exp CM2: {total_opt_cm2:.2f}")
    print(f"CM2 Uplift: {uplift_cm2:.2f}%")
    print(f"Avg Historic Conversion: {avg_hist_conv*100:.2f}%")
    print(f"Avg Optimized Conversion: {avg_opt_conv*100:.2f}%")
    print(f"Conversion Impact: {conv_impact:.2f} percentage points")

    print(f"Constraint Check: Drop > 3%? {'YES' if conv_impact < -3 else 'NO'}")

if __name__ == "__main__":
    evaluate_performance()

