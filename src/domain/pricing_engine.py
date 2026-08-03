import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any

from config.settings import settings
from src.domain.feature_pipeline import build_features

class PricingEngine:
    def __init__(self, model_path: str = None):
        """
        Pure domain pricing engine. No global state, dependencies passed in or loaded explicitly.
        """
        path = model_path if model_path else settings.get_model_path
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
        self.model = joblib.load(path)

    def optimize_fee(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find optimal fee optimizing expected CM2 subject to conversion constraint.
        """
        fee_candidates = np.arange(
            settings.fee_min, 
            settings.fee_max + settings.fee_step, 
            settings.fee_step
        )
        
        # Prepare batch dataframe across all candidates
        base_df = pd.DataFrame([request_data])
        df = base_df.loc[base_df.index.repeat(len(fee_candidates))].reset_index(drop=True)
        df['delivery_fee_charged'] = fee_candidates
        
        # Call single source of truth for features
        features_df = build_features(df)
        
        # Score candidates
        probs = self.model.predict_proba(features_df)[:, 1]
        
        margin = request_data['basket_margin']
        cost = request_data.get('delivery_cost_potential', 60)
        
        expected_cm2 = probs * (margin + fee_candidates - cost)

        results = pd.DataFrame({
            'fee': fee_candidates,
            'prob': probs,
            'expected_cm2': expected_cm2
        })

        # Calculate baseline (prob at fee=0 or max prob if fee=0 isn't available)
        baseline_prob = results.loc[results['fee'] == 0, 'prob'].values[0] if 0 in results['fee'].values else probs.max()

        # Apply constraint: conversion drop <= budget (e.g. 3%)
        min_prob = baseline_prob - settings.conversion_drop_budget
        valid_results = results[results['prob'] >= min_prob]
        
        if valid_results.empty:
            # If no fee satisfies constraint, fallback to max conversion fee
            best_row = results.loc[results['prob'].idxmax()]
        else:
            # Choose fee maximizing CM2
            best_row = valid_results.loc[valid_results['expected_cm2'].idxmax()]
            
        return {
            'optimal_fee': float(best_row['fee']),
            'expected_conversion': float(best_row['prob']),
            'expected_cm2': float(best_row['expected_cm2']),
            'baseline_conversion': float(baseline_prob),
            'cm2_uplift_pct': float((best_row['expected_cm2'] - (baseline_prob * (margin - cost))) / (baseline_prob * (margin - cost)) * 100) if (baseline_prob * (margin - cost)) != 0 else 0.0,
            'candidates': results.to_dict(orient='records')
        }
