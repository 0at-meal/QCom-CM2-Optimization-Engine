from src.pricing import PricingEngine

class QComPricingAPI:
    def __init__(self, model_path='models/conversion_model.pkl'):
        """
        Initialize the pricing engine.
        """
        self.engine = PricingEngine(model_path=model_path)
    
    def get_optimal_fee(self, request_data):
        """
        Calculate the optimal delivery fee for a given order request.
        
        Args:
            request_data (dict): Dictionary containing multiple order features:
                - basket_value (float)
                - basket_margin (float)
                - distance_km (float)
                - traffic_level (str): 'low', 'medium', 'high'
                - hour_of_day (int)
                - day_of_week (int)
                - price_sensitivity_score (float)
                - conversion_prob_stage1 (float): Base conversion probability
                - delivery_cost_potential (float): Estimated cost
        
        Returns:
            dict: Optimal fee recommendation and expected metrics.
        """
        return self.engine.optimize_fee(request_data)

if __name__ == "__main__":
    api = QComPricingAPI()
    sample_request = {
        'basket_value': 600, 
        'basket_margin': 120, 
        'basket_weight_kg': 4,
        'num_items': 6, 
        'distance_km': 2.5, 
        'estimated_delivery_time_min': 25,
        'hour_of_day': 19, 
        'day_of_week': 5, 
        'traffic_level': 'high',
        'price_sensitivity_score': 0.4, 
        'delivery_cost_potential': 45,
        'conversion_prob_stage1': 0.8
    }
    print(api.get_optimal_fee(sample_request))
