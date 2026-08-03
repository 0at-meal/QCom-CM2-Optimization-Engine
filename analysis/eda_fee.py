import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.settings import settings

file_path = settings.get_data_path
df = pd.read_csv(file_path)

print("Delivery Fee Stats:\n", df['delivery_fee_charged'].describe())
print("\nFee > 0 count:", (df['delivery_fee_charged'] > 0).sum())

print("\nIs Cheaper than Competitors:\n", df['is_cheaper_than_competitors'].value_counts())
print("\nPrice Difference vs Competition:\n", df['price_difference_vs_competition'].describe())

print("\nFee Correlation with Order Placed:", df['delivery_fee_charged'].corr(df['order_placed']))
