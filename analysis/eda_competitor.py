import pandas as pd

# Load data
file_path = r"d:\QCom Margin Optimization Engine\data\qcom_pune_dataset.csv"
df = pd.read_csv(file_path)

print("Delivery Fee Correlation with Order Placed:", df['delivery_fee_charged'].corr(df['order_placed']))

competitor_cols = ['competitor_blinkit_fee', 'competitor_zepto_fee', 'competitor_instamart_fee', 'competitor_avg_fee']
for col in competitor_cols:
    print(f"\n{col} Analysis:")
    print("Null Count:", df[col].isnull().sum())
    print("Unique Values:", df[col].unique()[:10])
    print("Description:", df[col].describe())
