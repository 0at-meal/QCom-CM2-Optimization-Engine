import pandas as pd
import numpy as np
import os

# Load data
file_path = r"d:\QCom Margin Optimization Engine\data\qcom_pune_dataset.csv"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Missing values
print("\nMissing Values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# Basic Stats
print("\nBasic Stats (Numerical):\n", df.describe())

# Target Analysis
print("\nConversion Rate (Overall):", df['order_placed'].mean())

# Fee Analysis
print("\nFee Distribution:\n", df['delivery_fee_charged'].describe())

# Correlation with Order Placed
correlations = df.select_dtypes(include=[np.number]).corrwith(df['order_placed']).sort_values(ascending=False)
print("\nTop Correlations with order_placed:\n", correlations.head(10))
print("\nBottom Correlations with order_placed:\n", correlations.tail(10))

# CM2 Analysis
print("\nCM2 Distribution:\n", df['cm2'].describe())

# Check categorical columns
print("\nCategorical Columns:", df.select_dtypes(include=['object']).columns.tolist())

# Check price sensitivity vs conversion
if 'price_sensitivity_score' in df.columns:
    print("\nConversion by Price Sensitivity:\n", df.groupby('price_sensitivity_score')['order_placed'].mean())
