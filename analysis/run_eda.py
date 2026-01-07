import pandas as pd
import numpy as np
import os

file_path = r"d:\QCom Margin Optimization Engine\data\qcom_pune_dataset.csv"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nMissing Values:\n", df.isnull().sum()[df.isnull().sum() > 0])

print("\nBasic Stats (Numerical):\n", df.describe())

print("\nConversion Rate (Overall):", df['order_placed'].mean())

print("\nFee Distribution:\n", df['delivery_fee_charged'].describe())

correlations = df.select_dtypes(include=[np.number]).corrwith(df['order_placed']).sort_values(ascending=False)
print("\nTop Correlations with order_placed:\n", correlations.head(10))
print("\nBottom Correlations with order_placed:\n", correlations.tail(10))

print("\nCM2 Distribution:\n", df['cm2'].describe())

print("\nCategorical Columns:", df.select_dtypes(include=['object']).columns.tolist())

if 'price_sensitivity_score' in df.columns:
    print("\nConversion by Price Sensitivity:\n", df.groupby('price_sensitivity_score')['order_placed'].mean())
