import pandas as pd
import numpy as np

file_path = r"d:\QCom Margin Optimization Engine\data\qcom_pune_dataset.csv"
df = pd.read_csv(file_path)

correlations = df.select_dtypes(include=[np.number]).corrwith(df['order_placed']).sort_values(ascending=False)
print("\nTop Correlations with order_placed:\n", correlations.head(15))
print("\nBottom Correlations with order_placed:\n", correlations.tail(15))
