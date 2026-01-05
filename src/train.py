import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import joblib
import os
from features import preprocess_features, load_data

def train_model():
    # Load and Preprocess
    print("Loading Data...")
    df = load_data(r"d:\QCom Margin Optimization Engine\data\qcom_pune_dataset.csv")
    X, y = preprocess_features(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective='binary:logistic',
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    
    print(f"Model Results:\nAccuracy: {acc:.4f}\nAUC: {auc:.4f}\nLogLoss: {ll:.4f}")
    
    # Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/conversion_model.pkl')
    print("Model saved to models/conversion_model.pkl")
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    print("\nFeature Importance:\n", importance.head(10))

if __name__ == "__main__":
    train_model()
