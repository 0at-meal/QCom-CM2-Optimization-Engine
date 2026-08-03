import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import joblib

from config.settings import settings
from src.domain.feature_pipeline import build_features

import mlflow
import mlflow.xgboost

def train_model():
    print("Loading Data...")
    df = pd.read_csv(settings.get_data_path)
    
    # Process features
    data = df.copy()
    
    # We must ensure order_placed target exists
    target_col = 'order_placed'
    data = data.dropna(subset=[target_col])
    
    X = build_features(data)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="conversion_model_v1"):
        mlflow.log_params({
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "fee_min": settings.fee_min,
            "fee_max": settings.fee_max,
            "conversion_drop_budget": settings.conversion_drop_budget,
        })

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

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        ll = log_loss(y_test, y_prob)
        
        print(f"Model Results:\nAccuracy: {acc:.4f}\nAUC: {auc:.4f}\nLogLoss: {ll:.4f}")

        mlflow.log_metrics({
            "auc": auc,
            "log_loss": ll,
            "accuracy": acc,
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

        mlflow.xgboost.log_model(
            model,
            artifact_path="conversion_model",
            registered_model_name="conversion"
        )
        print("Model saved to MLflow registry.")

        settings.get_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, settings.get_model_path)
        print(f"Model saved locally to {settings.get_model_path}")

        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        print("\nFeature Importance:\n", importance.head(10))

if __name__ == "__main__":
    train_model()
