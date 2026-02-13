# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 13:17:15 2025

@author: mjayant
"""

import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from data.load_data import load_credit_data
from features.feature_pipeline import create_features
from models.evaluate import get_credit_metrics

def train_logistic_baseline():
    """Requirement 3: Only Logistic Regression"""
    df = load_credit_data()
    X, y = create_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), X.select_dtypes(include=["number"]).columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes(include=["object", "category"]).columns)
    ])

    log_reg = LogisticRegression(max_iter=1000)

    pipeline = Pipeline([("preprocessing", preprocessor), ("model", log_reg)])

    mlflow.set_experiment("HMEQ_Logistic_Experiment")
    with mlflow.start_run(run_name="Logistic_Baseline"):
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]
        metrics = get_credit_metrics(y_test, probs)
        
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "logistic_model")
        joblib.dump(pipeline, "artifacts/credit_risk_pipeline.pkl")
        print(f" Logistic Model Trained. Gini: {metrics['Gini']:.3f}")

if __name__ == "__main__":
    train_logistic_baseline()
