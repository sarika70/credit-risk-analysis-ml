# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 13:37:26 2025

@author: mjayant
"""

import mlflow
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from data.load_data import load_credit_data
from features.feature_pipeline import create_features
from models.evaluate import get_credit_metrics

def train_model():
    df = load_credit_data()
    X, y = create_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    ensemble = VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier())
    ], voting='soft')

    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", ensemble)
    ])

    # =====================
    # MLflow tracking
    # =====================
    mlflow.set_experiment("Credit_Risk_Production")
    with mlflow.start_run(run_name="Credit_Risk_Ensemble"):
        model_pipeline.fit(X_train, y_train)

        probs = model_pipeline.predict_proba(X_test)[:, 1]
        metrics = get_credit_metrics(y_test, probs)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model_pipeline, "credit_risk_model")

        # =====================
        # Save Artifacts Safely
        # =====================
        # Use paths relative to project root (no '..')
        os.makedirs("artifacts", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        
        # Save the model
        joblib.dump(model_pipeline, "artifacts/credit_risk_pipeline.pkl")
        
        # Save training reference (Crucial for PSI drift calculation later)
        reference_path = "data/processed/training_reference.csv"
        X_train.to_csv(reference_path, index=False)
    
        print(f" Training complete | Gini: {metrics['Gini']:.3f} | KS: {metrics['KS_Statistic']:.3f}")
        print(f" Model saved to: artifacts/credit_risk_pipeline.pkl")
        print(f" Reference data saved to: {reference_path}")