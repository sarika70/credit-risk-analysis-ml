# -*- coding: utf-8 -*-

import os
import mlflow
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from data.load_data import load_credit_data
from features.feature_pipeline import create_features
from models.evaluate import get_credit_metrics

def train_model():
    df = load_credit_data()
    X, y = create_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    # Industry Best Practice: Stacking complex models into a simple Baseline (Meta-Learner)
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ]
    
    # Meta-Learner: Logistic Regression for probability calibration
    stack_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(),
        cv=5
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", stack_model)
    ])

    mlflow.set_experiment("Credit_Risk_PD_Engine")
    with mlflow.start_run():
        pipeline.fit(X_train, y_train) # trains the model
        probs = pipeline.predict_proba(X_test)[:, 1]
        metrics = get_credit_metrics(y_test, probs)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "pd_model")

        
        # =====================
        # Save Artifacts Safely
        # =====================
        # Use paths relative to project root (no '..')
        os.makedirs("artifacts", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        # Save the model
        joblib.dump(pipeline, "artifacts/credit_risk_pipeline.pkl")
        # Save training reference (Crucial for PSI drift calculation later)
        X_train.to_csv("data/processed/training_reference.csv", index=False)
        
        print(f"✅ Gini: {metrics['Gini']:.3f} | KS: {metrics['KS_Statistic']:.3f}")