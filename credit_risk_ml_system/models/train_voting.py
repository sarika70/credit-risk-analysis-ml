# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 13:15:23 2025

@author: mjayant
"""

import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

from data.load_data import load_credit_data
from features.feature_pipeline import create_features
from models.evaluate import get_credit_metrics

def train_voting_ensemble():
    """ Decision Trees, Random Forest and Voting Classifier"""
    df = load_credit_data()
    X, y = create_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), X.select_dtypes(include=["number"]).columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes(include=["object", "category"]).columns)
    ])

    clf1 = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)

    voter = VotingClassifier(
        estimators=[('dt', clf1), ('rf', clf2)],
        voting='soft'
    )

    pipeline = Pipeline([("preprocessing", preprocessor), ("model", voter)])

    mlflow.set_experiment("HMEQ_Voting_Experiment")
    with mlflow.start_run(run_name="Voting_Ensemble"):
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]
        metrics = get_credit_metrics(y_test, probs)
        
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "voting_model")
        joblib.dump(pipeline, "artifacts/credit_risk_pipeline.pkl")
        print(f" Voting Model Trained. Gini: {metrics['Gini']:.3f}")

if __name__ == "__main__":
    train_voting_ensemble()