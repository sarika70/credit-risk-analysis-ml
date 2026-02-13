# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 13:16:15 2025

@author: mjayant
"""

import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data.load_data import load_credit_data
from features.feature_pipeline import create_features
from models.evaluate import get_credit_metrics

def train_boosted_ensemble():
    """Requirement  Decision Trees, Random Forest, XGBoost"""
    df = load_credit_data()
    X, y = create_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), X.select_dtypes(include=["number"]).columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes(include=["object", "category"]).columns)
    ])

    # Using XGBoost as the primary champion
    #xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    
    model = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200)),
            ("xgb", XGBClassifier(
                eval_metric="logloss",
                max_depth=4,
                learning_rate=0.05,
                n_estimators=300
            ))
        ],
        voting="soft"
    )
    
    pipeline = Pipeline([("preprocessing", preprocessor), ("model", model)])

    mlflow.set_experiment("HMEQ_XGBoost_Voting_Experiment")
    with mlflow.start_run(run_name="XGBoost_Model"):
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]
        metrics = get_credit_metrics(y_test, probs)
        
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "xgb_model")
        joblib.dump(pipeline, "artifacts/credit_risk_pipeline.pkl")
        print(f" XGBoost Model Trained. Gini: {metrics['Gini']:.3f}")

if __name__ == "__main__":
    train_boosted_ensemble()

