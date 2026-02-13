# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 13:37:49 2025

@author: mjayant
"""



import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "HMEQ_Risk_Estimation_Engine"

def register_and_promote(run_id, current_gini, stage="Production"):
    """
    Registers the model and promotes it to Production only if it 
    outperforms the current champion.
    """
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/pd_model"
    
    # 1. Register the model version
    print(f" Registering model version from run: {run_id}")
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    
    # 2. Check for an existing Production model (The "Champion")
    try:
        production_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not production_versions:
            # No production model yet, promote immediately
            print(" No existing Production model. Promoting first version.")
            client.transition_model_version_stage(
                name=MODEL_NAME, version=mv.version, stage="Production"
            )
            return
        
        # 3. Champion vs Challenger logic
        champion_version = production_versions[0]
        # Fetch the Gini score of the current champion from its run
        champion_run = client.get_run(champion_version.run_id)
        champion_gini = float(champion_run.data.metrics.get("Gini", 0))
        
        print(f" Champion Gini: {champion_gini:.4f} | Challenger Gini: {current_gini:.4f}")
        
        if current_gini > champion_gini:
            print(f" Challenger wins! Promoting version {mv.version} to Production.")
            # Move the old champion to 'Archived' and new to 'Production'
            client.transition_model_version_stage(
                name=MODEL_NAME, 
                version=mv.version, 
                stage="Production", 
                archive_existing_versions=True
            )
        else:
            print(f"  Challenger did not outperform Champion. Staying in Staging.")
            client.transition_model_version_stage(
                name=MODEL_NAME, version=mv.version, stage="Staging"
            )
            
    except Exception as e:
        print(f" Registry Error: {e}. Defaulting to Staging.")
        client.transition_model_version_stage(
            name=MODEL_NAME, version=mv.version, stage="Staging"
        )

def get_production_model():
    """Fetches the latest Production model for deployment."""
    return mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/Production")