# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 13:36:23 2025

@author: mjayant
"""


import pandas as pd
import os
from sqlalchemy import create_mock_engine, create_engine

def load_credit_data(source='csv'):
    """
    Loads credit data from MySQL or a local CSV file.
    
    Args:
        source (str): 'mysql' or 'csv'
    """
    if source == 'mysql':
        try:
            # Database credentials (Update these for your Windows MySQL setup)
            USER = "root"
            PASSWORD = ""
            HOST = "localhost"
            PORT = "3306"
            DB_NAME = "credit_risk_db"
            
            connection_str = f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}"
            engine = create_engine(connection_str)
            
            print(f"🗄️ Querying data from MySQL: {DB_NAME}...")
            query = "SELECT * FROM hmeq_data"
            df = pd.read_sql(query, engine)
            
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
                
            print(f" Successfully loaded {len(df)} records from Database.")
            return df
            
        except Exception as e:
            print(f"  MySQL Error: {e}. Falling back to CSV...")
            source = 'csv'

    if source == 'csv':
        csv_path = "data/raw/hmeq.csv"
        if os.path.exists(csv_path):
            print(f"📄 Loading data from local file: {csv_path}")
            df = pd.read_csv(csv_path)
            # Ensure target naming consistency
            if 'BAD' in df.columns:
                df = df.rename(columns={'BAD': 'target'})
            return df
        else:
            raise FileNotFoundError(f"No CSV found at {csv_path}. Please place the hmeq.csv there.")

# Example usage within the script
if __name__ == "__main__":
    # Test the loader
    data = load_credit_data(source='csv')
    print(data.head())    