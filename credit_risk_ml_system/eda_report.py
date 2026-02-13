import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

def generate_eda_report(source='csv'):
    """
    EDA Generator for HMEQ Credit Risk.
    Analyzes labels, core predictors, and engineered financial ratios.
    """
    # 1. Load Data
    if source == 'mysql':
        from sqlalchemy import create_engine
        try:
            engine = create_engine("mysql+pymysql://root:your_password@localhost/credit_risk_db")
            df = pd.read_sql("SELECT * FROM hmeq_data", engine)
        except Exception as e:
            print(f" Database connection failed: {e}. Falling back to CSV.")
            df = pd.read_csv("data/raw/hmeq.csv")
    else:
        df = pd.read_csv("data/raw/hmeq.csv")
    
    # Standardize Column Names
    df = df.rename(columns={'BAD': 'target'})
    
    print(" EDA for HMEQ Dataset...")

    # 2.  Feature Engineering (Ratios & Collateral)
    # We use a copy for math to avoid modifying the original display strings
    df_calc = df.copy()
    
    # Collateral = Property Value - Mortgage Balance
    df_calc['COLLATERAL'] = df_calc['VALUE'] - df_calc['MORTDUE']
    
    # Loan-to-Collateral Ratio
    # We use a small epsilon to avoid division by zero
    df_calc['L_C_RATIO'] = df_calc['LOAN'] / (df_calc['COLLATERAL'].replace(0, np.nan))
    
    # Loan-to-Property Ratio (LTV)
    df_calc['L_P_RATIO'] = df_calc['LOAN'] / (df_calc['VALUE'].replace(0, np.nan))
    
    # Collateral-to-Property Ratio
    df_calc['C_P_RATIO'] = df_calc['COLLATERAL'] / (df_calc['VALUE'].replace(0, np.nan))

    # 3. Categorical Encoding for Correlation Math
    df_numeric = df_calc.copy()
    for col in df_numeric.select_dtypes(include=['object', 'category']).columns:
        df_numeric[col] = df_numeric[col].astype('category').cat.codes

    # 4. Exhaustive Visualizations
    os.makedirs('reports', exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    plt.subplots_adjust(hspace=0.4)

    # A. Label Distribution
    sns.countplot(ax=axes[0, 0], x='target', data=df, palette='viridis')
    axes[0, 0].set_title('Label Analysis: Default (1) vs Paid (0)')

    # B. Loan Amount Distribution (Predictor EDA)
    sns.histplot(ax=axes[0, 1], x='LOAN', hue='target', data=df, kde=True, element="step")
    axes[0, 1].set_title('Loan Amount Distribution')

    # C. Mortgage Balance vs Property Value
    sns.scatterplot(ax=axes[0, 2], x='MORTDUE', y='VALUE', hue='target', data=df, alpha=0.4)
    axes[0, 2].set_title('Mortgage Due vs Market Value')

    # D. Loan-to-Property Ratio (LTV) - Critical Predictor
    sns.boxplot(ax=axes[1, 0], x='target', y='L_P_RATIO', data=df_calc)
    axes[1, 0].set_title('Loan-to-Property Ratio (LTV)')
    axes[1, 0].set_ylim(0, 1.2)

    # E. Debt-to-Income Ratio - Critical Predictor
    sns.kdeplot(ax=axes[1, 1], x='DEBTINC', hue='target', data=df, fill=True)
    axes[1, 1].set_title('Debt-to-Income Ratio Impact')

    # F. Job vs Default Rate (Categorical EDA)
    job_risk = df.groupby('JOB')['target'].mean().sort_values()
    job_risk.plot(kind='barh', ax=axes[1, 2], color='skyblue')
    axes[1, 2].set_title('Default Rate by Professional Occupation')

    # G. Credit History: Oldest Trade Line (CLAGE)
    sns.boxenplot(ax=axes[2, 0], x='target', y='CLAGE', data=df)
    axes[2, 0].set_title('Age of Oldest Trade Line')

    # H. Behavioral Risk: Delinquent Lines (DELINQ)
    sns.pointplot(ax=axes[2, 1], x='DELINQ', y='target', data=df)
    axes[2, 1].set_title('Default Risk by Delinquent Lines')

    # I. Correlation Heatmap (Focus on Target)
    corr = df_numeric.corr()
    sns.heatmap(ax=axes[2, 2], data=corr[['target']].sort_values(by='target', ascending=False), 
                annot=True, cmap='RdYlGn', center=0)
    axes[2, 2].set_title('Predictor Correlation with Default')

    plt.savefig('reports/exhaustive_eda_visuals.png')
    print(" Exhaustive Visualizations saved to reports/exhaustive_eda_visuals.png")

    # 5. Statistical & Missing Value Report
    with open('reports/exhaustive_eda_summary.txt', 'w') as f:
        f.write("EXHAUSTIVE HMEQ EDA REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Missing Value Analysis
        f.write("1. MISSING VALUE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        f.write(pd.DataFrame({'Missing': missing, 'Percentage': missing_pct}).to_string())
        f.write("\n\n")

        # Descriptive Statistics
        f.write("2. DESCRIPTIVE STATISTICS (BY TARGET)\n")
        f.write("-" * 30 + "\n")
        stats_df = df_calc.groupby('target').mean(numeric_only=True).T
        f.write(stats_df.to_string())
        f.write("\n\n")

        # Skewness Analysis
        f.write("3. SKEWNESS & KURTOSIS (UNIVARIATE)\n")
        f.write("-" * 30 + "\n")
        for col in ['LOAN', 'VALUE', 'DEBTINC', 'CLAGE']:
            col_data = df[col].dropna()
            f.write(f"{col} - Skew: {col_data.skew():.2f}, Kurtosis: {col_data.kurt():.2f}\n")
        
        f.write("\n4. INSIGHTS ON RATIOS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Avg LTV (Paid): {df_calc[df_calc['target']==0]['L_P_RATIO'].mean():.2%}\n")
        f.write(f"Avg LTV (Default): {df_calc[df_calc['target']==1]['L_P_RATIO'].mean():.2%}\n")
        f.write(f"Avg Collateral-to-Property: {df_calc['C_P_RATIO'].mean():.2%}\n")

    print("  Exhaustive statistical report saved to reports/exhaustive_eda_summary.txt")

if __name__ == "__main__":
    generate_eda_report(source='csv')