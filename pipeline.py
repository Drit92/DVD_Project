import streamlit as st
import pandas as pd
import numpy as np
import gdown
import zipfile
import os

@st.cache_data(show_spinner="🔄 Downloading Home Credit Dataset...")
def load_raw_data():
    """EXACTLY Colab Cell 1-2: Download + Extract"""
    file_id = "1FSSkKQOJtvOpP1I1qyr4x6SYQg-kBnVw"
    output_filename = "dataset.zip"
    
    if not os.path.exists(output_filename):
        gdown.download(id=file_id, output=output_filename, quiet=False)
    
    extract_dir = "/tmp/file_zip"
    os.makedirs(extract_dir, exist_ok=True)
    if not os.path.exists(f"{extract_dir}/file_zip"):
        with zipfile.ZipFile(output_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    base_path = f"{extract_dir}/file_zip"
    app_data = pd.read_csv(f"{base_path}/application_data.csv")
    prev_data = pd.read_csv(f"{base_path}/previous_application.csv")
    
    return app_data, prev_data

@st.cache_data
def engineer_features(app_data, prev_data):
    """FIXED: All feature engineering with proper dtype handling"""
    
    # === PART 1: Basic Features ===
    app_data = app_data.copy()
    app_data['AGE_YEARS'] = (-app_data['DAYS_BIRTH'] / 365).round(1)
    app_data['EMP_YEARS'] = (-app_data['DAYS_EMPLOYED'] / 365).round(1)
    app_data.loc[app_data['EMP_YEARS'] > 60, 'EMP_YEARS'] = np.nan
    
    # Financial ratios
    app_data['CREDIT_INCOME_RATIO'] = app_data['AMT_CREDIT'] / app_data['AMT_INCOME_TOTAL']
    app_data['ANNUITY_INCOME_RATIO'] = app_data['AMT_ANNUITY'] / app_data['AMT_INCOME_TOTAL']
    app_data['INCOME_PER_PERSON'] = app_data['AMT_INCOME_TOTAL'] / app_data['CNT_FAM_MEMBERS']
    
    # === PART 1: Risk Flags ===
    app_data['FLAG_HIGH_CREDIT_STRESS'] = (app_data['CREDIT_INCOME_RATIO'] > 3).astype(int)
    app_data['FLAG_HIGH_EMI_STRESS'] = (app_data['ANNUITY_INCOME_RATIO'] > 0.20).astype(int)
    app_data['FLAG_LOW_INCOME_PERSON'] = (app_data['INCOME_PER_PERSON'] < 50000).astype(int)
    app_data['FLAG_SHORT_EMPLOYMENT'] = (app_data['EMP_YEARS'] < 2).astype(int)
    app_data['FLAG_YOUNG_BORROWER'] = (app_data['AGE_YEARS'] < 30).astype(int)
    
    # === PART 1: Behavioral Features ===
    prev_counts = prev_data.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique().rename('PREV_APP_COUNT')
    app_data = app_data.merge(prev_counts, on='SK_ID_CURR', how='left').fillna({'PREV_APP_COUNT': 0})
    
    refused_ids = prev_data[prev_data['NAME_CONTRACT_STATUS'] == 'Refused']['SK_ID_CURR'].unique()
    app_data['FLAG_EVER_REFUSED'] = app_data['SK_ID_CURR'].isin(refused_ids).astype(int)
    
    # Bins - CONVERT TO STRING immediately to avoid categorical issues
    app_data['PREV_APPS_BIN'] = pd.cut(app_data['PREV_APP_COUNT'], 
                                     bins=[-1, 0, 2, 4, 9, 1000], 
                                     labels=["0", "1-2", "3-4", "5-9", "10+"],
                                     ordered=False).astype(str)
    
    app_data['CREDIT_BIN'] = pd.cut(app_data['CREDIT_INCOME_RATIO'], 
                                  bins=[0, 1, 2, 3, 5, 10], 
                                  labels=['0-1x', '1-2x', '2-3x', '3-5x', '5x+'],
                                  ordered=False).astype(str)
    app_data['EMI_BIN'] = pd.cut(app_data['ANNUITY_INCOME_RATIO'], 
                               bins=[0, 0.10, 0.20, 0.30, 0.50, 1.0], 
                               labels=['<10%', '10-20%', '20-30%', '30-50%', '50%+'],
                               ordered=False).astype(str)
    app_data['INCOME_BIN'] = pd.cut(app_data['INCOME_PER_PERSON'], 
                                  bins=[0, 50000, 100000, 150000, 300000, 99999999], 
                                  labels=['<50k', '50-100k', '100-150k', '150-300k', '300k+'],
                                  ordered=False).astype(str)
    
    # === PART 2: External Score Bins ===
    app_data['EXT2_Q'] = pd.qcut(app_data['EXT_SOURCE_2'].fillna(0.5), 4, 
                               labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'],
                               ordered=False).astype(str)
    app_data['EXT2_BIN'] = pd.cut(app_data['EXT_SOURCE_2'], 
                                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01], 
                                labels=['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk'],
                                ordered=False).astype(str)
    
    # === PART 2: Risk Scores - FIXED NUMERIC CONVERSION ===
    credit_points = {'0-1x': 0, '1-2x': 1, '2-3x': 2, '3-5x': 3, '5x+': 4}
    app_data['FIN_SCORE'] = app_data['CREDIT_BIN'].map(credit_points).fillna(0).astype(float)
    
    app_data['BEHAV_SCORE'] = app_data['FLAG_EVER_REFUSED'] * 2
    
    ext_points = {'Q1 (low)': 3, 'Q2': 2, 'Q3': 1, 'Q4 (high)': 0}
    app_data['EXT_SCORE'] = app_data['EXT2_Q'].map(ext_points).fillna(0).astype(float)
    
    
    app_data['RISK_SCORE'] = (
        app_data['FIN_SCORE'].astype(float) + 
        app_data['BEHAV_SCORE'].astype(float) + 
        app_data['EXT_SCORE'].astype(float)
    ).astype(int)
    
    # === FINAL CLEAN DATASET (Cell 15) ===
    export_cols = [
        'SK_ID_CURR', 'TARGET', 'AGE_YEARS', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
        'INCOME_PER_PERSON', 'CREDIT_BIN', 'EMI_BIN', 'INCOME_BIN', 'FIN_SCORE',
        'EXT_SOURCE_2', 'EXT2_Q', 'EXT2_BIN', 'FLAG_EVER_REFUSED', 'PREV_APP_COUNT', 
        'PREV_APPS_BIN', 'BEHAV_SCORE', 'EXT_SCORE', 'RISK_SCORE'
    ]
    
    # Only keep columns that exist
    export_cols = [col for col in export_cols if col in app_data.columns]
    clean_df = app_data[export_cols].copy()
    
    return clean_df.sample(n=min(50000, len(clean_df)), random_state=42)
