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

# Download only if not cached
if not os.path.exists(output_filename):
gdown.download(id=file_id, output=output_filename, quiet=False)

# Extract to temp directory
extract_dir = "/tmp/file_zip"
os.makedirs(extract_dir, exist_ok=True)

if not os.path.exists(f"{extract_dir}/file_zip"):
with zipfile.ZipFile(output_filename, 'r') as zip_ref:
zip_ref.extractall(extract_dir)

# Load EXACTLY as Colab paths
base_path = f"{extract_dir}/file_zip"
app_path = f"{base_path}/application_data.csv"
prev_path = f"{base_path}/previous_application.csv"

app_data = pd.read_csv(app_path)
prev_data = pd.read_csv(prev_path)

return app_data, prev_data

@st.cache_data
def engineer_features(app_data, prev_data):
"""COMPLETE FIXED: All Colab Part 1+2 feature engineering"""

# Copy to avoid modifying original
app_data = app_data.copy()

# === PART 1: Basic Features (Colab Cells 3-4) ===
app_data['AGE_YEARS'] = (-app_data['DAYS_BIRTH'] / 365).round(1)
app_data['EMP_YEARS'] = (-app_data['DAYS_EMPLOYED'] / 365).round(1)
app_data.loc[app_data['EMP_YEARS'] > 60, 'EMP_YEARS'] = np.nan

# Financial ratios with zero division protection
app_data['CREDIT_INCOME_RATIO'] = app_data['AMT_CREDIT'] / app_data['AMT_INCOME_TOTAL'].replace(0, np.nan)
app_data['ANNUITY_INCOME_RATIO'] = app_data['AMT_ANNUITY'] / app_data['AMT_INCOME_TOTAL'].replace(0, np.nan)
app_data['INCOME_PER_PERSON'] = app_data['AMT_INCOME_TOTAL'] / app_data['CNT_FAM_MEMBERS'].replace(0, 1)

# === PART 1: Risk Flags (Colab Cell 5) ===
app_data['FLAG_HIGH_CREDIT_STRESS'] = (app_data['CREDIT_INCOME_RATIO'] > 3).fillna(False).astype(int)
app_data['FLAG_HIGH_EMI_STRESS'] = (app_data['ANNUITY_INCOME_RATIO'] > 0.20).fillna(False).astype(int)
app_data['FLAG_LOW_INCOME_PERSON'] = (app_data['INCOME_PER_PERSON'] < 50000).fillna(False).astype(int)
app_data['FLAG_SHORT_EMPLOYMENT'] = (app_data['EMP_YEARS'] < 2).fillna(False).astype(int)
app_data['FLAG_YOUNG_BORROWER'] = (app_data['AGE_YEARS'] < 30).fillna(False).astype(int)

# === PART 1: Behavioral Features (Colab Cells 9-11) ===
# Previous application counts
prev_counts = prev_data.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique().rename('PREV_APP_COUNT')
app_data = app_data.merge(prev_counts, on='SK_ID_CURR', how='left').fillna({'PREV_APP_COUNT': 0})

# Previous refusal flag
refused_ids = prev_data[prev_data['NAME_CONTRACT_STATUS'] == 'Refused']['SK_ID_CURR'].unique()
app_data['FLAG_EVER_REFUSED'] = app_data['SK_ID_CURR'].isin(refused_ids).astype(int)

# Previous apps bins
app_data['PREV_APPS_BIN'] = pd.cut(
app_data['PREV_APP_COUNT'].fillna(0),
bins=[-1, 0, 2, 4, 9, 1000],
labels=["0", "1-2", "3-4", "5-9", "10+"],
ordered=False
).astype(str).replace('nan', '0')

# === PART 1+2: Financial Stress Bins ===
app_data['CREDIT_BIN'] = pd.cut(
app_data['CREDIT_INCOME_RATIO'].fillna(0),
bins=[0, 1, 2, 3, 5, 10],
labels=['0-1x', '1-2x', '2-3x', '3-5x', '5x+'],
ordered=False
).astype(str).replace('nan', '0-1x')

app_data['EMI_BIN'] = pd.cut(
app_data['ANNUITY_INCOME_RATIO'].fillna(0),
bins=[0, 0.10, 0.20, 0.30, 0.50, 1.0],
labels=['<10%', '10-20%', '20-30%', '30-50%', '50%+'],
ordered=False
).astype(str).replace('nan', '<10%')

app_data['INCOME_BIN'] = pd.cut(
app_data['INCOME_PER_PERSON'].fillna(0),
bins=[0, 50000, 100000, 150000, 300000, 99999999],
labels=['<50k', '50-100k', '100-150k', '150-300k', '300k+'],
ordered=False
).astype(str).replace('nan', '<50k')

# === PART 2: External Score Bins (Colab Cells 2-4, 10) ===
# Fill missing EXT_SOURCE_2 with median
ext_median = app_data['EXT_SOURCE_2'].median()
app_data['EXT_SOURCE_2'] = app_data['EXT_SOURCE_2'].fillna(ext_median)

# EXT2 quartiles using pd.cut (fixed qcut issue)
app_data['EXT2_Q'] = pd.cut(
app_data['EXT_SOURCE_2'],
bins=4,
labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
).astype(str).replace('nan', 'Q1 (low)')

app_data['EXT2_BIN'] = pd.cut(
app_data['EXT_SOURCE_2'],
bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
labels=['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk'],
ordered=False
).astype(str).replace('nan', 'Very High Risk')

# === PART 2: Composite Risk Score (Colab Cell 14) ===
# Financial score
credit_points = {'0-1x': 0, '1-2x': 1, '2-3x': 2, '3-5x': 3, '5x+': 4}
app_data['FIN_SCORE'] = app_data['CREDIT_BIN'].map(credit_points).fillna(0).astype(float)

# Behavioral score
app_data['BEHAV_SCORE'] = app_data['FLAG_EVER_REFUSED'] * 2.0

# External score
ext_points = {'Q1 (low)': 3, 'Q2': 2, 'Q3': 1, 'Q4 (high)': 0}
app_data['EXT_SCORE'] = app_data['EXT2_Q'].map(ext_points).fillna(3).astype(float)

# Total risk score (ALL FLOAT before addition)
app_data['RISK_SCORE'] = (
app_data['FIN_SCORE'].astype(float) +
app_data['BEHAV_SCORE'].astype(float) +
app_data['EXT_SCORE'].astype(float)
).round().astype(int)

# === FINAL CLEAN DATASET (Colab Cell 15) ===
export_cols = [
'SK_ID_CURR', 'TARGET',
'AGE_YEARS', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE',
'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'INCOME_PER_PERSON',
'CREDIT_BIN', 'EMI_BIN', 'INCOME_BIN',
'EXT_SOURCE_2', 'EXT2_Q', 'EXT2_BIN',
'FLAG_EVER_REFUSED', 'PREV_APP_COUNT', 'PREV_APPS_BIN',
'FIN_SCORE', 'BEHAV_SCORE', 'EXT_SCORE', 'RISK_SCORE'
]

# Only keep existing columns
export_cols = [col for col in export_cols if col in app_data.columns]
clean_df = app_data[export_cols].copy()

# Sample for Streamlit performance
sample_size = min(50000, len(clean_df))
clean_df = clean_df.sample(n=sample_size, random_state=42)

return clean_df
