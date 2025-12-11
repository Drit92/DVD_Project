import os
import zipfile

import gdown
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
DATA_ID = "1FSSkKQOJtvOpP1I1qyr4x6SYQg-kBnVw"   # Google Drive file ID
ZIP_PATH = "dataset.zip"
EXTRACT_DIR = "file_zip"

st.set_page_config(page_title="Loan Dashboard", layout="wide")
st.title("📊 Loan Applicant Visual Insights Dashboard")


# ----------------------------------------------------
# Download & extract ZIP, return path to first CSV
# ----------------------------------------------------
def download_and_extract():
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Download only if ZIP is not present
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading dataset...")
        gdown.download(id=DATA_ID, output=ZIP_PATH, quiet=False)

    # Extract ZIP
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    except Exception as e:
        st.error(f"❌ ZIP extraction failed: {e}")
        return None

    # Find first CSV recursively
    csv_files = []
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        st.error("❌ No CSV file found inside ZIP.")
        return None

    return csv_files[0]


# ----------------------------------------------------
# MAIN LOGIC
# ----------------------------------------------------
csv_path = download_and_extract()

if not csv_path:
    st.error("Dataset could not be loaded.")
    st.stop()

st.success(f"Dataset loaded: `{csv_path}`")

# Read CSV
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    st.error(f"❌ Failed to read CSV: {e}")
    st.stop()

# ----------------------------------------------------
# FEATURE ENGINEERING (mirror of Colab)
# ----------------------------------------------------

# 1️ AGE_YEARS
if "DAYS_BIRTH" in df.columns:
    df["AGE_YEARS"] = (df["DAYS_BIRTH"] / -365).round(1)

# 2️ EMP_YEARS
if "DAYS_EMPLOYED" in df.columns:
    df["EMP_YEARS"] = (df["DAYS_EMPLOYED"] / -365).round(1)
    # Replace nonsense values in EMP_YEARS (people with 1000 years employment)
    df.loc[df["EMP_YEARS"] > 60, "EMP_YEARS"] = pd.NA

# 3️ Financial ratios
if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

if "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

# 4️ Basic NA handling (safe for visuals)
df = df.fillna(0)

# ----------------------------------------------------
# COLUMN CHECKS (adjust as needed)
# ----------------------------------------------------
required_cols = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AGE_YEARS",     # engineered
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns for visuals: {missing}")
    st.stop()

# ----------------------------------------------------
# PREVIEW
# ----------------------------------------------------
st.subheader("🔎 Data Preview")
st.dataframe(df.head())

# ----------------------------------------------------
# VISUALIZATIONS
# ----------------------------------------------------

# 1. Income vs Credit
st.subheader("💰 Income vs Credit")
fig1 = px.scatter(
    df,
    x="AMT_INCOME_TOTAL",
    y="AMT_CREDIT",
    opacity=0.7,
    title="Income vs Credit Amount",
)
st.plotly_chart(fig1, use_container_width=True)

# 2. Age Distribution
st.subheader("📈 Age Distribution")
fig2 = px.histogram(
    df,
    x="AGE_YEARS",
    nbins=40,
    title="Distribution of Applicant Age (Years)",
)
st.plotly_chart(fig2, use_container_width=True)

# 3. Optional ratio plot if present
if "CREDIT_INCOME_RATIO" in df.columns:
    st.subheader("📊 Credit-to-Income Ratio")
    fig3 = px.histogram(
        df,
        x="CREDIT_INCOME_RATIO",
        nbins=40,
        title="Distribution of Credit-to-Income Ratio",
    )
    st.plotly_chart(fig3, use_container_width=True)
