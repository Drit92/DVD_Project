import os
import zipfile
import gdown
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIG
# ============================================================
DATA_ID = "1FSSkKQOJtvOpP1I1qyr4x6SYQg-kBnVw"   # Google Drive file ID
ZIP_PATH = "dataset.zip"
EXTRACT_DIR = "file_zip"

st.set_page_config(page_title="Loan Applicant Insights Dashboard", layout="wide")
st.title("📊 Loan Applicant Visual Insights Dashboard")


# ============================================================
# Download & Extract ZIP
# ============================================================
@st.cache_data(show_spinner=True)
def download_and_extract():
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        gdown.download(id=DATA_ID, output=ZIP_PATH, quiet=False)

    # Extract ZIP
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Recursive search for CSV
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.lower().endswith(".csv"):
                return os.path.join(root, f)

    return None


# ============================================================
# Load Data + Apply EXACT Colab Feature Engineering
# ============================================================
@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    csv_path = download_and_extract()
    if csv_path is None:
        raise FileNotFoundError("❌ No CSV found inside ZIP!")

    df = pd.read_csv(csv_path)

    # ==================================================================
    # 1. AGE_YEARS — EXACT from Colab
    # ==================================================================
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (df["DAYS_BIRTH"] / -365).round(1)
    else:
        df["AGE_YEARS"] = np.nan

    # ==================================================================
    # 2. EMP_YEARS — EXACT from Colab (plus anomaly fixing)
    # ==================================================================
    if "DAYS_EMPLOYED" in df.columns:
        df["EMP_YEARS"] = (df["DAYS_EMPLOYED"] / -365).round(1)
        df.loc[df["EMP_YEARS"] > 60, "EMP_YEARS"] = np.nan
    else:
        df["EMP_YEARS"] = np.nan

    # ==================================================================
    # 3. Financial Ratios — EXACT from Colab
    # ==================================================================
    # Avoid division by zero
    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / income
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / income

    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)

    # Clean inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.copy()
    return df


# ============================================================
# MAIN APP
# ============================================================
try:
    df = load_and_prepare_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

st.success("Dataset loaded and processed successfully!")

# ============================================================
# Sidebar Filters
# ============================================================
st.sidebar.header("Filters")

# Income slider
income_min = float(df["AMT_INCOME_TOTAL"].min())
income_max = float(df["AMT_INCOME_TOTAL"].max())

income_slider = st.sidebar.slider(
    "Annual Income (USD)",
    income_min,
    income_max,
    (income_min, income_max),
    step=1000.0,
)

# Age slider
age_series = df["AGE_YEARS"].dropna()
age_min = int(age_series.min()) if len(age_series) else 18
age_max = int(age_series.max()) if len(age_series) else 70

age_slider = st.sidebar.slider(
    "Age (years)",
    age_min,
    age_max,
    (age_min, age_max)
)

# Apply filters
df_filtered = df[
    (df["AMT_INCOME_TOTAL"].between(income_slider[0], income_slider[1])) &
    (df["AGE_YEARS"].between(age_slider[0], age_slider[1], inclusive="both"))
]

st.caption(f"Showing **{len(df_filtered):,}** applicants (out of {len(df):,}).")

# ============================================================
# Data Preview
# ============================================================
st.subheader("🔍 Data Preview")
st.dataframe(df_filtered.head(), use_container_width=True)

# ============================================================
# VISUALS
# ============================================================

# --------------------- 1) Income vs Credit ---------------------
st.subheader("💰 Income vs Credit Amount")
fig1 = px.scatter(
    df_filtered,
    x="AMT_INCOME_TOTAL",
    y="AMT_CREDIT",
    opacity=0.6,
    trendline="ols",
    labels={"AMT_INCOME_TOTAL": "Income", "AMT_CREDIT": "Credit Amount"},
)
st.plotly_chart(fig1, use_container_width=True)


# --------------------- 2) Age Distribution ---------------------
st.subheader("📈 Age Distribution (Years)")
fig2 = px.histogram(
    df_filtered,
    x="AGE_YEARS",
    nbins=40,
    labels={"AGE_YEARS": "Age (Years)"},
)
st.plotly_chart(fig2, use_container_width=True)


# ----------------- 3) Credit-to-Income Ratio -------------------
st.subheader("📊 Credit-to-Income Ratio")
fig3 = px.histogram(
    df_filtered,
    x="CREDIT_INCOME_RATIO",
    nbins=40,
    labels={"CREDIT_INCOME_RATIO": "Credit / Income"},
)
st.plotly_chart(fig3, use_container_width=True)


# ----------------- 4) Income Per Person -------------------
st.subheader("👨‍👩‍👧 Income Per Family Member")
fig4 = px.histogram(
    df_filtered,
    x="INCOME_PER_PERSON",
    nbins=40,
    labels={"INCOME_PER_PERSON": "Income Per Person"},
)
st.plotly_chart(fig4, use_container_width=True)
