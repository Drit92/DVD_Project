import os
import zipfile

import gdown
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
DATA_ID = "1FSSkKQOJtvOpP1I1qyr4x6SYQg-kBnVw"   # Google Drive file ID
ZIP_PATH = "dataset.zip"
EXTRACT_DIR = "file_zip"

st.set_page_config(page_title="Loan Applicant Insights Dashboard", layout="wide")
st.title("📊 Loan Applicant Visual Insights Dashboard")
st.markdown(
    "Interactive views of urban loan applicants to explore income, age, credit, and risk segments."
)


# --------------------------------------------------------------------
# Download & extract ZIP, return path to first CSV
# --------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def download_and_extract():
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Download only if ZIP is not present
    if not os.path.exists(ZIP_PATH):
        gdown.download(id=DATA_ID, output=ZIP_PATH, quiet=False)

    # Extract ZIP
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Find first CSV recursively
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.lower().endswith(".csv"):
                return os.path.join(root, f)

    return None


# --------------------------------------------------------------------
# Load + feature engineering in a single cached function
# --------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    csv_path = download_and_extract()
    if csv_path is None:
        raise FileNotFoundError("No CSV file found inside the ZIP.")

    df = pd.read_csv(csv_path)

    # 1) AGE_YEARS  (from Colab logic)
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (df["DAYS_BIRTH"] / -365).round(1)
    else:
        # fallback so the app does not crash even if not present
        df["AGE_YEARS"] = np.nan

    # 2) EMP_YEARS
    if "DAYS_EMPLOYED" in df.columns:
        df["EMP_YEARS"] = (df["DAYS_EMPLOYED"] / -365).round(1)
        # Replace nonsense values (people with 1000 years employment)
        df.loc[df["EMP_YEARS"] > 60, "EMP_YEARS"] = np.nan
    else:
        df["EMP_YEARS"] = np.nan

    # 3) Financial ratios
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    else:
        df["CREDIT_INCOME_RATIO"] = np.nan

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    else:
        df["ANNUITY_INCOME_RATIO"] = np.nan

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    else:
        df["INCOME_PER_PERSON"] = np.nan

    # Basic NA handling for visuals
    df = df.copy()
    return df


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
try:
    df = load_and_prepare_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# Ensure AGE_YEARS exists and is numeric
if "AGE_YEARS" not in df.columns:
    st.error("AGE_YEARS column could not be created. Check source data.")
    st.stop()

# ---- Sidebar filters ------------------------------------------------
st.sidebar.header("Filters")

# Income range (guard against non-positive / missing values)
if "AMT_INCOME_TOTAL" in df.columns:
    income_min = float(df["AMT_INCOME_TOTAL"].min())
    income_max = float(df["AMT_INCOME_TOTAL"].max())
else:
    income_min, income_max = 0.0, 0.0

income_slider = st.sidebar.slider(
    "Annual Income Range",
    float(income_min),
    float(income_max),
    (float(income_min), float(income_max)),
    step=1000.0,
)

# Age range: drop NaNs to get valid bounds
age_series = df["AGE_YEARS"].dropna()
if len(age_series) > 0:
    age_min = int(age_series.min())
    age_max = int(age_series.max())
else:
    age_min, age_max = 18, 70  # fallback

age_slider = st.sidebar.slider(
    "Age Range (years)",
    int(age_min),
    int(age_max),
    (int(age_min), int(age_max)),
)

# Apply filters safely
df_filtered = df[
    (df["AMT_INCOME_TOTAL"].between(income_slider[0], income_slider[1]))
    & (df["AGE_YEARS"].between(age_slider[0], age_slider[1]))
]

st.caption(
    f"Showing {len(df_filtered):,} applicants after filters "
    f"(out of {len(df):,} total)."
)

# --------------------------------------------------------------------
# Data preview
# --------------------------------------------------------------------
st.subheader("🔎 Data Preview")
st.dataframe(df_filtered.head())

# --------------------------------------------------------------------
# Visuals
# --------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Income vs Credit")
    if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(df_filtered.columns):
        fig1 = px.scatter(
            df_filtered,
            x="AMT_INCOME_TOTAL",
            y="AMT_CREDIT",
            opacity=0.6,
            title="Income vs Credit Amount",
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Income or credit columns are missing.")

with col2:
    st.subheader("📈 Age Distribution")
    if "AGE_YEARS" in df_filtered.columns:
        fig2 = px.histogram(
            df_filtered,
            x="AGE_YEARS",
            nbins=40,
            title="Distribution of Applicant Age (Years)",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("AGE_YEARS column is missing.")

st.subheader("📊 Credit-to-Income Ratio")
if "CREDIT_INCOME_RATIO" in df_filtered.columns:
    fig3 = px.histogram(
        df_filtered,
        x="CREDIT_INCOME_RATIO",
        nbins=40,
        title="Distribution of Credit-to-Income Ratio",
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("CREDIT_INCOME_RATIO is not available.")
