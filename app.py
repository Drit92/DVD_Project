# app.py - Streamlit dashboard without AGE_YEARS
import os
import zipfile
import gdown
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------
# CONFIG
# -------------------
DATA_ID = "1FSSkKQOJtvOpP1I1qyr4x6SYQg-kBnVw"   # Google Drive file ID
ZIP_PATH = "dataset.zip"
EXTRACT_DIR = "file_zip"

st.set_page_config(page_title="Loan Applicant Visual Insights", layout="wide")
st.title("📊 Loan Applicant Visual Insights Dashboard (no AGE_YEARS)")
st.markdown("Interactive visualizations focusing on income, credit, employment years and ratios.")

# -------------------
# Helpers
# -------------------
def find_csvs(root_dir):
    csvs = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(root, f))
    return sorted(csvs)

def prefer_application_csv(csv_paths):
    if not csv_paths:
        return None
    priorities = ["application", "app", "app_data", "train", "app_train"]
    for p in priorities:
        for pth in csv_paths:
            if p in os.path.basename(pth).lower():
                return pth
    return csv_paths[0]

# -------------------
# Download & extract
# -------------------
@st.cache_data(show_spinner=True)
def download_and_extract():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        gdown.download(id=DATA_ID, output=ZIP_PATH, quiet=False)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    return EXTRACT_DIR

# -------------------
# Load + basic engineering (NO AGE_YEARS)
# -------------------
@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    extracted = download_and_extract()
    csv_list = find_csvs(extracted)
    if not csv_list:
        raise FileNotFoundError("No CSV files found inside the ZIP.")
    csv_path = prefer_application_csv(csv_list)

    # Try reading CSV with a few common encodings/separators
    read_opts = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "utf-8"},
    ]
    df = None
    for opts in read_opts:
        try:
            df = pd.read_csv(csv_path, low_memory=False, **opts)
            break
        except Exception:
            df = None
    if df is None:
        raise IOError(f"Failed to read CSV: {csv_path}")

    # Basic copy
    df = df.copy()

    # ---------- EMP_YEARS (from DAYS_EMPLOYED) ----------
    if "DAYS_EMPLOYED" in df.columns:
        # convert to numeric if needed
        df["DAYS_EMPLOYED"] = pd.to_numeric(df["DAYS_EMPLOYED"], errors="coerce")
        df["EMP_YEARS"] = (df["DAYS_EMPLOYED"] / -365).round(1)
        df.loc[df["EMP_YEARS"] > 60, "EMP_YEARS"] = np.nan
    else:
        df["EMP_YEARS"] = np.nan

    # ---------- Financial ratios (Colab code) ----------
    # Avoid division by zero or zeros -> replace with NaN for safe division
    if "AMT_INCOME_TOTAL" in df.columns:
        income_safe = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    else:
        income_safe = pd.Series(np.nan, index=df.index)

    if "AMT_CREDIT" in df.columns:
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / income_safe
    else:
        df["CREDIT_INCOME_RATIO"] = np.nan

    if "AMT_ANNUITY" in df.columns:
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / income_safe
    else:
        df["ANNUITY_INCOME_RATIO"] = np.nan

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace({0: np.nan})
    else:
        df["INCOME_PER_PERSON"] = np.nan

    # Clean infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df, csv_path

# -------------------
# MAIN
# -------------------
try:
    df, csv_used = load_and_prepare_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.success(f"Loaded CSV: `{os.path.basename(csv_used)}`")
st.write("Data columns (first 40):", list(df.columns)[:40])

# -------------------
# Sidebar filters (income + optional employment years)
# -------------------
st.sidebar.header("Filters")

# Income filter (if available)
if "AMT_INCOME_TOTAL" in df.columns and df["AMT_INCOME_TOTAL"].notna().any():
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

# Employment years filter (optional)
if df["EMP_YEARS"].notna().any():
    emp_min = int(np.nanmin(df["EMP_YEARS"]))
    emp_max = int(np.nanmax(df["EMP_YEARS"]))
    emp_slider = st.sidebar.slider(
        "Employment Years Range",
        emp_min,
        emp_max,
        (emp_min, emp_max),
    )
else:
    emp_slider = None

# Apply mask safely
mask = df["AMT_INCOME_TOTAL"].between(income_slider[0], income_slider[1]) if "AMT_INCOME_TOTAL" in df.columns else pd.Series(True, index=df.index)
if emp_slider is not None:
    mask = mask & df["EMP_YEARS"].between(emp_slider[0], emp_slider[1])
df_filtered = df[mask].copy()

st.caption(f"Showing **{len(df_filtered):,}** applicants (out of {len(df):,}).")

# -------------------
# Data preview
# -------------------
st.subheader("🔎 Data Preview")
st.dataframe(df_filtered.head(), use_container_width=True)

# -------------------
# Visual 1: Income vs Credit
# -------------------
st.subheader("💰 Income vs Credit")
if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(df_filtered.columns):
    fig1 = px.scatter(
        df_filtered,
        x="AMT_INCOME_TOTAL",
        y="AMT_CREDIT",
        opacity=0.6,
        trendline="ols",
        hover_data=["SK_ID_CURR"] if "SK_ID_CURR" in df_filtered.columns else None,
        labels={"AMT_INCOME_TOTAL": "Income", "AMT_CREDIT": "Credit"},
    )
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("AMT_INCOME_TOTAL or AMT_CREDIT is missing; cannot render scatter.")

# -------------------
# Visual 2: Credit-to-Income Ratio
# -------------------
st.subheader("📊 Credit-to-Income Ratio")
if "CREDIT_INCOME_RATIO" in df_filtered.columns:
    fig2 = px.histogram(df_filtered, x="CREDIT_INCOME_RATIO", nbins=40, labels={"CREDIT_INCOME_RATIO": "Credit / Income"})
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("CREDIT_INCOME_RATIO not available.")

# -------------------
# Visual 3: Income per person
# -------------------
st.subheader("👨‍👩‍👧 Income per Family Member")
if "INCOME_PER_PERSON" in df_filtered.columns:
    fig3 = px.histogram(df_filtered, x="INCOME_PER_PERSON", nbins=40, labels={"INCOME_PER_PERSON":"Income per person"})
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("INCOME_PER_PERSON not available.")

# -------------------
# Visual 4: Employment years distribution (optional)
# -------------------
if "EMP_YEARS" in df_filtered.columns and df_filtered["EMP_YEARS"].notna().any():
    st.subheader("🛠️ Employment Years Distribution")
    fig4 = px.histogram(df_filtered, x="EMP_YEARS", nbins=30, labels={"EMP_YEARS": "Employment years"})
    st.plotly_chart(fig4, use_container_width=True)

# -------------------
# End
# -------------------
st.markdown("---")
st.markdown("If you want AGE-based visuals restored later, tell me which column in your CSV actually contains birth/age info (column name) and I'll add age back.")
