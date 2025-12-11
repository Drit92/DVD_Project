import os
import zipfile

import gdown
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
DATA_ID = "1FSSkKQOJtvOpP1I1qyr4x6SYQg-kBnVw"   # Google Drive file ID
ZIP_PATH = "dataset.zip"
EXTRACT_DIR = "file_zip"

# Plotly template and color palette
pio.templates.default = "plotly_white"
PLOTLY_COLORS = ["#004c6d", "#00a1c6", "#f29f05", "#e03b8b"]

st.set_page_config(page_title="Loan Applicant Insights Dashboard", layout="wide")
st.title("📊 Loan Applicant Visual Insights Dashboard")
st.markdown(
    "Interactive views of urban loan applicants to explore income, age, credit, and risk-related segments."
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
# Load + feature engineering
# --------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    csv_path = download_and_extract()
    if csv_path is None:
        raise FileNotFoundError("No CSV file found inside the ZIP.")

    df = pd.read_csv(csv_path)

    # 1) AGE_YEARS (from Colab logic)
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (df["DAYS_BIRTH"] / -365).round(1)
    else:
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

    return df


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
try:
    df = load_and_prepare_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# Ensure AGE_YEARS exists
if "AGE_YEARS" not in df.columns:
    st.error("AGE_YEARS column could not be created. Check source data.")
    st.stop()

# ---- Sidebar filters ------------------------------------------------
st.sidebar.header("Filters")

# Income range
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

# Age range (drop NaNs)
age_series = df["AGE_YEARS"].dropna()
if len(age_series) > 0:
    age_min = int(age_series.min())
    age_max = int(age_series.max())
else:
    age_min, age_max = 18, 70

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
# Interactive Plotly visuals
# --------------------------------------------------------------------
col1, col2 = st.columns(2)

# 1. Income vs Credit, colored by age bucket
with col1:
    st.subheader("💰 Income vs Credit by Age Segment")

    df_scatter = df_filtered.copy()
    df_scatter["AGE_BUCKET"] = pd.cut(
        df_scatter["AGE_YEARS"],
        bins=[18, 30, 40, 50, 60, 80],
        labels=["18–30", "31–40", "41–50", "51–60", "60+"],
        include_lowest=True,
    )

    if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(df_scatter.columns):
        fig1 = px.scatter(
            df_scatter,
            x="AMT_INCOME_TOTAL",
            y="AMT_CREDIT",
            color="AGE_BUCKET",
            color_discrete_sequence=PLOTLY_COLORS,
            hover_data=["AGE_YEARS", "CREDIT_INCOME_RATIO", "INCOME_PER_PERSON"],
            opacity=0.7,
            trendline="ols",
            title="Income vs Credit Amount",
        )
        fig1.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig1.update_layout(legend_title_text="Age bucket")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Income or credit columns are missing for this view.")

# 2. Age distribution
with col2:
    st.subheader("📈 Age Distribution")

    if "AGE_YEARS" in df_filtered.columns:
        fig2 = px.histogram(
            df_filtered,
            x="AGE_YEARS",
            nbins=40,
            marginal="box",  # adds compact boxplot
            color_discrete_sequence=[PLOTLY_COLORS[1]],
            title="Distribution of Applicant Age (Years)",
        )
        fig2.update_layout(bargap=0.05)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("AGE_YEARS column is missing.")

# 3. Credit-to-income ratio
st.subheader("📊 Credit-to-Income Ratio")

if "CREDIT_INCOME_RATIO" in df_filtered.columns:
    fig3 = px.histogram(
        df_filtered,
        x="CREDIT_INCOME_RATIO",
        nbins=40,
        color_discrete_sequence=[PLOTLY_COLORS[2]],
        title="Distribution of Credit-to-Income Ratio",
    )
    fig3.update_layout(bargap=0.05)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("CREDIT_INCOME_RATIO is not available.")

# 4. Employment years by income bracket
st.subheader("👥 Employment Years by Income Bracket")

if {"AMT_INCOME_TOTAL", "EMP_YEARS"}.issubset(df_filtered.columns):
    df_cat = df_filtered.copy()
    df_cat["INCOME_BRACKET"] = pd.qcut(
        df_cat["AMT_INCOME_TOTAL"],
        q=4,
        labels=["Low", "Lower-Mid", "Upper-Mid", "High"],
        duplicates="drop",
    )

    fig4 = px.box(
        df_cat,
        x="INCOME_BRACKET",
        y="EMP_YEARS",
        color="INCOME_BRACKET",
        color_discrete_sequence=PLOTLY_COLORS,
        title="Employment Years by Income Bracket",
    )
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Required columns for employment view are missing.")

# --------------------------------------------------------------------
# Seaborn correlation heatmap
# --------------------------------------------------------------------
st.subheader("📌 Feature Correlations")

num_cols = [
    c
    for c in [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AGE_YEARS",
        "EMP_YEARS",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "INCOME_PER_PERSON",
    ]
    if c in df.columns
]

if len(num_cols) >= 2:
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    plt.title("Correlation Between Key Numeric Features")
    st.pyplot(fig)
else:
    st.info("Not enough numeric columns available for a correlation heatmap.")
