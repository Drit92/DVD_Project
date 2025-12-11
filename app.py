import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gdown
import zipfile
import os

DATA_ID = "1FSSkKQOJtvOpP1I1qyr4x6SYQg-kBnVw"   # Google Drive ID
ZIP_PATH = "dataset.zip"
EXTRACT_DIR = "file_zip"

# -----------------------------------------------
# Download & Extract ZIP Recursively
# -----------------------------------------------
def download_and_extract():
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading dataset...")
        gdown.download(id=DATA_ID, output=ZIP_PATH, quiet=False)

    # Extract ZIP
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    except:
        st.error("❌ ZIP extraction failed.")
        return None

    # Search recursively for ANY CSV
    csv_files = []
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        st.error("❌ No CSV file found inside ZIP.")
        return None

    return csv_files[0]  # First CSV found


# -----------------------------------------------
# UI
# -----------------------------------------------
st.set_page_config(page_title="Loan Dashboard", layout="wide")
st.title("📊 Loan Applicant Visual Insights Dashboard")


# -----------------------------------------------
# Load Data
# -----------------------------------------------
csv_path = download_and_extract()

if csv_path:
    st.success(f"Dataset loaded: `{csv_path}`")

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()

    # Preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # -------------------------------------------------------------------
    # PREPROCESSING — MUST BE INSIDE THIS BLOCK
    # -------------------------------------------------------------------
    # AGE_YEARS
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (df["DAYS_BIRTH"].abs() // 365).astype(int)

    # EMPLOY_YEARS
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOY_YEARS"] = (df["DAYS_EMPLOYED"].abs() // 365).astype(int)

    # Fill NA to avoid visualization errors
    df = df.fillna(0)

    # -------------------------------------------------------------------
    # Ensure required columns exist
    # -------------------------------------------------------------------
    required_cols = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT",
        "AGE_YEARS", "EMPLOY_YEARS"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # -------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------
    st.subheader("💰 Income vs Credit")

    fig1 = px.scatter(
        df, x="AMT_INCOME_TOTAL", y="AMT_CREDIT",
        opacity=0.7
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📈 Age Distribution")
    fig2 = px.histogram(df, x="AGE_YEARS")
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.error("Dataset could not be loaded.")
