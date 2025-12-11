# app.py — robust loader + exact feature engineering (AGE_YEARS debug)
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

st.set_page_config(page_title="Loan Applicant Insights Dashboard", layout="wide")
st.title("📊 Loan Applicant Visual Insights Dashboard (robust loader)")

# -------------------
# Utilities
# -------------------
def find_first_csv(root_dir):
    """Return list of all CSVs found (full paths)."""
    csvs = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(root, f))
    return sorted(csvs)

def prefer_application_csv(csv_paths):
    """If multiple CSVs, prefer filenames with 'app'/'application'/'train'."""
    if not csv_paths:
        return None
    priorities = ["application", "app", "app_data", "application_train", "train", "app_train"]
    for p in priorities:
        for pth in csv_paths:
            if p in os.path.basename(pth).lower():
                return pth
    return csv_paths[0]

def normalize_columns(df):
    """Return df with normalized column names and mapping (orig->norm)."""
    orig_cols = list(df.columns)
    norm_map = {}
    new_cols = []
    for c in orig_cols:
        nc = str(c).strip()            # remove leading/trailing spaces
        nc = "_".join(nc.split())      # collapse internal whitespace to single underscore
        nc = nc.upper()                # uppercase
        new_cols.append(nc)
        norm_map[nc] = c               # map normalized -> original
    df.columns = new_cols
    return df, norm_map

def detect_days_birth_column(col_names):
    """Heuristics to find DAYS_BIRTH-like column in normalized col names."""
    candidates = []
    for c in col_names:
        if "DAYS_BIRTH" == c:
            return c
    # looser matches
    for c in col_names:
        if "BIRTH" in c and "DAY" in c:
            candidates.append(c)
        if c.startswith("DAYS_") and "BIRTH" in c:
            candidates.append(c)
    # also accept column names that include 'DOB' or 'DATE_OF_BIRTH'
    for c in col_names:
        if "DOB" in c or "DATE_OF_BIRTH" in c or "BIRTHDATE" in c:
            candidates.append(c)
    return candidates[0] if candidates else None

# -------------------
# Download & extract
# -------------------
@st.cache_data(show_spinner=True)
def download_and_extract():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        gdown.download(id=DATA_ID, output=ZIP_PATH, quiet=False)
    # extract
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    return EXTRACT_DIR

# -------------------
# Load + prepare data (robust)
# -------------------
@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    extract_dir = download_and_extract()
    csv_list = find_first_csv(extract_dir)
    if not csv_list:
        raise FileNotFoundError("No CSV files found inside extracted ZIP.")
    csv_path = prefer_application_csv(csv_list)

    # read CSV (try common encodings, separators fallback)
    read_errors = []
    df = None
    read_attempts = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "utf-8"},
    ]
    for opts in read_attempts:
        try:
            df = pd.read_csv(csv_path, **opts, low_memory=False)
            break
        except Exception as e:
            read_errors.append((opts, str(e)))
    if df is None:
        raise IOError(f"Failed to read CSV {csv_path}. Attempts: {read_errors}")

    # Keep original copy for debugging if needed
    df_original = df.copy()

    # Normalize column names (strip/underscore/upper)
    df, norm_map = normalize_columns(df)

    # Detect DAYS_BIRTH-like column
    days_col = detect_days_birth_column(df.columns.tolist())

    # Debug info we'll return
    debug = {
        "csv_used": csv_path,
        "all_csvs": csv_list,
        "normalized_columns_sample": list(df.columns)[:30],
        "days_col_detected": days_col,
    }

    # Convert candidate column to numeric robustly
    if days_col is not None:
        # coerce strings with commas, spaces, parentheses, etc.
        ser = df[days_col].astype(str).str.replace(",", "").str.strip()
        # Replace empty strings with NaN
        ser = ser.replace({"": np.nan, "NA": np.nan, "NAN": np.nan})
        ser_num = pd.to_numeric(ser, errors="coerce")
        df[days_col] = ser_num
        debug["days_col_nonnull_count"] = int(df[days_col].notna().sum())
    else:
        debug["days_col_nonnull_count"] = 0

    # Create AGE_YEARS using detected column if possible
    if days_col is not None and df[days_col].notna().sum() > 0:
        # Following Colab logic: AGE_YEARS = (DAYS_BIRTH / -365).round(1)
        df["AGE_YEARS"] = (df[days_col] / -365).round(1)
        debug["age_computed_from"] = days_col
        debug["age_sample"] = df["AGE_YEARS"].dropna().head(5).tolist()
    else:
        df["AGE_YEARS"] = np.nan
        debug["age_computed_from"] = None
        debug["age_sample"] = []

    # EMP_YEARS: detect DAYS_EMPLOYED-like column similarly
    days_emp_col = None
    for c in df.columns:
        if c == "DAYS_EMPLOYED":
            days_emp_col = c
            break
    if days_emp_col is None:
        for c in df.columns:
            if "EMPLOY" in c and "DAY" in c:
                days_emp_col = c
                break

    if days_emp_col:
        ser = df[days_emp_col].astype(str).str.replace(",", "").str.strip()
        ser = ser.replace({"": np.nan, "NA": np.nan, "NAN": np.nan})
        ser_num = pd.to_numeric(ser, errors="coerce")
        df[days_emp_col] = ser_num
        df["EMP_YEARS"] = (df[days_emp_col] / -365).round(1)
        # Clean unrealistic values > 60 (as in Colab)
        df.loc[df["EMP_YEARS"] > 60, "EMP_YEARS"] = np.nan
        debug["emp_computed_from"] = days_emp_col
        debug["emp_sample"] = df["EMP_YEARS"].dropna().head(5).tolist()
    else:
        df["EMP_YEARS"] = np.nan
        debug["emp_computed_from"] = None
        debug["emp_sample"] = []

    # Financial ratios — proceed only if columns exist (columns have been normalized to upper)
    def safe_div(n, d):
        return n / d.replace({0: np.nan})

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    else:
        df["CREDIT_INCOME_RATIO"] = np.nan

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    else:
        df["ANNUITY_INCOME_RATIO"] = np.nan

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace({0: np.nan})
    else:
        df["INCOME_PER_PERSON"] = np.nan

    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Return df and debug info
    return df, debug, df_original

# -------------------
# Main
# -------------------
try:
    df, debug, df_original = load_and_prepare_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.success("Dataset loaded (robust).")
st.write("**Debug info (loader):**")
st.write({
    "csv_used": debug["csv_used"],
    "sample_normalized_columns": debug["normalized_columns_sample"][:40],
    "days_col_detected": debug["days_col_detected"],
    "days_col_nonnull_count": debug["days_col_nonnull_count"],
    "age_computed_from": debug["age_computed_from"],
    "age_sample (first 5)": debug["age_sample"],
    "emp_computed_from": debug["emp_computed_from"],
    "emp_sample (first 5)": debug["emp_sample"],
})

# At this point columns are uppercase; show first 20 normalized column names
st.write("First 30 normalized columns:", list(df.columns)[:30])

# -------------------
# If age couldn't be computed — show original few rows to help debug
# -------------------
if df["AGE_YEARS"].isna().all():
    st.warning("AGE_YEARS could not be computed — no DAYS_BIRTH-like column with numeric values was found.")
    st.write("Sample of original raw columns & values (first 5 rows):")
    st.dataframe(df_original.head().iloc[:, :20])
    st.stop()

# -------------------
# Downstream: convert column names back to friendly lower form for visuals
# (create a copy to use friendly names)
# -------------------
df_viz = df.copy()
# For convenience, create lower-case friendly names mapping if needed
friendly = {c: c for c in df_viz.columns}

# -------------------
# Sidebar filters (use uppercase names)
# -------------------
st.sidebar.header("Filters")
income_col = "AMT_INCOME_TOTAL" if "AMT_INCOME_TOTAL" in df_viz.columns else None
if income_col:
    income_min = float(df_viz[income_col].min())
    income_max = float(df_viz[income_col].max())
else:
    income_min, income_max = 0.0, 0.0

income_slider = st.sidebar.slider(
    "Annual Income Range",
    float(income_min),
    float(income_max),
    (float(income_min), float(income_max)),
    step=1000.0,
)

age_series = df_viz["AGE_YEARS"].dropna()
if len(age_series) > 0:
    age_min = int(age_series.min())
    age_max = int(age_series.max())
else:
    age_min, age_max = 18, 70

age_slider = st.sidebar.slider(
    "Age Range (years)",
    age_min,
    age_max,
    (age_min, age_max),
)

# Apply filters
mask_income = True
mask_age = True
if income_col:
    mask_income = df_viz[income_col].between(income_slider[0], income_slider[1])
mask_age = df_viz["AGE_YEARS"].between(age_slider[0], age_slider[1])
df_filtered = df_viz[mask_income & mask_age]

st.caption(f"Showing {len(df_filtered):,} rows after filters (from {len(df_viz):,}).")

# -------------------
# Visuals
# -------------------
st.subheader("🔎 Data preview (filtered)")
st.dataframe(df_filtered.head().astype(object), use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Income vs Credit")
    if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(df_filtered.columns):
        fig = px.scatter(df_filtered, x="AMT_INCOME_TOTAL", y="AMT_CREDIT",
                         opacity=0.6, trendline="ols",
                         labels={"AMT_INCOME_TOTAL":"Income", "AMT_CREDIT":"Credit"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("AMT_INCOME_TOTAL or AMT_CREDIT not available")

with col2:
    st.subheader("📈 Age distribution")
    fig2 = px.histogram(df_filtered, x="AGE_YEARS", nbins=40)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("📊 Credit / Income ratio")
if "CREDIT_INCOME_RATIO" in df_filtered.columns:
    fig3 = px.histogram(df_filtered, x="CREDIT_INCOME_RATIO", nbins=40)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("CREDIT_INCOME_RATIO not available")
