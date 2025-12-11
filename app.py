import os
import zipfile

import gdown
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
DATA_ID = "1FSSkKQOJtvOpP1I1qyr4x6SYQg-kBnVw"   # Google Drive ZIP ID
ZIP_PATH = "dataset.zip"
EXTRACT_DIR = "file_zip"

# Plotly template and color palette
pio.templates.default = "plotly_white"
PLOTLY_COLORS = ["#004c6d", "#00a1c6", "#f29f05", "#e03b8b"]

st.set_page_config(page_title="Loan Applicant Risk Insights", layout="wide")
st.title("📊 Loan Applicant Risk Insights Dashboard")
st.markdown(
    "Focused visual insights to help the credit team identify safe segments, watchlist groups, "
    "and red-flag borrowers based on financial stress, behaviour, and external scores."
)

# --------------------------------------------------------------------
# Download & extract ZIP, return base folder containing both CSVs
# --------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def download_and_extract() -> str:
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        gdown.download(id=DATA_ID, output=ZIP_PATH, quiet=False)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    base_dir = None
    for root, dirs, files in os.walk(EXTRACT_DIR):
        if "application_data.csv" in files and "previous_application.csv" in files:
            base_dir = root
            break

    if base_dir is None:
        raise FileNotFoundError(
            "Could not find application_data.csv and previous_application.csv inside the ZIP."
        )

    return base_dir

# --------------------------------------------------------------------
# Load + feature engineering with sampling
# --------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare_data(sample_n: int | None = 60_000) -> pd.DataFrame:
    """
    Load application_data and previous_application, optionally sample rows
    from application_data, and recreate the main engineered & behavioural
    features from the notebook. Returns a compact analytical dataset.
    """
    base_dir = download_and_extract()
    app_path = os.path.join(base_dir, "application_data.csv")
    prev_path = os.path.join(base_dir, "previous_application.csv")

    app_data = pd.read_csv(app_path)
    prev_data = pd.read_csv(prev_path)

    # Downsample to keep memory & plotting manageable
    if sample_n is not None and len(app_data) > sample_n:
        app_data = app_data.sample(sample_n, random_state=42)

    # ---------------------------
    # 1) Demographic & ratios
    # ---------------------------
    if "DAYS_BIRTH" in app_data.columns:
        app_data["AGE_YEARS"] = (app_data["DAYS_BIRTH"] / -365).round(1)

    if "DAYS_EMPLOYED" in app_data.columns:
        emp = app_data["DAYS_EMPLOYED"].replace(365243, np.nan)
        app_data["EMP_YEARS"] = (-emp / 365).clip(lower=0, upper=40)

    app_data["CREDIT_INCOME_RATIO"] = app_data["AMT_CREDIT"] / app_data["AMT_INCOME_TOTAL"]
    app_data["ANNUITY_INCOME_RATIO"] = app_data["AMT_ANNUITY"] / app_data["AMT_INCOME_TOTAL"]
    app_data["INCOME_PER_PERSON"] = app_data["AMT_INCOME_TOTAL"] / app_data["CNT_FAM_MEMBERS"]

    # ---------------------------
    # 2) Behavioural features from previous_application
    # ---------------------------
    refusal_flag = (
        prev_data.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"]
        .apply(lambda x: (x == "Refused").any())
        .astype(int)
        .rename("FLAG_EVER_REFUSED")
    )

    prev_app_count = (
        prev_data.groupby("SK_ID_CURR")["SK_ID_PREV"]
        .count()
        .rename("PREV_APP_COUNT")
    )

    prev_status = (
        prev_data.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
        .rename("PREV_MAIN_STATUS")
    )

    app_data = app_data.merge(refusal_flag, on="SK_ID_CURR", how="left")
    app_data = app_data.merge(prev_app_count, on="SK_ID_CURR", how="left")
    app_data = app_data.merge(prev_status, on="SK_ID_CURR", how="left")

    app_data["FLAG_EVER_REFUSED"] = app_data["FLAG_EVER_REFUSED"].fillna(0)
    app_data["PREV_APP_COUNT"] = app_data["PREV_APP_COUNT"].fillna(0)
    app_data["PREV_MAIN_STATUS"] = app_data["PREV_MAIN_STATUS"].fillna("No Previous History")

    app_data["PREV_APPS_BIN"] = pd.cut(
        app_data["PREV_APP_COUNT"],
        bins=[-1, 0, 2, 4, 9, 1000],
        labels=["0", "1-2", "3-4", "5-9", "10+"],
    )

    status_risk_map = {
        "Refused": 4,
        "Canceled": 3,
        "Unused offer": 2,
        "Approved": 1,
        "No Previous History": 0,
    }
    app_data["PREV_STATUS_RISK"] = app_data["PREV_MAIN_STATUS"].map(status_risk_map).fillna(0)

    # ---------------------------
    # 3) Stress bands
    # ---------------------------
    emi_bins = [0, 0.10, 0.20, 0.30, 0.50, 1.0]
    emi_labels = ["<10%", "10–20%", "20–30%", "30–50%", "50%+"]
    app_data["EMI_BIN"] = pd.cut(
        app_data["ANNUITY_INCOME_RATIO"],
        bins=emi_bins,
        labels=emi_labels,
        include_lowest=True,
    )

    credit_bins = [0, 1, 2, 3, 5, 10]
    credit_labels = ["0–1x", "1–2x", "2–3x", "3–5x", "5x+"]
    app_data["CREDIT_BIN"] = pd.cut(
        app_data["CREDIT_INCOME_RATIO"],
        bins=credit_bins,
        labels=credit_labels,
        include_lowest=True,
    )

    income_bins = [0, 50_000, 100_000, 150_000, 300_000, 99_999_999]
    income_labels = ["<50k", "50-100k", "100-150k", "150-300k", "300k+"]
    app_data["INCOME_BIN"] = pd.cut(
        app_data["INCOME_PER_PERSON"],
        bins=income_bins,
        labels=income_labels,
        include_lowest=True,
    )

    # ---------------------------
    # 4) External score features
    # ---------------------------
    if "EXT_SOURCE_2" in app_data.columns:
        app_data["EXT2_Q"] = pd.qcut(
            app_data["EXT_SOURCE_2"],
            4,
            labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
        )
    else:
        app_data["EXT2_Q"] = np.nan

    # ---------------------------
    # 5) Risk scores (FIN, BEHAV, EXT, total)
    # ---------------------------
    credit_points = {"0–1x": 0, "1–2x": 1, "2–3x": 2, "3–5x": 3, "5x+": 4}
    app_data["CREDIT_BIN"] = app_data["CREDIT_BIN"].astype(str)
    app_data["FIN_SCORE"] = app_data["CREDIT_BIN"].map(credit_points)

    app_data["FLAG_EVER_REFUSED"] = app_data["FLAG_EVER_REFUSED"].fillna(0).astype(int)
    app_data["BEHAV_SCORE"] = app_data["FLAG_EVER_REFUSED"] * 2

    ext_points = {"Q1 (low)": 3, "Q2": 2, "Q3": 1, "Q4 (high)": 0}
    app_data["EXT2_Q"] = app_data["EXT2_Q"].astype(str)
    app_data["EXT_SCORE"] = app_data["EXT2_Q"].map(ext_points)

    for c in ["FIN_SCORE", "BEHAV_SCORE", "EXT_SCORE"]:
        app_data[c] = pd.to_numeric(app_data[c], errors="coerce").fillna(0)

    app_data["RISK_SCORE"] = app_data["FIN_SCORE"] + app_data["BEHAV_SCORE"] + app_data["EXT_SCORE"]

    # ---------------------------
    # 6) Build clean analytical dataset
    # ---------------------------
    export_cols = [
        "SK_ID_CURR", "TARGET",

        "AGE_YEARS", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",

        "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "INCOME_PER_PERSON",
        "CREDIT_BIN", "EMI_BIN", "INCOME_BIN", "FIN_SCORE",

        "EXT_SOURCE_2", "EXT2_Q", "EXT_SCORE",

        "FLAG_EVER_REFUSED", "PREV_APP_COUNT", "PREV_APPS_BIN",
        "PREV_MAIN_STATUS", "PREV_STATUS_RISK", "BEHAV_SCORE",

        "RISK_SCORE",
    ]

    clean_df = app_data[export_cols].copy()
    return clean_df

# --------------------------------------------------------------------
# Helper: cap points before plotting
# --------------------------------------------------------------------
def sample_for_plot(df_in: pd.DataFrame, cols: list[str], max_points: int = 50_000):
    df_plot = df_in[cols].dropna()
    if len(df_plot) > max_points:
        df_plot = df_plot.sample(max_points, random_state=42)
    return df_plot

# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
try:
    df = load_and_prepare_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

if "AGE_YEARS" not in df.columns:
    st.error("AGE_YEARS column could not be created. Check source data.")
    st.stop()

st.sidebar.header("Filters")

# Income per person filter
income_min = float(df["INCOME_PER_PERSON"].min())
income_max = float(df["INCOME_PER_PERSON"].max())
income_slider = st.sidebar.slider(
    "Income per person range",
    float(income_min),
    float(income_max),
    (float(income_min), float(income_max)),
    step=5000.0,
)

# Age filter
age_series = df["AGE_YEARS"].dropna()
age_min = int(age_series.min()) if len(age_series) > 0 else 18
age_max = int(age_series.max()) if len(age_series) > 0 else 70
age_slider = st.sidebar.slider(
    "Age range (years)",
    int(age_min),
    int(age_max),
    (int(age_min), int(age_max)),
)

df_filtered = df[
    df["INCOME_PER_PERSON"].between(income_slider[0], income_slider[1])
    & df["AGE_YEARS"].between(age_slider[0], age_slider[1])
]

st.caption(
    f"Visuals below use {len(df_filtered):,} filtered applicants "
    f"(from {len(df):,} sampled rows of the portfolio)."
)

# --------------------------------------------------------------------
# Section 1: Financial stress and default
# --------------------------------------------------------------------
st.header("Financial stress vs default risk")

col1, col2 = st.columns(2)

# 1A) Default by credit / income band
with col1:
    rate_credit = (
        df_filtered.groupby("CREDIT_BIN")["TARGET"]
        .mean()
        .reset_index()
        .sort_values("CREDIT_BIN")
    )

    fig_credit = px.bar(
        rate_credit,
        x="CREDIT_BIN",
        y="TARGET",
        color="CREDIT_BIN",
        color_discrete_sequence=PLOTLY_COLORS,
        labels={"TARGET": "Default rate", "CREDIT_BIN": "Credit / Income band"},
        title="Default rate by credit / income ratio band",
    )
    fig_credit.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_credit, width="stretch")

# 1B) Default by EMI / income band
with col2:
    rate_emi = (
        df_filtered.groupby("EMI_BIN")["TARGET"]
        .mean()
        .reset_index()
        .sort_values("EMI_BIN")
    )

    fig_emi = px.bar(
        rate_emi,
        x="EMI_BIN",
        y="TARGET",
        color="EMI_BIN",
        color_discrete_sequence=PLOTLY_COLORS,
        labels={"TARGET": "Default rate", "EMI_BIN": "EMI / Income band"},
        title="Default rate by EMI / income ratio band",
    )
    fig_emi.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_emi, width="stretch")

st.markdown(
    "- Default risk increases as both total credit and EMI become larger relative to income.\n"
    "- Applicants with low credit/income and low EMI/income bands are strong candidates for fast-track approval."
)

# --------------------------------------------------------------------
# Section 2: Behavioural red flags
# --------------------------------------------------------------------
st.header("Behavioural risk signals")

col3, col4 = st.columns(2)

# 2A) Default by previous refusal history
with col3:
    rate_refusal = (
        df_filtered.groupby("FLAG_EVER_REFUSED")["TARGET"]
        .mean()
        .reset_index()
    )
    rate_refusal["Label"] = rate_refusal["FLAG_EVER_REFUSED"].map(
        {0: "Never Refused", 1: "Had Refusal"}
    )

    fig_ref = px.bar(
        rate_refusal,
        x="Label",
        y="TARGET",
        color="Label",
        color_discrete_sequence=PLOTLY_COLORS,
        labels={"TARGET": "Default rate", "Label": ""},
        title="Default rate by previous refusal history",
    )
    fig_ref.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_ref, width="stretch")

# 2B) Default by number of previous applications
with col4:
    rate_prev_apps = (
        df_filtered.groupby("PREV_APPS_BIN")["TARGET"]
        .mean()
        .reset_index()
        .sort_values("PREV_APPS_BIN")
    )

    fig_prev = px.bar(
        rate_prev_apps,
        x="PREV_APPS_BIN",
        y="TARGET",
        color="PREV_APPS_BIN",
        color_discrete_sequence=PLOTLY_COLORS,
        labels={"TARGET": "Default rate", "PREV_APPS_BIN": "Previous applications"},
        title="Default rate by number of previous applications",
    )
    fig_prev.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_prev, width="stretch")

st.markdown(
    "- A history of past refusals and many previous applications are clear red flags for future default.\n"
    "- Customers with past refusals or 10+ prior applications should be routed to manual review or stricter policy."
)

# --------------------------------------------------------------------
# Section 3: External score & profile quality
# --------------------------------------------------------------------
st.header("External score and profile quality")

col5, col6 = st.columns(2)

# 3A) Default by external score quartile
with col5:
    rate_ext = (
        df_filtered.groupby("EXT2_Q")["TARGET"]
        .mean()
        .reset_index()
        .sort_values("EXT2_Q")
    )

    fig_ext = px.bar(
        rate_ext,
        x="EXT2_Q",
        y="TARGET",
        color="EXT2_Q",
        color_discrete_sequence=PLOTLY_COLORS,
        labels={"TARGET": "Default rate", "EXT2_Q": "External score quartile"},
        title="Default rate by external risk score (EXT_SOURCE_2)",
    )
    fig_ext.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_ext, width="stretch")

# 3B) Default by education x external score (heatmap-style bar)
with col6:
    df_edu = df_filtered.dropna(subset=["NAME_EDUCATION_TYPE", "EXT2_Q"])
    # restrict to main education groups to keep chart readable
    top_edu = (
        df_edu["NAME_EDUCATION_TYPE"]
        .value_counts()
        .head(5)
        .index
    )
    df_edu = df_edu[df_edu["NAME_EDUCATION_TYPE"].isin(top_edu)]

    rate_edu_ext = (
        df_edu.groupby(["NAME_EDUCATION_TYPE", "EXT2_Q"])["TARGET"]
        .mean()
        .reset_index()
    )

    fig_edu_ext = px.imshow(
        rate_edu_ext.pivot(index="NAME_EDUCATION_TYPE", columns="EXT2_Q", values="TARGET") * 100,
        color_continuous_scale="Blues",
        labels={"color": "Default rate (%)"},
        aspect="auto",
        title="Default rate (%) by education level × external score",
    )
    st.plotly_chart(fig_edu_ext, width="stretch")

st.markdown(
    "- Lower external score quartiles show much higher default rates, even within the same education band.\n"
    "- Combining education level with external score helps separate very safe academic/high-score profiles from riskier low-score, low-education segments."
)

# --------------------------------------------------------------------
# Section 4: Combined risk score
# --------------------------------------------------------------------
st.header("Combined risk score bands")

rate_risk = (
    df_filtered.groupby("RISK_SCORE")["TARGET"]
    .mean()
    .reset_index()
    .sort_values("RISK_SCORE")
)

fig_risk = px.bar(
    rate_risk,
    x="RISK_SCORE",
    y="TARGET",
    color="RISK_SCORE",
    color_discrete_sequence=PLOTLY_COLORS,
    labels={"TARGET": "Default rate", "RISK_SCORE": "Risk score"},
    title="Default rate by combined risk score",
)
fig_risk.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_risk, width="stretch")

st.markdown(
    "- Lower risk scores (e.g., 0–2) correspond to safer borrowers with modest stress and clean behaviour.\n"
    "- Mid scores (3–5) are watchlist segments, while high scores (6+) represent red-flag customers with stacked financial, behavioural, and external risk."
)
