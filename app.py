import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
pio.templates.default = "plotly_white"
PLOTLY_COLORS = ["#004c6d", "#00a1c6", "#f29f05", "#e03b8b"]

st.set_page_config(
    page_title="Loan Applicant Risk Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for background, nav chips, section cards
st.markdown(
    """
    <style>
    /* Global page background */
    .stApp {
        background: #f5f7fb;
    }

    /* Section card look */
    .section-card {
        background-color: #ffffff;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
        border-radius: 0.6rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        border: 1px solid #e0e4f0;
    }

    /* Top nav chips */
    .nav-chip {
        display: inline-block;
        padding: 0.45rem 0.9rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        border-radius: 999px;
        border: 1px solid #cbd5f5;
        background: #eef2ff;
        color: #1e293b;
        font-size: 0.85rem;
        font-weight: 500;
        text-decoration: none;
    }

    .nav-chip:hover {
        border-color: #6366f1;
        color: #111827;
        background: #e0e7ff;
    }

    .nav-chip span {
        opacity: 0.85;
    }

    /* Titles */
    h1, h2, h3 {
        color: #0f172a;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top title + navigation chips
st.markdown(
    """
    <div class="section-card" style="margin-bottom: 0.8rem;">
      <h1 style="margin-bottom: 0.3rem;">📊 Loan Applicant Risk Insights Dashboard</h1>
      <p style="color:#4b5563; font-size:0.94rem; margin-bottom:0.6rem;">
        Conclusive, decision-focused risk views for urban loan applicants.
      </p>
      <div>
        <span class="nav-chip"><span>Overview</span></span>
        <span class="nav-chip"><span>Customer profile segment risk</span></span>
        <span class="nav-chip"><span>Financial stress & affordability</span></span>
        <span class="nav-chip"><span>Behavioural risk from previous loans</span></span>
        <span class="nav-chip"><span>External credit scores</span></span>
        <span class="nav-chip"><span>Combined risk score & radar</span></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# DATA LOADING
# ----------------------------------------------------
@st.cache_data(show_spinner=True)
def load_clean_data(sample_n: int | None = 100_000) -> pd.DataFrame:
    df = pd.read_csv("clean_loan_risk.csv.gz", compression="gzip")
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)
    return df

df = load_clean_data()

required_cols = {"TARGET", "AGE_YEARS", "INCOME_PER_PERSON"}
if not required_cols.issubset(df.columns):
    st.error(f"clean_loan_risk.csv.gz is missing required columns: {required_cols - set(df.columns)}")
    st.stop()

# ----------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------
st.sidebar.title("Filters")

income_min = float(df["INCOME_PER_PERSON"].min())
income_max = float(df["INCOME_PER_PERSON"].max())
income_slider = st.sidebar.slider(
    "Income per person range",
    float(income_min),
    float(income_max),
    (float(income_min), float(income_max)),
    step=5000.0,
)

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



def sample_for_plot(df_in: pd.DataFrame, max_points: int = 80_000) -> pd.DataFrame:
    if len(df_in) > max_points:
        return df_in.sample(max_points, random_state=42)
    return df_in

# ====================================================
# 1. PORTFOLIO OVERVIEW
# ====================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("1. Portfolio overview")

col1, col2 = st.columns(2)

with col1:
    target_counts = (
        df_filtered["TARGET"]
        .value_counts(normalize=True)
        .rename_axis("TARGET")
        .reset_index(name="share")
        .sort_values("TARGET")
    )
    target_counts["Label"] = target_counts["TARGET"].map({0: "Non-Defaulter (0)", 1: "Defaulter (1)"})

    fig_tgt = px.bar(
        target_counts,
        x="Label",
        y="share",
        color="Label",
        color_discrete_sequence=PLOTLY_COLORS,
        labels={"share": "Share of portfolio"},
        title="Overall default vs non-default share",
    )
    fig_tgt.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_tgt, width="stretch")

with col2:
    st.markdown(
        "- Portfolio is **highly imbalanced**, with a small share of defaulters.\n"
        "- Remaining visuals focus on **where risk spikes**, not raw EDA."
    )

st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# 2. CUSTOMER PROFILE SEGMENT RISK
# ====================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("2. Customer profile segment risk")

col3, col4 = st.columns(2)

# Education
with col3:
    if "NAME_EDUCATION_TYPE" in df_filtered.columns:
        edu_rate = (
            df_filtered.groupby("NAME_EDUCATION_TYPE")["TARGET"]
            .mean()
            .reset_index()
            .sort_values("TARGET", ascending=False)
        )
        fig_edu = px.bar(
            edu_rate,
            x="NAME_EDUCATION_TYPE",
            y="TARGET",
            color="NAME_EDUCATION_TYPE",
            color_discrete_sequence=PLOTLY_COLORS,
            labels={"TARGET": "Default rate", "NAME_EDUCATION_TYPE": "Education"},
            title="Default rate by education level",
        )
        fig_edu.update_yaxes(tickformat=".0%")
        fig_edu.update_xaxes(tickangle=30)
        st.plotly_chart(fig_edu, width="stretch")
    else:
        st.info("Education column not found in clean dataset.")

# Income type
with col4:
    if "NAME_INCOME_TYPE" in df_filtered.columns:
        inc_rate = (
            df_filtered.groupby("NAME_INCOME_TYPE")["TARGET"]
            .mean()
            .reset_index()
            .sort_values("TARGET", ascending=False)
        )
        fig_inc = px.bar(
            inc_rate,
            x="NAME_INCOME_TYPE",
            y="TARGET",
            color="NAME_INCOME_TYPE",
            color_discrete_sequence=PLOTLY_COLORS,
            labels={"TARGET": "Default rate", "NAME_INCOME_TYPE": "Income type"},
            title="Default rate by income type",
        )
        fig_inc.update_yaxes(tickformat=".0%")
        fig_inc.update_xaxes(tickangle=30)
        st.plotly_chart(fig_inc, width="stretch")
    else:
        st.info("Income type column not found in clean dataset.")

col5, col6 = st.columns(2)

# Family status
with col5:
    if "NAME_FAMILY_STATUS" in df_filtered.columns:
        fam_rate = (
            df_filtered.groupby("NAME_FAMILY_STATUS")["TARGET"]
            .mean()
            .reset_index()
            .sort_values("TARGET", ascending=False)
        )
        fig_fam = px.bar(
            fam_rate,
            x="NAME_FAMILY_STATUS",
            y="TARGET",
            color="NAME_FAMILY_STATUS",
            color_discrete_sequence=PLOTLY_COLORS,
            labels={"TARGET": "Default rate", "NAME_FAMILY_STATUS": "Family status"},
            title="Default rate by family status",
        )
        fig_fam.update_yaxes(tickformat=".0%")
        fig_fam.update_xaxes(tickangle=30)
        st.plotly_chart(fig_fam, width="stretch")
    else:
        st.info("Family status column not found in clean dataset.")

# Housing type
with col6:
    if "NAME_HOUSING_TYPE" in df_filtered.columns:
        house_rate = (
            df_filtered.groupby("NAME_HOUSING_TYPE")["TARGET"]
            .mean()
            .reset_index()
            .sort_values("TARGET", ascending=False)
        )
        fig_house = px.bar(
            house_rate,
            x="NAME_HOUSING_TYPE",
            y="TARGET",
            color="NAME_HOUSING_TYPE",
            color_discrete_sequence=PLOTLY_COLORS,
            labels={"TARGET": "Default rate", "NAME_HOUSING_TYPE": "Housing type"},
            title="Default rate by housing type",
        )
        fig_house.update_yaxes(tickformat=".0%")
        fig_house.update_xaxes(tickangle=30)
        st.plotly_chart(fig_house, width="stretch")
    else:
        st.info("NAME_HOUSING_TYPE not found in clean dataset.")

st.markdown(
    "- Lower education, informal or unstable income, and renting/parental housing **consistently show higher default rates**.\n"
    "- Higher education, state servants, married applicants, and home owners form **safer demographic segments**."
)

st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# 3. FINANCIAL STRESS & AFFORDABILITY
# ====================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("3. Financial stress & affordability")

col7, col8 = st.columns(2)

# Credit/income bands
with col7:
    if "CREDIT_BIN" in df_filtered.columns:
        credit_rate = (
            df_filtered.groupby("CREDIT_BIN")["TARGET"]
            .mean()
            .reset_index()
            .sort_values("CREDIT_BIN")
        )
        fig_credit = px.bar(
            credit_rate,
            x="CREDIT_BIN",
            y="TARGET",
            color="CREDIT_BIN",
            color_discrete_sequence=PLOTLY_COLORS,
            labels={"TARGET": "Default rate", "CREDIT_BIN": "Credit / income band"},
            title="Default rate by credit / income ratio",
        )
        fig_credit.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_credit, width="stretch")
    else:
        st.info("CREDIT_BIN not found in clean dataset.")

# EMI/income bands
with col8:
    if "EMI_BIN" in df_filtered.columns:
        emi_rate = (
            df_filtered.groupby("EMI_BIN")["TARGET"]
            .mean()
            .reset_index()
            .sort_values("EMI_BIN")
        )
        fig_emi = px.bar(
            emi_rate,
            x="EMI_BIN",
            y="TARGET",
            color="EMI_BIN",
            color_discrete_sequence=PLOTLY_COLORS,
            labels={"TARGET": "Default rate", "EMI_BIN": "EMI / income band"},
            title="Default rate by EMI / income ratio",
        )
        fig_emi.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_emi, width="stretch")
    else:
        st.info("EMI_BIN not found in clean dataset.")

# Income-per-person bands
st.subheader("Default by income-per-person bands")
if "INCOME_BIN" in df_filtered.columns:
    incp_rate = (
        df_filtered.groupby("INCOME_BIN")["TARGET"]
        .mean()
        .reset_index()
        .sort_values("INCOME_BIN")
    )
    fig_incp = px.bar(
        incp_rate,
        x="INCOME_BIN",
        y="TARGET",
        color="INCOME_BIN",
        color_discrete_sequence=PLOTLY_COLORS,
        labels={"TARGET": "Default rate", "INCOME_BIN": "Income per person band"},
        title="Default rate by income per person",
    )
    fig_incp.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_incp, width="stretch")
else:
    st.info("INCOME_BIN not found in clean dataset.")

# Defaulters-only CREDIT_INCOME_RATIO distribution
st.subheader("Defaulters – credit/income ratio distribution")
if "CREDIT_INCOME_RATIO" in df_filtered.columns:
    df_def = df_filtered[df_filtered["TARGET"] == 1].copy()
    df_def = df_def[df_def["CREDIT_INCOME_RATIO"].between(0, 10)]
    df_def = sample_for_plot(df_def)
    fig_def_ci = px.histogram(
        df_def,
        x="CREDIT_INCOME_RATIO",
        nbins=40,
        color_discrete_sequence=[PLOTLY_COLORS[3]],
        title="Defaulters: credit/income ratio (cleaned)",
    )
    st.plotly_chart(fig_def_ci, width="stretch")
else:
    st.info("CREDIT_INCOME_RATIO not found in clean dataset.")

# Default trend line across stress buckets
st.subheader("Default trend across credit/income stress buckets")
if "CREDIT_BIN" in df_filtered.columns:
    trend_df = (
        df_filtered.groupby("CREDIT_BIN")["TARGET"]
        .mean()
        .reset_index()
        .sort_values("CREDIT_BIN")
    )
    fig_trend = px.line(
        trend_df,
        x="CREDIT_BIN",
        y="TARGET",
        markers=True,
        color_discrete_sequence=["crimson"],
        labels={"TARGET": "Default rate", "CREDIT_BIN": "Credit / income stress group"},
        title="Default trend across financial stress buckets",
    )
    fig_trend.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_trend, width="stretch")
else:
    st.info("CREDIT_BIN not found in clean dataset for trend line.")

st.markdown(
    "- **Credit > ~3× income** and **EMI > ~20% of income** are clear stress thresholds where default rates jump.\n"
    "- Defaulters cluster in mid-to-high stress bands and **low income per person** segments."
)

st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# 4. BEHAVIOURAL RISK FROM PREVIOUS LOANS
# ====================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("4. Behavioural risk from previous loans")

col9, col10 = st.columns(2)

# Refusal history
with col9:
    if "FLAG_EVER_REFUSED" in df_filtered.columns:
        ref_rate = (
            df_filtered.groupby("FLAG_EVER_REFUSED")["TARGET"]
            .mean()
            .reset_index()
        )
        ref_rate["Label"] = ref_rate["FLAG_EVER_REFUSED"].map(
            {0: "Never refused", 1: "Had refusal"}
        )
        fig_ref = px.bar(
            ref_rate,
            x="Label",
            y="TARGET",
            color="Label",
            color_discrete_sequence=PLOTLY_COLORS,
            labels={"TARGET": "Default rate", "Label": ""},
            title="Default rate by refusal history",
        )
        fig_ref.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ref, width="stretch")
    else:
        st.info("FLAG_EVER_REFUSED not found in clean dataset.")

# Number of previous applications
with col10:
    if "PREV_APPS_BIN" in df_filtered.columns:
        apps_rate = (
            df_filtered.groupby("PREV_APPS_BIN")["TARGET"]
            .mean()
            .reset_index()
            .sort_values("PREV_APPS_BIN")
        )
        fig_apps = px.bar(
            apps_rate,
            x="PREV_APPS_BIN",
            y="TARGET",
            color="PREV_APPS_BIN",
            color_discrete_sequence=PLOTLY_COLORS,
            labels={"TARGET": "Default rate", "PREV_APPS_BIN": "Previous applications"},
            title="Default rate by number of previous applications",
        )
        fig_apps.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_apps, width="stretch")
    else:
        st.info("PREV_APPS_BIN not found in clean dataset.")

# Previous main status
st.subheader("Default by previous main contract status")
if "PREV_MAIN_STATUS" in df_filtered.columns:
    status_rate = (
        df_filtered.groupby("PREV_MAIN_STATUS")["TARGET"]
        .mean()
        .reset_index()
        .sort_values("TARGET", ascending=False)
    )
    fig_status = px.bar(
        status_rate,
        x="PREV_MAIN_STATUS",
        y="TARGET",
        color="PREV_MAIN_STATUS",
        color_discrete_sequence=PLOTLY_COLORS,
        labels={"TARGET": "Default rate", "PREV_MAIN_STATUS": "Previous status"},
        title="Default rate by previous contract status",
    )
    fig_status.update_yaxes(tickformat=".0%")
    fig_status.update_xaxes(tickangle=25)
    st.plotly_chart(fig_status, width="stretch")
else:
    st.info("PREV_MAIN_STATUS not found in clean dataset.")

st.markdown(
    "- Past refusals and **heavy prior application activity (5+ / 10+)** materially raise default risk.\n"
    "- Previous outcomes rank roughly: **Refused > Canceled/Unused > Approved/No history**."
)

st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# 5. EXTERNAL CREDIT SCORES (incl. bubble)
# ====================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("5. External credit scores")

col11, col12 = st.columns(2)

# Default by EXT2_Q
with col11:
    if "EXT2_Q" in df_filtered.columns:
        ext_rate = (
            df_filtered.groupby("EXT2_Q")["TARGET"]
            .mean()
            .reset_index()
            .sort_values("EXT2_Q")
        )
        fig_ext = px.bar(
            ext_rate,
            x="EXT2_Q",
            y="TARGET",
            color="EXT2_Q",
            color_discrete_sequence=PLOTLY_COLORS,
            labels={"TARGET": "Default rate", "EXT2_Q": "EXT_SOURCE_2 quartile"},
            title="Default rate by external score quartile (EXT_SOURCE_2)",
        )
        fig_ext.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ext, width="stretch")
    else:
        st.info("EXT2_Q not found in clean dataset.")

# EXT_SOURCE_2 distribution by default
with col12:
    if "EXT_SOURCE_2" in df_filtered.columns:
        df_ext = df_filtered[["EXT_SOURCE_2", "TARGET"]].dropna()
        df_ext = sample_for_plot(df_ext)
        fig_ext_dist = px.histogram(
            df_ext,
            x="EXT_SOURCE_2",
            color="TARGET",
            nbins=50,
            barmode="overlay",
            color_discrete_sequence=[PLOTLY_COLORS[0], PLOTLY_COLORS[3]],
            title="EXT_SOURCE_2 distribution by default status",
            labels={"TARGET": "Default flag"},
        )
        st.plotly_chart(fig_ext_dist, width="stretch")
    else:
        st.info("EXT_SOURCE_2 not found in clean dataset.")

# Bubble: EXT2_Q × FLAG_EVER_REFUSED
st.subheader("Bubble: external score × behavioural risk")
if {"EXT2_Q", "FLAG_EVER_REFUSED", "TARGET"}.issubset(df_filtered.columns):
    bubble_df = df_filtered[["EXT2_Q", "FLAG_EVER_REFUSED", "TARGET"]].dropna()
    bubble_group = (
        bubble_df
        .groupby(["EXT2_Q", "FLAG_EVER_REFUSED"])["TARGET"]
        .mean()
        .reset_index()
        .rename(columns={"TARGET": "DefaultRate"})
    )
    bubble_count = (
        bubble_df
        .groupby(["EXT2_Q", "FLAG_EVER_REFUSED"])["TARGET"]
        .count()
        .reset_index()
        .rename(columns={"TARGET": "Count"})
    )
    bubble_final = bubble_group.merge(
        bubble_count,
        on=["EXT2_Q", "FLAG_EVER_REFUSED"],
        how="inner",
    )
    bubble_final["RefusalLabel"] = bubble_final["FLAG_EVER_REFUSED"].map(
        {0: "No previous refusal", 1: "Had previous refusal"}
    )

    fig_bubble = px.scatter(
        bubble_final,
        x="EXT2_Q",
        y="DefaultRate",
        size="Count",
        color="RefusalLabel",
        color_discrete_sequence=[PLOTLY_COLORS[0], PLOTLY_COLORS[3]],
        size_max=60,
        labels={
            "EXT2_Q": "External score quartile (EXT_SOURCE_2)",
            "DefaultRate": "Default rate",
            "RefusalLabel": "Previous refusal",
        },
        title="External score × behavioural risk (bubble size = # customers)",
    )
    fig_bubble.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_bubble, width="stretch")
else:
    st.info("EXT2_Q or FLAG_EVER_REFUSED not found in clean dataset for bubble chart.")

st.markdown(
    "- Lower external score quartiles show **2–3× higher default** than top quartiles.\n"
    "- Within each score band, **past refusals further amplify risk**, as seen by larger, higher bubbles."
)

st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# 6. COMBINED RISK SCORE & RADAR
# ====================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("6. Combined risk score & overall risk profile")

# Default rate by RISK_SCORE
if "RISK_SCORE" in df_filtered.columns:
    risk_rate = (
        df_filtered.groupby("RISK_SCORE")["TARGET"]
        .mean()
        .reset_index()
        .sort_values("RISK_SCORE")
    )
    fig_risk = px.bar(
        risk_rate,
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
        "- **Low scores (0–2)**: safest borrowers – low stress, clean behaviour, strong external scores.\n"
        "- **Mid scores (3–5)**: watchlist segment – tighter limits and extra checks.\n"
        "- **High scores (6+)**: red-flag group combining high financial stress and negative history."
    )
else:
    st.info("RISK_SCORE not found in clean dataset.")



st.markdown('</div>', unsafe_allow_html=True)
