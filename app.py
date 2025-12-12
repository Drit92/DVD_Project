import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
pio.templates.default = "plotly_white"
PLOTLY_COLORS = ["#004c6d", "#00a1c6", "#f29f05", "#e03b8b"]

st.set_page_config(page_title="Loan Applicant Risk Insights", layout="wide")
st.title("📊 Loan Applicant Risk Insights Dashboard")
st.markdown(
    "Visual risk signals to help the credit team identify **safe segments**, "
    "**watchlist profiles**, and **red-flag borrowers**."
)

# ----------------------------------------------------
# DATA LOADING
# ----------------------------------------------------
@st.cache_data(show_spinner=True)
def load_clean_data(sample_n: int | None = 100_000) -> pd.DataFrame:
    """
    Load pre-engineered analytical dataset created in Colab:
    clean_loan_risk.csv.gz at repo root.

    Optionally sample rows to keep rendering light.
    """
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
# FILTERS
# ----------------------------------------------------
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



# ----------------------------------------------------
# HELPER: light sampling for charts
# ----------------------------------------------------
def sample_for_plot(df_in: pd.DataFrame, max_points: int = 80_000) -> pd.DataFrame:
    if len(df_in) > max_points:
        return df_in.sample(max_points, random_state=42)
    return df_in

# ====================================================
# 1. Portfolio overview
# ====================================================
st.header("1. Portfolio overview")

col1, col2 = st.columns(2)

# (1) TARGET distribution
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

# (2) Top-15 missing columns (from precomputed in Colab; here we approximate by non-null ratio if present)
with col2:
    # If original missing info not in clean file, approximate from available columns
    na_series = df_filtered.isnull().mean().sort_values(ascending=False).head(15)
    if len(na_series) > 0:
        miss_df = na_series.reset_index()
        miss_df.columns = ["column", "missing_pct"]
        fig_miss = px.bar(
            miss_df,
            y="column",
            x="missing_pct",
            orientation="h",
            color_discrete_sequence=[PLOTLY_COLORS[1]],
            labels={"missing_pct": "Missing %", "column": ""},
            title="Approx. top-15 missing columns (clean dataset)",
        )
        fig_miss.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig_miss, width="stretch")
    else:
        st.info("No missingness pattern visible in clean dataset.")

st.markdown(
    "- The portfolio is highly imbalanced with a small share of defaulters.\n"
    "- Most missingness is in low-value building/real-estate attributes that were dropped during cleaning."
)

# ====================================================
# 2. Profile-based segment risk
# ====================================================
st.header("2. Customer profile segment risk")

col3, col4 = st.columns(2)

# (3) Default by education
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

# (4) Default by income type
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

# (5) Default by family status
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

# (6) Default by housing type
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
        st.info("Housing type column not found in clean dataset.")

st.markdown(
    "- Lower education, unstable or informal income, and renting/parental housing are all associated with higher default risk.\n"
    "- Stable profiles (higher education, state servants, married, home-owners) form safer applicant segments."
)

# ====================================================
# 3. Financial stress & affordability
# ====================================================
st.header("3. Financial stress & affordability")

col7, col8 = st.columns(2)

# (7) Default by credit/income bands
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

# (8) Default by EMI/income bands
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

# (9) Default by income-per-person bands
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

# (10) Defaulters-only distribution of CREDIT_INCOME_RATIO (cleaned)
st.subheader("Defaulters – credit/income ratio distribution")
if "CREDIT_INCOME_RATIO" in df_filtered.columns:
    df_def = df_filtered[df_filtered["TARGET"] == 1].copy()
    df_def = df_def[df_def["CREDIT_INCOME_RATIO"].between(0, 10)]  # cleaned band
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

st.markdown(
    "- Default risk increases sharply when **total credit exceeds ~3× income** and **EMI exceeds ~20% of income**.\n"
    "- Low income per person and mid-to-high credit burden are where most defaulters cluster."
)

# ====================================================
# 4. Behavioural risk from previous loans
# ====================================================
st.header("4. Behavioural risk from previous loans")

col9, col10 = st.columns(2)

# (11) Default by previous refusal history
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

# (12) Default by number of previous applications
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

# (13) Default by previous main contract status
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
    "- Past refusals and heavy prior application activity both signal higher default risk.\n"
    "- Previous statuses rank roughly as: **Refused > Canceled/Unused > Approved/No history** in terms of risk."
)

# ====================================================
# 5. External scores
# ====================================================
st.header("5. External credit scores")

col11, col12 = st.columns(2)

# (14) Default by EXT_SOURCE_2 bands / quartiles
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

# (15) EXT_SOURCE_2 distribution by default vs non-default
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

st.markdown(
    "- Lower external score quartiles show **2–3× higher default** than top quartiles.\n"
    "- Defaulters are concentrated at the low end of EXT_SOURCE_2, confirming the power of external scores."
)

# ====================================================
# 6. Combined risk score
# ====================================================
st.header("6. Combined risk score")

# (16) Default rate by RISK_SCORE
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
        "- **Low scores (0–2)**: safest borrowers with low stress, clean behaviour, and strong external scores.\n"
        "- **Mid scores (3–5)**: watchlist customers needing tighter limits and closer review.\n"
        "- **High scores (6+)**: red-flag group combining high financial stress, negative behaviour, and weak scores."
    )
else:
    st.info("RISK_SCORE not found in clean dataset.")
