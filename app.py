import os
import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import streamlit as st




# ============================================
# STREAMLIT CONFIG
# ============================================

st.set_page_config(
    page_title="📊 Loan Applicant Risk Insights Dashboard",
    layout="wide",
)

st.title("📊 Loan Applicant Risk Insights Dashboard")
st.markdown("---")


# ============================================
# LOAD AGGREGATES FROM ZIP IN REPO ROOT
# ============================================

ZIP_PATH = "loan_risk_aggregates.zip"  # must sit next to app.py


@st.cache_data(show_spinner="🔄 Loading pre-aggregated loan risk data...")
def load_aggregates_from_disk(zip_path: str) -> dict:
    if not os.path.exists(zip_path):
        st.error(f"ZIP file '{zip_path}' not found in app root. Commit it to the repo.")
        st.stop()

    agg_dict = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("agg_") and name.endswith(".csv"):
                with zf.open(name) as f:
                    df = pd.read_csv(f)
                    agg_dict[name] = df
    return agg_dict


aggs = load_aggregates_from_disk(ZIP_PATH)


def get_agg(name: str, required: bool = True) -> pd.DataFrame:
    """Helper: fetch agg_<name>.csv or stop if required and missing."""
    fname = f"agg_{name}.csv"
    if fname not in aggs:
        if required:
            st.error(f"Missing {fname} in ZIP. Regenerate aggregates in Colab.")
            st.stop()
        return pd.DataFrame()
    return aggs[fname]


# ============================================
# BASIC METRICS
# ============================================

overview = get_agg("overview_metrics")
overview_map = dict(zip(overview["metric"], overview["value"]))

total_applicants = int(overview_map.get("total_applicants", 0))
total_defaulters = int(overview_map.get("total_defaulters", 0))
total_good = int(overview_map.get("total_good_borrowers", 0))
default_rate_overall = float(overview_map.get("default_rate_overall", 0.0))

target_dist = get_agg("target_distribution").copy()
target_dist["TARGET"] = target_dist["TARGET"].astype(int)


# ============================================
# 1. PORTFOLIO OVERVIEW – WHO DEFAULTS?
# ============================================

st.header("📈 Portfolio Overview – Who Defaults?")

col1, col2 = st.columns([1, 2])

with col1:
    labels = ["Good borrower (0)", "Defaulter (1)"]
    shares = [
        float(target_dist.loc[target_dist["TARGET"] == 0, "Share"].values[0]),
        float(target_dist.loc[target_dist["TARGET"] == 1, "Share"].values[0]),
    ]
    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=shares,
                hole=0.6,
                marker_colors=["#28a745", "#dc3545"],
                textinfo="label+percent",
                textposition="outside",
                showlegend=False,
            )
        ]
    )
    fig_donut.update_layout(
        title=dict(text="Share of Good Borrowers vs Defaulters", y=0.96),
        height=260,
        margin=dict(t=80, b=10, l=10, r=10),
        transition_duration=0,
    )
    st.plotly_chart(fig_donut, width="stretch")

with col2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Applicantions", f"{total_applicants:,}")
    c2.metric(
        "Total Defaulters",
        f"{total_defaulters:,}",
        f"{default_rate_overall:.1%}",
    )
    c3.metric("Total Good Borrowers", f"{total_good:,}")

st.markdown(
    """
    **Overview story:** Most customers repay on time; only a small share (about 8%) default, so the dataset is highly imbalanced.
    """
)
st.markdown("---")


# ============================================
# 1A. BORROWER PROFILES – GENDER & AGE
# ============================================

st.header("🧍 Borrower Profiles – Who Applies?")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Gender Mix")
    gender_mix = get_agg("gender_mix", required=False)
    if gender_mix.empty:
        st.info("Gender mix aggregate not available.")
    else:
        label_map = {"M": "Male", "F": "Female"}
        gender_mix["Label"] = gender_mix["CODE_GENDER"].map(label_map).fillna(
            gender_mix["CODE_GENDER"]
        )

        fig_gender_pie = go.Figure(
            data=[
                go.Pie(
                    labels=gender_mix["Label"],
                    values=gender_mix["Share"],
                    hole=0.4,
                    marker_colors=["#4C72B0", "#DD8452"],
                    textinfo="label+percent",
                    textposition="outside",
                    showlegend=False,
                )
            ]
        )
        fig_gender_pie.update_layout(
            title="Share of Applicants by Gender",
            margin=dict(t=40, l=10, r=10, b=10),
            height=260,
            transition_duration=0,
        )
        st.plotly_chart(fig_gender_pie, width="stretch")

with col2:
    st.subheader("Applicant Age Distribution")
    # Direct access - works since ZIP loads fine
    age_hist = aggs.get("agg_age_distribution.csv", pd.DataFrame())
    
    if age_hist.empty:
        st.info("age_distribution.csv not available in aggregates ZIP.")
    else:
        fig_age_all = px.bar(
            age_hist,
            x="Age_Years",
            y="Count",
            labels={
                "Age_Years": "Age (years)",
                "Count": "Number of Applicants",
            },
            title="Distribution of Loan Applicant Age",
            color_discrete_sequence=["#4C72B0"],
            text=age_hist["Count"].astype(int),
        )
        
        fig_age_all.update_traces(
            textposition="outside",
            textfont=dict(size=10),
            marker_line_width=1.2, 
            marker_line_color="black"
        )
        
        fig_age_all.add_scatter(
            x=age_hist["Age_Years"],
            y=age_hist["Count"],
            mode="lines",
            line=dict(color="crimson", width=2.5, shape="spline"),
            showlegend=False,
        )

        fig_age_all.update_layout(
            height=260, 
            transition_duration=0,
            showlegend=False,
            yaxis_title="Number of Applicants",
            xaxis_title="Age (years)"
        )
        st.plotly_chart(fig_age_all, width="stretch")


st.markdown(
    """
    **Insights:**
    - Male applicants show a **higher default rate (~10%)** than female applicants (~7%), even though both groups are large.
    - Most applicants fall in the **working‑age band (late 20s to early 50s)**; very young and very old borrowers are a small fraction.
    """
)
st.markdown("---")


# ============================================
# 2. DEMOGRAPHIC SEGMENTS – RISKIEST LEFT
# ============================================

st.header("👥 Demographic Segments – Who Is Riskier?")

fig_dem = make_subplots(
    rows=2,
    cols=2,
    horizontal_spacing=0.16,
    vertical_spacing=0.32,
    subplot_titles=(
        "Education level",
        "Type of income",
        "Family status",
        "Housing type",
    ),
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}]],
)

demo_mapping = {
    "NAME_EDUCATION_TYPE": (1, 1),
    "NAME_INCOME_TYPE": (1, 2),
    "NAME_FAMILY_STATUS": (2, 1),
    "NAME_HOUSING_TYPE": (2, 2),
}

for col, (r, c) in demo_mapping.items():
    df_demo = get_agg(f"demo_{col}", required=False)
    if df_demo.empty:
        continue

    df_demo = df_demo.copy()

    # Drop "Unknown" only for family status
    if col == "NAME_FAMILY_STATUS":
        df_demo = df_demo[df_demo[col] != "Unknown"]

    df_demo["DefaultRate"] = (df_demo["DefaultRate"] * 100).round(2)
    df_demo = df_demo.sort_values("DefaultRate", ascending=False)

    fig_tmp = px.bar(
        df_demo,
        x=col,
        y="DefaultRate",
        color="DefaultRate",
        color_continuous_scale="Blues",
        labels={
            col: col.replace("_", " ").title(),
            "DefaultRate": "Default rate (%)",
        },
    )
    fig_tmp.update_layout(
        margin=dict(t=30, b=50, l=40, r=40),
        xaxis=dict(tickangle=-35),
        coloraxis_showscale=False,
    )
    # ✅ SHOW VALUES ON BARS
    fig_tmp.update_traces(
        text=df_demo["DefaultRate"], 
        textposition="outside", 
        textfont=dict(size=11, color="white")
    )

    for trace in fig_tmp.data:
        fig_dem.add_trace(trace, row=r, col=c)

fig_dem.update_layout(
    height=720,
    showlegend=False,
    title="Default Rate by Demographic Group (riskiest categories on the left)",
    margin=dict(l=60, r=60, t=90, b=60),
    transition_duration=0,
)

st.plotly_chart(fig_dem, width="stretch")

st.markdown(
    """
    **Insights:**
    - Higher education, government income, being married, and owning a home are all linked to **lower default risk**.
    - Lower education, unstable income (e.g., unemployment, maternity leave) and renting or living with parents are associated with **higher default rates**.
    """
)
st.markdown("---")


# ============================================
# 2A. AGE PROFILE OF DEFAULTERS
# ============================================

st.header("📅 Age Profile of Defaulters")

age_hist = get_agg("age_defaulters_hist", required=False)
if age_hist.empty:
    st.info("Age histogram aggregate not available.")
else:
    age_hist = age_hist.copy()
    age_hist["bin_mid"] = (age_hist["bin_left"] + age_hist["bin_right"]) / 2

    fig_age_def = px.bar(
        age_hist,
        x="bin_mid",
        y="count",
        labels={"bin_mid": "Age (years)", "count": "Count"},
        title="Age Distribution of Defaulters (Binned)",
        color_discrete_sequence=["#E57373"],
    )
    fig_age_def.update_traces(marker_line_width=1.2, marker_line_color="black")

    # Smooth translucent trendline
    fig_age_def.add_scatter(
        x=age_hist["bin_mid"],
        y=age_hist["count"],
        mode="lines",
        name="Smoothed trend",
        line=dict(color="rgba(139,0,0,0.7)", width=2, shape="spline"),
    )

    fig_age_def.update_layout(height=400, transition_duration=0, showlegend=False)
    st.plotly_chart(fig_age_def, width="stretch")


st.markdown(
    """
    **Insight:** Most defaulters are between **about 28 and 45 years old**; default risk tapers off for older customers who tend to have more stable finances.
    """
)
st.markdown("---")


# ============================================
# 3. FINANCIAL STRESS – BANDS + TREND
# ============================================

st.header("💰 Financial Stress – How Much Debt Is Too Much?")

# ---- 3B. DEFAULTERS-ONLY BAND SHARES ----
st.subheader("📊 Where Are Defaulters Concentrated by Stress Band?")

credit_band_def = get_agg("defaulters_credit_band_share", required=False)
emi_band_def = get_agg("defaulters_emi_band_share", required=False)
incomepp_band_def = get_agg("defaulters_incomepp_band_share", required=False)

# First row: credit bands + EMI bands + income bands (narrow)
col_fs1, col_fs2, col_fs3 = st.columns([1.1, 1.1, 0.9])

with col_fs1:
    if not credit_band_def.empty:
        df_cb = credit_band_def.copy()
        df_cb["PercentDefaulters"] = df_cb["PercentDefaulters"].round(1)
        order_cb = ["0-1", "1-2", "2-3", "3-5", "5+"]
        df_cb["CREDIT_BIN"] = pd.Categorical(df_cb["CREDIT_BIN"], order_cb, ordered=True)
        df_cb = df_cb.sort_values("CREDIT_BIN")

        fig_cb = px.bar(
            df_cb,
            x="CREDIT_BIN",
            y="PercentDefaulters",
            title="Defaulters — Credit / Income Ratio Bands",
            labels={
                "CREDIT_BIN": "Credit / Income ratio band",
                "PercentDefaulters": "% of defaulters",
            },
            color_discrete_sequence=["#f66d6d"],
        )
        fig_cb.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_cb, width="stretch")

with col_fs2:
    if not emi_band_def.empty:
        df_eb = emi_band_def.copy()
        df_eb["PercentDefaulters"] = df_eb["PercentDefaulters"].round(1)
        order_eb = ["0-10%", "10-20%", "20-30%", "30-50%", "50%+"]
        df_eb["ANNUITY_BIN"] = pd.Categorical(df_eb["ANNUITY_BIN"], order_eb, ordered=True)
        df_eb = df_eb.sort_values("ANNUITY_BIN")

        fig_eb = px.bar(
            df_eb,
            x="ANNUITY_BIN",
            y="PercentDefaulters",
            title="Defaulters — EMI / Income Ratio Bands",
            labels={
                "ANNUITY_BIN": "EMI / Income ratio band",
                "PercentDefaulters": "% of defaulters",
            },
            color_discrete_sequence=["#f66d6d"],
        )
        fig_eb.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_eb, width="stretch")

with col_fs3:
    if not incomepp_band_def.empty:
        df_ip = incomepp_band_def.copy()
        df_ip["PercentDefaulters"] = df_ip["PercentDefaulters"].round(1)
        order_ip = ["<50k", "50-100k", "100-150k", "150-300k", "300k+"]
        df_ip["INCOME_PP_BIN"] = pd.Categorical(
            df_ip["INCOME_PP_BIN"], order_ip, ordered=True
        )
        df_ip = df_ip.sort_values("INCOME_PP_BIN")

        fig_ip = px.bar(
            df_ip,
            x="INCOME_PP_BIN",
            y="PercentDefaulters",
            title="Defaulters — Income per Person Bands",
            labels={
                "INCOME_PP_BIN": "Income per person band",
                "PercentDefaulters": "% of defaulters",
            },
            color_discrete_sequence=["#f66d6d"],
        )
        fig_ip.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_ip, width="stretch")

# Second row: full-width insight text, spans entire page under the three charts
st.markdown(
    """
    **Insights:**

    Default risk increases significantly as financial stress increases, but risk peaks at **mid‑level stress** rather than at the extremes.


    - Borrowers with **medium‑high debt load (credit/income 3–5×)** show the highest default probability (≈31% of defaulters).
    - EMI affordability risk is highest for **10–20% EMI borrowers (~47%)**, so moderately leveraged consumers can be riskier than heavily leveraged ones.
    - Per‑person income is a strong protective factor: above **300k per person**, default risk becomes almost negligible.
    """
)


# ---- 3C. LINE TREND ACROSS BUCKETS (FULL WIDTH) ----
st.subheader("📈 Default Trend Across Financial Stress Buckets")

credit_default = get_agg("credit_default").copy()
if credit_default.empty:
    st.info("Credit / income default aggregate not available.")
else:
    valid_bins = ["0–1x", "1–2x", "2–3x", "3–5x", "5x+"] # match Colab labels
    credit_default = credit_default.dropna(subset=["CREDIT_BIN"])
    credit_default = credit_default[credit_default["CREDIT_BIN"].isin(valid_bins)]

    if len(credit_default) < 2:
        st.info("Not enough non-empty credit bins to draw a trend line.")
    else:
        credit_default["DefaultRate"] = credit_default["DefaultRate"].astype(float) * 100
        order_map = {b: i for i, b in enumerate(valid_bins)}
        credit_default["order"] = credit_default["CREDIT_BIN"].map(order_map)
        credit_default = credit_default.sort_values("order")

        fig_trend = px.line(
            credit_default,
            x="CREDIT_BIN",
            y="DefaultRate",
            markers=True,
            title="Default Trend Across Financial Stress Buckets",
            labels={
                "CREDIT_BIN": "Credit / Income Stress Group",
                "DefaultRate": "Default Rate (%)",
            },
            color_discrete_sequence=["#dc3545"],
        )
        fig_trend.update_traces(line_shape="linear", marker=dict(size=8))
        fig_trend.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_trend, width="stretch")

       
        st.markdown("""
            **Insights:**
            - Default risk climbs from **low debt (0-1x income)** to **medium debt (2-3x income)**, hitting **8-9%**.
            - **Surprise**: Very high debt (5x+ income) is actually **safer** than medium debt.
            - **Why?** Moderately stressed borrowers are riskier than both safe and desperate ones.
        """)
st.markdown("---")



# ============================================
# 4. EXTERNAL SCORE + PAST REFUSAL – BUBBLE
# ============================================

st.header("🎯 External Score + Past Refusal – Combined Risk")

ext2_bubble = get_agg("ext2_refusal_bubble", required=False)
if ext2_bubble.empty:
    st.info("EXT2 × refusal aggregate not available.")
else:
    ext2_bubble = ext2_bubble.copy()

    # Map quartiles to numeric positions 1..4 (for x)
    label_to_num = {"Q1 (low)": 1, "Q2": 2, "Q3": 3, "Q4 (high)": 4}
    ext2_bubble["EXT2_Q_NUM"] = ext2_bubble["EXT2_Q"].map(label_to_num)
    ext2_bubble = ext2_bubble.dropna(subset=["EXT2_Q_NUM"])

    # Convert to percent
    ext2_bubble["DefaultRate"] = ext2_bubble["DefaultRate"] * 100

    # Label for legend
    ext2_bubble["REFUSAL_STR"] = ext2_bubble["FLAG_EVER_REFUSED"].map(
        {0: "No refusal", 1: "Had refusal"}
    )

    # Small vertical jitter to avoid perfect overlap when rates are very close
    jitter_map = {"No refusal": -0.3, "Had refusal": 0.3}
    ext2_bubble["DefaultRate_jitter"] = (
        ext2_bubble["DefaultRate"]
        + ext2_bubble["REFUSAL_STR"].map(jitter_map)
    )

    fig_bubble = px.scatter(
        ext2_bubble,
        x="EXT2_Q_NUM",
        y="DefaultRate_jitter",
        size="Count",
        color="REFUSAL_STR",
        size_max=25, # was 60 – much smaller bubbles
        color_discrete_map={"No refusal": "green", "Had refusal": "red"},
        labels={
            "EXT2_Q_NUM": "External score quartile (higher = safer)",
            "DefaultRate_jitter": "Default rate (%)",
            "REFUSAL_STR": "Previous refusal",
        },
        title="External Score vs Past Refusal (Bubble Size = Number of Customers)",
        hover_data={"DefaultRate": ':.2f'}
    )


    quartile_ticks = sorted(ext2_bubble["EXT2_Q_NUM"].unique())
    fig_bubble.update_xaxes(
        tickmode="array",
        tickvals=quartile_ticks,
        ticktext=[f"Q{int(q)}" for q in quartile_ticks],
        range=[0.5, 4.5],
    )
    fig_bubble.update_yaxes(ticksuffix="%")
    fig_bubble.update_layout(transition_duration=0)

    st.plotly_chart(fig_bubble, width="stretch")

    st.markdown(
        """
        **Insights:**
        - For any given external‑score quartile, customers with **past refusals (red)** default more often than those with **clean histories (green)**.
        - Big green bubbles in higher quartiles are **safe, high‑volume customers**; small red bubbles in low quartiles are **concentrated risk pockets**.
        """
    )

st.markdown("---")


# ============================================
# 5. BEHAVIOURAL RED FLAGS
# ============================================

st.header("🚩 Behavioural Red Flags – Past Actions Matter")

col1, col2 = st.columns(2)

with col1:
    refuse_def = get_agg("refuse_default")
    refuse_def = refuse_def.copy()
    refuse_def["DefaultRate"] = refuse_def["DefaultRate"] * 100
    refuse_def = refuse_def.set_index("FLAG_EVER_REFUSED").reindex([0, 1]).reset_index()
    labels_ref = ["No previous refusal", "Had previous refusal"]

    fig_refuse = px.bar(
        x=labels_ref,
        y=refuse_def["DefaultRate"].values,
        title="Default Rate by Refusal History",
        labels={"x": "Refusal history", "y": "Default rate (%)"},
        color_discrete_sequence=["#dc3545"],
    )
    fig_refuse.update_yaxes(ticksuffix="%")
    fig_refuse.update_layout(transition_duration=0)
    st.plotly_chart(fig_refuse, width="stretch")

with col2:
    apps_def = get_agg("prev_apps_default")
    apps_def = apps_def.copy()
    apps_def["DefaultRate"] = apps_def["DefaultRate"] * 100
    app_order = ["0", "1-2", "3-4", "5-9", "10+"]
    apps_def["PREV_APPS_BIN"] = pd.Categorical(
        apps_def["PREV_APPS_BIN"], categories=app_order, ordered=True
    )
    apps_def = apps_def.sort_values("PREV_APPS_BIN")

    fig_apps = px.bar(
        x=apps_def["PREV_APPS_BIN"].astype(str),
        y=apps_def["DefaultRate"].values,
        title="Default Rate by Number of Previous Applications",
        labels={"x": "Number of previous applications", "y": "Default rate (%)"},
        color_discrete_sequence=["#e83e8c"],
    )
    fig_apps.update_yaxes(ticksuffix="%")
    fig_apps.update_layout(transition_duration=0)
    st.plotly_chart(fig_apps, width="stretch")

st.markdown(
    """
    **Insights:**
    - Customers with **any past refusal** have a much higher chance of defaulting than those with a clean record.
    - Default risk stays low up to **4 previous applications**, then rises, especially at **10+ applications**, which signals credit‑shopping stress.
    """
)
st.markdown("---")

# ============================================
# 🔥5.B RISK INTERACTION HEATMAPS 
# ============================================

st.header("🔥 Risk Interactions – Combined Effects")

col1, col2 = st.columns(2)

# Heatmap 1: CREDIT_BIN × FLAG_EVER_REFUSED (Cell 12 from Colab)
credit_refusal = get_agg("credit_refusal_heatmap", required=False)
if not credit_refusal.empty:
    with col1:
        # Pivot for heatmap: rows=CREDIT_BIN, columns=refusal status
        pivot_credit = credit_refusal.set_index("CREDIT_BIN")[["No Refusal", "Had Refusal"]]
        fig_heat_credit = px.imshow(
            pivot_credit,
            color_continuous_scale="Reds",
            title="Credit Stress × Past Refusal<br><sup>Had refusal = 2-3x higher risk in all bands</sup>",
            aspect="auto",
            labels=dict(color="Default Rate (%)")
        )
        fig_heat_credit.update_layout(height=350, margin=dict(t=80, l=20, r=20))
        st.plotly_chart(fig_heat_credit, use_container_width=True)

# Heatmap 2: NAME_EDUCATION_TYPE × EXT2_Q (Cell 13 from Colab)
edu_ext = get_agg("education_ext_heatmap", required=False)
if not edu_ext.empty:
    with col2:
        # Pivot for heatmap: rows=education, columns=EXT quartiles
        pivot_edu = edu_ext.set_index("NAME_EDUCATION_TYPE")[["Q1 (low)", "Q2", "Q3", "Q4 (high)"]]
        fig_heat_edu = px.imshow(
            pivot_edu,
            color_continuous_scale="Blues",
            title="Education × External Score<br><sup>Low education + Q1 = 15%+ default risk</sup>",
            aspect="auto",
            labels=dict(color="Default Rate (%)")
        )
        fig_heat_edu.update_layout(height=350, margin=dict(t=80, l=20, r=20))
        st.plotly_chart(fig_heat_edu, use_container_width=True)

st.markdown("""
**Insights:**
- **High debt (3-5x income) + past rejection = 14-18% will default** (vs 7-9% for clean records)
- **High school education + worst credit score = 15%+ will default** (vs 3-4% for college grads + best scores)
- **Bad + bad = MUCH worse** – reject when multiple red flags appear together
""")
st.markdown("---")




# ============================================
# 6. EXTERNAL CREDIT SCORES – QUARTILES
# ============================================

st.header("⭐ External Credit Scores – Power of Bureau Data")

col1, col2 = st.columns([2, 1])

with col1:
    ext2_def = get_agg("ext2_quartile_default")
    ext2_def = ext2_def.copy()
    ext2_def = ext2_def.dropna(subset=["EXT2_Q"])
    ext2_def["DefaultRate"] = ext2_def["DefaultRate"] * 100

    order_q = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    ext2_def["EXT2_Q"] = pd.Categorical(ext2_def["EXT2_Q"], order_q, ordered=True)
    ext2_def = ext2_def.sort_values("EXT2_Q")

    fig_ext2 = px.bar(
        x=ext2_def["EXT2_Q"].astype(str),
        y=ext2_def["DefaultRate"],
        title="Default Rate by External Score Quartile",
        labels={"x": "External score quartile", "y": "Default rate (%)"},
        color_discrete_sequence=["#6f42c1"],
    )
    fig_ext2.update_yaxes(ticksuffix="%")
    fig_ext2.update_layout(transition_duration=0)
    st.plotly_chart(fig_ext2, width="stretch")

with col2:
    st.markdown(
        """
        **Insights:**
        - **Lowest quartile (Q1)** shows significantly higher default rates than **highest quartile (Q4)**.
        - Default probability decreases steadily as external scores improve, making this a powerful screening tool.
        """
    )

st.markdown("---")


# ============================================
# 7. COMBINED RISK SCORE
# ============================================

st.header("🎯 Combined Risk Score – One Number That Combines All Risk Flags")

st.markdown(
    """
    The Combined Risk Score brings three ideas into a single number:

    - **Financial stress** – loan and EMI vs income.
    - **Behaviour** – past refusals and application count.
    - **External risk** – external / bureau score bucket.

    Higher scores mean **more red flags** and higher default risk.
    """
)

risk_def = get_agg("risk_score_default")
risk_def = risk_def.copy()
risk_def["DefaultRate"] = risk_def["DefaultRate"] * 100
risk_def = risk_def.sort_values("RISK_SCORE")

# Classify 0–9 into bands
def band(score: int) -> str:
    if score <= 3:
        return "Safe (Low risk)"
    elif score <= 6:
        return "Medium risk"
    else:
        return "High risk"

risk_def["RiskBand"] = risk_def["RISK_SCORE"].apply(band)

fig_risk = px.bar(
    risk_def,
    x="RISK_SCORE",
    y="DefaultRate",
    title="Default Rate by Combined Risk Score",
    labels={
        "RISK_SCORE": "Combined risk score (0 = safest, 9 = riskiest)",
        "DefaultRate": "Default rate (%)",
    },
)

# Show all x labels 0–9
fig_risk.update_xaxes(
    tickmode="array",
    tickvals=list(range(10)),
    ticktext=[str(i) for i in range(10)],
)

# Gradient palette: green → yellow → red
fig_risk.update_traces(
    marker=dict(
        color=risk_def["RISK_SCORE"],
        colorscale=[
            [0.0, "rgb(0,160,0)"],
            [0.5, "rgb(255,210,0)"],
            [1.0, "rgb(200,0,0)"],
        ],
        colorbar=dict(
            title="Risk level",
            tickvals=[0, 4, 9],
            ticktext=["Low", "Medium", "High"],
        ),
    ),
    # Custom hover text: band + score + default rate
    hovertemplate=(
        "Risk band: %{customdata[0]}<br>"
        "Score: %{x}<br>"
        "Default rate: %{y:.2f}%<extra></extra>"
    ),
    customdata=risk_def[["RiskBand"]].to_numpy(),
)

fig_risk.update_yaxes(ticksuffix="%")
fig_risk.update_layout(showlegend=False, transition_duration=0)

st.plotly_chart(fig_risk, use_container_width=True)

st.markdown(
    """
    **Insight:** Low scores capture the **safest borrowers**, while high scores group together the **riskiest applicants**, so this single score can drive cut‑offs, pricing bands and watchlists.
    """
)
st.markdown("---")
# ============================================
# 8. INTERACTIVE RADAR CHART – RISK PROFILE COMPARISON
# ============================================

st.header("📈  Risk Profile Comparison – Radar Chart")

radar_means = get_agg("radar_means", required=False)
if radar_means.empty:
    st.info("Radar aggregate not available.")
else:
    radar_means = radar_means.copy()
    radar_means["TARGET"] = radar_means["TARGET"].astype(int)
    
    # Min-max normalize each feature to [0, 1]
    radar_norm = radar_means.set_index("TARGET").T.apply(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9), axis=1
    ).reset_index()
    
    labels = radar_norm["index"].tolist()
    values_0 = radar_norm[0].tolist() if 0 in radar_norm.columns else [0] * len(labels)
    values_1 = radar_norm[1].tolist() if 1 in radar_norm.columns else [0] * len(labels)
    
    # Close the polygons
    values_0 += values_0[:1]
    values_1 += values_1[:1]
    labels += labels[:1]
    
    # Create  radar
    fig_radar = go.Figure()
    
    # Non-defaulters (Green)
    if 0 in radar_norm.columns:
        fig_radar.add_trace(go.Scatterpolar(
            r=values_0,
            theta=labels,
            fill='toself',
            name='Non-Defaulters (0)',
            line_color="green",
            fillcolor="rgba(0, 255, 0, 0.2)",
            hovertemplate='<b>%{theta}</b><br>Value: %{r:.3f}<br>Group: Non-Defaulters<extra></extra>'
        ))
    
    # Defaulters (Red)
    if 1 in radar_norm.columns:
        fig_radar.add_trace(go.Scatterpolar(
            r=values_1,
            theta=labels,
            fill='toself',
            name='Defaulters (1)',
            line_color="red",
            fillcolor="rgba(255, 0, 0, 0.2)",
            hovertemplate='<b>%{theta}</b><br>Value: %{r:.3f}<br>Group: Defaulters<extra></extra>'
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0', '0.25', '0.5', '0.75', '1'],
                tickfont=dict(size=14, color="#1a1a1a")  # LARGER + DARKER
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color="#1a1a1a")  # LARGER + DARKER labels
            )
        ),
        showlegend=True,
        title=dict(
            text=" Risk Profile: Radar Chart",
            font=dict(size=16, color="#1a1a1a")  # DARKER title
        ),
        height=500,
        font=dict(size=12, color="#1a1a1a"),  # DARKER overall font
        margin=dict(t=80, b=20, l=20, r=20)
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown(
"""
**Insight:** The red shape (defaulters) bulges where debt burdens and refusals are higher and external scores weaker, while the green shape (non‑defaulters) shows lower leverage and stronger scores.

"""
)






