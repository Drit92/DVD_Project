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
# CONFIG
# ============================================

st.set_page_config(
    page_title="📊 Loan Applicant Risk Insights Dashboard",
    layout="wide",
)

st.title("📊 Loan Applicant Risk Insights Dashboard")
st.markdown("---")


# ============================================
# DATA LOADING – FROM AGG ZIP ONLY
# ============================================

@st.cache_data(show_spinner="🔄 Loading pre-aggregated loan risk data...")
def load_aggregates(zip_bytes: bytes):
    """Load all agg_*.csv from the provided ZIP into a dict of DataFrames."""
    agg_dict = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for name in zf.namelist():
            if name.startswith("agg_") and name.endswith(".csv"):
                with zf.open(name) as f:
                    df = pd.read_csv(f)
                agg_dict[name] = df
    return agg_dict


uploaded = st.file_uploader(
    "Upload loan_risk_aggregates.zip (from Colab)",
    type=["zip"],
)

if uploaded is None:
    st.info("⬆️ Please upload **loan_risk_aggregates.zip** to view the dashboard.")
    st.stop()

aggs = load_aggregates(uploaded.read())

# Helper: comfortable access by logical name
def get_agg(name: str, required: bool = True) -> pd.DataFrame:
    fname = f"agg_{name}.csv"
    if fname not in aggs:
        if required:
            st.error(f"Missing {fname} in the ZIP. Regenerate aggregates.")
            st.stop()
        else:
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

# For TARGET donut
target_dist = get_agg("target_distribution").copy()
# target_dist: columns ["TARGET","Share"]
target_dist["TARGET"] = target_dist["TARGET"].astype(int)


# ============================================
# 1. PORTFOLIO OVERVIEW – WHO DEFAULTS?
# ============================================

st.header("📈 Portfolio Overview – Who Defaults?")

col1, col2 = st.columns([1, 2])

with col1:
    # Donut: share of good vs defaulters
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
    st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Applicants", f"{total_applicants:,}")
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

# Gender mix
with col1:
    st.subheader("Applicant Gender Mix")
    gender_mix = get_agg("gender_mix", required=False)
    if not gender_mix.empty:
        # gender_mix: CODE_GENDER, Share
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
        st.plotly_chart(fig_gender_pie, use_container_width=True)
    else:
        st.info("Gender mix aggregate not available.")

# Age distribution (overall) – approximate from defaulter histogram bins
with col2:
    st.subheader("Applicant Age Distribution")
    age_hist = get_agg("age_defaulters_hist", required=False)
    if not age_hist.empty:
        # Use bin midpoints as x, counts as y. This is defaulters-only,
        # but still gives a shape. For exact overall histogram you’d need a separate agg.
        age_hist["bin_mid"] = (age_hist["bin_left"] + age_hist["bin_right"]) / 2
        fig_age_all = px.bar(
            age_hist,
            x="bin_mid",
            y="count",
            labels={
                "bin_mid": "Age (years)",
                "count": "Number of defaulters (binned)",
            },
            title="Distribution of Loan Applicant Age (Defaulters Approx.)",
            color_discrete_sequence=["#4C72B0"],
        )
        fig_age_all.update_traces(marker_line_width=1.2, marker_line_color="black")
        fig_age_all.update_layout(height=260, transition_duration=0)
        st.plotly_chart(fig_age_all, use_container_width=True)
    else:
        st.info("Age histogram aggregate not available.")

st.markdown(
    """
**Insights:**
- Male applicants show a **higher default rate (~10%)** than female applicants (~7%), even though both groups are large.
- Most applicants fall in the **working‑age band (late 20s to early 50s)**; very young and very old borrowers are a small fraction.
"""
)
st.markdown("---")


# ============================================
# 2. DEMOGRAPHIC SEGMENTS – DEFAULT RATES
# ============================================

st.header("👥 Demographic Segments – Who Is Riskier?")

fig_dem = make_subplots(
    rows=2,
    cols=2,
    horizontal_spacing=0.12,
    vertical_spacing=0.18,
    subplot_titles=(
        "Education level",
        "Type of income",
        "Family status",
        "Housing type",
    ),
    specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]],
)

demo_mapping = {
    "NAME_EDUCATION_TYPE": (1, 1),
    "NAME_INCOME_TYPE": (1, 2),
    "NAME_FAMILY_STATUS": (2, 1),
    "NAME_HOUSING_TYPE": (2, 2),
}

for col, (r, c) in demo_mapping.items():
    df_demo = get_agg(f"demo_{col}", required=False)
    if not df_demo.empty:
        df_demo = df_demo.copy()
        df_demo["DefaultRate"] = (df_demo["DefaultRate"] * 100).round(2)
        df_demo = df_demo.sort_values("DefaultRate", ascending=True)

        fig_tmp = px.bar(
            df_demo,
            x=col,
            y="DefaultRate",
            color="DefaultRate",
            color_continuous_scale="Reds",
            labels={
                col: col.replace("_", " ").title(),
                "DefaultRate": "Default rate (%)",
            },
        )
        fig_tmp.update_layout(xaxis=dict(tickangle=-35), coloraxis_showscale=False)

        for trace in fig_tmp.data:
            fig_dem.add_trace(trace, row=r, col=c)

fig_dem.update_layout(
    height=550,
    showlegend=False,
    title="Default Rate by Demographic Group (safest → riskiest)",
    margin=dict(l=30, r=30, t=70, b=40),
    transition_duration=0,
)
st.plotly_chart(fig_dem, use_container_width=True)

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
if not age_hist.empty:
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
    fig_age_def.update_layout(height=260, transition_duration=0)
    st.plotly_chart(fig_age_def, use_container_width=True)
else:
    st.info("Age histogram aggregate not available.")

st.markdown(
    """
**Insight:** Most defaulters are between **about 28 and 45 years old**; default risk tapers off for older customers who tend to have more stable finances.
"""
)
st.markdown("---")


# ============================================
# 3. FINANCIAL STRESS – CREDIT / INCOME TREND
# ============================================

st.header("💰 Financial Stress – How Much Debt Is Too Much?")

credit_default = get_agg("credit_default").copy()
# CREDIT_BIN, DefaultRate
credit_default["DefaultRate"] = credit_default["DefaultRate"] * 100

credit_order = ["0-1x", "1-2x", "2-3x", "3-5x", "5x+"]
credit_default["CREDIT_BIN"] = pd.Categorical(
    credit_default["CREDIT_BIN"],
    categories=credit_order,
    ordered=True,
)
credit_default = credit_default.sort_values("CREDIT_BIN")

fig_trend = px.line(
    credit_default,
    x="CREDIT_BIN",
    y="DefaultRate",
    markers=True,
    title="Default Rate Across Credit / Income Buckets",
    labels={"CREDIT_BIN": "Credit / income bucket", "DefaultRate": "Default rate (%)"},
    color_discrete_sequence=["#dc3545"],
)
fig_trend.update_traces(line_shape="linear", marker=dict(size=8))
fig_trend.update_yaxes(ticksuffix="%")
fig_trend.update_layout(transition_duration=0)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown(
    """
**Insights:**
- Default rate **rises steadily** as the loan amount grows relative to income, with a sharp jump beyond **3× income**.
- A practical guardrail is to keep loans below **4× income** and EMIs below **about 25% of income** wherever possible.
"""
)
st.markdown("---")


# ============================================
# 4. EXTERNAL SCORE + PAST REFUSAL – BUBBLE
# ============================================

st.header("🎯 External Score + Past Refusal – Combined Risk")

ext2_bubble = get_agg("ext2_refusal_bubble", required=False)
if not ext2_bubble.empty:
    ext2_bubble = ext2_bubble.copy()
    # EXT2_Q, FLAG_EVER_REFUSED, DefaultRate, Count
    label_to_num = {"Q1 (low)": 1, "Q2": 2, "Q3": 3, "Q4 (high)": 4}
    ext2_bubble["EXT2_Q_NUM"] = ext2_bubble["EXT2_Q"].map(label_to_num)
    # drop any rows where mapping failed
    ext2_bubble = ext2_bubble.dropna(subset=["EXT2_Q_NUM"])

    ext2_bubble["DefaultRate"] = ext2_bubble["DefaultRate"] * 100
    ext2_bubble["REFUSAL_STR"] = ext2_bubble["FLAG_EVER_REFUSED"].map(
        {0: "No refusal", 1: "Had refusal"}
    )

    offset_map = {"No refusal": -0.12, "Had refusal": 0.12}
    ext2_bubble["x_pos"] = ext2_bubble["EXT2_Q_NUM"] + ext2_bubble["REFUSAL_STR"].map(
        offset_map
    )

    fig_bubble = px.scatter(
        ext2_bubble,
        x="x_pos",
        y="DefaultRate",
        size="Count",
        color="REFUSAL_STR",
        size_max=40,
        color_discrete_map={"No refusal": "green", "Had refusal": "red"},
        labels={
            "x_pos": "External score quartile (higher = safer)",
            "DefaultRate": "Default rate (%)",
            "REFUSAL_STR": "Previous refusal",
        },
        title="External Score vs Past Refusal (Bubble Size = Number of Customers)",
    )

    quartile_ticks = sorted(ext2_bubble["EXT2_Q_NUM"].unique())
    # ensure only finite values and cast to int safely
    quartile_ticks = [int(q) for q in quartile_ticks if np.isfinite(q)]

    fig_bubble.update_xaxes(
        tickmode="array",
        tickvals=quartile_ticks,
        ticktext=[f"Q{q}" for q in quartile_ticks],
    )
    fig_bubble.update_yaxes(ticksuffix="%")
    fig_bubble.update_layout(transition_duration=0)
    st.plotly_chart(fig_bubble, width="stretch")
else:
    st.info("EXT2 × refusal aggregate not available.")


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
    refuse_labels = ["No previous refusal", "Had previous refusal"]

    fig_refuse = px.bar(
        x=refuse_labels,
        y=refuse_def["DefaultRate"].values,
        title="Default Rate by Refusal History",
        labels={"x": "Refusal history", "y": "Default rate (%)"},
        color_discrete_sequence=["#dc3545"],
    )
    fig_refuse.update_yaxes(ticksuffix="%")
    fig_refuse.update_layout(transition_duration=0)
    st.plotly_chart(fig_refuse, use_container_width=True)

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
    st.plotly_chart(fig_apps, use_container_width=True)

st.markdown(
    """
**Insights:**
- Customers with **any past refusal** have a much higher chance of defaulting than those with a clean record.
- Default risk stays low up to **4 previous applications**, then rises, especially at **10+ applications**, which signals credit‑shopping stress.
"""
)
st.markdown("---")


# ============================================
# 6. EXTERNAL CREDIT SCORES – QUARTILES
# ============================================

st.header("⭐ External Credit Scores – Power of Bureau Data")

col1, col2 = st.columns([2, 1])

with col1:
    ext2_def = get_agg("ext2_quartile_default")
    ext2_def = ext2_def.copy()
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
    st.plotly_chart(fig_ext2, use_container_width=True)

with col2:
    st.markdown(
        """
        **Key Insights:**
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
Combined Risk Score adds three components for each applicant:

- **Financial stress** – how large the loan and EMI are relative to income.  
- **Behaviour** – past refusals and how often the person has applied before.  
- **External score bucket** – quality of their external / bureau risk score.

In words:  
**Combined Risk Score = Financial Stress Score + Behaviour Score + External Score Component**.  
Higher scores mean **more red flags** across these areas.
"""
)

risk_def = get_agg("risk_score_default")
risk_def = risk_def.copy()
risk_def["DefaultRate"] = risk_def["DefaultRate"] * 100
risk_def = risk_def.sort_values("RISK_SCORE")

fig_risk = px.bar(
    x=risk_def["RISK_SCORE"].astype(str),
    y=risk_def["DefaultRate"],
    title="Default Rate by Combined Risk Score",
    labels={
        "x": "Combined risk score (0 = safest, higher = riskier)",
        "y": "Default rate (%)",
    },
    color_discrete_sequence=["#fd7e14"],
)
fig_risk.update_yaxes(ticksuffix="%")
fig_risk.update_layout(transition_duration=0)
st.plotly_chart(fig_risk, use_container_width=True)

st.markdown(
    """
**Insight:** Low scores capture the **safest borrowers**, while high scores group together the **riskiest applicants**, so this single score can drive cut‑offs, pricing bands and watchlists.
"""
)
st.markdown("---")


# ============================================
# 8. RADAR CHART – RISK PROFILE COMPARISON
# ============================================

st.header("📈 Risk Profile Comparison – Radar Chart")

radar_means = get_agg("radar_means", required=False)
if not radar_means.empty:
    radar_means = radar_means.copy()
    # radar_means: TARGET, col1, col2, ...
    radar_means["TARGET"] = radar_means["TARGET"].astype(int)
    radar_means = radar_means.set_index("TARGET").T  # rows=features, cols=TARGET

    # Normalize 0-1 per feature
    radar_norm = radar_means.apply(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9), axis=1
    )

    labels = radar_norm.index.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig_radar, ax_radar = plt.subplots(
        figsize=(3.8, 3.8), subplot_kw=dict(polar=True)
    )

    if 0 in radar_norm.columns:
        vals0 = radar_norm[0].tolist() + radar_norm[0].tolist()[:1]
        ax_radar.plot(
            angles, vals0, linewidth=2, label="Non-Defaulters (0)", color="green"
        )
        ax_radar.fill(angles, vals0, alpha=0.25, color="green")

    if 1 in radar_norm.columns:
        vals1 = radar_norm[1].tolist() + radar_norm[1].tolist()[:1]
        ax_radar.plot(
            angles, vals1, linewidth=2, label="Defaulters (1)", color="red"
        )
        ax_radar.fill(angles, vals1, alpha=0.25, color="red")

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=6)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticklabels([])

    ax_radar.set_title("Risk Profile Comparison – Radar Chart", pad=12, fontsize=11)
    ax_radar.legend(bbox_to_anchor=(1.05, 1.0), borderaxespad=0.0, fontsize=7)
    plt.tight_layout(pad=0.8)

    st.pyplot(fig_radar, use_container_width=True)

    st.markdown(
        """
**Insight:** The red shape (defaulters) bulges where debt burdens and refusals are higher and external scores weaker, while the green shape (non‑defaulters) shows lower leverage and stronger scores.
"""
    )
else:
    st.info("Radar aggregate not available.")
