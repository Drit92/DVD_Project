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

# CONFIG
st.set_page_config(page_title="📊 Loan Risk Dashboard", layout="wide")
st.title("📊 Loan Applicant Risk Insights Dashboard")
st.markdown("---")

# LOAD AGGREGATES
ZIP_PATH = "loan_risk_aggregates.zip"

@st.cache_data(show_spinner="🔄 Loading aggregates...")
def load_aggregates(zip_path):
    if not os.path.exists(zip_path):
        st.error(f"{zip_path} not found. Upload to repo root.")
        st.stop()
    aggs = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("agg_") and name.endswith(".csv"):
                aggs[name] = pd.read_csv(io.BytesIO(zf.read(name)))
    return aggs

def get_agg(name, required=True):
    fname = f"agg_{name}.csv"
    if fname not in aggs:
        if required: st.error(f"Missing {fname}"); st.stop()
        return pd.DataFrame()
    return aggs[fname]

aggs = load_aggregates(ZIP_PATH)

# METRICS
overview = get_agg("overview_metrics")
overview_map = dict(zip(overview["metric"], overview["value"]))
total_applicants, total_defaulters, total_good, default_rate = [
    int(overview_map.get(k, 0)) for k in ["total_applicants", "total_defaulters", "total_good_borrowers"]
] + [float(overview_map.get("default_rate_overall", 0))]

target_dist = get_agg("target_distribution").copy()
target_dist["TARGET"] = target_dist["TARGET"].astype(int)

# 1. PORTFOLIO OVERVIEW
st.header("📈 Portfolio Overview")
col1, col2 = st.columns([1, 2])
with col1:
    shares = [float(target_dist.loc[target_dist["TARGET"] == i, "Share"].values[0]) 
              for i in [0, 1]]
    fig_donut = go.Figure(go.Pie(labels=["Good (0)", "Defaulter (1)"], values=shares,
                                hole=0.6, marker_colors=["#28a745", "#dc3545"],
                                textinfo="label+percent"))
    fig_donut.update_layout(height=260, title="Default Rate", transition_duration=0)
    st.plotly_chart(fig_donut)

with col2:
    st.columns(3)[1].metric("Defaulters", f"{total_defaulters:,}", f"{default_rate:.1%}")
st.markdown("**8% defaulters = highly imbalanced. Focus on riskiest segments.**")

# 2. DEMOGRAPHICS
st.header("👥 Demographic Risk")
fig_dem = make_subplots(rows=2, cols=2, horizontal_spacing=0.18, vertical_spacing=0.25,
                       subplot_titles=["Education", "Income Type", "Family", "Housing"],
                       specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]])

for i, col in enumerate(["NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]):
    df = get_agg(f"demo_{col}", False)
    if df.empty: continue
    df = df.sort_values("DefaultRate", ascending=False)
    df["DefaultRate"] *= 100
    r, c = divmod(i, 2); r += 1; c += 1
    fig_tmp = px.bar(df, x=col, y="DefaultRate", color="DefaultRate", color_continuous_scale="Blues")
    for trace in fig_tmp.data: fig_dem.add_trace(trace, row=r, col=c)

fig_dem.update_layout(height=700, showlegend=False, title="Default Rate by Demographics (Riskiest → Safest)")
st.plotly_chart(fig_dem)
st.markdown("**Higher education, govt jobs, marriage, home ownership = 2-3x safer**")

# 3. FINANCIAL STRESS (Defaulters Only)
st.header("💰 Financial Stress - Defaulter Concentration")
col1, col2, col3 = st.columns(3)
for i, (name, agg_name) in enumerate([("Credit/Income", "defaulters_credit_band_share"),
                                     ("EMI/Income", "defaulters_emi_band_share"),
                                     ("Income/Person", "defaulters_incomepp_band_share")]):
    df = get_agg(agg_name, False)
    if df.empty: continue
    fig = px.bar(df.sort_values("PercentDefaulters"), x=df.columns[0], y="PercentDefaulters",
                title=f"{name} Bands", color_discrete_sequence=["#f66d6d"])
    fig.update_yaxes(ticksuffix="%")
    st.columns(3)[i].plotly_chart(fig, width="stretch")

credit_default = get_agg("credit_default")
if not credit_default.empty:
    st.subheader("Default Rate Trend")
    credit_default["DefaultRate"] *= 100
    fig_trend = px.line(credit_default.sort_values("CREDIT_BIN"), x="CREDIT_BIN", y="DefaultRate",
                       markers=True, color_discrete_sequence=["#dc3545"])
    fig_trend.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_trend)
st.markdown("**31% defaulters at 3-5x credit/income. EMI 10-20% = 47% of failures**")

# 4. BEHAVIORAL
st.header("🚩 Behavioral Red Flags")
col1, col2 = st.columns(2)
refuse = get_agg("refuse_default"); refuse["DefaultRate"] *= 100
fig_ref = px.bar(refuse, x=["No Refusal", "Had Refusal"], y="DefaultRate",
                title="Refusal History", color_discrete_sequence=["#dc3545"])
fig_ref.update_yaxes(ticksuffix="%"); col1.plotly_chart(fig_ref)

apps = get_agg("prev_apps_default"); apps["DefaultRate"] *= 100
fig_apps = px.bar(apps, x="PREV_APPS_BIN", y="DefaultRate", title="Previous Apps",
                 color_discrete_sequence=["#e83e8c"])
fig_apps.update_yaxes(ticksuffix="%"); col2.plotly_chart(fig_apps)
st.markdown("**Refusals: 10% vs 7%. 10+ apps: 9.7% default**")

# 🔥 NEW: INTERACTION HEATMAPS
st.header("🔥 Risk Interactions")
col1, col2 = st.columns(2)

# Credit × Refusal (Cell 12)
credit_ref = get_agg("credit_refusal_heatmap", False)
if not credit_ref.empty:
    fig_heat1 = px.imshow(credit_ref.set_index("CREDIT_BIN").T, color_continuous_scale="Reds",
                         title="Credit Stress × Refusal", aspect="auto")
    col1.plotly_chart(fig_heat1)

# Education × External Score (Cell 13)
edu_ext = get_agg("education_ext_heatmap", False)
if not edu_ext.empty:
    fig_heat2 = px.imshow(edu_ext.set_index("NAME_EDUCATION_TYPE").T, color_continuous_scale="Blues",
                         title="Education × External Score", aspect="auto")
    col2.plotly_chart(fig_heat2)
st.markdown("**Credit stress + refusal doubles risk. Low education + poor scores = 15%+ default**")

# 5. EXTERNAL SCORES
st.header("⭐ External Credit Scores")
ext2_q = get_agg("ext2_quartile_default"); ext2_q["DefaultRate"] *= 100
fig_ext = px.bar(ext2_q.sort_values("EXT2_Q"), x="EXT2_Q", y="DefaultRate",
                title="EXT_SOURCE_2 Quartiles", color_discrete_sequence=["#6f42c1"])
fig_ext.update_yaxes(ticksuffix="%")
st.plotly_chart(fig_ext)
st.markdown("**Q1=12% vs Q4=4% default**")

# 6. BUBBLE: EXT × REFUSAL
st.header("🎯 External Score × Refusal")
ext_bubble = get_agg("ext2_refusal_bubble")
if not ext_bubble.empty:
    ext_bubble["DefaultRate"] *= 100
    label_map = {"Q1 (low)": 1, "Q2": 2, "Q3": 3, "Q4 (high)": 4}
    ext_bubble["Q_NUM"] = ext_bubble["EXT2_Q"].map(label_map)
    ext_bubble["REFUSAL"] = ext_bubble["FLAG_EVER_REFUSED"].map({0: "No", 1: "Yes"})
    ext_bubble["DefaultRate_jitter"] = np.clip(ext_bubble["DefaultRate"] + 
                                              ext_bubble["FLAG_EVER_REFUSED"] * 0.6 - 0.3, 0, None)
    
    fig_bubble = px.scatter(ext_bubble, x="Q_NUM", y="DefaultRate_jitter", size="Count",
                           color="REFUSAL", size_max=30, color_discrete_map={"No": "green", "Yes": "red"},
                           title="EXT Score vs Refusal (Size=Clients)", hover_data=["DefaultRate"])
    fig_bubble.update_xaxes(tickvals=[1,2,3,4], ticktext=["Q1", "Q2", "Q3", "Q4"])
    fig_bubble.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_bubble)
st.markdown("**Refused clients default 2x more in every score quartile**")

# 7. RISK SCORE
st.header("🎯 Combined Risk Score (0-9)")
risk_df = get_agg("risk_score_default"); risk_df["DefaultRate"] *= 100
risk_df["Band"] = pd.cut(risk_df["RISK_SCORE"], [0,3,6,10], labels=["Safe", "Medium", "High"])
fig_risk = px.bar(risk_df, x="RISK_SCORE", y="DefaultRate", color="RISK_SCORE",
                 color_continuous_scale=[[0,"green"],[0.5,"yellow"],[1,"red"]],
                 title="Risk Score vs Default Rate",
                 hover_data=["Band"])
fig_risk.update_xaxes(tickmode="array", tickvals=list(range(10)))
fig_risk.update_traces(hovertemplate="Score: %{x}<br>Band: %{customdata[0]}<br>Default: %{y:.1f}%")
fig_risk.update_yaxes(ticksuffix="%")
st.plotly_chart(fig_risk)
st.markdown("**Score ≥5 = 22% default (55% of failures, 25% volume)**")

# 8. RADAR
st.header("📈 Risk Profile Radar")
radar = get_agg("radar_means", False)
if not radar.empty:
    radar = radar.set_index("TARGET").T
    radar_norm = radar.apply(lambda x: (x-x.min())/(x.max()-x.min()+1e-9), axis=1)
    labels = radar_norm.index.tolist()
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + angles[:1]
    
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    for t in [0,1]:
        if t in radar_norm: 
            vals = radar_norm[t].tolist() + radar_norm[t].tolist()[:1]
            ax.plot(angles, vals, linewidth=2, label=["Non-Defaulters", "Defaulters"][t], 
                   color=["green", "red"][t])
            ax.fill(angles, vals, alpha=0.25, color=["green", "red"][t])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0,1); ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.title("Defaulter vs Non-Defaulter Profile"); plt.tight_layout()
    st.pyplot(fig)
st.markdown("**Defaulters: high debt, refusals, weak scores vs non-defaulters**")
