import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import load_raw_data, engineer_features

st.set_page_config(page_title="📊 Loan Applicant Risk Insights Dashboard", layout="wide")

st.title("📊 Loan Applicant Risk Insights Dashboard")
st.markdown("---")

@st.cache_data(show_spinner="🔄 Loading + Engineering Data...")
def load_data():
    app_data, prev_data = load_raw_data()
    return engineer_features(app_data, prev_data)

df = load_data()
st.success(f"✅ Loaded {len(df):,} records | 🔴 Default Rate: {df['TARGET'].mean():.1%}")

# === 1. PORTFOLIO OVERVIEW ===
st.header("📈 Portfolio Overview – Who Defaults?")
col1, col2 = st.columns([1, 2])

with col1:
    target_counts = df['TARGET'].value_counts()
    fig_donut = go.Figure(data=[
        go.Pie(
            labels=['Good borrower (0)', 'Defaulter (1)'],
            values=target_counts.values,
            hole=0.6,
            marker_colors=['#28a745', '#dc3545'],
            textinfo='label+percent',
            textposition='outside',
            textfont_size=14,
            showlegend=False
        )
    ])
    fig_donut.update_layout(
        title="Share of Good Borrowers vs Defaulters",
        height=300,
        margin=dict(t=60)
    )
    st.plotly_chart(fig_donut, width='stretch')

with col2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Applicants", f"{len(df):,}")
    c2.metric("Total Defaulters", f"{df['TARGET'].sum():,}", f"{df['TARGET'].mean():.1%}")
    c3.metric("Total Good Borrowers", f"{(1-df['TARGET']).sum():,}")

st.markdown("""
**Plain‑language story:**
- Most customers repay on time; only a small share (about 8%) default.
- Because defaulters are rare, any risk model must treat this as an imbalanced problem.
""")

st.markdown("---")

# === 2. DEMOGRAPHICS – SORTED ASC ===
st.header("👥 Demographic Segments – Who Is Riskier?")

fig_dem = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Education level', 'Type of income', 'Family status', 'Housing type'),
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}]]
)

cat_cols = [
    'NAME_EDUCATION_TYPE',
    'NAME_INCOME_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE'
]

for i, col in enumerate(cat_cols):
    if col in df.columns:
        r, c = divmod(i, 2)
        r += 1; c += 1
        tmp = df.groupby(col)['TARGET'].mean().reset_index()
        tmp['DefaultRate'] = (tmp['TARGET'] * 100).round(2)
        tmp = tmp.sort_values('DefaultRate', ascending=True)

        fig_tmp = px.bar(
            tmp,
            x=col,
            y='DefaultRate',
            color='DefaultRate',
            color_continuous_scale='Reds',
            labels={col: col.replace('_', ' ').title(), 'DefaultRate': 'Default rate (%)'}
        )
        for trace in fig_tmp.data:
            fig_dem.add_trace(trace, row=r, col=c)

fig_dem.update_layout(
    height=550,
    showlegend=False,
    title="Default Rate by Demographic Group (sorted from safest to riskiest)"
)
st.plotly_chart(fig_dem, width='stretch')

st.markdown("""
**Story:**
- Higher education, government income, being married, and owning a home are all linked to **safer borrowers**.
- Lower education, unstable income and renting or living with parents increase default risk.
""")

st.markdown("---")

# === 2A. AGE DISTRIBUTION – DEFAULTERS ONLY ===
st.header("📅 Age Profile of Defaulters")

df_def = df[df['TARGET'] == 1].copy()
df_def['AGE_YEARS'] = df_def['AGE_YEARS'].clip(lower=18, upper=80)

fig_age, ax_age = plt.subplots(figsize=(5, 3))

sns.histplot(
    data=df_def,
    x='AGE_YEARS',
    kde=True,
    bins=40,
    stat="density",
    color="#E57373",
    alpha=0.65,
    ax=ax_age
)

ax_age.set_title("Age Distribution of Defaulters", fontsize=11)
ax_age.set_xlabel("Age in years")
ax_age.set_ylabel("Density")
ax_age.grid(True, axis="y", linestyle="--", alpha=0.3)
plt.tight_layout(pad=1.0)

st.pyplot(fig_age, width="content")

st.markdown("""
**Story:**
- Most defaulters fall between **about 28 and 45 years old**.
- Default risk tapers off for older customers, who tend to be more financially stable.
""")

st.markdown("---")

# === 3. FINANCIAL STRESS + TREND ===
st.header("💰 Financial Stress – How Much Debt Is Too Much?")

col1, col2 = st.columns(2)

with col1:
    credit_def = df.groupby('CREDIT_BIN', observed=True)['TARGET'].mean() * 100
    credit_order = ['0-1x', '1-2x', '2-3x', '3-5x', '5x+']
    credit_def = credit_def.reindex(credit_order)
    fig_credit = px.bar(
        x=credit_def.index.astype(str),
        y=credit_def.values,
        title="Default Rate by Credit / Income Multiple",
        labels={'x': 'Credit amount as multiple of annual income', 'y': 'Default rate (%)'},
        color_discrete_sequence=['#dc3545']
    )
    fig_credit.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_credit, width='stretch')

with col2:
    emi_def = df.groupby('EMI_BIN', observed=True)['TARGET'].mean() * 100
    emi_order = ['<10%', '10-20%', '20-30%', '30-50%', '50%+']
    emi_def = emi_def.reindex(emi_order)
    fig_emi = px.bar(
        x=emi_def.index.astype(str),
        y=emi_def.values,
        title="Default Rate by EMI / Income Percentage",
        labels={'x': 'Monthly EMI as % of income', 'y': 'Default rate (%)'},
        color_discrete_sequence=['#fd7e14']
    )
    fig_emi.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_emi, width='stretch')

st.subheader("📈 Trend: Credit Stress vs Default Rate")
trend_df = df.groupby('CREDIT_BIN', observed=True)['TARGET'].mean().reset_index()
trend_df['TARGET'] = trend_df['TARGET'] * 100
trend_df = trend_df.set_index('CREDIT_BIN').reindex(credit_order).reset_index()

fig_trend = px.line(
    trend_df,
    x='CREDIT_BIN',
    y='TARGET',
    markers=True,
    title="Default Trend Across Credit / Income Buckets",
    labels={'CREDIT_BIN': 'Credit / Income bucket', 'TARGET': 'Default rate (%)'},
    color_discrete_sequence=['#dc3545']
)
fig_trend.update_traces(line_shape='linear', marker=dict(size=10, color='#dc3545'))
fig_trend.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_trend, width='stretch')

st.markdown("""
**Story:**
- When loans reach **3–5× annual income**, default risk jumps sharply.
- EMIs above **20% of income** create serious repayment pressure.
- Sensible policy: avoid approving loans above **4× income** and EMI above **25% of income**.
""")

st.markdown("---")

# === 4. BUBBLE CHART (Seaborn/Matplotlib) ===
st.header("🎯 External Score + Past Refusal – Combined Risk")

bubble_df = df[['EXT2_Q', 'FLAG_EVER_REFUSED', 'TARGET']].dropna()
bubble_group = bubble_df.groupby(['EXT2_Q', 'FLAG_EVER_REFUSED'])['TARGET'].mean() * 100
bubble_count = bubble_df.groupby(['EXT2_Q', 'FLAG_EVER_REFUSED'])['TARGET'].count()

bubble_final = pd.DataFrame({
    'DefaultRate': bubble_group,
    'Count': bubble_count
}).reset_index()

fig_bubble, ax = plt.subplots(figsize=(5.5, 3.5))

sns.scatterplot(
    data=bubble_final,
    x='EXT2_Q',
    y='DefaultRate',
    size='Count',
    hue='FLAG_EVER_REFUSED',
    sizes=(40, 400),
    palette={0: 'green', 1: 'red'},
    alpha=0.6,
    ax=ax,
    legend=False
)

ax.set_title("External Score vs Past Refusal (Bubble Size = Number of Customers)", fontsize=11)
ax.set_xlabel("External score quartile (higher = safer)")
ax.set_ylabel("Default rate (%)")
ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.4)

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor='green', edgecolor='black', label='No previous refusal'),
    Patch(facecolor='red', edgecolor='black', label='Had previous refusal')
]
ax.legend(
    handles=legend_handles,
    title="Refusal history",
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0.,
    fontsize=9,
    title_fontsize=10
)

plt.tight_layout(pad=1.0)
st.pyplot(fig_bubble, width="content")

st.markdown("""
**Story:**
- For the same external score, customers with a **past refusal** (red bubbles) default more.
- The **largest safe group** is high external score + **no** refusal (big green bubbles on the right).
""")

st.markdown("---")

# === 5. BEHAVIOURAL RED FLAGS ===
st.header("🚩 Behavioural Red Flags – Past Actions Matter")

col1, col2 = st.columns(2)

with col1:
    refuse_def = df.groupby('FLAG_EVER_REFUSED')['TARGET'].mean() * 100
    x_labels = ['No previous refusal', 'Had previous refusal']
    fig_refuse = px.bar(
        x=x_labels,
        y=refuse_def.reindex([0, 1]).values,
        title="Default Rate by Refusal History",
        labels={'x': 'Refusal history', 'y': 'Default rate (%)'},
        color_discrete_sequence=['#dc3545']
    )
    fig_refuse.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_refuse, width='stretch')

with col2:
    apps_def = df.groupby('PREV_APPS_BIN', observed=True)['TARGET'].mean() * 100
    app_order = ['0', '1-2', '3-4', '5-9', '10+']
    apps_def = apps_def.reindex(app_order)
    fig_apps = px.bar(
        x=apps_def.index.astype(str),
        y=apps_def.values,
        title="Default Rate by Number of Previous Applications",
        labels={'x': 'Number of previous applications', 'y': 'Default rate (%)'},
        color_discrete_sequence=['#e83e8c']
    )
    fig_apps.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_apps, width='stretch')

st.markdown("""
**Story:**
- Customers who were **refused in the past** are noticeably more likely to default now.
- Risk is lowest for people with **0–4 past applications**, and rises for **5+**, especially **10+**.
""")

st.markdown("---")

# === 6. EXTERNAL CREDIT SCORES ===
st.header("⭐ External Credit Scores – Power of Bureau Data")

col1, col2 = st.columns([2, 1])

with col1:
    ext2_def = df.groupby('EXT2_Q')['TARGET'].mean() * 100
    ext2_def = ext2_def.reindex(['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
    fig_ext2 = px.bar(
        x=ext2_def.index.astype(str),
        y=ext2_def.values,
        title="Default Rate by External Score Quartile",
        labels={'x': 'External score quartile', 'y': 'Default rate (%)'},
        color_discrete_sequence=['#6f42c1']
    )
    fig_ext2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_ext2, width='stretch')

with col2:
    st.markdown("""
    **Story:**
    - The **lowest score band (Q1)** has roughly **four times** the default rate of the **highest band (Q4)**.
    - As the external score improves, default risk **falls smoothly**.
    - These scores are extremely useful for early screening.
    """)

st.markdown("---")

fig_hist = px.histogram(
    df,
    x='EXT_SOURCE_2',
    color='TARGET',
    nbins=50,
    marginal='violin',
    title="Distribution of External Scores for Good Borrowers vs Defaulters",
    labels={
        'EXT_SOURCE_2': 'External credit score',
        'count': 'Number of applicants',
        'TARGET': 'Default status'
    },
    color_discrete_map={0: '#28a745', 1: '#dc3545'},
)
fig_hist.update_traces(marker_line_width=1.5, marker_line_color='black')
fig_hist.update_layout(
    height=500,
    legend_title="Default status",
    legend=dict(itemsizing="constant")
)
for trace in fig_hist.data:
    if trace.name == '0':
        trace.name = 'Good borrower (0)'
    elif trace.name == '1':
        trace.name = 'Defaulter (1)'

st.plotly_chart(fig_hist, width='stretch')

st.markdown("---")

# === 7. COMBINED RISK SCORE ===
st.header("🎯 Combined Risk Score – One Number to Rank Risk")

risk_def = df.groupby('RISK_SCORE')['TARGET'].mean() * 100
risk_def = risk_def.sort_index()

fig_risk = px.bar(
    x=risk_def.index.astype(str),
    y=risk_def.values,
    title="Default Rate by Combined Risk Score",
    labels={'x': 'Risk score (0 = safest, higher = riskier)', 'y': 'Default rate (%)'},
    color_discrete_sequence=['#fd7e14']
)
fig_risk.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_risk, width='stretch')

st.markdown("""
**Story:**
- Low scores group the **safest** customers; high scores gather the **riskiest**.
- This single score can drive **approval thresholds, pricing tiers, or watchlists**.
""")

st.markdown("---")

# === 8. RADAR CHART – COLAB STYLE, SMALLER ===
st.header("📈 Risk Profile Comparison – Radar Chart")

radar_cols = [
    'CREDIT_INCOME_RATIO',
    'ANNUITY_INCOME_RATIO',
    'FLAG_EVER_REFUSED',
    'EXT_SOURCE_2',
    'AGE_YEARS',
    'AMT_INCOME_TOTAL'
]
radar_cols = [c for c in radar_cols if c in df.columns]

if len(radar_cols) >= 3:
    radar_data = (
        df.groupby('TARGET')[radar_cols]
        .mean()
        .T
    )

    radar_norm = radar_data.apply(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9),
        axis=1
    )

    labels = radar_norm.index.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig_radar, ax_radar = plt.subplots(
        figsize=(5, 5),
        subplot_kw=dict(polar=True)
    )

    if 0 in radar_norm.columns:
        vals0 = radar_norm[0].tolist() + radar_norm[0].tolist()[:1]
        ax_radar.plot(angles, vals0, linewidth=2, label='Non-Defaulters (0)', color='green')
        ax_radar.fill(angles, vals0, alpha=0.25, color='green')

    if 1 in radar_norm.columns:
        vals1 = radar_norm[1].tolist() + radar_norm[1].tolist()[:1]
        ax_radar.plot(angles, vals1, linewidth=2, label='Defaulters (1)', color='red')
        ax_radar.fill(angles, vals1, alpha=0.25, color='red')

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticklabels([])

    ax_radar.set_title("Risk Profile Comparison – Radar Chart", pad=20, fontsize=13)
    ax_radar.legend(
        bbox_to_anchor=(1.05, 1.0),
        borderaxespad=0.,
        fontsize=8,
        title_fontsize=9
    )
    plt.tight_layout(pad=1.2)

    st.pyplot(fig_radar, width="content")

    st.markdown("""
    **Story:**
    - Defaulters show higher debt and more refusal history, and weaker external scores.
    - Non‑defaulters sit in the opposite, safer region on most axes.
    """)
else:
    st.info("Not enough radar features available to draw the chart.")

st.markdown("---")
st.caption("Dashboard built for non‑technical users: each chart answers a simple question about loan risk.")
