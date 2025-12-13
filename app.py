import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pipeline import load_raw_data, engineer_features

st.set_page_config(page_title="📊 Loan Applicant Risk Insights Dashboard", layout="wide")

st.title("📊 Loan Applicant Risk Insights Dashboard")
st.markdown("---")

@st.cache_data(show_spinner="🔄 Loading + Engineering Data...")
def load_data():
    app_data, prev_data = load_raw_data()
    df = engineer_features(app_data, prev_data)
    return app_data, df

app_data, df = load_data()
st.success(f"✅ Loaded {len(df):,} records | 🔴 Default Rate: {df['TARGET'].mean():.1%}")

# Precompute defaulters once
df_def = df[df['TARGET'] == 1].copy()

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
            showlegend=False
        )
    ])
    fig_donut.update_layout(
        title="Share of Good Borrowers vs Defaulters",
        height=260,
        margin=dict(t=40, b=10)
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Applicants", f"{len(df):,}")
    c2.metric("Total Defaulters", f"{df['TARGET'].sum():,}", f"{df['TARGET'].mean():.1%}")
    c3.metric("Total Good Borrowers", f"{(1-df['TARGET']).sum():,}")

st.markdown("""
**Overview story:** Most customers repay on time; only a small share (about 8%) default, so the dataset is highly imbalanced.
""")
st.markdown("---")

# === 1A. BORROWER PROFILES ===
st.header("🧍 Borrower Profiles – Who Applies?")

col1, col2 = st.columns(2)

# Gender mix (Pie)
with col1:
    st.subheader("Applicant Gender Mix")
    gender_df = app_data[app_data['CODE_GENDER'] != 'XNA'].copy()
    gender_counts = gender_df['CODE_GENDER'].value_counts()
    label_map = {'M': 'Male', 'F': 'Female'}
    labels = [label_map.get(g, g) for g in gender_counts.index]

    fig_gender_pie = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=gender_counts.values,
            hole=0.4,
            marker_colors=['#4C72B0', '#DD8452'],
            textinfo='label+percent',
            textposition='outside',
            showlegend=False
        )]
    )
    fig_gender_pie.update_layout(
        title="Share of Applicants by Gender",
        margin=dict(t=40, l=10, r=10, b=10),
        height=260
    )
    st.plotly_chart(fig_gender_pie, use_container_width=True)

# Overall age distribution
with col2:
    st.subheader("Applicant Age Distribution")
    fig_age_all = px.histogram(
        df,
        x='AGE_YEARS',
        nbins=20,
        title="Distribution of Loan Applicant Age",
        labels={'AGE_YEARS': 'Age (years)', 'count': 'Number of applicants'},
        color_discrete_sequence=['#4C72B0']
    )
    fig_age_all.update_traces(marker_line_width=1.2, marker_line_color='black')
    fig_age_all.update_layout(height=260)
    st.plotly_chart(fig_age_all, use_container_width=True)

st.markdown("""
**Insights:**
- Male applicants show a **higher default rate (~10%)** than female applicants (~7%), even though both groups are large.
- Most applicants fall in the **working‑age band (late 20s to early 50s)**; very young and very old borrowers are a small fraction.
""")
st.markdown("---")

# === 2. DEMOGRAPHICS – ASCENDING BARS ===
st.header("👥 Demographic Segments – Who Is Riskier?")

fig_dem = make_subplots(
    rows=2,
    cols=2,
    horizontal_spacing=0.12,
    vertical_spacing=0.18,
    subplot_titles=(
        'Education level',
        'Type of income',
        'Family status',
        'Housing type'
    ),
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}]]
)

risk_columns = [
    'NAME_EDUCATION_TYPE',
    'NAME_INCOME_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE'
]

for i, col in enumerate(risk_columns):
    if col in app_data.columns:
        r, c = divmod(i, 2)
        r += 1; c += 1

        default_rate = (
            app_data.groupby(col)['TARGET']
                    .mean()
                    .sort_values(ascending=True)
        ).reset_index()
        default_rate.columns = [col, 'TARGET']
        default_rate['DefaultRate'] = (default_rate['TARGET'] * 100).round(2)

        fig_tmp = px.bar(
            default_rate,
            x=col,
            y='DefaultRate',
            color='DefaultRate',
            color_continuous_scale='Reds',
            labels={col: col.replace('_', ' ').title(), 'DefaultRate': 'Default rate (%)'}
        )
        fig_tmp.update_layout(xaxis=dict(tickangle=-35))

        for trace in fig_tmp.data:
            fig_dem.add_trace(trace, row=r, col=c)

fig_dem.update_layout(
    height=600,
    showlegend=False,
    title="Default Rate by Demographic Group (safest → riskiest)",
    margin=dict(l=30, r=30, t=70, b=40)
)
st.plotly_chart(fig_dem, use_container_width=True)

st.markdown("""
**Insights:**
- Higher education, government income, being married, and owning a home are all linked to **lower default risk**.
- Lower education, unstable income (e.g., unemployment, maternity leave) and renting or living with parents are associated with **higher default rates**.
""")
st.markdown("---")

# === 2A. AGE DISTRIBUTION – DEFAULTERS ONLY ===
st.header("📅 Age Profile of Defaulters")

fig_age_def = px.histogram(
    df_def,
    x='AGE_YEARS',
    nbins=40,
    histnorm='probability density',
    title="Age Distribution of Defaulters",
    labels={'AGE_YEARS': 'Age (years)', 'probability density': 'Density'},
    color_discrete_sequence=['#E57373']
)
fig_age_def.update_traces(marker_line_width=1.2, marker_line_color='black')
fig_age_def.update_layout(height=260)
st.plotly_chart(fig_age_def, use_container_width=True)

st.markdown("""
**Insight:** Most defaulters are between **about 28 and 45 years old**; default risk tapers off for older customers who tend to have more stable finances.
""")
st.markdown("---")

# === 3. FINANCIAL STRESS ===
st.header("💰 Financial Stress – How Much Debt Is Too Much?")

credit_order = ['0-1x', '1-2x', '2-3x', '3-5x', '5x+']
emi_order    = ['<10%', '10-20%', '20-30%', '30-50%', '50%+']

col1, col2 = st.columns(2)

with col1:
    credit_freq = df_def['CREDIT_BIN'].value_counts(normalize=True) * 100
    credit_freq = credit_freq.reindex(credit_order)

    fig_credit = go.Figure()
    fig_credit.add_bar(
        x=credit_freq.index.astype(str),
        y=credit_freq.values,
        marker_color='#dc3545',
        name='% of defaulters'
    )
    fig_credit.add_scatter(
        x=credit_freq.index.astype(str),
        y=credit_freq.values,
        mode='lines+markers',
        line=dict(color='black'),
        marker=dict(size=6),
        name='Trend'
    )
    fig_credit.update_layout(
        title="Defaulters by Credit / Income Ratio Band",
        yaxis=dict(range=[0, 60], ticksuffix="%", title='% of defaulters'),
        xaxis_title='Credit / income band'
    )
    st.plotly_chart(fig_credit, use_container_width=True)

with col2:
    emi_freq = df_def['EMI_BIN'].value_counts(normalize=True) * 100
    emi_freq = emi_freq.reindex(emi_order)

    fig_emi = go.Figure()
    fig_emi.add_bar(
        x=emi_freq.index.astype(str),
        y=emi_freq.values,
        marker_color='#fd7e14',
        name='% of defaulters'
    )
    fig_emi.add_scatter(
        x=emi_freq.index.astype(str),
        y=emi_freq.values,
        mode='lines+markers',
        line=dict(color='black'),
        marker=dict(size=6),
        name='Trend'
    )
    fig_emi.update_layout(
        title="Defaulters by EMI / Income Ratio Band",
        yaxis=dict(range=[0, 60], ticksuffix="%", title='% of defaulters'),
        xaxis_title='EMI / income band'
    )
    st.plotly_chart(fig_emi, use_container_width=True)

trend_df = df.groupby('CREDIT_BIN', observed=True)['TARGET'].mean().reset_index()
trend_df['TARGET'] = trend_df['TARGET'] * 100
trend_df = trend_df.set_index('CREDIT_BIN').reindex(credit_order).reset_index()

fig_trend = px.line(
    trend_df,
    x='CREDIT_BIN',
    y='TARGET',
    markers=True,
    title="Default Rate Across Credit / Income Buckets",
    labels={'CREDIT_BIN': 'Credit / income bucket', 'TARGET': 'Default rate (%)'},
    color_discrete_sequence=['#dc3545']
)
fig_trend.update_traces(line_shape='linear', marker=dict(size=8))
fig_trend.update_yaxes(ticksuffix="%")
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("""
**Insights:**
- A large share of defaulters sit in **3–5× income** credit bands and **10–30% EMI** bands.
- Default rate itself **rises steadily** with leverage; beyond **3× income** the risk climbs sharply.
- As a rule of thumb, keep loans below **4× income** and EMIs below **25% of income** where possible.
""")
st.markdown("---")

# === 4. BUBBLE CHART – EXTERNAL SCORE vs REFUSAL (NON‑OVERLAPPING) ===
st.header("🎯 External Score + Past Refusal – Combined Risk")

bubble_df = df[['EXT2_Q', 'FLAG_EVER_REFUSED', 'TARGET']].dropna()
bubble_group = bubble_df.groupby(['EXT2_Q', 'FLAG_EVER_REFUSED'])['TARGET'].mean() * 100
bubble_count = bubble_df.groupby(['EXT2_Q', 'FLAG_EVER_REFUSED'])['TARGET'].count()

bubble_final = pd.DataFrame({
    'DefaultRate': bubble_group,
    'Count': bubble_count
}).reset_index()

offset_map = {0: -0.12, 1: 0.12}
bubble_final['EXT2_Q_NUM'] = bubble_final['EXT2_Q'].astype(int)
bubble_final['x_pos'] = bubble_final['EXT2_Q_NUM'] + bubble_final['FLAG_EVER_REFUSED'].map(offset_map)

fig_bubble = px.scatter(
    bubble_final,
    x='x_pos',
    y='DefaultRate',
    size='Count',
    color='FLAG_EVER_REFUSED',
    size_max=40,
    color_discrete_map={0: 'green', 1: 'red'},
    labels={
        'x_pos': 'External score quartile (higher = safer)',
        'DefaultRate': 'Default rate (%)',
        'FLAG_EVER_REFUSED': 'Previous refusal (0 = No, 1 = Yes)'
    },
    title="External Score vs Past Refusal (Bubble Size = Number of Customers)"
)

quartile_ticks = sorted(bubble_final['EXT2_Q_NUM'].unique())
fig_bubble.update_xaxes(
    tickmode='array',
    tickvals=quartile_ticks,
    ticktext=[f"Q{q}" for q in quartile_ticks]
)
fig_bubble.update_yaxes(ticksuffix="%")
st.plotly_chart(fig_bubble, use_container_width=True)

st.markdown("""
**Insights:**
- For any given external‑score quartile, customers with **past refusals** (red) default more often than those with clean histories (green).
- Big green bubbles in higher quartiles are **safe, high‑volume customers**; small red bubbles in low quartiles are **concentrated risk pockets**.
""")
st.markdown("---")

# === 5. BEHAVIOURAL RED FLAGS ===
st.header("🚩 Behavioural Red Flags – Past Actions Matter")

col1, col2 = st.columns(2)

with col1:
    refuse_def = df.groupby('FLAG_EVER_REFUSED')['TARGET'].mean() * 100
    fig_refuse = px.bar(
        x=['No previous refusal', 'Had previous refusal'],
        y=refuse_def.reindex([0, 1]).values,
        title="Default Rate by Refusal History",
        labels={'x': 'Refusal history', 'y': 'Default rate (%)'},
        color_discrete_sequence=['#dc3545']
    )
    fig_refuse.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_refuse, use_container_width=True)

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
    fig_apps.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_apps, use_container_width=True)

st.markdown("""
**Insights:**
- Customers with **any past refusal** have a much higher chance of defaulting than those with a clean record.
- Default risk stays low up to **4 previous applications**, then rises, especially at **10+ applications**, which signals credit‑shopping stress.
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
    fig_ext2.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_ext2, use_container_width=True)

with col2:
    st.markdown("""
    **Insights:**
    - The **lowest external‑score quartile (Q1)** has several times the default rate of the **highest quartile (Q4)**.
    - Default probability **falls smoothly** as external score improves, making this one of the strongest early‑screening tools.
    """)

st.markdown("---")

fig_hist = px.histogram(
    df,
    x='EXT_SOURCE_2',
    color='TARGET',
    nbins=50,
    title="Distribution of External Scores for Good Borrowers vs Defaulters",
    labels={'EXT_SOURCE_2': 'External credit score', 'count': 'Number of applicants'},
    color_discrete_map={0: '#28a745', 1: '#dc3545'}
)
fig_hist.update_traces(marker_line_width=1.2, marker_line_color='black')
fig_hist.update_layout(height=350, legend_title="Default status")
for trace in fig_hist.data:
    if trace.name == '0':
        trace.name = 'Good borrower (0)'
    elif trace.name == '1':
        trace.name = 'Defaulter (1)'
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# === 7. COMBINED RISK SCORE ===
st.header("🎯 Combined Risk Score – One Number That Combines All Risk Flags")

st.markdown("""
Combined Risk Score adds three components for each applicant:

- **Financial stress** – how large the loan and EMI are relative to income.  
- **Behaviour** – past refusals and how often the person has applied before.  
- **External score bucket** – quality of their external / bureau risk score.

In words:  
**Combined Risk Score = Financial Stress Score + Behaviour Score + External Score Component**.  
Higher scores mean **more red flags** across these areas.
""")

risk_def = df.groupby('RISK_SCORE')['TARGET'].mean() * 100
risk_def = risk_def.sort_index()

fig_risk = px.bar(
    x=risk_def.index.astype(str),
    y=risk_def.values,
    title="Default Rate by Combined Risk Score",
    labels={'x': 'Combined risk score (0 = safest, higher = riskier)', 'y': 'Default rate (%)'},
    color_discrete_sequence=['#fd7e14']
)
fig_risk.update_yaxes(ticksuffix="%")
st.plotly_chart(fig_risk, use_container_width=True)

st.markdown("""
**Insight:** Low scores capture the **safest borrowers**, while high scores group together the **riskiest applicants**, so this single score can drive cut‑offs, pricing bands and watchlists.
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
    radar_data = df.groupby('TARGET')[radar_cols].mean().T
    radar_norm = radar_data.apply(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9),
        axis=1
    )

    labels = radar_norm.index.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig_radar, ax_radar = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True))

    if 0 in radar_norm.columns:
        vals0 = radar_norm[0].tolist() + radar_norm[0].tolist()[:1]
        ax_radar.plot(angles, vals0, linewidth=2, label='Non-Defaulters (0)', color='green')
        ax_radar.fill(angles, vals0, alpha=0.25, color='green')

    if 1 in radar_norm.columns:
        vals1 = radar_norm[1].tolist() + radar_norm[1].tolist()[:1]
        ax_radar.plot(angles, vals1, linewidth=2, label='Defaulters (1)', color='red')
        ax_radar.fill(angles, vals1, alpha=0.25, color='red')

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=6)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticklabels([])

    ax_radar.set_title("Risk Profile Comparison – Radar Chart", pad=12, fontsize=11)
    ax_radar.legend(bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=7)
    plt.tight_layout(pad=0.8)

    st.pyplot(fig_radar, use_container_width=False)

    st.markdown("""
    **Insight:** The red shape (defaulters) bulges where debt burdens and refusals are higher and external scores weaker, while the green shape (non‑defaulters) shows lower leverage and stronger scores.
    """)

st.caption("Dashboard built for non‑technical users: each chart answers a simple question about loan risk.")
