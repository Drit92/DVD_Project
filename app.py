import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
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

# === 1. DONUT CHART + METRICS ===
st.header("📈 Portfolio Overview")
col1, col2 = st.columns([1, 2])

with col1:
    target_counts = df['TARGET'].value_counts()
    fig_donut = go.Figure(data=[
        go.Pie(
            labels=['Non-Defaulter (0)', 'Defaulter (1)'],
            values=target_counts.values,
            hole=0.6,
            marker_colors=['#28a745', '#dc3545'],
            textinfo='label+percent',
            textposition='outside',
            textfont_size=14,
            showlegend=False
        )
    ])
    fig_donut.update_layout(title="Loan Default Distribution", height=300, margin=dict(t=60))
    st.plotly_chart(fig_donut, width='stretch')

with col2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Applicants", f"{len(df):,}")
    c2.metric("Defaulters", f"{df['TARGET'].sum():,}", f"{df['TARGET'].mean():.1%}")
    c3.metric("Good Borrowers", f"{(1-df['TARGET']).sum():,}")

st.markdown("""
**🔍 Insights:**
- **92% Good Borrowers** vs **8% Defaulters** (highly imbalanced)
- Need special handling for ML modeling (SMOTE/class weights)
""")

st.markdown("---")

# === 2. DEMOGRAPHICS ===
st.header("👥 Risk by Demographics")
fig_dem = make_subplots(rows=2, cols=2, 
                       subplot_titles=('Education', 'Income Type', 'Family Status', 'Housing'),
                       specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]])

cat_cols = ['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']
for i, col in enumerate(cat_cols):
    if col in df.columns:
        r, c = divmod(i, 2)
        r += 1; c += 1
        tmp = df.groupby(col)['TARGET'].mean().reset_index()
        tmp['TARGET'] = tmp['TARGET'] * 100
        tmp = tmp.sort_values('TARGET', ascending=True)
        fig_dem.add_trace(
            px.bar(tmp, x=col, y='TARGET', color='TARGET', 
                   color_continuous_scale='Reds').data[0], 
            row=r, col=c
        )

fig_dem.update_layout(height=550, showlegend=False, title="Default Rate by Demographics (%)")
st.plotly_chart(fig_dem, width='stretch')

st.markdown("""
**🔍 Insights:**
- **Education**: Lower secondary = 10.9% vs Academic degree = 1.8% (6x difference)
- **Income**: Maternity/Unemployed = ~40% vs State servant = 5.8%
- **Family**: Civil marriage/Single = ~10% vs Widow = 5.8%
- **Housing**: Rented/Parents = 12% vs Owners = 7.8%
""")

st.markdown("---")

# === 3. FINANCIAL STRESS + TREND LINE ===
st.header("💰 Financial Stress Analysis")
col1, col2 = st.columns(2)

with col1:
    credit_def = df.groupby('CREDIT_BIN', observed=True)['TARGET'].mean() * 100
    fig_credit = px.bar(
        x=credit_def.index.astype(str), 
        y=credit_def.values, 
        title="Credit/Income Ratio vs Default Rate",
        labels={'x': 'Credit/Income Multiple', 'y': 'Default Rate (%)'},
        color_discrete_sequence=['#dc3545']
    )
    fig_credit.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_credit, width='stretch')

with col2:
    emi_def = df.groupby('EMI_BIN', observed=True)['TARGET'].mean() * 100
    fig_emi = px.bar(
        x=emi_def.index.astype(str), 
        y=emi_def.values,
        title="EMI/Income Ratio vs Default Rate", 
        labels={'x': 'EMI as % of Income', 'y': 'Default Rate (%)'},
        color_discrete_sequence=['#fd7e14']
    )
    fig_emi.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_emi, width='stretch')

# NEW: Trend Line (Cell 17)
st.subheader("📈 Default Trend Across Financial Stress")
trend_df = df.groupby('CREDIT_BIN', observed=True)['TARGET'].mean().reset_index()
fig_trend = px.line(
    trend_df, 
    x='CREDIT_BIN', 
    y='TARGET',
    markers=True,
    title="Default Rate Trend: Credit Stress Buckets",
    labels={'CREDIT_BIN': 'Credit/Income Stress Group', 'TARGET': 'Default Rate (%)'},
    color_discrete_sequence=['#dc3545']
)
fig_trend.update_traces(line_shape='linear', marker=dict(size=10, color='#dc3545'))
fig_trend.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_trend, width='stretch')

st.markdown("""
**🔍 Insights:**
- **Credit 3-5x income**: Highest default peak (~12%)
- **EMI 10-20% income**: 47% of defaulters (moderate stress = biggest risk)
- **Trend**: Default risk increases steadily with financial pressure
""")

st.markdown("---")

# === 4. BUBBLE CHART (NEW - Cell 18) ===
st.header("🎯 External Score vs Behavioral Risk")
bubble_df = df[['EXT2_Q', 'FLAG_EVER_REFUSED', 'TARGET']].dropna()

bubble_group = bubble_df.groupby(['EXT2_Q', 'FLAG_EVER_REFUSED'])['TARGET'].mean() * 100
bubble_count = bubble_df.groupby(['EXT2_Q', 'FLAG_EVER_REFUSED'])['TARGET'].count()

bubble_final = pd.DataFrame({
    'DefaultRate': bubble_group,
    'Count': bubble_count,
    'EXT2_Q': bubble_group.index.get_level_values(0),
    'FLAG_EVER_REFUSED': bubble_group.index.get_level_values(1)
}).reset_index(drop=True)

fig_bubble = px.scatter(
    bubble_final,
    x='EXT2_Q',
    y='DefaultRate',
    size='Count',
    size_max=60,
    color='FLAG_EVER_REFUSED',
    color_continuous_scale=['#28a745', '#dc3545'],
    title="External Score vs Behavior (Bubble Size = # Customers)",
    labels={
        'EXT2_Q': 'External Score Quartile (Higher = Safer)',
        'DefaultRate': 'Default Rate (%)',
        'Count': 'Customer Count'
    },
    hover_data=['Count']
)
fig_bubble.update_traces(opacity=0.8)
st.plotly_chart(fig_bubble, width='stretch')

st.markdown("""
**🔍 Insights:**
- **Large bubbles** = high customer volume
- **Low EXT scores + prior refusals** = highest risk combinations
- **Size shows scale**: Low-score refused customers most dangerous
""")

st.markdown("---")

# === 5. BEHAVIORAL RED FLAGS ===
st.header("🚩 Behavioral Red Flags")
col1, col2 = st.columns(2)

with col1:
    refuse_def = df.groupby('FLAG_EVER_REFUSED')['TARGET'].mean() * 100
    fig_refuse = px.bar(
        x=['Never Refused', 'Had Refusal'], 
        y=refuse_def.values,
        title="Previous Refusal History",
        labels={'x': 'Refusal History', 'y': 'Default Rate (%)'},
        color_discrete_sequence=['#dc3545']
    )
    fig_refuse.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_refuse, width='stretch')

with col2:
    apps_def = df.groupby('PREV_APPS_BIN', observed=True)['TARGET'].mean() * 100
    fig_apps = px.bar(
        x=apps_def.index.astype(str), 
        y=apps_def.values,
        title="Previous Applications", 
        labels={'x': 'Number of Previous Apps', 'y': 'Default Rate (%)'},
        color_discrete_sequence=['#e83e8c']
    )
    fig_apps.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_apps, width='stretch')

st.markdown("""
**🔍 Insights:**
- **Previous refusal**: 10.3% vs 7% (48% higher risk)
- **10+ applications**: 9.9% default (credit shopping = stress signal)
- **Rule**: Manual review for refused history, reject 10+ apps
""")

st.markdown("---")

# === 6. EXTERNAL CREDIT SCORES ===
st.header("⭐ External Credit Scores")

# Bar chart on left + Insights on right
col1, col2 = st.columns([2, 1])

with col1:
    ext2_def = df.groupby('EXT2_Q')['TARGET'].mean() * 100
    fig_ext2 = px.bar(
        x=ext2_def.index.astype(str), 
        y=ext2_def.values,
        title="EXT_SOURCE_2 Quartiles",
        labels={'x': 'External Score Quartile', 'y': 'Default Rate (%)'},
        color_discrete_sequence=['#6f42c1']
    )
    fig_ext2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_ext2, width='stretch')

with col2:
    st.markdown("""
    **🔍 Insights:**
    
    **Q1 (lowest scores)**: 12% default  
    **Q4 (highest)**: 3% default
    
    **Monotonic gradient**:  
    Higher external scores = exponentially safer
    
    **Gold standard** predictor
    """)

# FULL WIDTH HISTOGRAM BELOW
st.markdown("---")
fig_hist = px.histogram(
    df, 
    x='EXT_SOURCE_2', 
    color='TARGET',
    nbins=50,
    marginal='violin',
    title="EXT_SOURCE_2 Distribution by Default Status",
    labels={'EXT_SOURCE_2': 'External Credit Score', 'count': 'Number of Applicants'},
    color_discrete_map={0: '#28a745', 1: '#dc3545'}
)
fig_hist.update_traces(marker_line_width=1.5, marker_line_color='black')
fig_hist.update_layout(height=500, legend_title="Default Status")
st.plotly_chart(fig_hist, width='stretch')


st.markdown("---")

# === 7. COMBINED RISK SCORE ===
st.header("🎯 Combined Risk Score")
risk_def = df.groupby('RISK_SCORE')['TARGET'].mean() * 100
fig_risk = px.bar(
    x=risk_def.index.astype(str), 
    y=risk_def.values,
    title="Default Rate by Composite Risk Score",
    labels={'x': 'Risk Score (0=Safe, Higher=Riskier)', 'y': 'Default Rate (%)'},
    color_discrete_sequence=['#fd7e14']
)
fig_risk.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_risk, width='stretch')

st.markdown("""
**🔍 Insights:**
- **Risk Score 0**: Safest bucket
- **Risk Score 6+**: Extreme danger zone
- **Perfect stratification**: Higher score = exponentially higher default
""")

st.markdown("---")

# === 8. RADAR CHART ===
st.header("📈 Risk Profile Comparison")
st.markdown("*Defaulters (🔴) vs Non-Defaulters (🟢) across 6 risk dimensions*")

radar_cols = ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'FLAG_EVER_REFUSED', 
              'EXT_SOURCE_2', 'AGE_YEARS', 'INCOME_PER_PERSON']
radar_cols = [c for c in radar_cols if c in df.columns]

if len(radar_cols) >= 3:
    radar_raw = df.groupby('TARGET')[radar_cols].mean()
    radar_norm = pd.DataFrame(index=radar_cols)
    
    for col in radar_cols:
        col_min = radar_raw[col].min()
        col_max = radar_raw[col].max()
        radar_norm.loc[col, 0] = (radar_raw.loc[0, col] - col_min) / (col_max - col_min + 1e-9)
        radar_norm.loc[col, 1] = (radar_raw.loc[1, col] - col_min) / (col_max - col_min + 1e-9)
    
    radar_norm = radar_norm.astype(float)
    
    fig_radar = go.Figure()
    
    values0 = radar_norm[0].tolist() + radar_norm[0].tolist()[:1]
    fig_radar.add_trace(go.Scatterpolar(
        r=values0, theta=radar_cols + [radar_cols[0]],
        fill='toself', name='🟢 Non-Defaulters',
        line_color='green', line=dict(width=4), fillcolor='rgba(0,255,0,0.2)'
    ))
    
    values1 = radar_norm[1].tolist() + radar_norm[1].tolist()[:1]
    fig_radar.add_trace(go.Scatterpolar(
        r=values1, theta=radar_cols + [radar_cols[0]],
        fill='toself', name='🔴 Defaulters',
        line_color='red', line=dict(width=4), fillcolor='rgba(255,0,0,0.2)'
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
        showlegend=True, title="Risk Profile Comparison", height=650
    )
    st.plotly_chart(fig_radar, width='stretch')
    
    st.markdown("""
    **🔍 Insights:**
    - Defaulters cluster in **high stress/low score** zone
    - Non-defaulters have **balanced low-risk profile**
    - Each spoke normalized separately (0=best, 1=worst per feature)
    """)

st.markdown("---")
st.caption("🎉 Production dashboard | 50k sample | Auto-download from Google Drive")
