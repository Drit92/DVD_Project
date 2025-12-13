import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from pipeline import engineer_features, load_raw_data

st.set_page_config(page_title="🏦 Home Credit Risk Dashboard", layout="wide")

st.title("🏦 Home Credit Risk Dashboard")
st.markdown("---")

@st.cache_data(show_spinner="🔄 Loading + Engineering Data...")
def load_data():
    """Fixed: Single cached function"""
    app_data, prev_data = load_raw_data()
    return engineer_features(app_data, prev_data)

df = load_data()
st.success(f"✅ Loaded {len(df):,} records | 🔴 Default Rate: {df['TARGET'].mean():.1%}")

# === 1. PORTFOLIO OVERVIEW ===
col1, col2, col3 = st.columns(3)
col1.metric("Total Applicants", f"{len(df):,}")
col2.metric("Defaulters", f"{df['TARGET'].sum():,}", f"{df['TARGET'].mean():.1%}")
col3.metric("Good Borrowers", f"{(1-df['TARGET']).sum():,}")

st.markdown("---")

# === 2. RISK BY KEY SEGMENTS ===
st.header("📊 Risk by Demographics")
fig_dem = make_subplots(rows=2, cols=2, 
                       subplot_titles=('Education', 'Income Type', 'Family Status', 'Housing'),
                       specs=[[{"type": "bar"}, {"type": "bar"}], 
                              [{"type": "bar"}, {"type": "bar"}]])

cat_cols = ['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']
for i, col in enumerate(cat_cols):
    if col in df.columns:
        r, c = divmod(i, 2)
        r += 1; c += 1
        tmp = df.groupby(col)['TARGET'].mean().reset_index()
        tmp['TARGET'] = tmp['TARGET'] * 100
        fig_dem.add_trace(
            px.bar(tmp, x=col, y='TARGET', color='TARGET', 
                   color_continuous_scale='Viridis').data[0], 
            row=r, col=c
        )

fig_dem.update_layout(height=500, showlegend=False, title="Default Rate by Demographics (%)")
st.plotly_chart(fig_dem, width='stretch')

# === 3. FINANCIAL STRESS BINS ===
st.header("💰 Financial Stress Analysis")
col1, col2 = st.columns(2)

with col1:
    credit_def = df.groupby('CREDIT_BIN', observed=True)['TARGET'].mean() * 100
    fig_credit = px.bar(x=credit_def.index.astype(str), y=credit_def.values, 
                       title="Credit/Income Ratio", color_discrete_sequence=['#d62728'])
    fig_credit.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_credit, width='stretch')

with col2:
    emi_def = df.groupby('EMI_BIN', observed=True)['TARGET'].mean() * 100
    fig_emi = px.bar(x=emi_def.index.astype(str), y=emi_def.values, 
                    title="EMI/Income Ratio", color_discrete_sequence=['#ff7f0e'])
    fig_emi.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_emi, width='stretch')

# === 4. RISK SCORE ===
st.header("🎯 Combined Risk Score")
risk_def = df.groupby('RISK_SCORE')['TARGET'].mean() * 100
fig_risk = px.bar(x=risk_def.index.astype(str), y=risk_def.values, 
                 title="Default Rate by Risk Score (0=Safe → Higher=Risky)",
                 color_discrete_sequence=['#9467bd'])
fig_risk.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_risk, width='stretch')

# === 5. BEHAVIORAL RISKS ===
st.header("🚩 Behavioral Red Flags")
col1, col2 = st.columns(2)

with col1:
    refuse_def = df.groupby('FLAG_EVER_REFUSED')['TARGET'].mean() * 100
    fig_refuse = px.bar(x=['Never Refused', 'Had Refusal'], y=refuse_def.values,
                       title="Previous Refusal History", color_discrete_sequence=['#2ca02c', '#d62728'])
    fig_refuse.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_refuse, width='stretch')

with col2:
    apps_def = df.groupby('PREV_APPS_BIN', observed=True)['TARGET'].mean() * 100
    fig_apps = px.bar(x=apps_def.index.astype(str), y=apps_def.values,
                     title="Previous Applications", color_discrete_sequence=['#1f77b4'])
    fig_apps.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_apps, width='stretch')

# === 6. EXTERNAL SCORES ===
st.header("⭐ External Credit Scores")
col1, col2 = st.columns(2)

with col1:
    ext2_def = df.groupby('EXT2_Q')['TARGET'].mean() * 100
    fig_ext2 = px.bar(x=ext2_def.index.astype(str), y=ext2_def.values,
                     title="EXT_SOURCE_2 Quartiles", color_discrete_sequence=['#17becf'])
    fig_ext2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_ext2, width='stretch')

with col2:
    fig_kde = px.histogram(df, x='EXT_SOURCE_2', color='TARGET', marginal='violin',
                          title="EXT_SOURCE_2 Distribution by Default", nbins=50)
    st.plotly_chart(fig_kde, width='stretch')

# === 7. FIXED RADAR CHART ===
st.header("📈 Risk Profile Comparison")
st.markdown("*Defaulters (red) vs Non-Defaulters (green) across 6 risk dimensions*")

# FIXED: Safe column selection + integer indexing
radar_cols = ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'FLAG_EVER_REFUSED', 
              'EXT_SOURCE_2', 'AGE_YEARS', 'INCOME_PER_PERSON']
radar_cols = [c for c in radar_cols if c in df.columns]

if len(radar_cols) > 0:
    radar_data = df.groupby('TARGET')[radar_cols].mean()
    radar_norm = radar_data.apply(lambda x: (x-x.min())/(x.max()-x.min()+1e-9), axis=1)
    
    fig_radar = go.Figure()
    
    # FIXED: Use column names instead of integer indexing
    for target_name, target_val in [('Non-Defaulters', 0), ('Defaulters', 1)]:
        if target_val in radar_norm.columns:
            values = radar_norm[target_val].tolist()
            values += values[:1]  # Close the polygon
            angles = np.linspace(0, 2*np.pi, len(radar_norm)+1)
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_norm.index.tolist() + [radar_norm.index[0]],
                fill='toself',
                name=target_name,
                line_color='green' if target_val == 0 else 'red',
                line=dict(width=3),
                opacity=0.7
            ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Risk Profile: Defaulters vs Non-Defaulters",
        height=500
    )
    st.plotly_chart(fig_radar, width='stretch')
else:
    st.warning("Radar columns not available")

# === 8. INTERACTION HEATMAP ===
st.header("🔥 Risk Interactions")
if 'CREDIT_BIN' in df.columns and 'FLAG_EVER_REFUSED' in df.columns:
    interaction = df.groupby(['CREDIT_BIN', 'FLAG_EVER_REFUSED'], observed=True)['TARGET'].mean().unstack(fill_value=0) * 100
    interaction.columns = interaction.columns.astype(str)
    interaction.index = interaction.index.astype(str)
    
    fig_heat = px.imshow(
        interaction.values, 
        x=interaction.columns, 
        y=interaction.index,
        title="Credit Stress × Previous Refusal",
        color_continuous_scale='Reds',
        labels=dict(color="Default Rate %")
    )
    fig_heat.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_heat, width='stretch')
else:
    st.warning("Interaction data not available")

st.markdown("---")
st.caption("🎉 All charts match Colab exactly | 50k sample for speed | Auto-download from Google Drive")
