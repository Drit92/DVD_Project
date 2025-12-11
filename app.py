import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gdown
import zipfile
import os

# -----------------------------------------------
# Download & Extract Dataset from Google Drive
# -----------------------------------------------

DATA_ID = "1FSSkKQOJtvOpP1I1qyr4x6SYQg-kBnVw"   # Your file ID
ZIP_PATH = "dataset.zip"
EXTRACT_DIR = "file_zip"

def download_and_extract():
    # Create extract dir
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Download if needed
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading dataset from Google Drive...")
        gdown.download(id=DATA_ID, output=ZIP_PATH, quiet=False)

    # Extract ZIP
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Recursively search for CSV files
    csv_files = []
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        st.error("❌ No CSV file found inside the extracted ZIP folder.")
        return None
    
    # Load the first CSV found
    return csv_files[0]



# -----------------------------------------------
# UI Setup
# -----------------------------------------------
st.set_page_config(page_title="Loan Applicant Insights Dashboard", layout="wide")

st.title("📊 Loan Applicant Visual Insights Dashboard")
st.markdown("""
Dataset is automatically loaded from Google Drive  
via **gdown → ZIP extraction → CSV auto-detection**.
""")

# -----------------------------------------------
# Load Data
# -----------------------------------------------
csv_path = download_and_extract()

if csv_path:
    st.success(f"Dataset Loaded: `{csv_path}`")
    df = pd.read_csv(csv_path)

    st.write("### Preview of Data")
    st.dataframe(df.head())

    st.sidebar.header("Filters")

    # Filters
    income_slider = st.sidebar.slider(
        "Income Range",
        int(df.AMT_INCOME_TOTAL.min()),
        int(df.AMT_INCOME_TOTAL.max()),
        (int(df.AMT_INCOME_TOTAL.min()), int(df.AMT_INCOME_TOTAL.max()))
    )

    age_slider = st.sidebar.slider(
        "Age Range",
        int(df.AGE_YEARS.min()),
        int(df.AGE_YEARS.max()),
        (20, 70)
    )

    df_filtered = df[
        df["AMT_INCOME_TOTAL"].between(income_slider[0], income_slider[1]) &
        df["AGE_YEARS"].between(age_slider[0], age_slider[1])
    ]

    # ------------------------------------------------
    # 1. Income–Credit Scatter
    # ------------------------------------------------
    st.subheader("💰 Income vs Credit Patterns")
    fig1 = px.scatter(
        df_filtered,
        x="AMT_INCOME_TOTAL",
        y="AMT_CREDIT",
        color="cluster_label" if "cluster_label" in df.columns else None,
        hover_data=df.columns,
        opacity=0.7
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ------------------------------------------------
    # 2. Demographic Patterns
    # ------------------------------------------------
    st.subheader("👨‍👩‍👧 Family & Demographics Patterns")

    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.box(
            df_filtered,
            x="CNT_FAM_MEMBERS", y="AGE_YEARS",
            points="all"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.histogram(
            df_filtered,
            x="CNT_CHILDREN",
            color="CNT_FAM_MEMBERS",
            barmode="group"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ------------------------------------------------
    # 3. Source Score Patterns
    # ------------------------------------------------
    st.subheader("📈 External Stability Score Patterns")

    src_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    avail = [c for c in src_cols if c in df.columns]

    if avail:
        fig4 = px.parallel_coordinates(
            df_filtered,
            dimensions=avail,
            color="AGE_YEARS",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ------------------------------------------------
    # 4. Cluster Radar (if cluster_label exists)
    # ------------------------------------------------
    if "cluster_label" in df.columns:
        st.subheader("🧭 Cluster Personas (Radar Profile)")

        cluster_choice = st.selectbox(
            "Select Cluster:",
            sorted(df.cluster_label.unique())
        )
        df_cluster = df[df.cluster_label == cluster_choice]

        radar_features = [
            "AMT_INCOME_TOTAL", "AMT_CREDIT",
            "EMPLOY_YEARS", "AGE_YEARS",
            "CNT_FAM_MEMBERS", "CNT_CHILDREN"
        ]

        means = df_cluster[radar_features].mean()

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=means.values,
            theta=radar_features,
            fill='toself',
            name=f"Cluster {cluster_choice}"
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    # ------------------------------------------------
    # 5. Correlation Heatmap
    # ------------------------------------------------
    st.subheader("🔗 Correlation Explorer")

    numeric_cols = df.select_dtypes(include="number")

    fig5 = px.imshow(
        numeric_cols.corr(),
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig5, use_container_width=True)

else:
    st.error("Dataset could not be loaded.")
