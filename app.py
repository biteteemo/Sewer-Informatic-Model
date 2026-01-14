import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import warnings
import plotly.graph_objects as go
import geopandas as gpd
import zipfile
import tempfile
import os

# ==========================================
# 0. 配置与初始化
# ==========================================
st.set_page_config(page_title="Urban Sewer Simulation (HRT Tracker)", layout="wide")

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. 建筑 SHP 读取函数
# ==========================================

@st.cache_data
def load_building_shp(uploaded_zip, target_crs=None):
    """读取 zip 格式的建筑 shp，并返回 GeoDataFrame"""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "bldg.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
        if not shp_files:
            return None

        gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))

        if target_crs is not None:
            gdf = gdf.to_crs(target_crs)

    return gdf


# ==========================================
# 2. 原有数据处理与模型（未改动）
# ==========================================

@st.cache_data
def process_uploaded_data(df):
    col_map = {
        'name': 'PipeID', 'start': 'UpstreamNode', 'end': 'DownstreamNode',
        'length': 'Length', 'diameter': 'Diameter', 'slope': 'Slope',
        'us_x': 'US_X', 'us_y': 'US_Y', 'ds_x': 'DS_X', 'ds_y': 'DS_Y',
        'inflow_baseline': 'inflow_baseline'
    }
    df = df.rename(columns=col_map)

    required = ['PipeID', 'UpstreamNode', 'DownstreamNode', 'Length', 'Diameter', 'Slope']
    if any(c not in df.columns for c in required):
        return None

    df['UpstreamNode'] = df['UpstreamNode'].astype(str)
    df['DownstreamNode'] = df['DownstreamNode'].astype(str)
    df['Slope'] = df['Slope'].clip(lower=0.001)
    if 'Manning' not in df.columns:
        df['Manning'] = 0.013

    if 'inflow_baseline' not in df.columns:
        df['inflow_baseline'] = 0.0
    else:
        df['inflow_baseline'] = df['inflow_baseline'].fillna(0.0)

    if 'US_X' in df.columns and 'DS_X' in df.columns:
        df['Mid_X'] = (df['US_X'] + df['DS_X']) / 2
        df['Mid_Y'] = (df['US_Y'] + df['DS_Y']) / 2

    return df


@st.cache_data
def build_graph(df_pipe):
    G = nx.DiGraph()
    for _, row in df_pipe.iterrows():
        G.add_edge(row['UpstreamNode'], row['DownstreamNode'],
                   pipe_id=row['PipeID'], length=row['Length'])
    return G


# ==========================================
# 3. Network Map（加入建筑 Polygon）
# ==========================================

def create_interactive_map(df_pipe, gdf_buildings=None, show_buildings=True):
    fig = go.Figure()

    # --- 管道 ---
    x_lines, y_lines = [], []
    for _, row in df_pipe.iterrows():
        x_lines += [row['US_X'], row['DS_X'], None]
        y_lines += [row['US_Y'], row['DS_Y'], None]

    fig.add_trace(go.Scatter(
        x=x_lines, y=y_lines,
        mode='lines',
        line=dict(color='#bdc3c7', width=2),
        hoverinfo='skip',
        name='Pipes'
    ))

    fig.add_trace(go.Scatter(
        x=df_pipe['Mid_X'], y=df_pipe['Mid_Y'],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Pipes',
        customdata=df_pipe.index,
        hovertemplate="Pipe %{customdata}<extra></extra>"
    ))

    # --- 建筑（Polygon → Centroid）---
    if gdf_buildings is not None and show_buildings:
        centroids = gdf_buildings.geometry.centroid

        hover_text = []
        for idx, row in gdf_buildings.iterrows():
            attrs = "<br>".join(
                [f"{c}: {row[c]}" for c in gdf_buildings.columns
                 if c != "geometry"][:5]
            )
            hover_text.append(attrs)

        fig.add_trace(go.Scatter(
            x=centroids.x,
            y=centroids.y,
            mode='markers',
            marker=dict(size=4, color='rgba(52,152,219,0.6)'),
            name='Buildings',
            text=hover_text,
            hovertemplate="%{text}<extra></extra>"
        ))

    fig.update_layout(
        title="Network Map (Pipes + Buildings)",
        height=600,
        showlegend=True,
        hovermode='closest',
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(scaleanchor="y"),
        plot_bgcolor="white"
    )
    return fig


# ==========================================
# 4. Streamlit 界面
# ==========================================

st.title("🏙️ Urban Drainage Network Simulation (HRT Tracker)")

with st.sidebar:
    st.header("1️⃣ Upload Pipe CSV")
    pipe_file = st.file_uploader("Pipe Network CSV", type=["csv"])

    st.header("2️⃣ Upload Building SHP")
    bldg_file = st.file_uploader("Building SHP (zip)", type=["zip"])
    show_buildings = st.toggle("Show Buildings", value=True)

    st.header("3️⃣ Simulation")
    sim_hours = st.slider("Duration (hours)", 24, 168, 48, 12)

# ==========================================
# 5. 主逻辑
# ==========================================

if pipe_file:
    df_pipe = process_uploaded_data(pd.read_csv(pipe_file))

    gdf_bldg = None
    if bldg_file:
        # ⚠️ 这里假设和管道是同一投影
        gdf_bldg = load_building_shp(bldg_file)

    if df_pipe is not None:
        fig = create_interactive_map(
            df_pipe,
            gdf_buildings=gdf_bldg,
            show_buildings=show_buildings
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Upload pipe CSV to start.")
