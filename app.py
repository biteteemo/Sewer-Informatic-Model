import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import geopandas as gpd
import zipfile
import tempfile
import os
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Urban Sewer Model")

# =====================================================
# 1. 工具函数：读取建筑 SHP（稳定版，强制 Fiona）
# =====================================================

@st.cache_data(show_spinner=False)
def load_building_shp(uploaded_zip):
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        shp_path = None
        for root, _, files in os.walk(tmpdir):
            for f in files:
                if f.lower().endswith(".shp"):
                    shp_path = os.path.join(root, f)
                    break

        if shp_path is None:
            raise ValueError("No .shp file found in zip")

        gdf = gpd.read_file(shp_path, engine="fiona")
        gdf = gdf[gdf.geometry.notnull()]
        gdf = gdf[gdf.is_valid]

        return gdf


# =====================================================
# 2. 管道数据处理
# =====================================================

@st.cache_data
def process_pipe_data(df):
    df = df.copy()

    df['UpstreamNode'] = df['UpstreamNode'].astype(str)
    df['DownstreamNode'] = df['DownstreamNode'].astype(str)

    if 'inflow_baseline' not in df.columns:
        df['inflow_baseline'] = 0.0

    if 'COD_load' not in df.columns:
        df['COD_load'] = 0.0

    df['Mid_X'] = (df['US_X'] + df['DS_X']) / 2
    df['Mid_Y'] = (df['US_Y'] + df['DS_Y']) / 2

    return df


# =====================================================
# 3. 建筑人口 / 流量 / COD 生成器
# =====================================================

def generate_building_loads(
    gdf_bldg,
    df_pipe,
    pop_density,
    water_lpd,
    wastewater_ratio,
    cod_gpd,
    max_dist
):
    gdf = gdf_bldg.copy()

    # ---------- 人口 ----------
    if 'population' in gdf.columns:
        gdf['population_calc'] = gdf['population']
    elif 'POP' in gdf.columns:
        gdf['population_calc'] = gdf['POP']
    else:
        area_ha = gdf.geometry.area / 10_000
        gdf['population_calc'] = area_ha * pop_density

    # ---------- 流量 (m3/s) ----------
    gdf['flow_m3s'] = (
        gdf['population_calc']
        * water_lpd
        * wastewater_ratio
        / 1000
        / 86400
    )

    # ---------- COD (kg/d) ----------
    gdf['COD_kgd'] = gdf['population_calc'] * cod_gpd / 1000

    # ---------- 最近管道 ----------
    pipe_xy = np.column_stack([df_pipe['Mid_X'], df_pipe['Mid_Y']])
    bldg_xy = np.column_stack([
        gdf.geometry.centroid.x,
        gdf.geometry.centroid.y
    ])

    tree = cKDTree(pipe_xy)
    dist, idx = tree.query(bldg_xy, k=1)

    gdf['pipe_idx'] = idx
    gdf['dist_to_pipe'] = dist
    gdf = gdf[gdf['dist_to_pipe'] <= max_dist]

    # ---------- 汇总 ----------
    flow_sum = gdf.groupby('pipe_idx')['flow_m3s'].sum()
    cod_sum = gdf.groupby('pipe_idx')['COD_kgd'].sum()

    df_pipe = df_pipe.copy()
    df_pipe['inflow_baseline'] += df_pipe.index.map(flow_sum).fillna(0)
    df_pipe['COD_load'] += df_pipe.index.map(cod_sum).fillna(0)

    return df_pipe, gdf


# =====================================================
# 4. Plotly Network Map
# =====================================================

def create_map(df_pipe, gdf_bldg=None, show_buildings=True):
    fig = go.Figure()

    # --- 管道 ---
    x, y = [], []
    for _, r in df_pipe.iterrows():
        x += [r.US_X, r.DS_X, None]
        y += [r.US_Y, r.DS_Y, None]

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color="#bdc3c7", width=2),
        hoverinfo="skip",
        name="Pipes"
    ))

    # --- 建筑 ---
    if gdf_bldg is not None and show_buildings:
        c = gdf_bldg.geometry.centroid
        fig.add_trace(go.Scatter(
            x=c.x,
            y=c.y,
            mode="markers",
            marker=dict(
                size=np.clip(gdf_bldg['flow_m3s'] * 5000, 4, 16),
                color=gdf_bldg['COD_kgd'],
                colorscale="YlOrRd",
                colorbar=dict(title="COD (kg/d)")
            ),
            name="Buildings",
            hovertemplate=(
                "Population: %{customdata[0]:.1f}<br>"
                "Flow: %{customdata[1]:.5f} m³/s<br>"
                "COD: %{customdata[2]:.2f} kg/d<extra></extra>"
            ),
            customdata=np.column_stack([
                gdf_bldg['population_calc'],
                gdf_bldg['flow_m3s'],
                gdf_bldg['COD_kgd']
            ])
        ))

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
        title="Sewer Network with Building Loads",
        xaxis=dict(scaleanchor="y"),
        plot_bgcolor="white"
    )
    return fig


# =====================================================
# 5. UI
# =====================================================

st.title("🏙️ Urban Sewer Network – Building Load Generator")

with st.sidebar:
    st.header("1️⃣ Upload Pipe CSV")
    pipe_file = st.file_uploader("Pipe Network CSV", type="csv")

    st.header("2️⃣ Upload Building SHP")
    bldg_file = st.file_uploader("Building SHP (zip)", type="zip")
    show_buildings = st.toggle("Show Buildings", True)

    st.header("3️⃣ Load Parameters")
    pop_density = st.slider("Population Density (person/ha)", 50, 300, 120)
    water_lpd = st.slider("Water Use (L/person/day)", 100, 300, 180)
    wastewater_ratio = st.slider("Wastewater Ratio", 0.6, 1.0, 0.85)
    cod_gpd = st.slider("COD Generation (g/person/day)", 60, 150, 110)
    max_dist = st.slider("Max Connect Distance (m)", 10, 200, 50)

    apply_loads = st.button("🚀 Generate Building Loads")


# =====================================================
# 6. 主逻辑
# =====================================================

if pipe_file:
    df_pipe = process_pipe_data(pd.read_csv(pipe_file))

    gdf_bldg = None
    if bldg_file:
        gdf_bldg = load_building_shp(bldg_file)

    if apply_loads and gdf_bldg is not None:
        df_pipe, gdf_bldg = generate_building_loads(
            gdf_bldg,
            df_pipe,
            pop_density,
            water_lpd,
            wastewater_ratio,
            cod_gpd,
            max_dist
        )
        st.success("✅ Building loads successfully applied")

    fig = create_map(df_pipe, gdf_bldg, show_buildings)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Please upload a pipe network CSV to start.")
