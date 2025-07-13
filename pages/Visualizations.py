# pages/Map.py – ESG Dashboard without SHAP

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

try:
    from fpdf import FPDF  # lightweight PDF generator
    from io import BytesIO
except ImportError:
    FPDF = None
    BytesIO = None

# Page configuration and styles
st.set_page_config(page_title="🌍 Africa ESG Dashboard", layout="wide")

st.markdown(
    """
    <style>

    h1, h2, h3, h4, h5, h6 {
        color: #d002f0; /* Bright cyan for headings */
        font-weight: 600;
        text-shadow: 0 0 5px rgba(0, 234, 255, 0.5);
    }

    .stButton>button {
        background-color: #0f4c75; /* Darker blue for buttons */
        color: #e0e0e0;
        border-radius: 8px;
        border: 1px solid #3282b8;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3282b8; /* Lighter blue on hover */
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 234, 255, 0.3);
    }

    .stRadio > label > div {
        color: #88ccee; /* Lighter blue for radio options */
        font-weight: 500;
    }

    .stSelectbox > label, .stMultiSelect > label, .stSlider > label {
        color: #88ccee;
        font-weight: 500;
    }

    .stMarkdown p {
        font-size: 1.05rem;
        line-height: 1.6;
    }

    .stInfo {
        background-color: rgba(0, 234, 255, 0.1);
        border-left: 5px solid #00eaff;
        padding: 10px;
        border-radius: 5px;
    }
    .stSuccess {
        background-color: rgba(60, 179, 113, 0.1);
        border-left: 5px solid #3cb371;
        padding: 10px;
        border-radius: 5px;
    }
    .stWarning {
        background-color: rgba(255, 165, 0, 0.1);
        border-left: 5px solid #ffa500;
        padding: 10px;
        border-radius: 5px;
    }
    .stError {
        background-color: rgba(255, 69, 0, 0.1);
        border-left: 5px solid #ff4500;
        padding: 10px;
        border-radius: 5px;
    }

    .fade-in {
        animation: fadeIn 1.5s ease forwards;
        opacity: 0;
        transform: translateY(20px);
    }
    @keyframes fadeIn {
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="fade-in">
        <h1>🗜️ Africa ESG Dashboard</h1>
        <p class="subtitle">Map & country-level ESG analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_and_preprocess_data(file_path):
    df_raw = pd.read_csv(file_path)
    if df_raw.empty:
        st.error("`fss.csv` returned an empty DataFrame.")
        st.stop()
    esg_display_vars = [c for c in df_raw.columns if c not in ["Country", "Year"] and not c.endswith('_norm')]
    if "FSS" not in esg_display_vars:
        st.error("Composite ESG score column `FSS` missing in dataset.")
        st.stop()
    for col in esg_display_vars:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    df = df_raw.copy()
    scaler = MinMaxScaler((0, 100))
    df[esg_display_vars] = scaler.fit_transform(df[esg_display_vars])
    return df, esg_display_vars

df, esg_display_vars = load_and_preprocess_data('fss.csv')
esg_model_features = [col for col in esg_display_vars if col != "FSS"]

country_name_to_iso = {"Algeria":"DZA","Angola":"AGO","Benin":"BEN","Botswana":"BWA","Burkina Faso":"BFA","Burundi":"BDI","Cabo Verde":"CPV","Cameroon":"CMR","Central African Republic":"CAF","Chad":"TCD","Comoros":"COM","Congo, Dem. Rep.":"COD","Congo, Rep.":"COG","Côte d'Ivoire":"CIV","Cote d'Ivoire":"CIV","Djibouti":"DJI","Egypt":"EGY","Egypt, Arab Rep.":"EGY","Equatorial Guinea":"GNQ","Eritrea":"ERI","Eswatini":"SWZ","Ethiopia":"ETH","Gabon":"GAB","Gambia":"GMB","Gambia, The":"GMB","Ghana":"GHA","Guinea":"GIN","Guinea-Bissau":"GNB","Kenya":"KEN","Lesotho":"LSO","Liberia":"LBR","Libya":"LBY","Madagascar":"MDG","Malawi":"MWI","Mali":"MLI","Mauritania":"MRT","Mauritius":"MUS","Morocco":"MAR","Mozambique":"MOZ","Namibia":"NAM","Niger":"NER","Nigeria":"NGA","Rwanda":"RWA","Sao Tome and Principe":"STP","Senegal":"SEN","Seychelles":"SYC","Sierra Leone":"SLE","Somalia":"SOM","South Africa":"ZAF","South Sudan":"SSD","Sudan":"SDN","Tanzania":"TZA","Togo":"TGO","Tunisia":"TUN","Uganda":"UGA","Zambia":"ZMB","Zimbabwe":"ZWE"}
df["ISO"] = df["Country"].map(country_name_to_iso)
df_africa = df[df["ISO"].isin(country_name_to_iso.values())].copy()
if df_africa.empty:
    st.error("No African ISO codes found or data for mapped countries is empty.")
    st.stop()

st.subheader("🗜️ Choropleth Map")
col_map_year, col_map_metric = st.columns(2)
with col_map_year:
    year_sel = st.selectbox("Year", sorted(df_africa["Year"].astype(int).unique(), reverse=True))
with col_map_metric:
    metric_sel = st.selectbox("Metric", esg_display_vars)

map_df = df_africa[df_africa["Year"] == year_sel]
if not map_df.empty and metric_sel in map_df.columns:
    fig = px.choropleth(map_df, locations="ISO", locationmode="ISO-3", color=metric_sel,
                        hover_name="Country", color_continuous_scale="Plasma", scope="africa",
                        title=f"{metric_sel} – {year_sel}", template="seaborn", height=600)
    fig.update_layout(margin=dict(l=0, r=0, t=80, b=0))
    st.plotly_chart(fig, use_container_width=True)

overall_tab, country_tab, corr_tab = st.tabs(["🌍 Overall Summary", "📌 Country Breakdown", "📈 Correlation / Trends"])

with overall_tab:
    st.header("Overview of ESG Indicators")
    st.write("Select and explore trends in the tabs.")

with country_tab:
    st.header("Country-Specific ESG Breakdown")
    ctry = st.selectbox("Country", sorted(df_africa["Country"].unique()))
    c_df = df_africa[df_africa["Country"] == ctry].sort_values("Year")
    if not c_df.empty:
        fig_fss = px.line(c_df, x="Year", y="FSS", markers=True,
                          title=f"{ctry} – Composite ESG Score (FSS) over time (0-100)",
                          template="plotly_dark")
        st.plotly_chart(fig_fss, use_container_width=True)
        latest = c_df.iloc[-1]
        radar_vals = latest[esg_model_features].tolist()
        radar_lbls = esg_model_features
        if radar_vals:
            radar_vals += [radar_vals[0]]
            radar_lbls += [radar_lbls[0]]
        else:
            radar_vals = [0, 0]
            radar_lbls = ["N/A", "N/A"]
        fig_c_radar = go.Figure(go.Scatterpolar(r=radar_vals, theta=radar_lbls,
                                                mode="lines", fill="toself", line=dict(color="orange")))
        fig_c_radar.update_layout(title=f"{ctry} – ESG Snapshot ({int(latest['Year'])}) (0-100)",
                                  polar=dict(radialaxis=dict(range=[0, 100])),
                                  template="plotly_dark", showlegend=False, height=500)
        st.plotly_chart(fig_c_radar, use_container_width=True)

with corr_tab:
    st.header("Correlation Matrix & Trendlines")
    sel_year = st.slider("Select year for correlation", int(df_africa["Year"].min()), int(df_africa["Year"].max()), int(df_africa["Year"].max()))
    corr_df = df_africa[df_africa["Year"] == sel_year][esg_display_vars].corr()
    fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r",
                          title=f"Correlation Matrix – {sel_year}")
    fig_corr.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.subheader("Trendlines for selected indicators")
    sel_inds = st.multiselect("Choose indicators", esg_display_vars, default=[esg_display_vars[0]] if esg_display_vars else [])
    if sel_inds:
        tr_df = df_africa.groupby("Year")[sel_inds].mean().reset_index()
        fig_tr = px.line(tr_df, x="Year", y=sel_inds, markers=True,
                          template="plotly_dark", title="Average trend across Africa (0-100)")
        st.plotly_chart(fig_tr, use_container_width=True)
    else:
        st.info("Please select at least one indicator to display trendlines.")

st.markdown(
    """
    <hr style='border-color:#333' />
    <p style='font-size:0.85rem;color:#666'>🚀 Built for clarity, precision & sustainability insights.</p>
    """,
    unsafe_allow_html=True,
)
