# pages/Map.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from utils import load_data # Assuming utils.py correctly loads imputed_data.csv

# --- Page Configuration and Custom Styling (Consistent with other pages) ---
st.set_page_config(page_title="🌍 Africa ESG Map", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Roboto Mono', monospace;
    }

    .fade-in {
        animation: fadeIn ease 1.5s;
        opacity: 0;
        transform: translateY(20px);
        animation-fill-mode: forwards;
    }

    @keyframes fadeIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        color: #00eaff;
        text-shadow: 0 0 8px #00eaff99;
    }

    p.subtitle {
        color: #88ccee;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }

    .stSelectbox > label, .stSlider > label, .stNumberInput > label {
        color: #00eaff;
        font-weight: bold;
        letter-spacing: 0.5px;
    }

    /* Adjust Streamlit specific elements for dark theme consistency */
    .stSelectbox > div > div, .stSlider > div > div > div, .stNumberInput > div > div {
        background-color: #1a1a1a;
        border: 1px solid #00eaff;
        border-radius: 5px;
    }
    .stSelectbox > div > div > div > span, .stNumberInput > div > div > input {
        color: #ffffff;
    }
    .stSlider > div > div > div > div > div > div {
        background-color: #00eaff;
    }
    .stSlider > div > div > div > div {
        background-color: #333;
    }
    .stButton > button {
        background-color: #00eaff;
        color: #0e1117;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #00fff7;
        color: #0e1117;
    }

    @media screen and (max-width: 768px) {

        h1 {
            font-size: 1.8rem;
        }

        p.subtitle {
            font-size: 1rem;
        }
    }
    </style>

""", unsafe_allow_html=True)

st.markdown("""
<div class='fade-in'>
    <h1>🗺️ Africa ESG Map Visualization</h1>
    <p class='subtitle'>Explore ESG indicator values across African countries.</p>
</div>
""", unsafe_allow_html=True)


# --- Load and Preprocess Data ---
df = load_data()

# Ensure required columns for ESG variables are numeric
# Coerce to numeric, converting non-numeric values to NaN
for col in df.columns:
    if col not in ['Country', 'Year']:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# ESG variable groups (copied for consistency across pages)
env_vars = ['Adjusted savings: net forest depletion (% of GNI)',
            'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
            'Agriculture, forestry, and fishing, value added (% of GDP)',
            'Level of water stress: freshwater withdrawal as a proportion of available freshwater resources',
            'Fossil fuel energy consumption (% of total)']

soc_vars = ['School enrollment, primary (% gross)',
            'Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total)',
            'Annualized average growth rate in per capita real survey mean consumption or income, total population (%)',
            'Poverty headcount ratio at national poverty lines (% of population)',
            'Unmet need for contraception (% of married women ages 15-49)']

gov_vars = ['Economic and Social Rights Performance Score',
            'Voice and Accountability: Estimate',
            'Gini index',
            'Control of Corruption: Estimate',
            'School enrollment, primary and secondary (gross), gender parity index (GPI)']
esg_vars = env_vars + soc_vars + gov_vars

# Filter out ESG variables that are not in the DataFrame
esg_vars = [col for col in esg_vars if col in df.columns]
env_vars = [col for col in env_vars if col in df.columns]
soc_vars = [col for col in soc_vars if col in df.columns]
gov_vars = [col for col in gov_vars if col in df.columns]

if not esg_vars:
    st.error("No valid ESG indicator columns found in the loaded data for mapping. Please check 'imputed_data.csv'.")
    st.stop()

# Normalize function (re-defined as it's not imported from utils)
def normalize_columns(df_to_scale, columns, fitted_scaler=None):
    if fitted_scaler:
        df_scaled = df_to_scale.copy()
        df_scaled[columns] = fitted_scaler.transform(df_to_scale[columns])
        return df_scaled, fitted_scaler
    else:
        scaler = MinMaxScaler(feature_range=(0, 100))
        df_scaled = df_to_scale.copy()
        df_scaled[columns] = scaler.fit_transform(df_to_scale[columns])
        return df_scaled, scaler

# Composite ESG score (re-defined as it's not imported from utils)
def compute_esg_score(row, env_v, soc_v, gov_v):
    env_score = row[env_v].mean() if not row[env_v].isnull().all() else np.nan
    soc_score = row[soc_v].mean() if not row[soc_v].isnull().all() else np.nan
    gov_score = row[gov_v].mean() if not row[gov_v].isnull().all() else np.nan

    scores = [s for s in [env_score, soc_score, gov_score] if not np.isnan(s)]
    return np.mean(scores) if scores else np.nan

# Calculate composite ESG score for all relevant data
df_clean = df.dropna(subset=esg_vars + ['Year']).copy()
if not df_clean.empty:
    df_normalized, scaler = normalize_columns(df_clean, esg_vars)
    df_normalized["ESG_Score"] = df_normalized.apply(lambda row: compute_esg_score(row, env_vars, soc_vars, gov_vars), axis=1)
else:
    st.warning("No complete data available to compute ESG scores for mapping.")
    st.stop()

# --- ISO Alpha-3 Mapping for African Countries ---
# This map ensures consistency with Plotly's internal country codes.
# Add more mappings here if your dataset uses country names not covered.
country_name_to_iso_a3_map = {
    "Algeria": "DZA", "Angola": "AGO", "Benin": "BEN", "Botswana": "BWA",
    "Burkina Faso": "BFA", "Burundi": "BDI", "Cabo Verde": "CPV", "Cameroon": "CMR",
    "Central African Republic": "CAF", "Chad": "TCD", "Comoros": "COM",
    "Congo, Dem. Rep.": "COD", "Congo, Rep.": "COG", "Côte d'Ivoire": "CIV",
    "Djibouti": "DJI", "Egypt": "EGY", "Equatorial Guinea": "GNQ", "Eritrea": "ERI",
    "Eswatini": "SWZ", "Ethiopia": "ETH", "Gabon": "GAB", "Gambia": "GMB",
    "Ghana": "GHA", "Guinea": "GIN", "Guinea-Bissau": "GNB", "Kenya": "KEN",
    "Lesotho": "LSO", "Liberia": "LBR", "Libya": "LBY", "Madagascar": "MDG",
    "Malawi": "MWI", "Mali": "MLI", "Mauritania": "MRT", "Mauritius": "MUS",
    "Morocco": "MAR", "Mozambique": "MOZ", "Namibia": "NAM", "Niger": "NER",
    "Nigeria": "NGA", "Rwanda": "RWA", "Sao Tome and Principe": "STP", "Senegal": "SEN",
    "Seychelles": "SYC", "Sierra Leone": "SLE", "Somalia": "SOM", "South Africa": "ZAF",
    "South Sudan": "SSD", "Sudan": "SDN", "Tanzania": "TZA", "Togo": "TGO",
    "Tunisia": "TUN", "Uganda": "UGA", "Zambia": "ZMB", "Zimbabwe": "ZWE"
}

# Create a new column with ISO Alpha-3 codes
df_normalized['Country_ISO_A3'] = df_normalized['Country'].map(country_name_to_iso_a3_map)

# List of African ISO Alpha-3 codes to filter the data
african_iso_a3_list = list(country_name_to_iso_a3_map.values())

# Filter data to include only recognized African countries with valid ISO codes
df_africa = df_normalized[df_normalized['Country_ISO_A3'].isin(african_iso_a3_list)].copy()

if df_africa.empty:
    st.warning("No African countries with valid ISO Alpha-3 codes found in the dataset with complete ESG data after filtering.")
    st.stop()


# --- Map Filters ---
st.subheader("🗺️ Map Filters")

col1, col2 = st.columns(2)

with col1:
    # Select year for mapping
    available_years = sorted(df_africa['Year'].unique(), reverse=True)
    if available_years:
        selected_year = st.selectbox("Select Year", available_years)
    else:
        st.warning("No years available for African countries in the dataset.")
        st.stop()

with col2:
    # Select metric to visualize
    map_metrics = ['ESG_Score'] + esg_vars
    selected_map_metric = st.selectbox("Select Metric to Visualize", map_metrics)


# --- Prepare data for plotting ---
df_map_data = df_africa[df_africa['Year'] == selected_year].copy()

if df_map_data.empty:
    st.info(f"No data available for {selected_map_metric} in {selected_year} for African countries.")
else:
    # Ensure the selected metric column is present and not all NaNs for the selected year
    if selected_map_metric not in df_map_data.columns or df_map_data[selected_map_metric].isnull().all():
        st.warning(f"No valid data for '{selected_map_metric}' for African countries in {selected_year}.")
    else:
        # Create the Choropleth map using ISO-3 codes
        fig_map = px.choropleth(
            df_map_data,
            locations="Country_ISO_A3", # Use the new ISO Alpha-3 column
            locationmode="ISO-3",      # Specify that locations are ISO-3 codes
            color=selected_map_metric, # Column to color the map by
            hover_name="Country",      # Original country name to show on hover
            color_continuous_scale=px.colors.sequential.Plasma, # A nice color scale
            title=f"{selected_map_metric} in Africa ({selected_year})",
            scope="africa",            # Focus on Africa
            template="plotly_dark",    # Dark theme for the map
            height=600                 # Adjust height as needed
        )

        fig_map.update_layout(
            title_font_color="#00eaff",
            title_font_size=24,
            margin={"r":0,"l":0,"t":80,"b":0}, # Adjust margins
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='natural earth', # A good projection for maps
                bgcolor='rgba(0,0,0,0)', # Transparent background
            ),
            coloraxis_colorbar=dict(
                title=selected_map_metric,
                tickfont_color="#ffffff",
                title_font_color="#ffffff"
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

# --- Optional Footer (Consistent with other pages) ---
st.markdown("""
<hr style='border-color: #333;'/>
<p style='font-size: 0.85rem; color: #666;'>🛰️ Data visualized using AI-enhanced dashboards. Built for clarity, precision, and sustainability insights.</p>
""", unsafe_allow_html=True)