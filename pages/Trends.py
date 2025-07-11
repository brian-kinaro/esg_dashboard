import streamlit as st
import pandas as pd
import plotly.express as px

# Define the load_data function, as 'utils' module is not available
@st.cache_data
def load_data():
    """Loads the CSV data into a pandas DataFrame."""
    try:
        df = pd.read_csv("gpt_analyze.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'gpt_analyze.csv' not found. Please ensure the file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Page setup
st.set_page_config(page_title="📊 ESG Trends", layout="wide")

# Custom styling: futuristic + responsive
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

    .stSelectbox > label, .stSlider > label, .stMultiSelect > label {
        color: #00eaff;
        font-weight: bold;
        letter-spacing: 0.5px;
    }

    /* Adjust Streamlit specific elements for dark theme consistency */
    .stSelectbox > div > div, .stSlider > div > div > div, .stMultiSelect > div > div {
        background-color: #1a1a1a; /* Darker background for input widgets */
        border: 1px solid #00eaff;
        border-radius: 5px;
    }
    .stSelectbox > div > div > div > span, .stMultiSelect > div > div > div > div > span {
        color: #ffffff; /* Text color in select/multiselect */
    }
    .stSlider > div > div > div > div > div > div {
        background-color: #00eaff; /* Slider handle color */
    }
    .stSlider > div > div > div > div {
        background-color: #333; /* Slider track color */
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

    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Roboto+Mono&display=swap" rel="stylesheet">

""", unsafe_allow_html=True)

# Header with animation
st.markdown("""
<div class='fade-in'>
    <h1>📈 ESG Indicator Trends</h1>
    <p class='subtitle'>Visualize historical patterns across key sustainability metrics, powered by intelligent insights.</p>
</div>
""", unsafe_allow_html=True)

# Load data
df = load_data()

# User inputs
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        # Ensure 'Country' column exists before accessing
        if "Country" in df.columns:
            country = st.selectbox("🌍 Select Country", sorted(df["Country"].unique()))
        else:
            st.error("Error: 'Country' column not found in data.")
            st.stop()

    with col2:
        # Ensure 'Year' column exists before accessing
        if "Year" in df.columns:
            year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
            year_range = st.slider("📅 Select Year Range", year_min, year_max, (year_min, year_max))
        else:
            st.error("Error: 'Year' column not found in data.")
            st.stop()

# Filter dataset by Country and Year range
# Assuming 'Value' column exists for plotting numeric values
if "Value" not in df.columns or "Indicators" not in df.columns:
    st.error("Error: 'Value' or 'Indicators' column not found in data. Please check your CSV file.")
    st.stop()

filtered_by_country_year = df[
    (df["Country"] == country) &
    (df["Year"] >= year_range[0]) &
    (df["Year"] <= year_range[1])
].copy() # Use .copy() to avoid SettingWithCopyWarning

# Get unique indicators available after country and year filtering
available_indicators = filtered_by_country_year["Indicators"].unique().tolist()
selected_indicators = st.multiselect(
    "📌 Choose ESG Indicators",
    available_indicators,
    default=available_indicators[:min(len(available_indicators), 3)] # Default to first 3 or fewer if available
)

# Filter further by selected indicators
filtered_final = filtered_by_country_year[
    filtered_by_country_year["Indicators"].isin(selected_indicators)
].copy()

# Dynamic charts for each selected indicator
if not filtered_final.empty and selected_indicators:
    for ind in selected_indicators:
        indicator_data = filtered_final[filtered_final["Indicators"] == ind].sort_values(by="Year")

        if not indicator_data.empty:
            fig = px.line(
                indicator_data,
                x="Year",
                y="Value", # Y-axis is always 'Value' in the long format
                title=f"📊 {ind} in {country}",
                markers=True,
                template="plotly_dark"
            )
            fig.update_layout(
                title_font_color="#00eaff",
                title_font_size=20,
                margin=dict(t=60, b=40),
                hoverlabel=dict(bgcolor="#222", font_size=12, font_color="#00eaff"),
                xaxis_title="Year",
                yaxis_title="Value" # Consistent Y-axis title
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data available for indicator: **{ind}** in the selected country and year range.")
else:
    st.info("Please select a country, year range, and at least one ESG indicator to display trends.")

# Optional footer (consistent styling)
st.markdown("""
<hr style='border-color: #333;'/>
<p style='font-size: 0.85rem; color: #666;'>🛰️ Data visualized using AI-enhanced dashboards. Built for clarity, precision, and sustainability insights.</p>
""", unsafe_allow_html=True)