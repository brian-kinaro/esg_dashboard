import streamlit as st
import pandas as pd
import plotly.express as px # Using Plotly for better integration with Streamlit and dark theme
import numpy as np
from sklearn.linear_model import LinearRegression # For forecasting
from utils import load_data

# Page Configuration
st.set_page_config(page_title="🌍 Real-Time ESG Forecasting", layout="wide")

# Custom styling: futuristic + responsive (from previous code)
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

    .floating-nav {
        position: fixed;
        top: 65px;
        left: 20px;
        z-index: 100;
        padding: 12px 20px;
        border-radius: 16px;
        box-shadow: 0 0 20px #00eaff99;
        backdrop-filter: blur(2px);
        font-family: 'Orbitron', sans-serif;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        background-color: rgba(14, 17, 23, 0.8); /* Semi-transparent background for nav */
    }

    .floating-nav a {
        color: #00eaff;
        text-decoration: none;
        font-weight: bold;
        font-size: 1.05rem;
        letter-spacing: 1.1px;
        transition: color 0.3s ease, text-shadow 0.3s ease;
    }

    .floating-nav a:hover {
        color: #00fff7;
        text-shadow: 0 0 6px #00fff7;
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

    .stSelectbox > label, .stSlider > label { /* Target specific Streamlit labels */
        color: #00eaff;
        font-weight: bold;
        letter-spacing: 0.5px;
    }

    /* Adjust Streamlit specific elements for dark theme consistency */
    .stSelectbox > div > div, .stSlider > div > div > div {
        background-color: #1a1a1a; /* Darker background for input widgets */
        border: 1px solid #00eaff;
        border-radius: 5px;
    }
    .stSelectbox > div > div > div > span {
        color: #ffffff; /* Text color in select */
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
        .floating-nav {
            top: 65px;
            left: 10px;
            padding: 8px 12px; /* Adjusted padding */
            border-radius: 8px;
            box-shadow: 0 0 10px #00eaff99;
            font-size: 0.8rem; /* Adjusted font size */
        }

        .floating-nav a {
            display: inline-block; /* Keep items in a row, but allow wrapping */
            margin: 4px 6px; /* Reduced margin */
            font-size: 0.85rem; /* Adjusted font size for links */
        }

        h1 {
            font-size: 1.8rem;
        }

        p.subtitle {
            font-size: 1rem;
        }
    }
    </style>

    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Roboto+Mono&display=swap" rel="stylesheet">

    <div class="floating-nav fade-in">
        <a href="/" target="_self">🏠 Home</a>
        <a href="/Trends" target="_self">📊 Trends</a>
        <a href="/Forecast" target="_self">🔮 Forecast</a>
    </div>
""", unsafe_allow_html=True)

# Header with animation
st.markdown("""
<div class='fade-in'>
    <h1>🌍 Real-Time ESG Forecasting & Actionable Intelligence</h1>
    <p class='subtitle'>Gain foresight into future ESG performance and make data-driven decisions.</p>
</div>
""", unsafe_allow_html=True)


# Load Data
df = load_data()

# Ensure essential columns for core functionality exist
required_core_columns = ['Country', 'Year']
if not all(col in df.columns for col in required_core_columns):
    st.error(f"Error: Missing one or more required columns ({', '.join(required_core_columns)}) in your CSV data. Please ensure 'Country' and 'Year' columns are present.")
    st.stop()


# UI Filters (moved from sidebar to main page)
st.subheader("🔎 Filter ESG Data")
col_filter_1, col_filter_2 = st.columns(2)

with col_filter_1:
    country = st.selectbox("Select Country", sorted(df['Country'].unique()))

with col_filter_2:
    year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
    year_range = st.slider("Select Year Range", year_min, year_max, (2000, 2020)) # Default range as per original code

# ESG Metric Selection
# Metrics are now all columns except 'Country' and 'Year' as per the wide format
metrics = df.columns.difference(['Country', 'Year']).tolist()
if not metrics:
    st.error("No ESG metrics found in the data besides 'Country' and 'Year'. Please check your CSV file.")
    st.stop()
metric = st.selectbox("Select ESG Metric", metrics)

# Ensure the selected metric column is numeric for plotting and forecasting
if not pd.api.types.is_numeric_dtype(df[metric]):
    st.warning(f"Warning: The selected metric '{metric}' is not numeric. Attempting to convert to numeric, non-numeric values will become NaN.")
    df[metric] = pd.to_numeric(df[metric], errors='coerce')


# Filtered Data
# Filter by Country and Year range
filtered_df = df[
    (df['Country'] == country) &
    (df['Year'].between(*year_range))
].copy() # Using .copy() to avoid SettingWithCopyWarning

# Ensure selected metric column exists in filtered_df and has non-null values
if metric not in filtered_df.columns or filtered_df[metric].isnull().all():
    st.warning(f"No valid data available for '{metric}' in {country} for the selected year range ({year_range[0]}-{year_range[1]}). Please adjust your filters or select a different metric.")
    filtered_df_for_plot = pd.DataFrame() # Empty DataFrame to prevent errors
else:
    # Filter for the specific metric after country/year, ensuring it's not all NaNs
    filtered_df_for_plot = filtered_df[['Year', metric]].dropna().sort_values(by='Year').reset_index(drop=True)


if filtered_df_for_plot.empty:
    st.info("No data available to display trends or forecasts based on current selections.")
else:
    # Show current ESG snapshot (most recent year)
    st.subheader(f"📊 ESG Snapshot for {country} (Latest Data Point)")
    latest_year_data = filtered_df_for_plot[filtered_df_for_plot['Year'] == filtered_df_for_plot['Year'].max()]
    if not latest_year_data.empty:
        st.metric(label=f"Latest Value for {metric} in {latest_year_data['Year'].iloc[0]}",
                  value=f"{latest_year_data[metric].iloc[0]:,.2f}")
    else:
        st.info("No latest data point found for the selected filters.")

    # ESG Trend Chart (using Plotly Express for better styling and interactivity)
    st.subheader(f"📈 {metric} Trend in {country} ({year_range[0]} - {year_range[1]})")
    fig_trend = px.line(
        filtered_df_for_plot,
        x='Year',
        y=metric, # Y-axis is now the selected metric column name
        title=f"{metric} over Time",
        markers=True,
        template="plotly_dark"
    )
    fig_trend.update_layout(
        title_font_color="#00eaff",
        title_font_size=20,
        margin=dict(t=60, b=40),
        hoverlabel=dict(bgcolor="#222", font_size=12, font_color="#00eaff"),
        xaxis_title="Year",
        yaxis_title=metric # Y-axis title is the metric name
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Forecasting (Linear Regression)
    st.subheader(f"📉 Forecasting {metric} (Linear Regression)")

    # Prepare data for forecasting
    X = filtered_df_for_plot[['Year']]
    y = filtered_df_for_plot[metric].astype(float)

    if len(X) < 2:
        st.warning("Not enough data points to perform reliable forecasting (need at least 2 years with non-null values for the selected metric).")
    else:
        # Fit model
        model = LinearRegression()
        model.fit(X, y)

        # Predict future (5 years beyond the selected range end)
        forecast_horizon = 5 # Predict for 5 years into the future
        future_years_start = year_range[1] + 1
        future_years_end = future_years_start + forecast_horizon
        future_years = np.arange(future_years_start, future_years_end).reshape(-1, 1)
        future_preds = model.predict(future_years)

        # Create a DataFrame for forecast results
        forecast_df = pd.DataFrame({
            'Year': future_years.flatten(),
            metric: future_preds, # Use metric name as column for forecast
            'Type': 'Forecast'
        })
        actual_df = filtered_df_for_plot[['Year', metric]].copy()
        actual_df['Type'] = 'Actual'

        # Combine actual and forecast data for plotting
        combined_df = pd.concat([actual_df, forecast_df])

        # Plot forecast
        fig_forecast = px.line(
            combined_df,
            x='Year',
            y=metric, # Y-axis is the metric name
            color='Type', # Differentiate actual vs. forecast
            title=f"Forecast of {metric} for {country} (Next {forecast_horizon} Years)",
            markers=True,
            template="plotly_dark",
            color_discrete_map={'Actual': '#00eaff', 'Forecast': '#ffcc00'} # Custom colors
        )
        fig_forecast.update_layout(
            title_font_color="#00eaff",
            title_font_size=20,
            margin=dict(t=60, b=40),
            hoverlabel=dict(bgcolor="#222", font_size=12, font_color="#00eaff"),
            xaxis_title="Year",
            yaxis_title=metric, # Y-axis title is the metric name
            legend_title_text="" # Remove legend title
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("🔮 Projected Values")
        st.dataframe(forecast_df.set_index('Year'), use_container_width=True)


# Download filtered data (moved from sidebar)
st.markdown("---") # Separator
st.subheader("Download Data")
# Ensure filtered_df is defined before trying to download
if 'filtered_df' in locals() and not filtered_df.empty:
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name=f"{country}_{metric.replace(' ', '_').replace('/', '_').replace('%', '').replace('(', '').replace(')', '')}_ESG_data.csv",
        mime='text/csv'
    )
else:
    st.info("Filter data and select a valid metric to enable download.")

# Optional footer (consistent styling)
st.markdown("""
<hr style='border-color: #333;'/>
<p style='font-size: 0.85rem; color: #666;'>🛰️ Data visualized using AI-enhanced dashboards. Built for clarity, precision, and sustainability insights.</p>
""", unsafe_allow_html=True)