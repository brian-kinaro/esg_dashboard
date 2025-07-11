# pages/Predict.py
import xgboost as xgb
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from utils import load_data # Assuming utils.py correctly loads imputed_data.csv

# --- Page Configuration and Custom Styling (Consistent with other pages) ---
st.set_page_config(page_title="🔮 Predict FSS Score", layout="wide")

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

    .stSelectbox > label, .stSlider > label, .stNumberInput > label { /* Target specific Streamlit labels */
        color: #00eaff;
        font-weight: bold;
        letter-spacing: 0.5px;
    }

    /* Adjust Streamlit specific elements for dark theme consistency */
    .stSelectbox > div > div, .stSlider > div > div > div, .stNumberInput > div > div {
        background-color: #1a1a1a; /* Darker background for input widgets */
        border: 1px solid #00eaff;
        border-radius: 5px;
    }
    .stSelectbox > div > div > div > span, .stNumberInput > div > div > input {
        color: #ffffff; /* Text color in select/number input */
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

""", unsafe_allow_html=True)

# Header with animation
st.markdown("""
<div class='fade-in'>
    <h1>🔮 Predict Financial Sustainability Score (FSS)</h1>
    <p class='subtitle'>Enter parameters to predict FSS and visualize future trends for a selected country.</p>
</div>
""", unsafe_allow_html=True)


# --- Load and Preprocess Data ---
data = load_data()
data['Financial Sustainability Score (FSS)'] = data

df = data

# Ensure required columns are numeric
for col in df.columns:
    if col not in ['Country', 'Year']:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# --- Define Variables based on user's list ---
# Dependent Variable
dependent_var = 'Financial Sustainability Score (FSS)'

# Independent Variables (ESG)
env_vars = ['CO2 emissions (metric tons per capita)',
            'Renewable electricity output (% of total)',
            'Renewable energy consumption (% of final energy use)',
            'Forest area (% of land area)',
            'Net forest depletion',
            'Agricultural land (% of land area)']

soc_vars = ['Life expectancy at birth',
            'Under-5 mortality rate',
            'Literacy rate, adult total (% ages 15+)',
            'Unemployment, total (%)',
            'Children in employment (% ages 7–14)']

gov_vars = ['Control of Corruption (index)',
            'Government Effectiveness (index)',
            'Regulatory Quality (index)',
            'Rule of Law (index)',
            'Voice and Accountability (index)']

# Control Variables
control_vars = [
    'Cooling Degree Days',
    'Heating Degree Days',
    'Heat Index >35°C days',
    'Land Surface Temperature',
    'Population ages 65+ (% of population)',
    'Fertility rate (births per woman)',
    'Net migration (people net)',
    'School enrollment (% gross enrollment)'
]

# Combine all independent and control variables for the model's features
all_model_features = env_vars + soc_vars + gov_vars + control_vars

# Filter out variables that are not present in the DataFrame
# This makes the code robust if some listed variables are missing in the CSV
available_features = [col for col in all_model_features if col in df.columns]
if dependent_var not in df.columns:
    st.error(f"Dependent variable '{dependent_var}' not found in the loaded data. Please check 'imputed_data.csv'.")
    st.stop()
if not available_features:
    st.error("No valid independent or control indicator columns found in the loaded data. Please check 'imputed_data.csv'.")
    st.stop()

# Re-filter env/soc/gov_vars to only include those actually in the data
env_vars = [col for col in env_vars if col in df.columns]
soc_vars = [col for col in soc_vars if col in df.columns]
gov_vars = [col for col in gov_vars if col in df.columns]
control_vars = [col for col in control_vars if col in df.columns]


# Normalize function
def normalize_columns(df_to_scale, columns_to_normalize, fitted_scaler=None):
    # Only normalize the specified columns, others are passed through
    df_scaled = df_to_scale.copy()
    if fitted_scaler:
        df_scaled[columns_to_normalize] = fitted_scaler.transform(df_to_scale[columns_to_normalize])
        return df_scaled, fitted_scaler
    else:
        scaler = MinMaxScaler() # Default range 0-1 for general features
        df_scaled[columns_to_normalize] = scaler.fit_transform(df_to_scale[columns_to_normalize])
        return df_scaled, scaler


# Country selection
selected_country = st.selectbox("🌍 Select Country", sorted(df['Country'].unique()))

# Filter data for the selected country
df_country = df[df['Country'] == selected_country].copy()

# Filter data with valid FSS and all model features for the selected country
required_cols_for_training = [dependent_var, 'Year'] + available_features
df_clean_country = df_country.dropna(subset=required_cols_for_training).copy()


# Check for sufficient data
if df_clean_country.empty:
    st.warning(f"No complete data available for {selected_country} after removing rows with missing values for the selected indicators and FSS.")
    st.stop()

if len(df_clean_country) < 2: # At least 2 data points for meaningful training/forecasting
    st.warning(f"Insufficient data points ({len(df_clean_country)}) for {selected_country} to train the prediction model. Need at least 2 non-null data points for FSS and all features.")
    st.stop()

# Normalize features (independent and control variables) on country-specific data
# The dependent variable (FSS) should NOT be normalized if we want to predict its original scale
features_to_normalize = available_features # These are already filtered to what's in DF
df_normalized_country, scaler = normalize_columns(df_clean_country, features_to_normalize)


# XGBoost model training on country-specific data
X_train = df_normalized_country[available_features + ['Year']] # Features + Year
y_train = df_normalized_country[dependent_var] # Financial Sustainability Score (FSS)

model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Streamlit UI for user input
st.markdown("Enter values for ESG indicators and control variables to predict the **Financial Sustainability Score** and forecast based on current outlook.")

with st.form("xgb_input_form"):
    user_input = {}

    st.markdown("### 🌱 Environmental Indicators")
    for col in env_vars:
        # Provide more realistic initial values based on min/max in the data, or a sensible mid-point
        col_min = df_clean_country[col].min() if col in df_clean_country.columns and not df_clean_country[col].isnull().all() else 0.0
        col_max = df_clean_country[col].max() if col in df_clean_country.columns and not df_clean_country[col].isnull().all() else 100.0
        user_input[col] = st.number_input(f"{col}", min_value=float(col_min), max_value=float(col_max), value=float(col_min + (col_max - col_min) / 2))

    st.markdown("### 🧑‍🤝‍🧑 Social Indicators")
    for col in soc_vars:
        col_min = df_clean_country[col].min() if col in df_clean_country.columns and not df_clean_country[col].isnull().all() else 0.0
        col_max = df_clean_country[col].max() if col in df_clean_country.columns and not df_clean_country[col].isnull().all() else 100.0
        user_input[col] = st.number_input(f"{col}", min_value=float(col_min), max_value=float(col_max), value=float(col_min + (col_max - col_min) / 2))

    st.markdown("### 🏛️ Governance Indicators")
    for col in gov_vars:
        col_min = df_clean_country[col].min() if col in df_clean_country.columns and not df_clean_country[col].isnull().all() else 0.0
        col_max = df_clean_country[col].max() if col in df_clean_country.columns and not df_clean_country[col].isnull().all() else 100.0
        user_input[col] = st.number_input(f"{col}", min_value=float(col_min), max_value=float(col_max), value=float(col_min + (col_max - col_min) / 2))

    st.markdown("### ⚙️ Control Variables")
    for col in control_vars:
        col_min = df_clean_country[col].min() if col in df_clean_country.columns and not df_clean_country[col].isnull().all() else 0.0
        col_max = df_clean_country[col].max() if col in df_clean_country.columns and not df_clean_country[col].isnull().all() else 100.0 # Use a sensible default max
        user_input[col] = st.number_input(f"{col}", min_value=float(col_min), max_value=float(col_max), value=float(col_min + (col_max - col_min) / 2))


    current_year_for_prediction = st.number_input("📅 Input Year for Prediction", min_value=int(df['Year'].min()), max_value=2100, value=int(df['Year'].max()))
    submitted = st.form_submit_button("🚀 Predict FSS and Forecast")

if submitted:
    # Create DataFrame from user input
    user_df = pd.DataFrame([user_input])
    user_df['Year'] = current_year_for_prediction

    # Normalize user input for features (independent + control variables) using the training scaler
    user_scaled_features, _ = normalize_columns(user_df, features_to_normalize, fitted_scaler=scaler)

    # Predict using XGBoost for the current input year
    prediction_input = user_scaled_features[available_features + ['Year']] # Features + Year for prediction
    prediction = model.predict(prediction_input)[0]
    st.success(f"🌍 Predicted Financial Sustainability Score (FSS) for **{selected_country}** in {current_year_for_prediction}: **{prediction:.2f}**")


    st.markdown("---")
    st.subheader(f"📈 FSS Forecast for {selected_country} (Next 5 Years)")

    # Prepare data for forecasting the next 5 years based on user input
    forecast_years = np.arange(current_year_for_prediction + 1, current_year_for_prediction + 6)
    forecast_data = []

    # Use the same user_input values for all features, only change the year
    for year in forecast_years:
        future_user_input = user_input.copy()
        future_user_input['Year'] = year
        forecast_data.append(future_user_input)

    forecast_df_raw = pd.DataFrame(forecast_data)
    forecast_df_scaled, _ = normalize_columns(forecast_df_raw, features_to_normalize, fitted_scaler=scaler)

    future_predictions = model.predict(forecast_df_scaled[available_features + ['Year']])

    # Create a DataFrame for historical data (with computed FSS scores)
    historical_df_plot = df_clean_country[['Year', dependent_var]].rename(columns={dependent_var: 'FSS_Score'}).copy() # Rename for plot consistency
    historical_df_plot['Type'] = 'Historical'

    # Create a DataFrame for current year prediction
    current_prediction_df = pd.DataFrame({
        'Year': [current_year_for_prediction],
        'FSS_Score': [prediction],
        'Type': 'Predicted'
    })

    # Create a DataFrame for forecast results
    forecast_results_df = pd.DataFrame({
        'Year': forecast_years,
        'FSS_Score': future_predictions,
        'Type': 'Forecast'
    })

    # Combine all data for plotting
    combined_plot_df = pd.concat([historical_df_plot, current_prediction_df, forecast_results_df]).sort_values('Year')

    # Plotting with Plotly Express
    fig = px.line(
        combined_plot_df,
        x="Year",
        y="FSS_Score",
        color="Type",
        title=f"Financial Sustainability Score Historical Trend, Current Prediction, and 5-Year Forecast for {selected_country}",
        markers=True,
        template="plotly_dark",
        color_discrete_map={'Historical': '#00eaff', 'Predicted': '#FFD700', 'Forecast': '#ff6347'} # Gold for predicted, Tomato for forecast
    )
    fig.update_layout(
        title_font_color="#00eaff",
        title_font_size=20,
        margin=dict(t=60, b=40),
        hoverlabel=dict(bgcolor="#222", font_size=12, font_color="#00eaff"),
        xaxis_title="Year",
        yaxis_title="Financial Sustainability Score (FSS)",
        legend_title_text=""
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔮 Projected FSS Scores (Next 5 Years)")
    st.dataframe(forecast_results_df[['Year', 'FSS_Score']].set_index('Year').round(2), use_container_width=True)


# --- Optional Footer (Consistent with other pages) ---
st.markdown("""
<hr style='border-color: #333;'/>
<p style='font-size: 0.85rem; color: #666;'>🛰️ Data visualized using AI-enhanced dashboards. Built for clarity, precision, and sustainability insights.</p>
""", unsafe_allow_html=True)