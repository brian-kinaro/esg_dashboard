import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="ESG FSS Forecasting",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "This application forecasts future FSS (Financial Stability Score) using historical ESG (Environmental, Social, Governance) indicators."
    }
)
st.markdown(
    """
    <style>
    /* General Styles */
    body {
        color: #e0e0e0;
        background-color: #0c1a2e; /* Dark blue background */
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #197ffc; 
        font-weight: 700;
        text-shadow: 0 0 8px rgba(0, 234, 255, 0.4);
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }
    h1 { font-size: 2.8rem; }
    h2 { font-size: 2.2rem; }
    h3 { font-size: 1.8rem; }

    /* Streamlit components specific styling */
    .stButton>button {
        background-color: #1a5e8a; /* Darker blue for buttons */
        color: #ffffff;
        border-radius: 8px;
        border: 1px solid #3282b8;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #3282b8; /* Lighter blue on hover */
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 234, 255, 0.4);
    }

    .stRadio > label > div {
        color: #aaddff; /* Lighter blue for radio options */
        font-weight: 500;
    }

    .stSelectbox > label, .stMultiSelect > label, .stSlider > label, .stTextInput > label, .stDateInput > label {
        color: #aaddff;
        font-weight: 600;
        margin-bottom: 0.5em;
    }

    .stMarkdown p, .stMarkdown li {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #c0c0c0;
    }

    /* Info, Success, Warning, Error messages */
    .stInfo {
        background-color: rgba(0, 150, 200, 0.15);
        border-left: 5px solid #00eaff;
        padding: 10px 15px;
        border-radius: 8px;
        color: #aaddff;
    }
    .stSuccess {
        background-color: rgba(60, 179, 113, 0.15);
        border-left: 5px solid #3cb371;
        padding: 10px 15px;
        border-radius: 8px;
        color: #90ee90;
    }
    .stWarning {
        background-color: rgba(255, 165, 0, 0.15);
        border-left: 5px solid #ffa500;
        padding: 10px 15px;
        border-radius: 8px;
        color: #ffda80;
    }
    .stError {
        background-color: rgba(255, 69, 0, 0.15);
        border-left: 5px solid #ff4500;
        padding: 10px 15px;
        border-radius: 8px;
        color: #ff9999;
    }

    /* Custom classes for animations */
    .fade-in {
        animation: fadeIn 1.5s ease forwards;
        opacity: 0;
        transform: translateY(20px);
    }
    @keyframes fadeIn {
        to { opacity: 1; transform: translateY(0); }
    }

    .subtitle {
        font-size: 1.3rem;
        color: #90c0ff;
        margin-top: -10px;
        margin-bottom: 20px;
        font-style: italic;
    }

    /* Plotly chart container styling */
    .stPlotlyChart {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 234, 255, 0.2);
        margin-bottom: 2em;
    }

    hr {
        border-top: 1px solid #336699;
        margin: 2em 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Title and Introduction ---
st.title("📈 ESG FSS Forecasting")
st.markdown(
    """
    Predict future **Financial Stability Scores (FSS)** using raw **ESG indicators** and an **XGBoost model**.
    This tool leverages historical ESG data to project likely future FSS performance.
    """
)
st.info("💡 Select a country from the dropdown to view its historical FSS and forecasted values up to 2029.")

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data(file_path="fss.csv"):
    """
    Loads the FSS dataset, converts 'Year' to integer, and identifies raw ESG features.
    Caches the data to improve performance on subsequent runs.
    """
    try:
        df = pd.read_csv(file_path)
        df['Year'] = df['Year'].astype(int)
        # Identify raw features by excluding 'Country', 'Year', 'FSS', and normalized columns
        raw_features = [col for col in df.columns if col not in ["Country", "Year", "FSS"] and not col.endswith("_norm")]
        return df, raw_features
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure 'fss.csv' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        st.stop()

# Load data globally once
main_df, raw_esg_features = load_and_preprocess_data()

# --- Country Selection ---
st.header("🌍 Country Selection")
selected_country = st.selectbox(
    "Choose a Country for Forecasting",
    sorted(main_df["Country"].unique()),
    help="Select a country to analyze its ESG FSS trends and predictions."
)

# Filter data for the selected country
country_data = main_df[main_df["Country"] == selected_country].sort_values("Year").copy()

# Display historical data points for FSS
fss_historical_points = country_data["FSS"].notnull().sum()
st.metric(label="📊 Historical FSS Data Points Available", value=fss_historical_points)

# Enforce minimum data requirement for reliable forecasting
MIN_FSS_DATA_POINTS = 10
if fss_historical_points < MIN_FSS_DATA_POINTS:
    st.warning(
        f"⚠️ **Insufficient Data!** {selected_country} has only **{fss_historical_points}** historical FSS data points. "
        f"A minimum of **{MIN_FSS_DATA_POINTS}** points are required for reliable forecasting. Please select another country."
    )
    st.stop() # Stop the app execution if data is insufficient

# --- Historical FSS Visualization ---
st.subheader(f"📈 Historical FSS Trends for {selected_country}")
fig_historical_fss = px.line(
    country_data,
    x="Year",
    y="FSS",
    title=f"Historical Financial Stability Score – {selected_country}",
    markers=True,
    hover_name="Year",
    labels={"FSS": "FSS Value"}
)
fig_historical_fss.update_layout(hovermode="x unified")
st.plotly_chart(fig_historical_fss, use_container_width=True)


# Prepare training data
features_for_model = raw_esg_features + ["Year"]
X_train = country_data[features_for_model].copy()
y_train = country_data["FSS"].copy()

# Impute missing values using median strategy
# This ensures that even if some ESG indicators have missing values, they can be used.
imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

# Initialize and train the XGBoost model
# Parameters are chosen to provide a good balance of performance and generalization.
model = XGBRegressor(
    n_estimators=200,          # Number of boosting rounds
    learning_rate=0.05,        # Step size shrinkage to prevent overfitting
    max_depth=4,               # Maximum depth of a tree
    random_state=42,           # For reproducibility
    subsample=0.8,             # Subsample ratio of the training instance
    colsample_bytree=0.8       # Subsample ratio of columns when constructing each tree
)
model.fit(X_train_imputed, y_train)

# --- Forecasting Future FSS Values ---
st.subheader("🔮 Forecasting FSS (2024–2029)")
st.markdown(
    """
    Future ESG indicator values are projected based on linear trends from historical data.
    These projected indicators are then fed into the trained XGBoost model to predict future FSS.
    """
)

future_years = list(range(2024, 2030)) # Years from 2024 to 2029
future_predictions_df = pd.DataFrame({"Year": future_years})

# Project ESG features for future years using linear regression
for col in raw_esg_features:
    valid_feature_data = country_data[["Year", col]].dropna()
    if len(valid_feature_data) >= 2:
        # Fit a linear trend to existing data
        coefficients = np.polyfit(valid_feature_data["Year"], valid_feature_data[col], 1)
        # Apply the trend to future years
        future_predictions_df[col] = np.poly1d(coefficients)(future_years)
    elif len(valid_feature_data) == 1:
        # If only one data point, use that value for future years
        future_predictions_df[col] = valid_feature_data[col].iloc[0]
    else:
        # Fallback: if no valid data, assume 0 or handle as appropriate (e.g., median of all countries)
        # For this example, we'll set to 0, but a more robust solution might impute from global data.
        future_predictions_df[col] = 0

future_predictions_df["Country"] = selected_country

# Ensure all feature columns expected by the model are present in the future DataFrame
# and impute any missing values in these projected features
for col in features_for_model:
    if col not in future_predictions_df.columns:
        future_predictions_df[col] = 0 # Add missing columns, initializing with 0

X_future_imputed = pd.DataFrame(
    imputer.transform(future_predictions_df[features_for_model]),
    columns=features_for_model
)

# Predict FSS for future years
future_predictions_df["FSS_Predicted"] = model.predict(X_future_imputed)

# Clip predictions to the observed historical FSS range to prevent unrealistic values
min_fss_observed, max_fss_observed = country_data["FSS"].min(), country_data["FSS"].max()
future_predictions_df["FSS_Predicted"] = np.clip(
    future_predictions_df["FSS_Predicted"], min_fss_observed, max_fss_observed
)

# --- Combined Historical and Forecasted FSS Visualization ---
st.subheader("📊 Combined Historical & Forecasted FSS")
st.markdown("Visualizing the historical FSS alongside the projected future FSS to observe trends.")

combined_fss_df = pd.concat([
    country_data[["Year", "FSS"]].rename(columns={"FSS": "Value"}).assign(Type="Historical FSS"),
    future_predictions_df[["Year", "FSS_Predicted"]].rename(columns={"FSS_Predicted": "Value"}).assign(Type="Forecasted FSS")
])

fig_combined_fss = px.line(
    combined_fss_df,
    x="Year",
    y="Value",
    color="Type",
    title=f"{selected_country} – FSS: Historical vs. Forecasted (2024–2029)",
    markers=True,
    color_discrete_map={"Historical FSS": "blue", "Forecasted FSS": "red"},
    hover_name="Year",
    labels={"Value": "FSS Value", "Type": "Data Type"}
)
fig_combined_fss.update_layout(hovermode="x unified")
st.plotly_chart(fig_combined_fss, use_container_width=True)

# --- Forecast-Only Visualization ---
st.subheader("📈 Forecasted FSS Only (2024–2029)")
st.markdown("A focused view of the predicted FSS values for the upcoming years.")
fig_forecast_only = px.line(
    future_predictions_df,
    x="Year",
    y="FSS_Predicted",
    markers=True,
    title=f"{selected_country} – Predicted FSS (2024-2029)",
    labels={"FSS_Predicted": "Predicted FSS Value"},
    hover_name="Year"
)
fig_forecast_only.update_traces(line_color="red", line_width=3)
st.plotly_chart(fig_forecast_only, use_container_width=True)

# --- Forecast Table and Download ---
st.subheader("📋 Forecasted FSS Table")
st.dataframe(future_predictions_df[["Year", "FSS_Predicted"]].round(3), use_container_width=True)

# Prepare CSV for download
csv_data = future_predictions_df[["Country", "Year", "FSS_Predicted"]].to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Forecast as CSV",
    data=csv_data,
    file_name=f"{selected_country}_fss_forecast.csv",
    mime="text/csv",
    help="Download the forecasted FSS values for the selected country."
)

# --- Feature Importance ---
st.subheader("🔍 Feature Importance")
with st.expander("Click to view how each feature contributed to the predictions"):
    st.write(
        """
        These values indicate the relative importance of each ESG indicator (and 'Year')
        in making the FSS predictions. Higher importance means a greater impact on the model's output.
        """
    )
    importance_df = pd.DataFrame({
        "Feature": features_for_model,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df.head(10), use_container_width=True)

    fig_feature_importance = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Most Important Features for FSS Prediction",
        color="Importance",
        color_continuous_scale="Plasma",
        labels={"Importance": "Feature Importance Score", "Feature": "ESG Feature"}
    )
    fig_feature_importance.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_feature_importance, use_container_width=True)

st.markdown("---")