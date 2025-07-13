import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from utils import load_data

df = load_data()

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


st.title("🔮 Dynamic ESG Prediction")
st.write("Predict ESG performance (FSS) and understand which inputs most influence the prediction.")

# Define features (raw indicators only)
non_features = ['Country_ID','Country', 'Year', 'FSS']
feature_cols = [col for col in df.columns if col not in non_features and not col.endswith('_norm')]
target_col = 'FSS'

# Select country and get latest record
country_sel = st.selectbox("Select a Country", sorted(df["Country"].unique()))
latest_data = df[df["Country"] == country_sel].sort_values("Year").iloc[-1]

# Sliders for indicator adjustments
st.subheader("📊 Adjust ESG Indicator Values (Raw Data)")
user_input = {}
for col in feature_cols:
    col_min = float(df[col].min(skipna=True))
    col_max = float(df[col].max(skipna=True))
    default_val = float(latest_data[col]) if pd.notnull(latest_data[col]) else (col_min + col_max) / 2
    user_input[col] = st.slider(f"{col}", col_min, col_max, default_val)

X_input = pd.DataFrame([user_input])

# Train XGBoost model
X = df[feature_cols]
y = df[target_col]

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X_input_imputed = imputer.transform(X_input)

model = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
model.fit(X_imputed, y)

# Forecast
prediction = model.predict(X_input_imputed)[0]
st.success(f"📈 Forecasted ESG Performance (FSS): **{prediction:.2f}**")

# Feature importances (from adjusted input)
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Indicator": feature_cols,
    "Importance": importances
}).sort_values("Importance", ascending=False)

# Plot
st.subheader("🧮 Feature Contribution (Forecasted ESG Performance)")
fig = px.bar(importance_df.head(15), x="Importance", y="Indicator", orientation="h",
             color="Importance", color_continuous_scale="Blues_r",
             title="Top ESG Indicators Contributing to Forecast (XGBoost Feature Importance)")

fig.update_layout(yaxis=dict(title=""), xaxis=dict(title="Relative Importance"), height=600)
st.plotly_chart(fig, use_container_width=True)

st.caption("Note: This shows overall feature importance from XGBoost")
