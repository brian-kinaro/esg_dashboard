import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from utils import load_data

# Page setup
st.set_page_config(page_title="🔮 ESG Forecast", layout="wide")

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

    .stSelectbox > label, .stSlider > label, .stMultiSelect > label {
        color: #00eaff;
        font-weight: bold;
        letter-spacing: 0.5px;
    }

    @media screen and (max-width: 768px) {
        .floating-nav {
            top: 65px;
            left: 10px;
            padding: 2px 3px;
            border-radius: 8px;
            box-shadow: 0 0 10px #00eaff99;
            font-size: 0.4rem;
        }

        .floating-nav a {
            display: block;
            margin: 8px 0;
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

# Header
st.markdown("""
<div class='fade-in'>
    <h1>🔮 ESG Forecasting</h1>
    <p class='subtitle'>Predictive analytics for sustainability—powered by AI.</p>
</div>
""", unsafe_allow_html=True)

# Load and preprocess data
df = load_data()
pivot_df = df.pivot_table(index=["Country", "Year"], columns="Indicators", values="Value").reset_index()
target = "Control of Corruption: Estimate"
pivot_df = pivot_df.dropna(subset=[target])

X = pivot_df.drop(columns=["Country", "Year", target])
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X)
y = pivot_df[target]

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# UI inputs
country = st.selectbox("🌍 Select Country", sorted(pivot_df["Country"].unique()))
year_range = st.slider("📅 Select Years", int(pivot_df["Year"].min()), int(pivot_df["Year"].max()), (2010, 2020))

# Filter data
filtered = pivot_df[
    (pivot_df["Country"] == country) &
    (pivot_df["Year"] >= year_range[0]) &
    (pivot_df["Year"] <= year_range[1])
]

X_input = imputer.transform(filtered.drop(columns=["Country", "Year", target]))
predictions = model.predict(X_input)

# Results
results = filtered[["Year", target]].copy()
results["Predicted"] = predictions

# Metrics
col1, col2 = st.columns(2)
with col1:
    st.metric("Latest Actual", f"{results[target].iloc[-1]:.2f}")
with col2:
    st.metric("Latest Prediction", f"{results['Predicted'].iloc[-1]:.2f}")

# Plot: Actual vs Predicted
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=results["Year"], y=results[target],
    name="Actual", mode="lines+markers",
    line=dict(color="#1f77b4", width=3),
    marker=dict(symbol="circle", size=8)
))
fig.add_trace(go.Scatter(
    x=results["Year"], y=results["Predicted"],
    name="Predicted", mode="lines+markers",
    line=dict(color="#00eaff", width=3, dash="dash"),
    marker=dict(symbol="diamond", size=8)
))
fig.update_layout(
    title="📈 ESG Indicator Forecast: Actual vs Predicted",
    title_font_color="#00eaff",
    template="plotly_dark",
    margin=dict(t=60, b=40),
    hoverlabel=dict(bgcolor="#222", font_color="#00eaff", font_size=12)
)
st.plotly_chart(fig, use_container_width=True)

# Optional footer
st.markdown("""
<hr style='border-color: #333;'/>
<p style='font-size: 0.85rem; color: #666;'>🔗 AI Forecast Engine | RandomForestRegressor | ESG Insight Platform</p>
""", unsafe_allow_html=True)
