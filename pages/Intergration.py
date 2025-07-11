from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import streamlit as st
import pandas as pd
import numpy as np
import io
import requests

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(layout="wide")
st.title("Data Integration Layer")

st.markdown("""
This section elucidates the foundational 'Data Integration Layer' of our ESG dashboard, a critical component engineered for the robust, continuous, and automated ingestion of diverse ESG indicator data.  

**Designed to handle diverse data sources** (e.g., *World Bank APIs, national statistics, proprietary bank data*) and formats, it transforms raw data into the **structured wide format required for the predictive models**.  

This layer is paramount in converting heterogeneous, messy input into a meticulously structured dataset — a pristine analytical base for predictive modeling and data-driven ESG insights.
""")

# -------------------------
# Data Upload or API Integration
# -------------------------
st.subheader("🔌 1. Upload or Integrate ESG Data")

upload_option = st.radio(
    "Choose data ingestion method:",
    ("📁 Upload CSV File", "🌐 Fetch via API (Simulated)"),
    horizontal=True
)

df = None

if upload_option == "📁 Upload CSV File":
    uploaded_file = st.file_uploader("Upload your ESG dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

elif upload_option == "🌐 Fetch via API (Simulated)":
    st.info("This simulates fetching ESG data from a public or private URL (e.g., World Bank data, CSV endpoint).")

    user_url = st.text_input("Enter a direct CSV URL to fetch data:", placeholder="https://example.com/data.csv")

    if st.button("📡 Fetch Data"):
        if user_url.strip() == "":
            st.warning("Please enter a valid URL before fetching.")
        else:
            try:
                df = pd.read_csv(user_url)
                st.success("✅ Data fetched successfully from the provided URL!")
            except Exception as e:
                st.error(f"❌ Failed to fetch or parse data. Error: {e}")

# -------------------------
# Data Preview
# -------------------------
if df is not None and not df.empty:
    st.write("#### ✅ Raw Data Preview (First 5 Rows):")
    st.dataframe(df.head())

    # --- Dataset Dimensions ---
    st.write("#### 📐 Dataset Dimensions:")
    st.markdown(f"""
    - **Rows:** {df.shape[0]}
    - **Columns:** {df.shape[1]}
    """)
    
    # --- Data Schema and Types (Collapsible) ---
    with st.expander("🧬 View Full Data Schema and Column Types"):
        st.markdown("""
        This expandable section provides a full overview of the dataset structure, including column types, non-null counts, and memory usage.
        """)
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)


    # -------------------------
    # Preprocessing
    # -------------------------
    st.subheader("🧹 2. Data Preprocessing and Transformation")

    st.write("#### 🕳️ Missing Value Analysis:")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_data, 'Missing %': missing_percentage})
    st.dataframe(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing %', ascending=False))

    # Impute or drop
    if missing_df['Missing Count'].sum() > 0:
        st.markdown("We will use simple imputation: median for numerics, mode for categoricals.")
        df_cleaned = df.copy()
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    median_val = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median_val)
                    st.write(f"🧮 Imputed numerical column '{col}' with median = {median_val:.2f}")
                else:
                    mode_val = df_cleaned[col].mode()[0]
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                    st.write(f"🗂️ Imputed categorical column '{col}' with mode = '{mode_val}'")
    else:
        df_cleaned = df.copy()
        st.info("✅ No missing values detected.")

    st.write("#### 🔁 Data Type Coercion:")
    if 'Year' in df_cleaned.columns:
        df_cleaned['Year'] = pd.to_numeric(df_cleaned['Year'], errors='coerce').fillna(0).astype(int)
        st.write("🗓️ Converted 'Year' column to integer.")

    for col in df_cleaned.select_dtypes(include='object').columns:
        df_cleaned[col] = df_cleaned[col].astype('category')
        st.write(f"🔄 Converted '{col}' to categorical type.")

    st.write("#### 🧼 Cleaned Data Schema Overview:")
    st.markdown(f"""
    - **Rows:** {df_cleaned.shape[0]}
    - **Columns:** {df_cleaned.shape[1]}
    """)
    
    with st.expander("📂 View Detailed Cleaned Data Types and Structure"):
        st.markdown("""
        This section displays the full data schema after cleaning, showing types, non-null counts, and memory usage.
        """)
        buffer_cleaned = io.StringIO()
        df_cleaned.info(buf=buffer_cleaned)
        s_cleaned = buffer_cleaned.getvalue()
        st.text(s_cleaned)

    # -------------------------
    # Structured Output
    # -------------------------
    st.subheader("📦 3. Structured Wide Format Output")

    st.markdown("""
The culmination of the data integration layer is the generation of a structured wide-format dataset.  
This format is optimized for feeding into ESG prediction models.
""")

    st.write("#### 👀 Preview of Cleaned Data:")
    st.dataframe(df_cleaned.head())

    st.write("#### 📊 Descriptive Statistics of Numerical Features:")
    st.dataframe(df_cleaned.describe())

    st.success("✅ Data is cleaned and structured. Ready for modeling and visualization.")

else:
    st.warning("Please upload or fetch data to proceed.")




st.subheader("📈 ESG Predictive Analytics with XGBoost")

target_col = st.selectbox("🎯 Select Target Variable (what you want to predict):", df_cleaned.select_dtypes(include=['float64', 'int64']).columns)

feature_cols = st.multiselect("🧮 Select Features to Use for Prediction:", 
                              [col for col in df_cleaned.columns if col != target_col and df_cleaned[col].dtype != 'category'])

if st.button("🚀 Train XGBoost Model"):
    if len(feature_cols) == 0:
        st.warning("Please select at least one feature.")
    else:
        # Prepare data
        X = df_cleaned[feature_cols]
        y = df_cleaned[target_col]

        # Encode any categorical columns
        X = pd.get_dummies(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train XGBoost model
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        st.success("✅ Model trained successfully!")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Show feature importances
        st.subheader("🔍 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.dataframe(importance_df)

        st.bar_chart(importance_df.set_index('Feature').head(10))

