from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import requests
import shap
import matplotlib.pyplot as plt # Kept for SHAP plots as they are tightly integrated with matplotlib
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────── Page Configuration & Styling ───────────────────────────
st.set_page_config(page_title="ESG Data Explorer & Modeler", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a refined look
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
        <h1>📊 ESG Data Explorer & Modeler</h1>
        <p class="subtitle" style="color:#88ccee; font-size:1.1rem; letter-spacing:0.5px;">
            Upload, clean, model, and interpret your ESG data with ease.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── Data Upload or API Integration ───────────────────────────
st.subheader("🔌 1. Data Ingestion: Upload or Fetch")

upload_opt = st.radio(
    "Choose your data source:",
    ("📁 Upload CSV File", "🌐 Fetch via Public URL (CSV)", "📂 Google Drive (Shareable Link)"),
    horizontal=True,
    key="data_source_radio"
)

df = None

if upload_opt == "📁 Upload CSV File":
    uploaded_file = st.file_uploader("Drag & drop your ESG dataset (CSV format) here, or click to browse", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("🎉 CSV file uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}. Please ensure it's a valid CSV file.")

elif upload_opt == "🌐 Fetch via Public URL (CSV)":
    st.info("💡 This simulates fetching ESG data from a public CSV endpoint (e.g., a GitHub raw CSV, World Bank data).")
    user_url = st.text_input("Enter a direct CSV URL to fetch data:", placeholder="https://www.example/data.csv", key="csv_url_input")

    if st.button("📡 Fetch Data from URL", key="fetch_data_button"):
        if user_url.strip() == "":
            st.warning("Please enter a valid URL before attempting to fetch data.")
        else:
            with st.spinner("Fetching data... This might take a moment."):
                try:
                    response = requests.get(user_url)
                    response.raise_for_status()
                    df = pd.read_csv(io.StringIO(response.text))
                    st.success("✅ Data fetched successfully from the provided URL!")
                except requests.exceptions.RequestException as err:
                    st.error(f"❌ Request Error: {err}")
                except pd.errors.EmptyDataError:
                    st.error("❌ The URL returned an empty dataset. Please check the URL content.")
                except Exception as e:
                    st.error(f"❌ Failed to parse data from URL: {e}. Ensure it's a direct link to a CSV.")

elif upload_opt == "📂 Google Drive (Shareable Link)":
    st.info("🔗 Paste a **public Google Drive shareable link** to a CSV file.")
    gdrive_link = st.text_input("Google Drive CSV Link", placeholder="https://drive.google.com/file/d/FILE_ID/view?usp=sharing")

    if st.button("📥 Load from Google Drive", key="load_gdrive_button"):
        if gdrive_link.strip() == "":
            st.warning("Please paste a Google Drive shareable link.")
        else:
            # Extract file ID
            match = re.search(r"/d/([a-zA-Z0-9_-]+)", gdrive_link)
            if not match:
                st.error("❌ Invalid Google Drive link format. Please check the URL.")
            else:
                file_id = match.group(1)
                gdrive_raw_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                with st.spinner("Fetching from Google Drive..."):
                    try:
                        response = requests.get(gdrive_raw_url)
                        response.raise_for_status()
                        df = pd.read_csv(io.StringIO(response.text))
                        st.success("✅ Data loaded successfully from Google Drive!")
                    except requests.exceptions.RequestException as err:
                        st.error(f"❌ Error accessing Google Drive: {err}")
                    except pd.errors.EmptyDataError:
                        st.error("❌ The file is empty or not in CSV format.")
                    except Exception as e:
                        st.error(f"❌ Could not read the CSV file: {e}")

# ─────────────────────────── Data Preview and Cleaning ───────────────────────────
if df is not None and not df.empty:
    st.subheader("🧹 2. Data Overview & Preprocessing")

    st.write("#### 👀 Raw Data Preview:")
    st.dataframe(df.head())

    st.markdown(f"""
    **Dataset Dimensions:**
    - **Rows:** `{df.shape[0]}`
    - **Columns:** `{df.shape[1]}`
    """)

    with st.expander("🧬 View Detailed Data Schema (Raw)"):
        st.markdown("""
        This section provides a full overview of the raw dataset structure, including column names, non-null counts, and data types.
        """)
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    st.write("#### 🕳️ Missing Value Analysis:")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_data, 'Missing %': missing_percentage})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing %', ascending=False)

    if not missing_df.empty:
        st.dataframe(missing_df.style.format({"Missing %": "{:.2f}%"}))
        st.markdown("---")
        st.info("💡 **Imputation Strategy:** Missing numerical values will be filled with the column's median. Missing categorical values will be filled with the column's mode. This is a simple but effective strategy for initial analysis.")
        df_clean = df.copy()
        with st.spinner("Applying imputation and type conversions..."):
            for col in df_clean.columns:
                if df_clean[col].isnull().sum() > 0:
                    if df_clean[col].dtype in ['int64', 'float64']:
                        median_val = df_clean[col].median()
                        df_clean[col].fillna(median_val, inplace=True)
                        # st.write(f"🧮 Imputed numerical column '{col}' with median = `{median_val:.2f}`") # Too verbose
                    else:
                        mode_val = df_clean[col].mode()[0]
                        df_clean[col].fillna(mode_val, inplace=True)
                        # st.write(f"🗂️ Imputed categorical column '{col}' with mode = `'{mode_val}'`") # Too verbose
    else:
        df_clean = df.copy()
        st.success("✅ No missing values detected in your dataset!")

    st.write("#### 🔁 Data Type Coercion:")
    with st.spinner("Converting data types..."):
        if 'Year' in df_clean.columns:
            df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce').fillna(0).astype(int)
            st.write("🗓️ Converted 'Year' column to integer type.")

        for col in df_clean.select_dtypes(include='object').columns:
            df_clean[col] = df_clean[col].astype('category')
            st.write(f"🔄 Converted column `'{col}'` to categorical type.")
    st.success("Data types adjusted!")

    st.write("#### 🧼 Cleaned Data Preview:")
    st.dataframe(df_clean.head())

    with st.expander("📂 View Detailed Data Schema (Cleaned)"):
        st.markdown("""
        This section displays the full data schema after cleaning and type conversions, showing types, non-null counts, and memory usage.
        """)
        buffer_cleaned = io.StringIO()
        df_clean.info(buf=buffer_cleaned)
        s_cleaned = buffer_cleaned.getvalue()
        st.text(s_cleaned)

    # ─────────────────────────── Feature Selection ───────────────────────────
    st.subheader("🎯 3. Target & Feature Selection")

    all_cols = df_clean.columns.tolist()
    
    # Ensure 'FSS' is a default target if present, otherwise suggest first numeric
    default_target = 'FSS' if 'FSS' in all_cols else (df_clean.select_dtypes(include=np.number).columns.tolist()[0] if not df_clean.select_dtypes(include=np.number).empty else all_cols[0])
    
    tgt_col = st.selectbox(
        "Select your **target variable** (what you want to predict):",
        all_cols,
        index=all_cols.index(default_target) if default_target in all_cols else 0,
        key="target_selector"
    )

    # Automatically exclude target column and its normalized version from features
    # Also exclude 'Year' and any other identifier columns if they are not features
    excluded_from_features = [tgt_col, tgt_col + '_norm', 'Year'] # Add 'Year' as a common non-feature
    
    # Filter out columns ending with '_norm' for cleaner feature selection, unless they are the target itself
    feature_candidates = [col for col in all_cols if col not in excluded_from_features and not col.endswith('_norm')]

    # Provide a reasonable default for feature selection
    default_features = [col for col in feature_candidates if df_clean[col].dtype in ['int64', 'float64']][:5] # Select first 5 numeric by default

    feat_cols = st.multiselect(
        "Select **input features** (variables used for prediction):",
        feature_candidates,
        default=default_features,
        key="feature_selector"
    )

    if feat_cols and tgt_col:
        X = df_clean[feat_cols]
        y = df_clean[tgt_col]

        # One-hot encode categorical features in X
        # Ensure that X contains only numerical data for model training
        categorical_features_in_X = X.select_dtypes(include='category').columns
        if not categorical_features_in_X.empty:
            st.info(f"⚙️ One-hot encoding categorical features: {', '.join(categorical_features_in_X.tolist())}")
            X = pd.get_dummies(X, columns=categorical_features_in_X, drop_first=True)
        
        # Convert target variable to numeric if it's categorical
        if y.dtype.name == 'category':
            st.info(f"⚙️ Encoding categorical target variable `'{tgt_col}'` using LabelEncoder.")
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.markdown(f"**Note:** Categorical target `'{tgt_col}'` has been converted to numerical labels: `{le.classes_.tolist()}` mapped to `{list(range(len(le.classes_)))}` respectively.")
            
        # Final check to ensure X and y are numeric
        X = X.select_dtypes(include=np.number)
        if not pd.api.types.is_numeric_dtype(y):
            st.error(f"❌ Error: Target variable '{tgt_col}' is not numeric after preprocessing. Please select a numeric target or ensure it can be encoded.")
            st.stop()

        if X.empty:
            st.error("❌ No numerical features remaining after selection and encoding. Please select valid numerical or categorical features.")
            st.stop()

        # ─────────────────────────── Model Training & Evaluation ───────────────────────────
        st.subheader("🧠 4. Model Training & Evaluation")

        st.markdown("Select the machine learning models you'd like to train and evaluate:")

        models = {}
        model_options = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42, n_jobs=-1),
            "Support Vector Regressor (SVR)": SVR(),
            "XGBoost Regressor": xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
        }

        selected_model_names = []
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.checkbox("Linear Regression", key="check_lr"):
                selected_model_names.append("Linear Regression")
        with col2:
            if st.checkbox("Random Forest", key="check_rf"):
                selected_model_names.append("Random Forest Regressor")
                n_estimators_rf = st.slider("RF: n_estimators", 10, 500, 100, 10, key="rf_n_est")
                max_depth_rf = st.slider("RF: max_depth", 1, 50, 10, key="rf_max_depth")
                model_options["Random Forest Regressor"].set_params(n_estimators=n_estimators_rf, max_depth=max_depth_rf)
        with col3:
            if st.checkbox("SVR", key="check_svr"):
                selected_model_names.append("Support Vector Regressor (SVR)")
                c_svr = st.slider("SVR: C", 0.1, 10.0, 1.0, key="svr_c")
                eps_svr = st.slider("SVR: epsilon", 0.01, 1.0, 0.1, key="svr_eps")
                model_options["Support Vector Regressor (SVR)"].set_params(C=c_svr, epsilon=eps_svr)
        with col4:
            if st.checkbox("XGBoost", key="check_xgb"):
                selected_model_names.append("XGBoost Regressor")
                n_estimators_xgb = st.slider("XGB: n_estimators", 10, 500, 100, 10, key="xgb_n_est")
                lr_xgb = st.slider("XGB: learning_rate", 0.01, 0.5, 0.1, key="xgb_lr")
                model_options["XGBoost Regressor"].set_params(n_estimators=n_estimators_xgb, learning_rate=lr_xgb)

        for name in selected_model_names:
            models[name] = model_options[name]

        if models:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            results = []
            with st.spinner("Training models and calculating performance metrics..."):
                for name, mdl in models.items():
                    mdl.fit(X_train, y_train)
                    preds = mdl.predict(X_test)

                    rmse = mean_squared_error(y_test, preds, squared=False)
                    mae = mean_absolute_error(y_test, preds)
                    r2 = r2_score(y_test, preds)
                    
                    # Cross-validation can be computationally intensive, adding a warning
                    try:
                        cv = cross_val_score(mdl, X, y, cv=5, scoring='r2', n_jobs=-1).mean()
                    except Exception as e:
                        cv = np.nan
                        st.warning(f"⚠️ Could not perform cross-validation for {name}: {e}. This might happen with SVR or if data is too small.")

                    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2, "CV R²": cv})

            results_df = pd.DataFrame(results).sort_values(by="RMSE")

            st.subheader("📊 Model Performance Summary")
            st.dataframe(results_df.style.format({
                "RMSE": "{:.3f}",
                "MAE": "{:.3f}",
                "R²": "{:.3f}",
                "CV R²": "{:.3f}"
            }))

            st.markdown("---")
            st.subheader("📈 Visualizing Model Performance")
            
            # Plotly Bar Charts for Metrics
            metrics_to_plot = ["RMSE", "MAE", "R²", "CV R²"]
            cols_plot = st.columns(len(metrics_to_plot))

            for i, met in enumerate(metrics_to_plot):
                with cols_plot[i]:
                    fig = go.Figure(data=[go.Bar(x=results_df['Model'], y=results_df[met], marker_color='#3282b8')])
                    fig.update_layout(
                        title=f"<b>{met} by Model</b>",
                        xaxis_title="Model",
                        yaxis_title=met,
                        template="plotly_dark",
                        height=350,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # -------------------------
            # Model Evaluation Explanation
            # -------------------------
            st.subheader("📘 Model Evaluation Metrics: Explained")
            st.markdown("""
            Understanding these metrics helps you choose the best model for your specific needs:

            - **RMSE (Root Mean Squared Error):** This measures the average magnitude of the errors. It's sensitive to large errors, so a lower RMSE indicates a model that makes fewer large mistakes. **Lower is better.**
            - **MAE (Mean Absolute Error):** This is the average of the absolute differences between predictions and actual observations. It's less sensitive to outliers than RMSE. **Lower is better.**
            - **R² (R-squared):** Also known as the coefficient of determination, it represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A value closer to 1.0 indicates that the model explains a large proportion of the variance in the target variable. **Higher is better.**
            - **CV R² (Cross-Validation R²):** This is the average R² score obtained from multiple train-test splits (here, 5-fold cross-validation). It provides a more robust estimate of how well the model generalizes to unseen data, reducing the chance of overfitting to a single train-test split. **Higher and more consistent (less variance) is better.**

            **Which model is best?** Look for models with low RMSE/MAE and high R²/CV R². Consider the trade-off between model complexity and interpretability.
            """)

            # ─────────────────────────── SHAP Feature Importance ───────────────────────────
            st.subheader("💡 5. Feature Importance with SHAP (SHapley Additive exPlanations)")
            st.markdown("""
            SHAP values help us understand how each feature contributes to the model's prediction for individual instances, as well as overall feature importance. They provide a game-theoretic approach to explain the output of any machine learning model.
            """)

            # Allow user to select a model for SHAP analysis
            shap_model_name = st.selectbox(
                "Select a trained model for SHAP analysis:",
                selected_model_names,
                key="shap_model_selector"
            )

            if shap_model_name:
                selected_model_for_shap = models[shap_model_name]

                # Check if the selected model is tree-based for TreeExplainer
                if isinstance(selected_model_for_shap, (RandomForestRegressor, xgb.XGBRegressor)):
                    with st.spinner(f"Calculating SHAP values for {shap_model_name}... This may take a moment."):
                        explainer = shap.TreeExplainer(selected_model_for_shap, X_train)
                        shap_values = explainer.shap_values(X_test)

                        st.write(f"#### SHAP Summary Plot (Beeswarm) for {shap_model_name}:")
                        fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(12, 8))
                        shap.summary_plot(shap_values, X_test, plot_type="beeswarm", show=False)
                        st.pyplot(fig_beeswarm)
                        plt.close(fig_beeswarm) # Close the plot to prevent display issues

                        st.markdown("""
                        The **beeswarm plot** illustrates the distribution of SHAP values for each feature across the dataset:
                        - **Color:** Red indicates higher feature values, blue indicates lower feature values.
                        - **X-axis (SHAP value):** Represents the impact of that feature on the model's output. Positive SHAP values push the prediction higher, while negative values push it lower.
                        - **Vertical dispersion:** Each dot represents an instance from the dataset. The spread shows how the feature's impact varies across different data points.
                        """)

                        st.write(f"#### SHAP Bar Plot (Overall Feature Importance) for {shap_model_name}:")
                        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                        st.pyplot(fig_bar)
                        plt.close(fig_bar) # Close the plot

                        st.markdown("""
                        The **bar plot** displays the average absolute SHAP value for each feature, providing an overall ranking of feature importance. Features with larger average absolute SHAP values are more important to the model's predictions.
                        """)
                else:
                    st.warning(f"⚠️ SHAP TreeExplainer is primarily designed for tree-based models (Random Forest, XGBoost). For {shap_model_name}, other SHAP explainers (e.g., KernelExplainer, DeepExplainer) would be needed, which are more computationally intensive and require specific model types. Please select a tree-based model for SHAP analysis.")
            else:
                st.info("Please select a model above to view its SHAP feature importance.")

        else:
            st.warning("Please select at least one model to train and evaluate.")

    else:
        st.warning("Please select a target variable and at least one input feature to proceed with modeling.")

else:
    st.info("⬆️ Please upload a CSV file or fetch data from a URL to begin your ESG data exploration!")

# ──────────────────────────── Footer ────────────────────────────
st.markdown(
    """
    <hr style='border-color:#333' />
    <p style='font-size:0.85rem;color:#666; text-align: center;'>
        Built using Streamlit, Scikit-learn, Plotly, and SHA.
    </p>
    """,
    unsafe_allow_html=True,
)