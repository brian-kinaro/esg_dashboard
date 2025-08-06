import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress specific FutureWarnings from Plotly
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly.graph_objs")

try:
    from fpdf import FPDF
    from io import BytesIO
except ImportError:
    FPDF = None
    BytesIO = None
    st.warning("`fpdf` library not found. PDF export functionality will be disabled. "
               "Please install it using `pip install fpdf2` if you need this feature.")

# --- 1. Page Configuration and Custom Styling ---

# Page configuration and styles

st.set_page_config(page_title="🌍 Africa ESG Visualization", layout="wide")


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

# --- Header Section ---
st.markdown(
    """
    <div class="fade-in">
        <h1> 🌍 ESG Visualizations </h1>
        <p class="subtitle">Exploring Environmental, Social, and Governance Insights Across African Nations</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 2. Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data(file_path: str):
    """
    Loads, preprocesses, and scales ESG data.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing the processed DataFrame and a list of ESG display variables.
    """
    try:
        df_raw = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    if df_raw.empty:
        st.error("`fss.csv` returned an empty DataFrame. Please check your data source.")
        st.stop()

    # Identify ESG variables, excluding 'Country', 'Year', and normalized columns
    esg_display_vars = [c for c in df_raw.columns if c not in ["Country", "Year"] and not c.endswith('_norm')]

    if "FSS" not in esg_display_vars:
        st.error("Composite ESG score column `FSS` missing in dataset. Please ensure your data contains this column.")
        st.stop()

    # Convert ESG columns to numeric, coercing errors
    for col in esg_display_vars:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df = df_raw.copy()

    # Apply Min-Max Scaling to all ESG display variables
    scaler = MinMaxScaler((0, 100))
    # Drop rows with any NaN in ESG columns before scaling, then re-merge or handle
    # A more robust approach might be to impute NaNs if appropriate for the data
    df_scaled = df[esg_display_vars].dropna()
    if not df_scaled.empty:
        df[esg_display_vars] = scaler.fit_transform(df[esg_display_vars])
    else:
        st.warning("All ESG data appears to be missing for scaling. Displayed values might be incorrect.")

    return df, esg_display_vars

# Load data
df, esg_display_vars = load_and_preprocess_data('fss.csv')
esg_model_features = [col for col in esg_display_vars if col != "FSS"]

# --- Country ISO Mapping ---
# It's better to define this as a constant outside functions if it's static
COUNTRY_NAME_TO_ISO = {
    "Algeria":"DZA","Angola":"AGO","Benin":"BEN","Botswana":"BWA","Burkina Faso":"BFA","Burundi":"BDI",
    "Cabo Verde":"CPV","Cameroon":"CMR","Central African Republic":"CAF","Chad":"TCD","Comoros":"COM",
    "Congo, Dem. Rep.":"COD","Congo, Rep.":"COG","Côte d'Ivoire":"CIV","Cote d'Ivoire":"CIV","Djibouti":"DJI",
    "Egypt":"EGY","Egypt, Arab Rep.":"EGY","Equatorial Guinea":"GNQ","Eritrea":"ERI","Eswatini":"SWZ",
    "Ethiopia":"ETH","Gabon":"GAB","Gambia":"GMB","Gambia, The":"GMB","Ghana":"GHA","Guinea":"GIN",
    "Guinea-Bissau":"GNB","Kenya":"KEN","Lesotho":"LSO","Liberia":"LBR","Libya":"LBY","Madagascar":"MDG",
    "Malawi":"MWI","Mali":"MLI","Mauritania":"MRT","Mauritius":"MUS","Morocco":"MAR","Mozambique":"MOZ",
    "Namibia":"NAM","Niger":"NER","Nigeria":"NGA","Rwanda":"RWA","Sao Tome and Principe":"STP",
    "Senegal":"SEN","Seychelles":"SYC","Sierra Leone":"SLE","Somalia":"SOM","South Africa":"ZAF",
    "South Sudan":"SSD","Sudan":"SDN","Tanzania":"TZA","Togo":"TGO","Tunisia":"TUN","Uganda":"UGA",
    "Zambia":"ZMB","Zimbabwe":"ZWE"
}

df["ISO"] = df["Country"].map(COUNTRY_NAME_TO_ISO)
df_africa = df[df["ISO"].isin(COUNTRY_NAME_TO_ISO.values())].copy()

if df_africa.empty:
    st.error("No African countries with valid ISO codes found in the dataset after mapping. Please check 'Country' names in your `fss.csv` file.")
    st.stop()

# --- 3. Choropleth Map Section ---
st.markdown("---")
st.subheader("🗺️ Africa ESG Choropleth Map")
st.write("Visualize ESG indicators across African countries for a selected year.")

col_map_year, col_map_metric = st.columns([1, 2]) # Adjust column width ratio
with col_map_year:
    # Ensure year selection is clean
    available_years = sorted(df_africa["Year"].astype(int).unique(), reverse=True)
    if not available_years:
        st.warning("No years available for mapping.")
        year_sel = None
    else:
        year_sel = st.selectbox("Select Year", available_years, help="Choose the year to display data for on the map.")

with col_map_metric:
    metric_sel = st.selectbox("Select Metric", esg_display_vars, help="Choose the ESG indicator to visualize.")

if year_sel is not None:
    map_df = df_africa[df_africa["Year"] == year_sel].dropna(subset=[metric_sel])

    if not map_df.empty:
        fig = px.choropleth(map_df, locations="ISO", locationmode="ISO-3", color=metric_sel,
                            hover_name="Country",
                            color_continuous_scale=px.colors.sequential.Plasma, # Use a more vibrant, perceptually uniform colormap
                            scope="africa",
                            title=f"<b>{metric_sel} in Africa ({year_sel})</b>",
                            template="plotly_dark", # Consistent dark theme for charts
                            height=650) # Slightly increased height
        fig.update_layout(margin=dict(l=0, r=0, t=100, b=0),
                          coloraxis_colorbar=dict(title=f"Score (0-100)", thicknessmode="pixels", thickness=30),
                          title_font_size=24,
                          geo=dict(
                              bgcolor='rgba(0,0,0,0)', # Transparent background for map
                              landcolor='#334756', # Darker land color
                              coastlinecolor='#ffffff', # White coastlines
                              coastlinewidth=0.8,
                              subunitcolor='#ffffff', # White subunits
                              subunitwidth=0.4,
                              showland=True,
                              showlakes=True, lakecolor='#0c1a2e', # Dark lake color
                              showocean=True, oceancolor='#0c1a2e', # Dark ocean color
                              showcountries=True, countrycolor="#ffffff" # White country borders
                          ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No data available for **{metric_sel}** in **{year_sel}** to display on the map. Please select a different year or metric.")

# --- 4. Tabbed Navigation for Detailed Analytics ---
st.markdown("---")
overall_tab, country_tab, corr_tab = st.tabs(["🌍 Overall Summary", "📌 Country Breakdown", "📈 Correlation & Trends"])

with overall_tab:
    st.header("Overview of ESG Indicators in Africa")
    st.write(f"""
        This section provides a high-level summary of the ESG landscape across African countries.
        The data covers {df_africa['Country'].nunique()} countries from {int(df_africa['Year'].min())} to {int(df_africa['Year'].max())}.
        All scores are normalized to a scale of 0-100, where higher values indicate better performance.
    """)
    st.info("Navigate to the 'Country Breakdown' tab for detailed country-specific insights, or 'Correlation & Trends' for deeper analytical views.")

    # Display some key statistics or a summary table (e.g., latest year's averages)
    latest_year_data = df_africa[df_africa["Year"] == df_africa["Year"].max()]
    if not latest_year_data.empty:
        st.subheader(f"Key Averages for {int(df_africa['Year'].max())}")
        avg_esg_scores = latest_year_data[esg_display_vars].mean().sort_values(ascending=False)
        st.dataframe(avg_esg_scores.rename("Average Score (0-100)").reset_index().rename(columns={'index': 'ESG Indicator'}),
                     hide_index=True,
                     use_container_width=True)
    else:
        st.info("No data available for the latest year to display overall averages.")

with country_tab:
    st.header("Country-Specific ESG Breakdown")
    st.write("Dive into the ESG performance of individual African nations over time.")
    st.markdown("---")

    countries_sorted = sorted(df_africa["Country"].unique())
    ctry = st.selectbox("Select a Country", countries_sorted, help="Choose an African country to see its ESG performance.")

    if ctry:
        c_df = df_africa[df_africa["Country"] == ctry].sort_values("Year")

        if not c_df.empty:
            st.subheader(f"📈 {ctry} – Composite ESG Score (FSS) Over Time")
            fig_fss = px.line(c_df, x="Year", y="FSS", markers=True,
                              title=f"<b>{ctry} – Composite ESG Score (FSS) (0-100)</b>",
                              template="plotly_dark",
                              labels={"FSS": "FSS Score"},
                              line_shape="spline") # Smooth the line
            fig_fss.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')),
                                  line=dict(color='#00eaff', width=3))
            fig_fss.update_layout(xaxis_title="Year", yaxis_title="Score", title_x=0.5)
            st.plotly_chart(fig_fss, use_container_width=True)

            st.markdown("---")
            st.subheader(f"🕸️ {ctry} – Latest ESG Snapshot (Radar Chart)")
            latest = c_df.iloc[-1]
            radar_vals = latest[esg_model_features].tolist()
            radar_lbls = esg_model_features

            if radar_vals and all(not pd.isna(val) for val in radar_vals):
                # Append first value to close the radar chart loop
                radar_vals.append(radar_vals[0])
                radar_lbls.append(radar_lbls[0])

                fig_c_radar = go.Figure(go.Scatterpolar(
                    r=radar_vals,
                    theta=radar_lbls,
                    mode="lines+markers", # Show markers on radar
                    fill="toself",
                    fillcolor="rgba(255,165,0,0.2)", # Lighter fill for radar
                    line=dict(color="#ffa500", width=3), # Orange line for radar
                    marker=dict(size=8, color="#ffa500")
                ))
                fig_c_radar.update_layout(
                    title=f"<b>{ctry} – ESG Component Scores ({int(latest['Year'])}) (0-100)</b>",
                    polar=dict(
                        radialaxis=dict(
                            range=[0, 100],
                            showticklabels=True,
                            tickcolor="#666",
                            gridcolor="#333",
                            linecolor="#666"
                        ),
                        angularaxis=dict(
                            linecolor="#666",
                            tickfont=dict(size=12, color="#c0c0c0")
                        )
                    ),
                    template="plotly_dark",
                    showlegend=False,
                    height=550, # Adjusted height for better radar chart display
                    title_x=0.5
                )
                st.plotly_chart(fig_c_radar, use_container_width=True)
            else:
                st.info(f"No complete ESG component data available for {ctry} in {int(latest['Year'])} for the radar chart.")
        else:
            st.warning(f"No ESG data available for {ctry}.")

with corr_tab:
    st.header("Correlation Matrix & Average Trendlines")
    st.write("Understand the relationships between different ESG indicators and observe their average trends across Africa.")
    st.markdown("---")

    st.subheader("📊 Correlation Matrix")
    sel_year = st.slider("Select year for correlation matrix",
                         int(df_africa["Year"].min()),
                         int(df_africa["Year"].max()),
                         int(df_africa["Year"].max()), # Default to latest year
                         help="Choose a year to calculate the correlation between ESG indicators for all countries.")

    corr_df = df_africa[df_africa["Year"] == sel_year][esg_display_vars].corr()

    if not corr_df.empty:
        fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", # Reverse diverging colormap
                             title=f"<b>Correlation Matrix of ESG Indicators ({sel_year})</b>",
                             labels=dict(color="Correlation"), # Add colorbar label
                             aspect="auto") # Adjust aspect for better readability
        fig_corr.update_layout(template="plotly_dark", height=650, title_x=0.5,
                               xaxis_showgrid=False, yaxis_showgrid=False, # Hide gridlines for cleaner look
                               xaxis_tickangle=-45) # Angle x-axis labels
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # --- Automatic highlights: top positive & negative correlations ---
        try:
            # Ensure corr_df is numeric and has matching shape
            if corr_df.empty:
                st.info("Correlation matrix is empty — no highlights to show.")
            else:
                # Keep only upper triangle to avoid duplicate pairs and self-correlations
                mask = np.triu(np.ones(corr_df.shape), k=1).astype(bool)
                corr_upper = corr_df.where(mask)
        
                # Stack and reset index to get pairwise rows
                corr_pairs = corr_upper.stack().reset_index()
                corr_pairs.columns = ["Variable A", "Variable B", "Correlation"]
        
                if corr_pairs.empty:
                    st.info("No pairwise correlations found (matrix may be degenerate).")
                else:
                    # Sort to get top positive and top negative correlations
                    top_pos = corr_pairs.sort_values("Correlation", ascending=False).head(5).copy()
                    top_neg = corr_pairs.sort_values("Correlation", ascending=True).head(5).copy()
        
                    # Round correlation values for display
                    top_pos["Correlation"] = top_pos["Correlation"].round(3)
                    top_neg["Correlation"] = top_neg["Correlation"].round(3)
        
                    # Display side-by-side
                    st.markdown("### 🔝 Top correlations highlights")
                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.subheader("Top 5 Positive Correlations")
                        st.dataframe(top_pos.reset_index(drop=True), use_container_width=True)
                    with col_neg:
                        st.subheader("Top 5 Negative Correlations")
                        st.dataframe(top_neg.reset_index(drop=True), use_container_width=True)
        
                    # Expandable interpretation section (automatically references the table values)
                    with st.expander(f"🔍 Interpretations & Key Observations for {sel_year}", expanded=False):
                        st.markdown(
                            f"**Summary for {sel_year}:** The table above lists the 5 strongest positive and negative pairwise linear correlations "
                            "between the selected ESG indicators across countries. Positive values indicate variables that increase together; "
                            "negative values indicate variables that move in opposite directions."
                        )
        
                        st.write("**Top Positive Correlations (interpretation):**")
                        for _, row in top_pos.iterrows():
                            st.write(f"- **{row['Variable A']} ↔ {row['Variable B']}**: {row['Correlation']} — variables that tend to move together.")
        
                        st.write("**Top Negative Correlations (interpretation):**")
                        for _, row in top_neg.iterrows():
                            st.write(f"- **{row['Variable A']} ↔ {row['Variable B']}**: {row['Correlation']} — variables that tend to move in opposite directions.")
        
                        st.markdown(
                            "⚠️ *Note:* Correlation is linear and does not imply causation. Consider investigating outliers, sample sizes, "
                            "or using domain knowledge before drawing strong conclusions."
                        )
        
        except Exception as e:
            st.warning(f"Could not compute automatic correlation highlights: {e}")


    st.markdown("---")
    st.subheader("📈 Average Trendlines for Selected Indicators")
    st.write("Observe how the average scores for specific ESG indicators have evolved over time across all African countries.")

    sel_inds = st.multiselect("Choose indicators to display trendlines",
                              esg_display_vars,
                              default=["FSS"] if "FSS" in esg_display_vars else [], # Default to FSS
                              help="Select one or more ESG indicators to view their average trend across Africa.")

    if sel_inds:
        # Calculate the mean for selected indicators, dropping NaNs before mean calculation
        tr_df = df_africa.groupby("Year")[sel_inds].mean(numeric_only=True).reset_index()

        if not tr_df.empty:
            fig_tr = px.line(tr_df, x="Year", y=sel_inds, markers=True,
                             template="plotly_dark",
                             title="<b>Average Trend Across Africa (0-100 Scale)</b>",
                             labels={"value": "Average Score", "variable": "ESG Indicator"},
                             line_shape="spline") # Smooth the line
            fig_tr.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
            fig_tr.update_layout(xaxis_title="Year", yaxis_title="Average Score", title_x=0.5)
            st.plotly_chart(fig_tr, use_container_width=True)
        else:
            st.info("No data to display trendlines for the selected indicators.")
    else:
        st.info("Please select at least one indicator to display trendlines.")

# --- 5. Footer Section ---
st.markdown("---")
st.markdown(
    """
    <p style='font-size:0.9rem;color:#888888; text-align: center;'>
        Data-driven sustainability visualization.
    </p>
    """,
    unsafe_allow_html=True,
)

# Optional: Add a PDF export functionality if fpdf is available
if FPDF:
    def create_pdf_report(dataframe, country, year):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"ESG Report for {country} ({year})", 0, 1, "C")
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)

        pdf.multi_cell(0, 10, f"This report provides an overview of the Environmental, Social, and Governance (ESG) performance for {country} in {year}, based on the available data.")
        pdf.ln(5)

        # Add FSS score
        fss_score = dataframe[dataframe['Country'] == country][dataframe['Year'] == year]['FSS'].iloc[0]
        pdf.cell(0, 10, f"Composite ESG Score (FSS): {fss_score:.2f} (0-100 scale)", 0, 1)
        pdf.ln(5)

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Individual ESG Component Scores:", 0, 1)
        pdf.set_font("Arial", "", 12)

        latest_data = dataframe[(dataframe['Country'] == country) & (dataframe['Year'] == year)].iloc[0]
        for feature in esg_model_features:
            score = latest_data.get(feature, 'N/A')
            if score != 'N/A':
                pdf.cell(0, 7, f"- {feature}: {score:.2f}", 0, 1)
            else:
                pdf.cell(0, 7, f"- {feature}: N/A", 0, 1)
        pdf.ln(10)
        
        pdf_output = BytesIO()
        pdf.output(pdf_output)
        return pdf_output.getvalue()

    # Place the PDF button in the country breakdown tab or at the end
    with country_tab:
        if ctry: # Only show if a country is selected
            if st.button(f"Download ESG Report for {ctry}"):
                latest_year_for_report = df_africa[df_africa["Country"] == ctry]["Year"].max()
                if latest_year_for_report is not None:
                    pdf_bytes = create_pdf_report(df_africa, ctry, int(latest_year_for_report))
                    st.download_button(
                        label="Click to Download PDF",
                        data=pdf_bytes,
                        file_name=f"ESG_Report_{ctry}_{int(latest_year_for_report)}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.warning(f"No data available to generate a report for {ctry}.")
