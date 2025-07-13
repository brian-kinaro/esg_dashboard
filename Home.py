import streamlit as st

# Page configuration
st.set_page_config(page_title="🌐 AI-Driven ESG Dashboard", layout="wide")

# Custom styles and fonts
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

# Header
st.markdown("""
<div style='max-width: 1280px; margin: 0 auto;' class='fade-in'>
    <h1 class='header-title'>🌍 ESG Dashboard</h1>
    <p class='header-subtitle'>Harnessing advanced AI & real-time data to illuminate sustainable futures.</p>
</div>
""", unsafe_allow_html=True)

# Interactive Introduction - Scholarly Version
with st.expander("🤖 What makes this dashboard?", expanded=False):
    st.markdown("""
    <div style='font-family:"Roboto Mono", monospace; color:#333333; max-width: 1280px; margin: 0 auto;'>

    <ul>

    <li><p style='color:#006699;'>
    <b>Interactive Visualization & User-Centric Design:</b>  
    Designed for both accessibility and analytical depth, the user interface facilitates seamless interaction with ESG metrics. Users can explore longitudinal trends, visualize real-time scores, and manipulate predictive parameters through a suite of interactive charts and query tools.
    </p></li>

    <li><p style='color:#006699;'>
    <b>Real-time Predictive Modeling:</b>  
    The dashboard integrates a fully operational regression-based machine learning model deployed via Streamlit, enabling real-time ESG forecasting. This implementation offloads computation from the client side, providing immediate, high-fidelity predictions suitable for dynamic scenario analysis and rapid strategic decision-making.
    </p></li>

    <li><p style='color:#006699;'>
    <b>Forecasting Capabilities:</b>  
    A core functionality of the dashboard is its ability to generate multi-horizon forecasts for ESG performance metrics across countries and sectors. Leveraging historical patterns captured through machine learning models, the platform projects future ESG trajectories under various policy or investment scenarios—empowering users to anticipate risks, evaluate sustainability trends, and plan proactively.
    </p></li>

    <li><p style='color:#006699;'>
    <b>Strategic Insight Generation:</b>  
    Beyond visual reporting, the dashboard integrates advanced interpretability techniques such as feature importance analysis and scenario simulation. These tools provide stakeholders with actionable intelligence, enabling targeted ESG interventions and evidence-based financial planning.
    </p></li>

    <li><p style='color:#006699;'>
    <b>Advanced ESG Data Integration:</b>  
    This platform incorporates a robust data ingestion pipeline that systematically harmonizes ESG indicators from heterogeneous global and proprietary sources. The data is preprocessed to resolve structural inconsistencies, handle missing values, and align temporal dimensions—thereby ensuring analytical validity and readiness for machine learning deployment.
    </p></li>

    </ul>

    <p style='font-style: italic; color:#00eaff; text-align:center;'>“Transforming complex data into strategic foresight through intelligent and interactive systems.”</p>

    </div>
    """, unsafe_allow_html=True)



# Footer
st.markdown("---")
st.markdown("""
<footer>
  🤖 Built with by <strong>Brian</strong> | Powered by Streamlit
</footer>
""", unsafe_allow_html=True)
