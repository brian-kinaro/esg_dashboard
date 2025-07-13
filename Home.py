import streamlit as st

# Page configuration
st.set_page_config(page_title="🌐 AI-Driven ESG Dashboard", layout="wide")

# Custom styles and fonts
st.markdown("""
<style>

  /* Fade-in animation */
  .fade-in {
      animation: fadeIn ease 1.8s forwards;
      text-align: center;
      opacity: 0;
      transform: translateY(20px);
  }
  @keyframes fadeIn {
      to {
          opacity: 1;
          transform: translateY(0);
      }
  }
    
    /* Optional: subtle fade-in animation */
    @keyframes glowFadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    
/* Header styles */
.header-title {
    font-family: 'Orbitron', sans-serif;
    color: #008cff;
    text-shadow:
      0 0 1px #00eaff,
      0 0 1.5px #00eaffaa,
      0 0 2px #00eaffcc;
    letter-spacing: 2px;
    font-size: 2.8rem;
    margin: 0.5rem 0;
}

.header-subtitle {
    font-family: 'Roboto Mono', monospace;
    color: #88ccee;
    font-weight: 50;
    font-size: 1.2rem;
    margin-top: -12px;
    letter-spacing: 1.1px;
}


  /* Footer */
  footer {
      font-family: 'Roboto Mono', monospace;
      color: #666;
      font-size: 0.9rem;
      padding: 1rem 0;
      text-align: center;
      border-top: 1px solid #222;
      margin-top: 3rem;
  }

  /* Responsive styles */

@media screen and (max-width: 768px) {
    .header-title {
        font-size: 6vw;
        letter-spacing: 1px;
    }

    .header-subtitle {
        font-size: 3.5vw;
        margin-top: -8px;
    }

    div[style*="margin-top: 3rem"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
}

@media screen and (max-width: 480px) {
    .header-title {
        font-size: 7vw;
    }

    .header-subtitle {
        font-size: 3vw;
    }

    footer {
        font-size: 0.75rem;
        padding: 0.5rem;
    }
}

</style>

""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='max-width: 1280px; margin: 0 auto;' class='fade-in'>
    <h1 class='header-title'>🌍 ESG Forecast Dashboard</h1>
    <p class='header-subtitle'>Harnessing advanced AI & real-time data to illuminate sustainable futures.</p>
</div>
""", unsafe_allow_html=True)

# Interactive Introduction
with st.expander("🤖 What makes this dashboard?", expanded=False):
    st.markdown("""
    <div style='font-family:"Roboto Mono", monospace; color:#0099cc; max-width: 1280px; margin: 0 auto;'>
      <ul>
        <li><p style='font-family:"Roboto Mono", monospace; color:#0099cc;'><b>Sophisticated Data Integration:</b> Our foundational layer meticulously ingests and transforms disparate ESG indicator data from diverse global and proprietary sources into a structured format, ensuring data integrity and readiness for advanced analytical processes.</p></li>
        <li><p style='font-family:"Roboto Mono", monospace; color:#0099cc;'><b>Real-time Predictive Analytics:</b> Leveraging an optimally deployed machine learning model via Streamlit, the dashboard offers instantaneous, high-fidelity ESG performance forecasts, enabling dynamic scenario analysis and responsive decision-making without client-side computational burden.</p></li>
        <li><p style='font-family:"Roboto Mono", monospace; color:#0099cc;'><b>Intuitive Visualization and Interactive UI:</b> Engineered for exceptional user experience, the interface provides comprehensive visualization capabilities, facilitating intuitive exploration of historical trends, current ESG standing, and future projections through dynamic querying.</p></li>
        <li><p style='font-family:"Roboto Mono", monospace; color:#0099cc;'><b>Actionable Strategic Insights:</b> Beyond forecasting, the dashboard integrates advanced features such as scenario simulation, performance benchmarking against peers, and key indicator impact analysis using model interpretability techniques, empowering stakeholders with profound, actionable insights for strategic planning and targeted interventions.</p></li>
      </ul>
      <p style='font-style: italic; color:#00eaff;'>“Transforming data into foresight.”</p>
    </div>
    """, unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("""
<footer>
  🤖 Built with by <strong>Brian</strong> | Powered by Streamlit
</footer>
""", unsafe_allow_html=True)
