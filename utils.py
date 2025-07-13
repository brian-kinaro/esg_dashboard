import streamlit as st
import pandas as pd

# Define the load_data function, as 'utils' module is not available
@st.cache_data
def load_data():
    """Loads the CSV data into a pandas DataFrame."""
    try:
        df = pd.read_csv("fss.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'gpt_analyze.csv' not found. Please ensure the file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
