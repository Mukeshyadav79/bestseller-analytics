import streamlit as st

from data_processing import load_and_preprocess_data
from model_training import train_model
from view_overview import render_overview
from view_eda import render_eda
from view_predictor import render_predictor

# Set page layout to wide and add title
st.set_page_config(page_title="BestsellerAnalytics",
                   layout="wide", page_icon="📚")

# ------------------------------------------------------------------------------
# Dashboard App Logic / Controller
# ------------------------------------------------------------------------------
st.title("📚 BestsellerAnalytics Dashboard")

# Load Data & Train Model
with st.spinner('Loading data and training pricing engine...'):
    df = load_and_preprocess_data()
    model, encoder, rmse, mae, r2, feat_imp_df, feature_cols = train_model(df)

# Sidebar Navigation
st.sidebar.title("Navigation")
tabs = st.sidebar.radio(
    "Go to:", ["Data Overview", "EDA Dashboard", "Price Predictor"])

# Route to the appropriate view module
if tabs == "Data Overview":
    render_overview(df)
elif tabs == "EDA Dashboard":
    render_eda(df)
elif tabs == "Price Predictor":
    render_predictor(df, model, encoder, rmse, mae, r2, feat_imp_df)
