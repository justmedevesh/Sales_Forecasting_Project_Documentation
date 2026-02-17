# ============================================================
# SALES FORECASTING STREAMLIT APPLICATION
# FINAL DEVELOPMENT VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# ============================================================
# BASE DIRECTORY SETUP (Corrected for app folder structure)
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "Random_Forest_17-02-2026-08-15-19.pkl")
feature_path = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
store_path = os.path.join(BASE_DIR, "data", "raw_data", "store.csv")
# ============================================================
# LOAD MODEL + FEATURES + STORE DATA
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_path)

    if os.path.exists(store_path):
        store_df = pd.read_csv(store_path)
    else:
        store_df = pd.DataFrame()

    return model, feature_columns, store_df


rf_model, feature_columns, store_df = load_artifacts()

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📊 Rossmann Store Sales Prediction Dashboard")

st.sidebar.header("Model Information")
st.sidebar.write("Model Type: Random Forest Regressor")
st.sidebar.write("Version: 17-02-2026")
st.sidebar.write("Forecast Horizon: 6 Weeks")

# ============================================================
# STORE ID INPUT
# ============================================================

store_id = st.number_input("Enter Store ID", min_value=1, step=1)

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload CSV File (Must contain 'Date' column)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if df.empty:
        st.error("Uploaded CSV file is empty.")
        st.stop()

    if "Date" not in df.columns:
        st.error("CSV must contain a 'Date' column.")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    if st.checkbox("Show Uploaded Data Preview"):
        st.write(df.head())

    # ========================================================
    # DATE FEATURE ENGINEERING
    # ========================================================

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1
    df["IsWeekend"] = df["DayOfWeek"].isin([6,7]).astype(int)

    df["Store"] = store_id

    # ========================================================
    # MERGE STORE-LEVEL DATA
    # ========================================================

    if not store_df.empty and "Store" in store_df.columns:
        df = df.merge(store_df, on="Store", how="left")

    # ========================================================
    # HANDLE MISSING STORE VALUES
    # ========================================================

    store_related_cols = [
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
        "StoreType",
        "Assortment"
    ]

    for col in store_related_cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("None")
            else:
                df[col] = df[col].fillna(0)

    # ========================================================
    # CATEGORICAL ENCODING (Must Match Training Method)
    # ========================================================

    # Apply same dummy encoding as training
    df = pd.get_dummies(df)

    # Strictly match training feature columns
    X = df.reindex(columns=feature_columns, fill_value=0)

    # ========================================================
    # ALIGN FEATURES EXACTLY AS TRAINING
    # ========================================================

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_columns]

    # ========================================================
    # PREDICTION
    # ========================================================

    try:
        predictions = rf_model.predict(X)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    df["Predicted_Sales"] = predictions

    # ========================================================
    # SALES SUMMARY METRICS
    # ========================================================

    st.subheader("📈 Sales Summary")

    total_sales = df["Predicted_Sales"].sum()
    avg_sales = df["Predicted_Sales"].mean()
    max_sales = df["Predicted_Sales"].max()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Predicted Sales", f"{total_sales:,.0f}")
    col2.metric("Average Daily Sales", f"{avg_sales:,.0f}")
    col3.metric("Max Predicted Sales", f"{max_sales:,.0f}")

    # ========================================================
    # SHOW RESULTS TABLE
    # ========================================================

    st.subheader("📅 Prediction Results")

    st.write(df[["Date", "Predicted_Sales"]])

    # ========================================================
    # VISUALIZATION
    # ========================================================

    st.subheader("📊 Sales Trend")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["Date"], df["Predicted_Sales"], marker="o")
    ax.set_title("Predicted Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.grid(True)

    st.pyplot(fig)

    # ========================================================
    # DOWNLOAD BUTTON
    # ========================================================

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predicted_sales.csv",
        mime="text/csv"
    )

st.write("------------------------------------------------------------")
st.write("Developed for Internship Project - Sales Forecasting System")