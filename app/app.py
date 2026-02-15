import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------
# Base Directory Setup
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
feature_path = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
store_path = os.path.join(BASE_DIR, "data", "raw_data", "store.csv")

# ------------------------------------------------
# Load Model + Features + Store Data
# ------------------------------------------------
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

st.title("📊 Store Sales Prediction Dashboard")

# ------------------------------------------------
# Store ID Input
# ------------------------------------------------
store_id = st.number_input("Enter Store ID", min_value=1, step=1)

# ------------------------------------------------
# Upload CSV
# ------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV File (Must contain Date column)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "Date" not in df.columns:
        st.error("CSV must contain a 'Date' column.")
        st.stop()

    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    # ------------------------------------------------
    # Date Processing
    # ------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1

    df["Store"] = store_id

    # ------------------------------------------------
    # Merge Store-Level Data Safely
    # ------------------------------------------------
    if not store_df.empty and "Store" in store_df.columns:
        df = df.merge(store_df, on="Store", how="left")

    # ------------------------------------------------
    # Fill Missing Store Columns Safely
    # ------------------------------------------------
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

    # ------------------------------------------------
    # Encode Categorical Columns Safely
    # ------------------------------------------------
    categorical_cols = ["StoreType", "Assortment", "PromoInterval"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    # ------------------------------------------------
    # Align Features EXACTLY with Training
    # ------------------------------------------------
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_columns]

    # ------------------------------------------------
    # Make Prediction
    # ------------------------------------------------
    try:
        predictions = rf_model.predict(X)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    df["Predicted_Sales"] = predictions

    # ------------------------------------------------
    # Show Results
    # ------------------------------------------------
    st.subheader("Prediction Results")
    st.write(df[["Date", "Predicted_Sales"]])

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Predicted_Sales"])
    ax.set_title("Predicted Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # ------------------------------------------------
    # Download Button
    # ------------------------------------------------
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predicted_sales.csv",
        mime="text/csv"
    )