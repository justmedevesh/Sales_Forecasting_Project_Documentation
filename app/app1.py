# -------------------------
# 1️⃣ Imports
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# -------------------------
# 2️⃣ Base Directory Setup (PASTE HERE)
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "lstm_model.h5")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")


# -------------------------
# 3️⃣ Load Model & Scaler
# -------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()


# -------------------------
# 4️⃣ Streamlit UI Starts
# -------------------------
st.title("📊 Sales Forecasting Dashboard")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    if "Sales" not in df.columns:
        st.error("CSV must contain 'Sales' column.")
    else:
        sales_values = df["Sales"].values.reshape(-1,1)
        sales_scaled = scaler.transform(sales_values)

        TIME_STEPS = 30
        X = []

        for i in range(TIME_STEPS, len(sales_scaled)):
            X.append(sales_scaled[i-TIME_STEPS:i, 0])

        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        predictions = model.predict(X)
        predictions_inv = scaler.inverse_transform(predictions)

        df_result = df.iloc[TIME_STEPS:].copy()
        df_result["Predicted_Sales"] = predictions_inv

        st.subheader("Prediction Results")
        st.write(df_result[["Date", "Predicted_Sales"]])

        fig, ax = plt.subplots()
        ax.plot(df_result["Date"], df_result["Predicted_Sales"])
        ax.set_title("Predicted Sales Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        st.pyplot(fig)

        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )