# 📊 Sales Forecasting Project – Rossmann Store Sales

## 📌 Project Overview

This project focuses on predicting daily sales for Rossmann stores using Machine Learning and Time Series techniques.

The objective is to help store managers forecast future sales based on historical data, promotions, holidays, competition distance, and store characteristics.

The project includes:

- Data understanding
- Data cleaning & EDA
- Feature engineering
- Machine Learning modeling (Random Forest)
- Time Series analysis
- Deep Learning (LSTM)
- Model comparison
- Deployment using Streamlit

---

## 🎯 Problem Statement

Rossmann operates over 1,000 stores across Europe. Store sales are influenced by:

- Promotions
- Holidays
- Seasonality
- Store type
- Competition distance
- Assortment type

The goal is to build predictive models that estimate future sales accurately.

---

## 📂 Dataset Description

The dataset consists of:

### 1️⃣ train.csv
Historical daily sales data including:
- Store
- DayOfWeek
- Date
- Sales (Target)
- Customers
- Open
- Promo
- StateHoliday
- SchoolHoliday

### 2️⃣ test.csv
Similar structure as train but without Sales.

### 3️⃣ store.csv
Store-level information:
- StoreType
- Assortment
- CompetitionDistance
- CompetitionOpenSinceMonth
- CompetitionOpenSinceYear
- Promo2
- Promo2SinceWeek
- Promo2SinceYear
- PromoInterval

---

## 🧪 Project Workflow (Notebook Structure)

### 📘 Notebook 01 – Data Understanding
- Loaded datasets
- Checked duplicates
- Checked missing values
- Merged train + store
- Merged test + store

---

### 📘 Notebook 02 – Data Cleaning & EDA
- Missing value treatment
- Outlier detection
- Univariate, Bivariate, Multivariate analysis
- Sales distribution analysis
- Correlation heatmap
- Skewness analysis
- Seasonality analysis
- Holiday effect analysis
- Promo impact analysis
- Competition distance impact
- Weekend vs weekday analysis

---

### 📘 Notebook 03 – Feature Engineering
- Date feature extraction (Year, Month, Week, Day)
- Competition duration calculation
- Promo duration calculation
- Encoding categorical variables
- Feature selection

---

### 📘 Notebook 04 – Machine Learning Modeling
Models trained:
- Baseline Model
- Linear Regression
- Random Forest Regressor

Evaluation Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Random Forest performed best and was selected for deployment.

---

### 📘 Notebook 05 – Time Series Analysis
Performed:
- Stationarity check (ADF Test)
- Autocorrelation (ACF)
- Partial Autocorrelation (PACF)
- Trend and Seasonality analysis

This confirmed that past sales influence future sales.

---

### 📘 Notebook 06 – Deep Learning (LSTM)
Steps:
1. Converted data into time series format
2. Checked stationarity
3. Created sliding window sequences
4. Scaled data to (-1, 1)
5. Built 2-layer LSTM model
6. Trained model
7. Evaluated performance

Random Forest was more stable and selected for deployment.

---

## 📊 Model Comparison

| Model | MAE | RMSE | R² |
|--------|------|-------|------|
| Baseline | High | High | Low |
| Linear Regression | Medium | Medium | Moderate |
| Random Forest | Lowest | Lowest | Highest |
| LSTM | Competitive | Slightly Higher | Good |

✅ Random Forest selected as best model.

---

## 🚀 Deployment (Streamlit App)

The application allows:

- Input: Store ID
- Upload CSV (test.csv format)
- Automatic merging with store-level data
- Feature engineering
- Sales prediction
- Sales visualization
- Download predictions as CSV

---

## 📂 Project Structure

```
sales_forecasting_project/
│
├── app/                          # Streamlit Deployment Application
│   ├── app.py                    # Main Streamlit app
│   └── requirements.txt          # App dependencies
│
├── data/
│   ├── raw_data/                 # Original datasets
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── store.csv
│   │
│   └── processed_data/           # Cleaned & engineered datasets
│       ├── train_merged.csv
│       ├── test_merged.csv
│       ├── train_features.csv
│       ├── train_target.csv
│       ├── X_test.npy
│       └── y_test.npy
│
├── models/                       # Saved trained models
│   ├── random_forest_model.pkl
│   ├── feature_columns.pkl
│   └── lstm_model.h5
│
├── notebooks/                    # Jupyter Notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning_and_EDA.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_machine_learning_modeling.ipynb
│   ├── 05_time_series_analysis.ipynb
│   └── 06_deep_learning_LSTM.ipynb
│
├── README.md
└── .gitignore
```
---

## ⚙️ How to Run the App

### Step 1: Create virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependency
```bash
pip install -r requirements.txt
```

### Step 3: Run StreamLit
```bash
cd app
streamlit run app.py
```

## 📂 Project Structure

The project follows a modular and organized structure separating raw data, processed data, models, notebooks, and deployment code to ensure reproducibility and scalability.

## 📂 Google Drive
All the files and folder where uploaded in google drive because there is large file which is not uploaded in github because it doesnot have capacity for uploade so.
drive link:- 
