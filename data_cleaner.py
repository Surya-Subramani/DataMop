import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

# -------------------------------
# CLEANING FUNCTIONS
# -------------------------------

def basic_summary(df):
    return {
        "Shape": df.shape,
        "Missing Values": df.isnull().sum().to_dict(),
        "Data Types": df.dtypes.astype(str).to_dict()
    }

def handle_missing_values(df, method="mean"):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["int64", "float64"]:
                if method == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == "median":
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)
    return df

def remove_outliers(df, z_thresh=3):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    mask = (z_scores < z_thresh).all(axis=1)
    return df[mask]

def clean_data(df):
    df = handle_missing_values(df)
    df = remove_outliers(df)
    return df

# -------------------------------
# STREAMLIT INTERFACE
# -------------------------------

st.set_page_config(page_title="🧹 Data Cleaning Assistant", layout="wide")
st.title("🧹 Data Cleaning Assistant")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Raw Data")
    st.dataframe(df)

    st.subheader("📋 Summary Before Cleaning")
    summary = basic_summary(df)
    st.json(summary)

    if st.button("🧼 Clean the Data"):
        cleaned_df = clean_data(df.copy())

        st.subheader("✅ Cleaned Data")
        st.dataframe(cleaned_df)

        st.subheader("📋 Summary After Cleaning")
        cleaned_summary = basic_summary(cleaned_df)
        st.json(cleaned_summary)

        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")
else:
    st.info("👆 Upload a CSV file to get started.")
