# -*- coding: utf-8 -*-
"""
Simple dashboard to explore the data and churn by segment. Run with: streamlit run dashboard.py
"""
from pathlib import Path

import streamlit as st
import pandas as pd

# Run from the project root so it finds data/
base = Path(__file__).resolve().parent.parent
csv_path = base / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

if not csv_path.exists():
    st.warning("Put the Telco CSV in the data/ folder to view the dashboard.")
    st.stop()

df = pd.read_csv(csv_path)
df = df.dropna(how="any")

st.title("Churn – Quick data overview")
st.write("Number of customers:", len(df))
st.write("Percentage that churned:", round(df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100, 1), "%")

# Filter by contract type
contract = st.selectbox("Filter by contract", ["All"] + list(df["Contract"].unique()))
if contract != "All":
    df = df[df["Contract"] == contract]

st.dataframe(df.head(100))
