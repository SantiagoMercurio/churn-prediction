# -*- coding: utf-8 -*-
"""
Gets the data ready for training: we clean it and turn text columns into numbers.
"""
import pandas as pd
from pathlib import Path


def load(csv_path=None):
    # If no path is given, we look for the usual Telco CSV in data/
    if csv_path is None:
        base = Path(__file__).resolve().parent.parent
        csv_path = base / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(csv_path)
    return df


def clean(df):
    # Drop rows that are missing values in key columns
    df = df.dropna(how="any")
    # TotalCharges sometimes comes as text; we turn it into a number (blanks are already gone after dropna)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    return df


def to_numeric(df):
    # Yes/No columns become 0 and 1 so the model can use them
    churn_map = {"No": 0, "Yes": 1}
    if "Churn" in df.columns:
        df = df.copy()
        df["Churn"] = df["Churn"].map(churn_map)
    return df


def prepare_for_model(df, target="Churn"):
    # Drop columns we don't use (customer ID doesn't help predict)
    drop_cols = ["customerID", target]
    cols = [c for c in df.columns if c not in drop_cols]
    X = df[cols].copy()
    y = df[target] if target in df.columns else None

    # Any text column gets turned into numbers (one-hot)
    X = pd.get_dummies(X, drop_first=True)
    return X, y
