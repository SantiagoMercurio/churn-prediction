# -*- coding: utf-8 -*-
"""
We train the model and save it so we can use it later without training again.
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb

# Pull in the data-prep logic (so it finds the other file when you run from the project folder)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preparar_datos import load, clean, to_numeric, prepare_for_model


def train(csv_path=None, save_model=True):
    # Load and clean everything
    df = load(csv_path)
    df = clean(df)
    df = to_numeric(df)
    X, y = prepare_for_model(df)

    # Split into train and test (we keep 20% aside to see how it really does)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # We use XGBoost with weights so it doesn't ignore the churners (they're fewer)
    model = xgb.XGBClassifier(
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # See how it did on the part we didn't use for training
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("F1 (overall):", round(f1_score(y_test, y_pred), 4))

    if save_model:
        base = Path(__file__).resolve().parent.parent
        (base / "models").mkdir(exist_ok=True)
        with open(base / "models" / "modelo_churn.pkl", "wb") as f:
            pickle.dump(model, f)
        # Also save column names for when we run predictions later
        with open(base / "models" / "columnas.pkl", "wb") as f:
            pickle.dump(list(X.columns), f)
        print("Model saved to models/modelo_churn.pkl")

    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train()
