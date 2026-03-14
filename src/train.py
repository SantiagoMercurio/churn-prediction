# -*- coding: utf-8 -*-
"""
We train the model and save it so we can use it later without training again.
Also saves visuals: confusion matrix, ROC curve, and metrics bar chart.
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import xgboost as xgb

# Pull in the data-prep logic (so it finds the other file when you run from the project folder)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preparar_datos import load, clean, to_numeric, prepare_for_model


def train(csv_path=None, save_model=True, tune_hyperparams=True):
    # Load and clean everything
    df = load(csv_path)
    df = clean(df)
    df = to_numeric(df)
    X, y = prepare_for_model(df)

    # Split into train and test (we keep 20% aside to see how it really does)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    base_clf = xgb.XGBClassifier(
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="logloss",
    )

    if tune_hyperparams:
        # Search for better hyperparameters to improve accuracy
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }
        search = RandomizedSearchCV(
            base_clf,
            param_grid,
            n_iter=20,
            scoring="accuracy",
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print("Best params (accuracy):", search.best_params_)
        print("Best validation accuracy:", round(search.best_score_, 4))
    else:
        model = base_clf
        model.fit(X_train, y_train)

    # See how it did on the part we didn't use for training
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("F1 (overall):", round(f1_score(y_test, y_pred), 4))

    base = Path(__file__).resolve().parent.parent
    reports_dir = base / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 1. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (XGBoost)")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved reports/confusion_matrix.png")

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=150)
    plt.close()
    print("Saved reports/roc_curve.png")

    # 3. Metrics bar chart (Accuracy, F1, Recall churn, Precision churn)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, pos_label=1)
    prec = precision_score(y_test, y_pred, pos_label=1)
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1", "Recall (churn)", "Precision (churn)"],
        "Value": [acc, f1, rec, prec],
    })
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=metrics_df, x="Metric", y="Value", color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model metrics (test set)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(reports_dir / "metrics_summary.png", dpi=150)
    plt.close()
    print("Saved reports/metrics_summary.png")

    if save_model:
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
