# Churn prediction

Predict which customers are most likely to leave (churn), with cohort analysis and explainability (SHAP). Ready for portfolio and GitHub.

---

## Proof it works

The project has been run end-to-end. Here are the results:

**Model comparison (test set)**  
| Model | Accuracy | F1 | Recall (churn) |
|-------|----------|-----|---------------|
| Baseline (dummy) | 0.50 | 0.39 | 0.50 |
| Logistic Regression | 0.78 | 0.57 | 0.62 |
| **XGBoost** | **0.75** | **0.60** | **0.68** |

XGBoost gives the best recall on churners (who we care most about). Full table: [reports/model_comparison.csv](reports/model_comparison.csv).

**Sample predictions**  
Each customer gets a churn prediction (0/1) and a probability. Example: [reports/predictions_sample.csv](reports/predictions_sample.csv).

**Pipeline**  
1. Data in `data/` → 2. `python src/train.py` → 3. `python src/predict.py` → 4. Predictions in `reports/`.

---

## Cómo lo corro (resumen)

**Sí tienes que instalar** las librerías una vez. Luego solo necesitas el CSV de datos.

1. **Abre terminal** en la carpeta del proyecto: `01_churn_prediction` (donde está este README).

2. **Instala dependencias (solo la primera vez):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Descarga el dataset** de Kaggle [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) y guarda el CSV dentro de la carpeta `data/`. Si el archivo se llama distinto, renómbralo a `WA_Fn-UseC_-Telco-Customer-Churn.csv` o cambia el nombre en el código.

4. **Entrenar el modelo:**
   ```bash
   python src/train.py
   ```
   Se crean los archivos en `models/`.

5. **Hacer predicciones:**
   ```bash
   python src/predict.py
   ```
   Se genera `reports/predictions.csv`.

**Opcional:** para ver todo el análisis y la comparación de modelos, abre y ejecuta los notebooks en orden: `01_eda_churn.ipynb` → `02_modelado_churn.ipynb`.

---

## What’s in the repo

- **Notebooks:** EDA (`01_eda_churn.ipynb`) and modeling with baseline comparison (`02_modelado_churn.ipynb`).
- **Scripts:** `src/train.py` (train and save model), `src/predict.py` (run predictions), `src/dashboard.py` (Streamlit).
- **Data:** You add the CSV (not in repo). **Models** and **reports** are generated when you run the project.

## Dataset

Download **Telco Customer Churn** from Kaggle and save the CSV in `data/`:

- [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Use the default filename `WA_Fn-UseC_-Telco-Customer-Churn.csv` or change it in the code.

## How to run (show it works)

From the project root (`01_churn_prediction/`):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put the Telco CSV in data/

# 3. Train (saves model to models/)
python src/train.py

# 4. Predict (writes reports/predictions.csv)
python src/predict.py
```

Optional:

- Run the notebooks in order for full EDA and model comparison (baseline, logistic regression, XGBoost).
- Dashboard: `streamlit run src/dashboard.py`

## Results you’ll get

- **reports/model_comparison.csv** – Accuracy, F1, Recall (churn) for baseline, logistic regression, and XGBoost (from the notebook).
- **reports/shap_summary.png** – Which variables drive the prediction (from the notebook).
- **reports/predictions.csv** – Churn prediction and probability per customer (from `predict.py`).

After running once, you can commit these files so the repo clearly shows the project works.

## Project structure

```
├── data/              # Put Telco CSV here (not in repo)
├── models/             # Saved model after train.py (not in repo)
├── notebooks/          # EDA + modeling
├── reports/            # model_comparison.csv, predictions_sample.csv, etc.
├── src/
│   ├── preparar_datos.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── dashboard.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Publish on GitHub

1. **Initialize git** (if not already):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: churn prediction project"
   ```

2. **Create a new repo on GitHub:**  
   Go to [github.com/new](https://github.com/new). Name it e.g. `churn-prediction`. Do **not** add a README or .gitignore (you already have them).

3. **Push this folder to GitHub:**
   ```bash
   git remote add origin https://github.com/TU_USUARIO/churn-prediction.git
   git branch -M main
   git push -u origin main
   ```
   Replace `TU_USUARIO` with your GitHub username.
