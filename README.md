# 🏠 California House Price Predictor

An AI-powered web application that estimates the **median house value** for a
California census block group using an XGBoost regression model trained on
the classic California Housing dataset.

---

## 📖 Project Description

Housing prices in California are driven by a complex mix of location,
demographics, income, and proximity to amenities. This project builds an
end-to-end machine-learning pipeline — from raw census data to a live
prediction interface — enabling users to enter block-level details and
instantly receive an estimated property value.

---

## 🤖 About the Machine-Learning Model

| Property | Detail |
|---|---|
| **Algorithm** | XGBoost Regressor |
| **Selection method** | 5-fold GridSearchCV over `n_estimators ∈ {500, 800}`, `learning_rate ∈ {0.05, 0.08}`, `max_depth ∈ {6, 8}` |
| **Optimisation metric** | R² Score |
| **Preprocessing** | OneHotEncoder (drop=first) for `ocean_proximity` · numeric features passed through as-is |
| **Feature engineering** | Three ratio features derived from raw counts: `rooms_per_household`, `bedrooms_per_room`, `population_per_household` |
| **Target** | `median_house_value` (USD, block group median) |
| **Test R²** | ~0.83 |
| **Test MAE** | ~$26 500 |
| **Test RMSE** | ~$40 900 |
| **Dataset** | California Housing — 20 640 block groups; 19 675 after removing capped values |

The full pipeline (OHE preprocessing + XGBoost) is serialised in
`California_House_Price_Prediction.pkl`. Feature engineering is performed
inside the Streamlit app before calling the pipeline, exactly mirroring the
training notebook.

---

## ✨ Application Features

- 📍 **Location inputs** — longitude, latitude, and ocean proximity with
  California-range validation.
- 🏘️ **Housing details** — median house age, total rooms, and total bedrooms
  with sensible defaults derived from dataset medians.
- 👥 **Population & income** — population, households, and median income sliders.
- 🔧 **Derived features preview** — expandable panel showing the three
  engineered ratios calculated from your inputs in real time.
- 💰 **Colour-coded prediction card** — four price bands (Budget / Mid-range /
  Premium / Luxury) with distinct visual styling.
- 📊 **Model metrics panel** — displays MAE and R² alongside the prediction,
  plus an approximate ±1 MAE confidence interval.
- 📍 **Context table** — compares your block's key ratios against California
  dataset medians.
- ✅ **Input validation** — catches impossible values (bedrooms > rooms,
  households > population, out-of-California coordinates).
- 🔍 **Input summary expander** — review every submitted value alongside the
  result.
- 🚀 **Production-ready** — cached model loading, explicit dtype handling to
  prevent sklearn/pandas compatibility issues, and clean error messaging.

---

## 📁 Project Structure

```
california-house-price-predictor/
│
├── app.py                                  # Streamlit application (main entry point)
├── California_House_Price_Prediction.pkl   # Pre-trained XGBoost pipeline
├── dataset.csv                             # Original training dataset (reference only)
├── California_House_Price_Predictor.ipynb  # Model-training notebook (reference only)
├── requirements.txt                        # Pinned Python dependencies
└── README.md                               # This file
```

> **Note:** `dataset.csv` and `California_House_Price_Predictor.ipynb` are
> not required at runtime. Only `app.py`, `California_House_Price_Prediction.pkl`,
> and `requirements.txt` are needed for deployment.

---

## ⚙️ Installation & Local Setup

### Prerequisites

- Python **3.11** or **3.12** (recommended)
- `pip` (bundled with Python)
- `git` (optional, for cloning)

### 1 — Clone or download the repository

```bash
git clone https://github.com/your-username/california-house-price-predictor.git
cd california-house-price-predictor
```

Or place `app.py`, `California_House_Price_Prediction.pkl`, and
`requirements.txt` in the same folder.

### 2 — Create a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4 — Run the application

```bash
streamlit run app.py
```

Streamlit will automatically open `http://localhost:8501` in your browser.

---

## 🖱️ How to Use the Application

1. **Enter coordinates** — type or adjust the longitude and latitude for the
   California location you want to evaluate.
2. **Select ocean proximity** — choose the category that best describes the
   block's relationship to the coast or bay.
3. **Set housing details** — use the slider for house age and number inputs
   for total rooms and bedrooms.
4. **Enter population figures** — fill in population, households, and median
   income (in tens of thousands of USD).
5. **Preview derived features** — expand the collapsible panel to see the
   three ratio features the model will actually receive.
6. **Click "Predict House Price"** — the model runs instantly.
7. **Read your result** — a colour-coded card shows the predicted median value
   with its price band.
8. **Check model accuracy context** — the MAE, R², and approximate confidence
   range are shown alongside the prediction.
9. **Compare with California medians** — the context table shows how your
   block's ratios compare to the dataset-wide medians.

---

## ⚠️ Assumptions & Limitations

| Item | Detail |
|---|---|
| **Geographic scope** | The model was trained exclusively on California data (lon: -124.35 to -114.31, lat: 32.54 to 41.95). Inputs outside this range are rejected. |
| **Value ceiling** | The training data excludes houses valued at $500 001 (a census cap), so predictions near or above $500 000 should be treated with extra caution. |
| **Block-level, not house-level** | All features are block-group aggregates. The model predicts the *median* for the block, not the price of a single house. |
| **Static model** | The model reflects 1990 California census data and does not account for modern market conditions, inflation, or recent developments. |
| **Income scale** | `median_income` is in units of **tens of thousands of USD** (e.g. 3.5 = $35 000). |
| **Model version sensitivity** | The pickle was retrained with scikit-learn 1.9.0 and XGBoost 3.3.0. Using different versions may produce warnings or errors. |

---

## 🛠️ Technologies & Libraries

| Library | Version | Purpose |
|---|---|---|
| [Streamlit](https://streamlit.io) | 1.59.2 | Web application framework |
| [scikit-learn](https://scikit-learn.org) | 1.9.0 | OneHotEncoder, ColumnTransformer, Pipeline, GridSearchCV |
| [XGBoost](https://xgboost.readthedocs.io) | 3.3.0 | Gradient-boosting regression model |
| [pandas](https://pandas.pydata.org) | 3.0.2 | DataFrame construction for model input |
| [NumPy](https://numpy.org) | 2.4.4 | Numerical operations |
| [joblib](https://joblib.readthedocs.io) | 1.5.3 | Model serialisation / deserialisation |
| [PyArrow](https://arrow.apache.org/docs/python/) | 24.0.0 | pandas 3.x string-dtype backend |

---

## 🌐 Live Demo

**Live Demo: https://your-streamlit-app-url.streamlit.app**

> Replace the URL above after deploying to
> [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## 🚀 Deploying to Streamlit Community Cloud

1. Push your project folder to a **public GitHub repository** containing:
   - `app.py`
   - `California_House_Price_Prediction.pkl`  ← use the **retrained** version
   - `requirements.txt`
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository and branch, set the main file
   path to `app.py`.
4. Click **Deploy** — Streamlit installs the requirements automatically.
5. Copy the generated URL and update the **Live Demo** link above.

---

*Made with ❤️ using Streamlit and XGBoost.*
