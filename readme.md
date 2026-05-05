# Corporación Favorita Grocery Sales Forecasting

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Streamlit](https://img.shields.io/badge/streamlit-ready-orange)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

## Project Overview

This repository contains a streamlined time series forecasting pipeline for grocery sales of a specific item in a store. The goal is to analyze historical sales, test multiple forecasting families, and deliver a minimal Streamlit app for business users to explore future unit sales forecasts.

The solution focuses on:
- cleaning and resampling daily sales data,
- engineering temporal and lag-based features,
- comparing classical and machine learning forecasting approaches,
- serving predictions through a lightweight Streamlit interface.

## Data Architecture & Provenance

The dataset is derived from the Kaggle "Corporación Favorita Grocery Sales Forecasting" competition. The repository includes a simplified daily sales view and supporting external signals to improve forecast quality.

Primary data sources:
- `data/raw/timeseries.csv` — daily unit sales series used as the target and for lag-based feature generation.
- `data/raw/oil.csv` — oil price time series used as an exogenous signal.
- `data/raw/holidays.csv` — holiday calendar used for seasonal and event-aware features.
- `data/processed/timeseries_ABT.csv` — engineered analytic base table used for modeling.

Target and features:
- Target variable: `unit_sales` (daily sales count for the item/store combination)
- Features: date-derived fields, lagged sales, rolling statistics, holiday indicators, oil price, and other exogenous covariates.

## Project Structure

```
.
├── data
│   ├── processed
│   │   ├── holidays.csv
│   │   └── timeseries_ABT.csv
│   └── raw
│       ├── holidays.csv
│       ├── oil.csv
│       └── timeseries.csv
├── ipynb
│   ├── EDA.ipynb
│   ├── ML_Model_Experimentation.ipynb
│   ├── Time_Seris_Model_Experimentation.ipynb
│   ├── hyperparameter_tunnung.ipynb
│   └── register_best_models.ipynb
├── src
│   ├── evaluation
│   │   └── objective.py
│   ├── processing
│   │   ├── numeric.py
│   │   ├── temporal.py
│   │   └── wrangler.py
│   └── prod
│       └── ml_forecast.py
├── streamlit_app
│   └── main.py
├── requirements.txt
├── readme.md
└── .gitignore
```

## Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/Moritz-Binder/corporacion_favorita_grocery_sales_forecasting.git
cd corporacion_favorita_grocery_sales_forecasting
```

2. Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

4. Ensure the data files exist in `data/raw/` and `data/processed/`.

5. Start the app:

```bash
streamlit run streamlit_app/main.py
```

> Note: The Streamlit app currently connects to an MLflow tracking server at `http://127.0.0.1:5000`. Make sure your MLflow server is running and your best models are registered if you want the full production workflow to function.

## Modeling Approach

The repository is designed to compare multiple forecasting approaches, including:
- ARIMA / classical autoregressive methods
- Exponential Smoothing (ETS)
- Prophet
- XGBoost
- RandomForest
- LinearRegression

The primary performance metric used across experiments is Mean Absolute Error (MAE), which is intuitive and robust for business planning where absolute forecast deviation matters.

## Usage / Web App

The Streamlit app provides a simple interface to:
- load the prepared dataset,
- select a forecasting horizon,
- visualize historical sales,
- display future sales forecasts from the trained model.

## Notebook Execution Flow
Follow this recommended order before launching the app:

- `EDA` → `Time_Seris_Model_Experimentation` → `ML_Model_Experimentation`
- `hyperparameter_tunnung` → `register_best_models` → running Streamlit App

To launch the app:

```bash
streamlit run streamlit_app/main.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

## Future Improvements

1. Add a simple model selection panel to compare forecasts from multiple registered models side-by-side.
2. Implement a caching strategy for the MLflow model registry and local fallback models when registry access is unavailable.
3. Extend the app with scenario analysis for promotions, holiday impact, and inventory safety stock recommendations.

