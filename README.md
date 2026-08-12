# Appliance Energy Forecasting -- Time Series Case Study

Forecasting short-term household appliance energy demand using the UCI
Appliance Energy Prediction dataset (Candanedo, Feldheim & Deramaix, 2017).
Covers data preparation and EDA, benchmark models, SARIMAX, weather/sensor
covariates, a feature-based ML model (XGBoost/LightGBM), and a zero-shot
time-series foundation model (Chronos-Bolt-small).

## Repository contents

| File | Description |
|---|---|
| `appliance_energy_forecasting_parts1to7.ipynb` | Main notebook -- run this. Fully self-contained: downloads data, runs all analysis, and reproduces every plot and metric for Parts 1-7. |
| `Time_Series_Case_Study_Report.pdf` | Full written report (problem definition, methodology, results, discussion, limitations). |
| `report/` | Source files used to build the report (optional, for reference). |

## How to run

**Recommended: Google Colab** (fastest, no local setup needed)

1. Upload `appliance_energy_forecasting_parts1to7.ipynb` to
   [Google Colab](https://colab.research.google.com).
2. Run all cells top to bottom (`Runtime -> Run all`).

**Local (VS Code / Jupyter)**

1. Use Python 3.13.
2. Install dependencies:
   ```
   pip install -U statsmodels scikit-learn xgboost lightgbm chronos-forecasting torch
   ```
3. Open the notebook in VS Code or Jupyter Lab and run all cells.

> Note: `%pip install` / `%matplotlib inline` are Jupyter magic commands and
> only work inside a notebook environment (Colab, Jupyter, or VS Code's
> notebook interface) -- not in a plain `.py` script or terminal.

## Requirements

- **Python 3.13**
- **Internet connection** -- required to download the dataset and, in Part 7,
  the pretrained Chronos-Bolt-small model weights from Hugging Face (first
  run only).
- **Runtime:** approx. 25-40 minutes end to end. The SARIMAX grid search in
  Part 4 is the slowest single step (~15-20 minutes on a single CPU core).
- No GPU required, though it speeds up Part 7.

## Structure of the notebook

1. **Part 1** -- Data retrieval, cleaning, resampling, EDA, stationarity tests
2. **Part 2** -- Forecasting problem definition and shared evaluation harness
3. **Part 3** -- Benchmark models (naive, seasonal naive, mean, drift)
4. **Part 4** -- SARIMAX (grid search, diagnostics, rolling-origin backtest)
5. **Part 5** -- Feature engineering with weather/sensor covariates
6. **Part 6** -- XGBoost / LightGBM models and feature-group ablation
7. **Part 7** -- Chronos-Bolt-small (zero-shot foundation model)

Full discussion of results, model comparison, and answers to the assignment's
discussion questions are in the accompanying report.

## Dataset

Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). *Data driven
prediction models of energy use of appliances in a low-energy house.*
Energy and Buildings, 140, 81-97.
