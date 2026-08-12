# Appliance Energy Forecasting — Time Series Case Study

A 24-hour-ahead forecasting pipeline for household appliance electricity use,
progressing from naive benchmarks through SARIMAX, a feature-based ML model, and a
zero-shot time-series foundation model — all evaluated on the same rolling-origin
backtest so results are directly comparable across every model.

## Dataset

**Appliance Energy Prediction** (Candanedo, Feldheim & Deramaix, 2017) — UCI Machine
Learning Repository. 10-minute-resolution appliance energy use (Wh), indoor sensor
readings (temperature/humidity across 9 rooms), and outdoor weather data from a
low-energy house in Belgium, resampled to hourly resolution for this analysis.

> Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models
> of energy use of appliances in a low-energy house. *Energy and Buildings*, 140,
> 81–97.

The script downloads the dataset automatically (UCI source, with an automatic
fallback to a GitHub mirror of the identical file if the primary source is
unreachable) — no manual download needed.

## Repository structure

```
.
├── appliance_energy_forecasting_pipeline.py  # Main deliverable — run this
├── appliance_energy_forecasting.ipynb        # Equivalent notebook version
├── requirements.txt
├── README.md
├── Figures/     # Generated on run — 11 PNGs covering EDA through model diagnostics
├── Results/     # Generated on run — CSVs for every model's backtest metrics
└── report/      # Written report (6–8 pages) covering analysis, discussion, and
                  # the assignment's required analysis questions
```

`Figures/` and `Results/` are **not checked in** — they're generated fresh each time
the script is run (see below), so anyone cloning the repo can reproduce every number
and plot from scratch.

## How to run

```bash
pip install -r requirements.txt
python appliance_energy_forecasting_pipeline.py
```

By default this reproduces the exact assignment specification (14 rolling-origin
backtest windows; full SARIMAX AIC grid search over p∈[0,6], d∈[0,2], q∈[0,6] — 147
combinations). That full run takes roughly 25–40 minutes on a single CPU core,
mostly the SARIMAX grid search. Useful flags:

- `--quick` — shrink the grid/origins for a fast smoke test before committing to the
  full run
- `--skip-sarimax-search` — skip the slow grid search, use a fixed order instead
- `--skip-chronos` — skip the foundation model step (Part 7)
- `--output-dir DIR` — where `Figures/` and `Results/` are written

Run with `--help` for the full list. Console output doubles as a run log — every
model's backtest metrics and every figure's save path are printed as it goes.

**Alternative — notebook:** `appliance_energy_forecasting.ipynb` covers the identical
Parts 1–7 with the same functions, structured for cell-by-cell inline output instead
of a single script run (`jupyter notebook appliance_energy_forecasting.ipynb`, run
all cells). Useful if you want to inspect intermediate results interactively, but the
`.py` script is the version actually submitted for this assignment.

## The foundation model (Part 7)

Part 7 uses **Chronos-Bolt-small** (Amazon), applied zero-shot (no fine-tuning). It
needs `torch` and `chronos-forecasting` installed, plus internet access the first
time it runs to download the pretrained weights from Hugging Face:

```bash
pip install chronos-forecasting torch
```

Install with a plain command like the one above and let pip resolve compatible
versions on its own — do not manually pin an old exact `transformers` version
alongside it, as that produces a version conflict that breaks the import. If
`torch`/`chronos-forecasting` aren't installed, this step is skipped automatically
with a clear message; the rest of the pipeline is unaffected.

## Methodology summary

- **Target:** `Appliances` (hourly Wh) · **Horizon:** 24 hours ahead
- **Evaluation:** rolling-origin backtest — 14 separate 24h-ahead forecasts, one per
  day, each model trained only on data before its own origin, together spanning
  exactly the last 14 days of the series
- **Metrics:** MAE, RMSE, sMAPE, and MASE (scale-free; primary metric for
  cross-model comparison)
- **Models, in increasing complexity:** mean / naive / seasonal-naive / drift
  benchmarks → SARIMAX (AIC-grid-searched order) → XGBoost & LightGBM (engineered
  lag/time/weather features) → Chronos-Bolt-small (zero-shot foundation model)
- **Data-leakage check:** weather/sensor covariates are built in two variants —
  `realistic` (lagged 24h, genuinely available at forecast time) and `conditional`
  (true future values, an oracle upper bound never available in practice) — with the
  gap between them quantified directly rather than left as an assumption

Full methodological rationale (why each design choice was made, not just what it
does) is documented inline in the script's docstrings and in the written report.

## Results

See `Results/all_models_summary.csv` after running for the full cross-model
comparison table, and `Figures/` for every diagnostic and forecast plot. A narrative
discussion of these results is in the written report.

## Requirements

Python 3.10+. See `requirements.txt` for the full dependency list (numpy, pandas,
matplotlib, scipy, statsmodels, scikit-learn, xgboost, lightgbm; `torch` and
`chronos-forecasting` optional, for Part 7 only).
