# Retail demand forecast → inventory decision

This repository forecasts daily Olist product-category line-item demand and turns the resulting distributional forecast into a stocking policy. Results are deliberately generated only after the public source data are downloaded: this repository contains no fabricated headline metrics.

## Result

Run `python scripts/00_prepare.py` through `python scripts/03_report.py` (below). The pipeline writes the real model ladder to `reports/metrics_models.csv`, including WAPE, MAE, RMSE, bias, and improvement versus SeasonalNaive; it writes the actual q50/q75/q90/q95 policy comparison to `reports/newsvendor_policies.csv`. WAPE is pooled over the portfolio, and bias accompanies it because absolute error can hide systematic understocking.

With understock:overstock cost of 3:1, the critical ratio is 75%; the recommended available forecast is q75. This is an assumption-driven decision, not a claimed saving: change costs in `NewsvendorCosts` and rerun the decision script.

## Run

```powershell
pip install -r requirements.txt
# Download the four Olist files to data/raw; see data/README.md
python scripts/00_prepare.py
python scripts/01_backtest.py
python scripts/02_decision.py
python scripts/03_report.py
pytest
```

The backtest is six disjoint 28-day rolling origins. DirectGBM uses only target features lagged at least to the forecast bucket, so no prediction can consume an unobservable future actual.

## Layout

- `src/rdf/data`: Olist construction, schema validation, and dense zero-day calendar.
- `src/rdf/features`: known-ahead calendar and leakage-safe lag features.
- `src/rdf/models`: baseline ladder, ETS, and global direct Tweedie LightGBM.
- `src/rdf/evaluation`: rolling-origin evaluation and zero-safe metrics.
- `src/rdf/decision`: newsvendor policies and cost-ratio sensitivity.

## Limits

This is one Brazilian marketplace over roughly two years, at category rather than SKU level. It has no promotion or price covariates. Orders approximate observed sales rather than latent demand under stockouts. The Olist data licence is CC BY-NC-SA 4.0; raw data are not included.
