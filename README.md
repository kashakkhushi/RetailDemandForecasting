<!-- Every number below comes from the committed outputs in reports/ (metrics_models.csv,
     newsvendor_policies.csv, newsvendor_sensitivity.csv, backtest_gbm.csv).
     Regenerate with: python scripts/01_backtest.py && python scripts/02_decision.py -->

# Retail Demand Forecasting & Inventory Planning

**Forecasting daily demand for a Brazilian e-commerce marketplace, and turning the
forecast into an order quantity.**

A forecast is not a decision. This project runs a full baseline ladder over 32 product
categories, picks a winner honestly, and then answers the question an operations team
actually asks: *given this forecast, how much should we order?*

The answer is not "order the forecast."

---

## Results

Rolling-origin backtest, 28-day horizon, 6 disjoint origins, 5,376 scored series-days
per model.

| Model | WAPE | MAE | RMSE | Bias | vs seasonal naive |
|---|---:|---:|---:|---:|---:|
| **Holt-Winters (ETS)** — damped trend, weekly seasonality | **0.406** | **3.06** | **4.82** | **+0.01** | **+17.9%** |
| Croston-SBA | 0.416 | 3.13 | 4.92 | −0.34 | +15.9% |
| Moving average (28d) | 0.422 | 3.18 | 4.96 | −0.13 | +14.7% |
| Seasonal moving average (7×4) | 0.431 | 3.24 | 4.98 | −0.13 | +12.9% |
| DirectGBM — global LightGBM, Tweedie | 0.433 | 3.26 | 5.31 | −0.76 | +12.5% |
| Seasonal naive (7d) | 0.495 | 3.73 | 5.81 | −0.20 | — |
| Naive | 0.495 | 3.73 | 5.90 | +0.55 | −0.1% |

Bias is the mean of (forecast − actual): negative means the model under-forecasts.

Three things worth reading off this table.

**A well-configured classical model wins.** Holt-Winters is the most accurate model and
essentially unbiased (+0.01 units per category-day). The global LightGBM model is less
accurate and systematically under-forecasts (−0.76). With 32 series, under two years of
history and no price or promotion covariates, there is little shared structure for a
global model to exploit — and in this implementation the gradient-boosted model only uses
lags of 28+ days for every horizon day (see *Direct features* below), while Holt-Winters
updates on the most recent day.

**Seasonal naive barely beats plain naive** (0.4948 vs 0.4954). At category level the
weekly cycle is weaker relative to noise than the aggregate series suggests — a 28-day
moving average beats both. Anyone who assumed strong weekly seasonality and skipped the
ladder would have shipped the wrong baseline.

**Accuracy and bias tell different stories.** The direction of error is invisible in any
absolute-error metric, and it matters for stocking: a model that under-forecasts causes
stock-outs even when its WAPE looks acceptable. That is exactly why the stocking decision
below orders to a *quantile* rather than to the point forecast.

## The decision layer

Ordering to the mean forecast is the default habit and it is wrong whenever being short
costs more than being long. The newsvendor critical ratio, cu ÷ (cu + co), gives the
service level the cost structure implies — and the cost-minimising order quantity is that
quantile of the demand distribution, not its mean.

The decision layer uses the DirectGBM quantile forecasts (q50, q75, q90, q95). At a 3:1
understock-to-overstock ratio the critical ratio is **0.75**:

| Policy | Fill rate | Stockout days | Cost/day | vs ordering to mean |
|---|---:|---:|---:|---:|
| Order to mean forecast | 73.3% | 51.5% | 7.28 | — |
| Order to q50 | 72.1% | 55.1% | 7.22 | +0.8% |
| **Order to q75** | **84.3%** | **33.2%** | **5.60** | **+23.0%** |
| Order to q90 | 91.6% | 17.8% | 5.57 | +23.5% |
| Order to q95 | 94.6% | 11.6% | 6.16 | +15.4% |
| *Perfect foresight* | *100%* | *0%* | *0.00* | *+100%* |

Cost/day is total cost divided by the 5,376 held-out category-days, in the relative units
of the assumed cost ratio (understock = 3, overstock = 1).

**Ordering to q75 instead of the mean cuts expected cost 23.0% and lifts fill rate from
73.3% to 84.3% on held-out demand** — with no change to the model. The gain comes from
using the forecast correctly, not from forecasting better.

**Why q90 is marginally cheaper than q75.** Theory says the 0.75 quantile is optimal, and
it would be if the quantile forecasts were calibrated. They are not: on the held-out data,
actual demand fell at or below the forecast q50 / q75 / q90 / q95 on 44.9% / 66.8% /
82.2% / 88.4% of category-days (computed from `reports/backtest_gbm.csv`). Every quantile
is too low, consistent with the model's negative bias, so the model's "q90" behaves like
the true ~0.8 quantile — closer to the 0.75 critical ratio than its "q75". Calibrating the
quantiles (for example with conformal prediction) before ordering is the natural next step.

Sweeping the cost ratio, the recommended quantile — the available quantile nearest the
critical ratio — rises as shortages become more expensive, as theory predicts:

| Understock : overstock | Critical ratio | Recommended quantile |
|---|---:|---|
| 1.5 : 1 | 0.60 | q50 |
| 2 : 1 | 0.67 | q75 |
| 3 : 1 | 0.75 | q75 |
| 5 : 1 | 0.83 | q90 |
| 9 : 1 | 0.90 | q90 |

Because the quantiles under-cover, the policy that is cheapest in hindsight is often one
step higher than the nominal recommendation.

**Measured vs assumed.** The forecast accuracy, fill rates and stockout counts are
measured on held-out data. The cost ratio is an *assumption* — it is a business input, not
something this dataset can reveal, which is why it is swept rather than asserted. No
figure here is a claim about realised business impact.

→ **[reports/DECISION_MEMO.md](reports/DECISION_MEMO.md)** is the write-up for an
operations reader.

---

## Why the results are trustworthy

Most of the work in this repo is spent on not fooling itself.

**Zeros are restored, not dropped.** The transactional export contains a row only for days
a category actually sold, so the zeros are *missing*, not absent. `fill_calendar` reindexes
every series onto a dense daily calendar; 22.8% of category-days are genuine zeros. Omitting
them inflates every accuracy number, because the metric never sees the days the forecast
should have said "nothing."

**Direct features, not recursive.** Every lag and rolling feature is shifted by a minimum
lag of at least the forecast horizon before any rolling window is applied, so no feature
ever uses a value that will not exist at scoring time, and the model's own predictions are
never fed back as inputs. `DirectGBM` trains models for horizon buckets of 7, 14 and 28
days; the backtest currently requests the full 28-day horizon, so all 28 days are scored by
the 28-day bucket. Routing days 1–7 and 8–14 to their own buckets would let the model use
more recent sales and is the first planned improvement.

**Rolling origin, disjoint windows, fresh model per fold.** No random train/test split —
that leaks the future into training and reports an accuracy nobody can reach. Test windows
do not overlap, so no day is scored twice. `model_factory()` is called once per fold on
purpose: reusing a fitted object across origins leaks later data into earlier folds.

**Leakage is tested, not assumed.** A test sets every value after a date to 999 and checks
that every feature on or before that date is unchanged; another checks the smallest lag is
never below the horizon.

**Tweedie, not squared error on a log target.** Daily demand is non-negative, zero-inflated
and right-skewed. Training on `log1p(y)` and inverting estimates the conditional *median*,
so it under-forecasts systematically — retro-transformation bias, which shows up in the
`bias` column and is easy to miss if you only read WAPE. Tweedie models the zero mass and
the positive tail together on the original scale.

**MAPE is deliberately not implemented.** It is undefined on a large share of these
observations, and the usual fixes — adding epsilon, dropping zero days — quietly change
the question being asked. WAPE is used for the portfolio and pinball loss where the
forecast feeds a decision. Scale-free MASE and RMSSE are implemented in
`evaluation/metrics.py` (scaled by training-period seasonal-naive error only) for
cross-series comparison.

**WAPE is pooled, not averaged.** One ratio of summed errors to summed actuals. Averaging
per-series percentages lets a low-volume category with 300% error dominate a portfolio
number it barely contributes to.

---

## Data

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(CC BY-NC-SA 4.0). See [data/README.md](data/README.md) for download instructions.

The demand panel is built with these choices, each of which changes the answer:

| Choice | Why |
|---|---|
| Aggregate to (category, day) | Per-SKU daily series are almost entirely zeros — nothing to learn |
| Key on **purchase** date, not delivery | Demand is when the customer ordered; delivery date smears one day across the logistics lag |
| Exclude `canceled` / `unavailable` | Not demand that was met |
| 2017-01-01 → 2018-08-15 | Pre-2017 is marketplace ramp-up, a different process; the last fortnight has incomplete order capture |
| Drop categories < 365 total units | Too sparse to forecast; 42 of 74 categories are dropped, and the count is reported, not hidden |

Resulting panel: **32 categories × 592 days = 18,944 series-days, 103,841 units,
22.8% zeros.** The panel is committed as `data/panel.parquet`, so the backtest can run
without the raw files.

---

## Running it

```bash
pip install -r requirements.txt
# place the Olist CSVs in data/raw/ — see data/README.md

python scripts/00_prepare.py     # build and validate the demand panel
python scripts/01_backtest.py    # full ladder through rolling-origin backtest (~1 minute)
python scripts/02_decision.py    # newsvendor policy comparison + sensitivity
python scripts/03_report.py      # figures

pytest                           # 40 tests
```

Seeds are fixed, so repeated runs in the same environment produce identical numbers.
Package versions are not pinned; newer library versions can shift results slightly
(for example, DirectGBM WAPE 0.4329 vs 0.4326), while the other models reproduce exactly.

## Layout

```
src/rdf/
  data/schema.py        canonical panel contract — unique_id | ds | y
  data/olist.py         loader: order lines → dense daily demand panel
  features/calendar.py  day-of-week, Fourier, payday and holiday windows
  features/lags.py      lag/rolling features, min_lag enforced per horizon
  models/base.py        Forecaster protocol — fit(train) / predict(h, futr)
  models/baselines.py   Naive · SeasonalNaive · MovingAverage · SeasonalMA · Croston-SBA
  models/ets.py         Holt-Winters, damped additive trend, additive weekly seasonality
  models/gbm.py         DirectGBM — global, horizon-bucketed, Tweedie + quantile heads
  evaluation/metrics.py WAPE · MAE · RMSE · bias · MASE · RMSSE · pinball · coverage
  evaluation/backtest.py rolling-origin harness
  decision/newsvendor.py critical ratio → order quantity → realised cost
scripts/                00_prepare · 01_backtest · 02_decision · 03_report
tests/                  40 tests, including an explicit leakage test
reports/                metric tables, figures, DECISION_MEMO.md
```

Every model implements the same two-method interface, which is what lets the backtest
treat a two-line seasonal naive and a global gradient booster identically — and what makes
the comparison fair rather than rigged.

---

## Limits

- **One marketplace, two years, one country.** Nothing here should be assumed stationary
  or transferable.
- **Category level, not SKU.** Per-SKU series in this dataset are too sparse to model; the
  method would transfer, the results would not.
- **No price or promotion covariates.** This export has no promotion flag, so a real
  driver of retail demand is simply absent. Adding it is the highest-value extension.
- **Orders approximate demand.** Demand censored by a stockout is unobservable here — an
  item that could not be bought looks like an item nobody wanted.
- **The cost ratio is assumed, not measured.** Every cost figure is conditional on it,
  which is why it is swept across a plausible range rather than fixed.
- **Quantiles are not yet calibrated.** Their coverage is below nominal, so the decision
  layer should be recalibrated before use.
- **Single-period decision.** The newsvendor model ignores lead times, pack sizes, shelf
  life and capacity.
