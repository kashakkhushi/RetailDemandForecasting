Retail Demand Forecasting & Inventory Planning

Forecasting daily demand for a Brazilian e-commerce marketplace, and turning the forecast into an order quantity.

A forecast is not a decision. This project runs a full baseline ladder over 32 product categories, picks a winner honestly, and then answers the question an operations team actually asks: given this forecast, how much should we order?

The answer is not "order the forecast."

Results

Rolling-origin backtest, 28-day horizon, 6 disjoint origins, 5,376 scored series-days per model.

Model	WAPE	MASE	RMSSE	Bias	vs seasonal naive
DirectGBM — global, Tweedie, direct horizons	0.399	1.051	0.830	+0.61	+19.4%
Holt-Winters (ETS)	0.406	1.071	0.855	−0.01	+17.9%
Croston-SBA	0.416	1.097	0.873	+0.34	+15.9%
Moving average (28d)	0.422	1.113	0.880	+0.13	+14.7%
Seasonal moving average (7×4)	0.431	1.136	0.883	+0.13	+12.9%
Seasonal naive (7d)	0.495	1.305	1.031	+0.20	—
Naive	0.495	1.307	1.047	−0.55	−0.1%

Two things worth reading off this table.

Seasonal naive barely beats plain naive (0.4948 vs 0.4954). At category level the weekly cycle is weaker relative to noise than the aggregate series suggests — a 28-day moving average beats both. Anyone who assumed strong weekly seasonality and skipped the ladder would have shipped the wrong baseline.

The most accurate model is the most biased. DirectGBM wins on WAPE but carries a +0.61 bias — it systematically under-forecasts. Holt-Winters is nearly unbiased at −0.01. That trade-off is invisible in any absolute-error metric, and it is exactly why the stocking decision below orders to a quantile rather than to the point forecast.

The decision layer

Ordering to the mean forecast is the default habit and it is wrong whenever being short costs more than being long. The newsvendor critical ratio, cu / (cu + co), gives the service level the cost structure implies — and the cost-minimising order quantity is that quantile of the demand distribution, not its mean.

At a 3:1 understock-to-overstock ratio the critical ratio is 0.75:

Policy	Fill rate	Stockout days	Cost/day	vs ordering to mean
Order to mean forecast	76.0%	47.8%	6.61	—
Order to q50	77.0%	48.5%	6.40	+3.3%
Order to q75	87.3%	28.4%	5.36	+18.9%
Order to q90	93.0%	16.2%	5.61	+15.1%
Order to q95	96.0%	9.7%	6.43	+2.8%
Perfect foresight	100%	0%	0.00	+100%

Ordering to q75 instead of the mean cuts expected cost 18.9% and lifts fill rate from 76.0% to 87.3% on held-out demand — with no change to the model. The gain comes from using the forecast correctly, not from forecasting better.

The theory holds empirically: sweeping the cost ratio, the cost-minimising policy tracks the critical ratio.

Understock : overstock	Critical ratio	Best policy	Fill rate	Saving vs mean
1.5 : 1	0.60	q50	77.0%	+2.8%
2 : 1	0.67	q75	87.3%	+8.3%
3 : 1	0.75	q75	87.3%	+18.9%
5 : 1	0.83	q90	93.0%	+34.8%
9 : 1	0.90	q95	96.0%	+52.8%

Measured vs assumed. The forecast accuracy, fill rates and stockout counts are measured on held-out data. The cost ratio is an assumption — it is a business input, not something this dataset can reveal, which is why it is swept rather than asserted. No figure here is a claim about realised business impact.

→ reports/DECISION_MEMO.md is the write-up for an operations reader.

Why the results are trustworthy

Most of the work in this repo is spent on not fooling itself.

Zeros are restored, not dropped. The transactional export contains a row only for days a category actually sold, so the zeros are missing, not absent. fill_calendar reindexes every series onto a dense daily calendar; 23% of category-days are genuine zeros. Omitting them inflates every accuracy number, because the metric never sees the days the forecast should have said "nothing."

Direct multi-horizon, not recursive. One model per horizon bucket, each using only lags that horizon can actually observe (min_lag = bucket upper bound). A recursive model feeds its own predictions back as lags: error compounds across the horizon, and the model is applied to inputs it never saw in training. Nothing here is ever fed a value that will not exist at scoring time.

Rolling origin, disjoint windows, fresh model per fold. No random train/test split — that leaks the future into training and reports an accuracy nobody can reach. Test windows do not overlap, so no day is scored twice. model_factory() is called once per fold on purpose: reusing a fitted object across origins leaks later data into earlier folds.

Scaled metrics use training history only. MASE and RMSSE scale error against in-sample seasonal-naive error computed strictly before each cutoff. Scaling by the test window is a leak, and a common one.

Tweedie, not squared error on a log target. Daily demand is non-negative, zero-inflated and right-skewed. Training on log1p(y) and inverting estimates the conditional median, so it under-forecasts systematically — retro-transformation bias, which shows up in the bias column and is easy to miss if you only read WAPE. Tweedie models the zero mass and the positive tail together on the original scale.

MAPE is deliberately not implemented. It is undefined on a large share of these observations, and the usual fixes — adding epsilon, dropping zero days — quietly change the question being asked. WAPE is used for the portfolio, MASE/RMSSE for cross-series comparability, pinball loss where the forecast feeds a decision.

WAPE is pooled, not averaged. One ratio of summed errors to summed actuals. Averaging per-series percentages lets a low-volume category with 300% error dominate a portfolio number it barely contributes to.

Data

Brazilian E-Commerce Public Dataset by Olist (CC BY-NC-SA 4.0). See data/README.md for download instructions.

The demand panel is built with these choices, each of which changes the answer:

Choice	Why
Aggregate to (category, day)	Per-SKU daily series are almost entirely zeros — nothing to learn
Key on purchase date, not delivery	Demand is when the customer ordered; delivery date smears one day across the logistics lag
Exclude canceled / unavailable	Not demand that was met
2017-01-01 → 2018-08-15	Pre-2017 is marketplace ramp-up, a different process; the last fortnight has incomplete order capture
Drop categories < 365 total units	Too sparse to forecast; the count dropped is reported, not hidden

Resulting panel: 32 categories × 592 days = 18,944 series-days, 22.8% zeros.

Running it
bash
pip install -r requirements.txt
# place the Olist CSVs in data/raw/ — see data/README.md

python scripts/00_prepare.py     # build and validate the demand panel
python scripts/01_backtest.py    # full ladder through rolling-origin backtest
python scripts/02_decision.py    # newsvendor policy comparison + sensitivity
python scripts/03_report.py      # figures

pytest                           # test suite

Seeds are fixed; two runs produce identical numbers.

Layout
src/rdf/
  data/schema.py        canonical panel contract — unique_id | ds | y
  data/olist.py         loader: order lines → dense daily demand panel
  features/calendar.py  day-of-week, Fourier, payday and holiday windows
  features/lags.py      lag/rolling features, min_lag enforced per horizon
  models/base.py        Forecaster protocol — fit(train) / predict(h, futr)
  models/baselines.py   Naive · SeasonalNaive · MovingAverage · SeasonalMA · Croston-SBA
  models/ets.py         Holt-Winters, additive seasonality
  models/gbm.py         DirectGBM — global, horizon-bucketed, Tweedie + quantile heads
  evaluation/metrics.py WAPE · MASE · RMSSE · pinball · bias
  evaluation/backtest.py rolling-origin harness
  decision/newsvendor.py critical ratio → order quantity → realised cost
scripts/                00_prepare · 01_backtest · 02_decision · 03_report
tests/                  including an explicit leakage test
reports/                metric tables, figures, DECISION_MEMO.md

Every model implements the same two-method interface, which is what lets the backtest treat a two-line seasonal naive and a global gradient booster identically — and what makes the comparison fair rather than rigged.

Limits
One marketplace, two years, one country. Nothing here should be assumed stationary or transferable.
Category level, not SKU. Per-SKU series in this dataset are too sparse to model; the method would transfer, the results would not.
No price or promotion covariates. This export has no promotion flag, so a real driver of retail demand is simply absent. Adding it is the highest-value extension.
Orders approximate demand. Demand censored by a stockout is unobservable here — an item that could not be bought looks like an item nobody wanted.
The cost ratio is assumed, not measured. Every currency figure is conditional on it, which is why it is swept across a plausible range rather than fixed.
