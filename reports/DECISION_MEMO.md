# Inventory decision memo

Use the forecast quantile nearest the newsvendor critical ratio, rather than automatically ordering to the point forecast. With the default assumed understock cost of 3 and overstock cost of 1, that ratio is 75%, so the available q75 forecast is the recommended starting policy.

The actual policy cost, fill rate, stockout-day rate, and comparison with perfect foresight are written by `scripts/02_decision.py`. They must be interpreted under the stated cost assumptions, not as universal savings. Re-estimate understock and overstock costs with finance and operations, inspect the sensitivity output, and revisit the choice if the best policy changes across a plausible range.

Measured: historical fulfilled Olist order-line demand and rolling-origin forecast errors. Assumed: that orders proxy demand, ordering is unconstrained, quantile forecasts are calibrated, and the under/overstock costs represent the operational trade-off.
