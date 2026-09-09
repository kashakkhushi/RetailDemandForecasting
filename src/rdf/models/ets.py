from __future__ import annotations
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .base import LocalForecaster, seasonal_naive
class ETS(LocalForecaster):
 """Additive weekly seasonality is defined at zero demand; multiplicative is not."""
 def _fit_one(self,y):
  if len(y)<14 or np.allclose(y,y[0]): raise ValueError("insufficient variation")
  return ExponentialSmoothing(y,trend="add",damped_trend=True,seasonal="add",seasonal_periods=7).fit(optimized=True)
 def _predict_one(self,m,h,hist): return np.maximum(0,m.forecast(h))
