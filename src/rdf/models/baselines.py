from __future__ import annotations
import numpy as np
from .base import LocalForecaster, seasonal_naive
class Naive(LocalForecaster):
 def _predict_one(self,m,h,hist): return np.repeat(hist[-1] if len(hist) else 0.,h)
class SeasonalNaive(LocalForecaster):
 def _predict_one(self,m,h,hist): return seasonal_naive(hist,h,7)
class MovingAverage(LocalForecaster):
 def _predict_one(self,m,h,hist): return np.repeat(np.mean(hist[-28:]) if len(hist) else 0.,h)
class SeasonalMovingAverage(LocalForecaster):
 def _predict_one(self,m,h,hist):
  a=hist[-28:]; return np.array([np.mean(a[i::7]) if len(a[i::7]) else 0. for i in range(7)]* ((h+6)//7))[:h]
class CrostonSBA(LocalForecaster):
 def __init__(self,alpha:float=.1): super().__init__(); self.alpha=alpha
 def _fit_one(self,y):
  nz=np.flatnonzero(y>0)
  if not len(nz): return 0.
  z=y[nz[0]]; p=max(1,nz[0]+1); last=nz[0]
  for i in nz[1:]: z=self.alpha*y[i]+(1-self.alpha)*z; p=self.alpha*(i-last)+(1-self.alpha)*p; last=i
  return (1-self.alpha/2)*z/p
 def _predict_one(self,m,h,hist): return np.repeat(m,h)
