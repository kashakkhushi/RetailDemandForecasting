"""Metrics. MAPE is intentionally absent: zero-demand days make it undefined and dropping them changes the business question."""
from __future__ import annotations
import numpy as np
def mae(y,yh): return float(np.mean(np.abs(np.asarray(y)-np.asarray(yh))))
def rmse(y,yh): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(yh))**2)))
def bias(y,yh): return float(np.mean(np.asarray(yh)-np.asarray(y)))
def wape(y,yh):
 d=np.abs(np.asarray(y)).sum(); return float(np.abs(np.asarray(y)-np.asarray(yh)).sum()/d) if d else np.nan
def smape(y,yh):
 y=np.asarray(y); h=np.asarray(yh); d=np.abs(y)+np.abs(h); return float(np.mean(np.divide(2*np.abs(y-h),d,out=np.zeros_like(d,dtype=float),where=d!=0)))
def mase(y,yh,train,m=7):
 t=np.asarray(train); scale=np.mean(np.abs(t[m:]-t[:-m])) if len(t)>m else np.nan; return mae(y,yh)/scale if scale and not np.isnan(scale) else np.nan
def rmsse(y,yh,train,m=7):
 t=np.asarray(train); scale=np.mean((t[m:]-t[:-m])**2) if len(t)>m else np.nan; return rmse(y,yh)/np.sqrt(scale) if scale and not np.isnan(scale) else np.nan
def pinball_loss(y,qhat,q):
 e=np.asarray(y)-np.asarray(qhat); return float(np.mean(np.maximum(q*e,(q-1)*e)))
def coverage(y,lo,hi): return float(np.mean((np.asarray(y)>=np.asarray(lo))&(np.asarray(y)<=np.asarray(hi))))
def summarise(y,yhat,train=None):
 d={"mae":mae(y,yhat),"rmse":rmse(y,yhat),"bias":bias(y,yhat),"wape":wape(y,yhat),"smape":smape(y,yhat)}
 if train is not None: d|={"mase":mase(y,yhat,train),"rmsse":rmsse(y,yhat,train)}
 return d
