"""Global direct Tweedie LightGBM: direct buckets prevent recursive target leakage."""
from __future__ import annotations
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from ..features.calendar import add_calendar_features
from ..features.lags import build_lag_features, feature_columns
class DirectGBM:
 def __init__(self,horizon:int=28,buckets:tuple[int,...]=(7,14,28),quantiles:tuple[float,...]|None=None,params:dict|None=None): self.horizon=horizon; self.buckets=buckets; self.quantiles=quantiles or (); self.params=params or {}; self.models={}; self.columns={}
 def _bucket(self,h): return next(b for b in self.buckets if h<=b)
 def fit(self,train:pd.DataFrame):
  for b in self.buckets:
   x=build_lag_features(add_calendar_features(train),b); cols=feature_columns(x); x=x.dropna(subset=cols); X=x[cols].copy(); X["unique_id"]=x.unique_id.astype("category")
   common=dict(objective="tweedie",tweedie_variance_power=1.2,n_estimators=200,learning_rate=.05,num_leaves=31,random_state=42,verbosity=-1,**self.params)
   self.models[b]={"point":LGBMRegressor(**common).fit(X,x.y)}; self.columns[b]=cols
   for q in self.quantiles: self.models[b][q]=LGBMRegressor(objective="quantile",alpha=q,n_estimators=200,learning_rate=.05,num_leaves=31,random_state=42,verbosity=-1,**self.params).fit(X,x.y)
  self.train=train.copy(); return self
 def predict(self,h:int,futr:pd.DataFrame)->pd.DataFrame:
  b=self._bucket(h); allx=pd.concat([self.train,futr.assign(y=np.nan)],ignore_index=True).sort_values(["unique_id","ds"]); x=build_lag_features(add_calendar_features(allx),b); x=x.merge(futr[["unique_id","ds"]],on=["unique_id","ds"])
  cols=self.columns[b]; X=x[cols].copy(); X["unique_id"]=x.unique_id.astype("category"); out=x[["unique_id","ds"]].copy(); out["yhat"]=np.maximum(0,self.models[b]["point"].predict(X))
  vals=[]
  for q in self.quantiles: vals.append(np.maximum(0,self.models[b][q].predict(X)))
  if vals:
   vals=np.sort(np.vstack(vals),axis=0)
   for q,v in zip(sorted(self.quantiles),vals): out[f"yhat_q{int(q*100):02d}"]=v
  return out
 def feature_importance(self,bucket:int,top:int=20)->pd.DataFrame:
  m=self.models[bucket]["point"]; return pd.DataFrame({"feature":self.columns[bucket]+["unique_id"],"importance":m.feature_importances_}).sort_values("importance",ascending=False).head(top)
