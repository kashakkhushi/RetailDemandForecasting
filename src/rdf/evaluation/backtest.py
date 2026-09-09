from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .metrics import wape,mae,rmse,bias
@dataclass(frozen=True)
class BacktestConfig: horizon:int=28; n_folds:int=6; step:int|None=None; initial_train:int=270; mode:str="expanding"
def make_cutoffs(df:pd.DataFrame,cfg:BacktestConfig)->list[pd.Timestamp]:
 dates=pd.DatetimeIndex(sorted(pd.to_datetime(df.ds).unique())); step=cfg.step or cfg.horizon; last=len(dates)-cfg.horizon-1; first=last-(cfg.n_folds-1)*step
 if first<cfg.initial_train-1: raise ValueError("Not enough history for requested complete folds")
 return [dates[first+i*step] for i in range(cfg.n_folds)]
def rolling_origin(df,model_factory,cfg:BacktestConfig)->pd.DataFrame:
 out=[]
 for fold,c in enumerate(make_cutoffs(df,cfg)):
  train=df[df.ds<=c]; test=df[(df.ds>c)&(df.ds<=c+pd.Timedelta(days=cfg.horizon))]; model=model_factory().fit(train); pred=model.predict(cfg.horizon,test.drop(columns="y")); z=test.merge(pred,on=["unique_id","ds"],how="left")
  if z.yhat.isna().any(): raise ValueError(f"Missing forecasts in fold {fold}")
  z["fold"]=fold; z["cutoff"]=c; out.append(z)
 return pd.concat(out,ignore_index=True)
def score(results,history=None,by=None):
 keys=[by] if isinstance(by,str) else (by or []); g=results.groupby(keys,observed=True) if keys else [((),results)]
 rows=[]
 for k,x in g:
  row=dict(zip(keys,(k,) if not isinstance(k,tuple) else k)); row|={"wape":wape(x.y,x.yhat),"mae":mae(x.y,x.yhat),"rmse":rmse(x.y,x.yhat),"bias":bias(x.y,x.yhat),"n":len(x)}; rows.append(row)
 return pd.DataFrame(rows)
