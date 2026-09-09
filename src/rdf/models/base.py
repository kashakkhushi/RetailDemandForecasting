from __future__ import annotations
from typing import Protocol
import numpy as np
import pandas as pd
class Forecaster(Protocol):
    def fit(self, train: pd.DataFrame) -> "Forecaster": ...
    def predict(self, h: int, futr: pd.DataFrame) -> pd.DataFrame: ...
def seasonal_naive(hist: np.ndarray, steps: int, m: int=7) -> np.ndarray:
    """Repeat the latest season; gracefully handles short histories."""
    a=np.asarray(hist,dtype=float); return np.maximum(0, np.resize(a[-min(len(a),m):] if len(a) else np.array([0.]),steps))
class LocalForecaster:
    """Per-series base that exposes failures instead of silently dropping series."""
    def __init__(self): self.models={}; self.histories={}; self.n_fallbacks=0
    def fit(self, train: pd.DataFrame):
        self.histories={u:g.sort_values("ds").y.to_numpy(float) for u,g in train.groupby("unique_id",observed=True)}
        for u,a in self.histories.items():
            try: self.models[u]=self._fit_one(a)
            except Exception: self.models[u]=None; self.n_fallbacks+=1
        return self
    def _fit_one(self, y: np.ndarray): return y
    def _predict_one(self, model: object, h:int, hist:np.ndarray)->np.ndarray: return seasonal_naive(hist,h)
    def predict(self,h:int,futr:pd.DataFrame)->pd.DataFrame:
        rows=[]
        for u,g in futr.groupby("unique_id",observed=True):
            hist=self.histories.get(u,np.array([])); pred=seasonal_naive(hist,len(g)) if self.models.get(u) is None else self._predict_one(self.models[u],len(g),hist)
            z=g[["unique_id","ds"]].copy(); z["yhat"]=np.maximum(0,np.asarray(pred,float)); rows.append(z)
        return pd.concat(rows,ignore_index=True)
