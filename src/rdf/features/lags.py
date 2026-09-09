"""Leakage-safe target history features."""
from __future__ import annotations
import numpy as np
import pandas as pd
def build_lag_features(df: pd.DataFrame, min_lag: int, lag_offsets: tuple[int,...]=(0,1,6,7,13,14,20,21), windows: tuple[int,...]=(7,14,28)) -> pd.DataFrame:
    """Shift before rolling so every feature at t ends no later than t-min_lag."""
    if min_lag<1: raise ValueError("min_lag must be >=1")
    x=df.sort_values(["unique_id","ds"]).copy(); g=x.groupby("unique_id",observed=True).y
    for off in sorted(set(lag_offsets)): x[f"lag_{min_lag+off}"]=g.shift(min_lag+off).astype(float)
    shifted=g.shift(min_lag)
    for w in windows: x[f"roll_mean_{w}"]=shifted.groupby(x.unique_id,observed=True).transform(lambda s:s.rolling(w,min_periods=1).mean()).astype(float); x[f"roll_std_{w}"]=shifted.groupby(x.unique_id,observed=True).transform(lambda s:s.rolling(w,min_periods=2).std()).astype(float)
    for weeks in range(1,5): x[f"same_weekday_{weeks}w"]=g.shift(min_lag+7*(weeks-1)).astype(float)
    return x.replace({pd.NA:np.nan})
def feature_columns(df: pd.DataFrame, extra: list[str]|None=None) -> list[str]:
    """Return only safe engineered inputs, never labels, ids, or raw timestamps."""
    excluded={"unique_id","ds","y"}; cols=[c for c in df.columns if c not in excluded]
    return cols + [c for c in (extra or []) if c not in cols and c not in excluded]
