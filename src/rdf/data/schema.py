"""Canonical demand-panel validation and dense-calendar utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd

class SchemaError(ValueError):
    """Raised when a panel cannot safely enter the forecasting pipeline."""

def validate_panel(df: pd.DataFrame, allow_gaps: bool = False) -> pd.DataFrame:
    """Validate the shared shape; dense dates are required unless explicitly waived."""
    needed = {"unique_id", "ds", "y"}
    if not needed.issubset(df.columns): raise SchemaError(f"Panel needs {needed}; got {set(df.columns)}")
    out = df.copy(); out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    if out[["unique_id", "ds", "y"]].isna().any().any(): raise SchemaError("unique_id, ds, and y must be non-null and parseable")
    if out.duplicated(["unique_id", "ds"]).any(): raise SchemaError("Duplicate (unique_id, ds) rows; aggregate before validation")
    out = out.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    if not allow_gaps:
        for uid, part in out.groupby("unique_id", observed=True):
            if len(part) and not part["ds"].equals(pd.Series(pd.date_range(part.ds.min(), part.ds.max(), freq="D"), index=part.index)):
                raise SchemaError(f"{uid!r} has missing calendar days; call fill_calendar first")
    return out

def fill_calendar(df: pd.DataFrame, fill_value: float = 0.0, start: object = None, end: object = None) -> pd.DataFrame:
    """Restore unobserved zero-sale days; exports omit them but the model must learn them."""
    base = validate_panel(df, allow_gaps=True); pieces=[]
    for uid, part in base.groupby("unique_id", observed=True, sort=False):
        lo=pd.Timestamp(start) if start is not None else part.ds.min(); hi=pd.Timestamp(end) if end is not None else part.ds.max()
        dates=pd.date_range(lo, hi, freq="D"); x=part.set_index("ds").reindex(dates)
        x["unique_id"]=uid; x["y"]=x["y"].fillna(fill_value); x.index.name="ds"; pieces.append(x.reset_index()[["unique_id","ds","y"]])
    return validate_panel(pd.concat(pieces, ignore_index=True), allow_gaps=False) if pieces else base

def panel_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise intermittency first because it determines valid metrics and model choices."""
    x=validate_panel(df); g=x.groupby("unique_id", observed=True).y
    out=g.agg(n_days="size", total="sum", mean="mean", std="std"); out["zero_share"]=g.apply(lambda s: (s==0).mean()); out["cv"]=out["std"]/out["mean"].replace(0,np.nan)
    return out[["n_days","total","zero_share","mean","std","cv"]].reset_index()
