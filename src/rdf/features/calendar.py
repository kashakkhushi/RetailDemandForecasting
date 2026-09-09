"""Features known before the forecast origin, hence safe at every horizon."""
from __future__ import annotations
import numpy as np
import pandas as pd
def add_calendar_features(df: pd.DataFrame, fourier_order: int=3) -> pd.DataFrame:
    """Add Brazil-relevant calendar proxies; holidays are windows because dates move yearly."""
    x=df.copy(); d=pd.to_datetime(x.ds); x["dayofweek"]=d.dt.dayofweek; x["is_weekend"]=(x.dayofweek>=5).astype(int); x["month"]=d.dt.month; x["day"]=d.dt.day; x["weekofyear"]=d.dt.isocalendar().week.astype(int)
    x["is_month_start"]=d.dt.is_month_start.astype(int); x["is_month_end"]=d.dt.is_month_end.astype(int); x["near_payday"]=(d.dt.day.isin([1,2,3,4,5,28,29,30,31])).astype(int)
    doy=d.dt.dayofyear; 
    for k in range(1,fourier_order+1): x[f"sin_year_{k}"]=np.sin(2*np.pi*k*doy/365.25); x[f"cos_year_{k}"]=np.cos(2*np.pi*k*doy/365.25)
    md=d.dt.strftime("%m-%d"); x["is_holiday_window"]=(md.between("11-19","12-02")|md.between("12-10","12-25")|md.between("12-26","01-05")|md.between("04-25","05-15")).astype(int)
    return x
