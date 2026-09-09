"""Olist loader: purchase-day line-item demand is the observable demand proxy."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .schema import fill_calendar, validate_panel

REQUIRED=("olist_orders_dataset.csv","olist_order_items_dataset.csv","olist_products_dataset.csv","product_category_name_translation.csv")
def load_demand_panel(csv_dir: str|Path|None=None, level: str="category", start: str="2017-01-01", end: str="2018-08-15", min_total_units: int=365) -> pd.DataFrame:
    """Build dense category-day units; purchase date avoids smearing demand over delivery lag."""
    if level != "category": raise ValueError("Only category level is supported by this portfolio")
    root=Path(csv_dir or "data/raw"); missing=[f for f in REQUIRED if not (root/f).exists()]
    if missing: raise FileNotFoundError(f"Missing {missing} in {root}. See data/README.md for the Kaggle download step.")
    orders=pd.read_csv(root/REQUIRED[0], parse_dates=["order_purchase_timestamp"])
    items=pd.read_csv(root/REQUIRED[1]); products=pd.read_csv(root/REQUIRED[2]); trans=pd.read_csv(root/REQUIRED[3])
    x=items.merge(orders[["order_id","order_status","order_purchase_timestamp"]], on="order_id").merge(products[["product_id","product_category_name"]],on="product_id").merge(trans,on="product_category_name",how="left")
    x=x[~x.order_status.isin(["canceled","unavailable"])].copy(); x["ds"]=x.order_purchase_timestamp.dt.normalize()
    x=x[(x.ds>=pd.Timestamp(start))&(x.ds<=pd.Timestamp(end))]; x["unique_id"]=x.product_category_name_english.fillna(x.product_category_name).fillna("unknown")
    panel=x.groupby(["unique_id","ds"],as_index=False).size().rename(columns={"size":"y"})
    totals=panel.groupby("unique_id").y.sum(); keep=totals[totals>=min_total_units].index
    panel=panel[panel.unique_id.isin(keep)]; panel=fill_calendar(panel,start=start,end=end); panel.attrs["dropped_categories"]=int((~totals.index.isin(keep)).sum())
    return validate_panel(panel)
