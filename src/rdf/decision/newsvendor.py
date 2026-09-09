from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
@dataclass(frozen=True)
class NewsvendorCosts:
 understock:float=3.; overstock:float=1.
 @property
 def critical_ratio(self)->float: return self.understock/(self.understock+self.overstock)
def policy_cost(actual,order_qty,costs:NewsvendorCosts):
 a=np.asarray(actual,float); q=np.maximum(0,np.asarray(order_qty,float)); short=np.maximum(a-q,0); over=np.maximum(q-a,0); cost=short.sum()*costs.understock+over.sum()*costs.overstock
 return {"units_short":short.sum(),"units_over":over.sum(),"fill_rate":1-short.sum()/a.sum() if a.sum() else 1.,"stockout_day_rate":np.mean(short>0),"total_cost":cost,"cost_per_day":cost/len(a) if len(a) else np.nan}
def recommended_quantile(costs,available): return min(available,key=lambda q:abs(q-costs.critical_ratio))
def compare_policies(results,costs):
 rows=[]; choices={"point":"yhat","perfect":"y"}|{c:c for c in results if c.startswith("yhat_q")}
 for name,col in choices.items(): rows.append({"policy":name,**policy_cost(results.y,results[col],costs)})
 return pd.DataFrame(rows)
def sensitivity(results,ratios=(1.5,2,3,5,9)):
 rows=[]
 for r in ratios:
  c=NewsvendorCosts(r,1); best=recommended_quantile(c,[float(x[6:])/100 for x in results if x.startswith("yhat_q")]); rows.append({"under_over_ratio":r,"critical_ratio":c.critical_ratio,"recommended_quantile":best})
 return pd.DataFrame(rows)
