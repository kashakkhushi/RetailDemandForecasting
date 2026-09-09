from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
out=Path('reports/figures'); out.mkdir(parents=True,exist_ok=True); m=pd.read_csv('reports/metrics_models.csv'); ax=m.plot.bar(x='model',y='wape',legend=False); ax.axhline(m.loc[m.model=='SeasonalNaive','wape'].iloc[0],color='r',ls='--'); plt.tight_layout(); plt.savefig(out/'ladder_wape.png',dpi=150); plt.close()
r=pd.read_csv('reports/backtest_gbm.csv'); top=r.groupby('unique_id').y.sum().nlargest(6).index; fig,ax=plt.subplots();
for u in top: x=r[r.unique_id==u]; ax.plot(pd.to_datetime(x.ds),x.y,label=u,alpha=.6); ax.plot(pd.to_datetime(x.ds),x.yhat,ls='--',alpha=.6)
ax.legend(fontsize=6,ncol=2); plt.tight_layout(); plt.savefig(out/'actual_forecast_top_categories.png',dpi=150); plt.close()
q=[c for c in r if c.startswith('yhat_q')]; costs=[]
from rdf.decision.newsvendor import NewsvendorCosts,policy_cost
for c in q: costs.append((c,policy_cost(r.y,r[c],NewsvendorCosts())['total_cost']))
plt.plot([x[0] for x in costs],[x[1] for x in costs],marker='o'); plt.tight_layout(); plt.savefig(out/'cost_vs_quantile.png',dpi=150); plt.close()
